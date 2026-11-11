#!/usr/bin/env python3
"""方案三 OmniGuard GRPO。

每步用当前策略采样 5 条，再按组内 advantage 更新。参考策略是同一份 SFT
adapter，挂成名为 ref 的第二套 LoRA，更新时不动它。

传入 --ipc-dir 时，采样交给另一张卡上的 grpo_sample.py（完整权重）。
本进程只做 ZeRO-3 更新，并在每步之后把 LoRA 写到 policy_latest。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

TVP = Path(__file__).resolve().parent
sys.path.insert(0, str(TVP))

try:
    import fla  # noqa: F401
    import transformers.utils.import_utils as _tfu

    _tfu.is_flash_linear_attention_available = lambda: True
except ImportError:
    pass

import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from tokens import add_special_tokens

MODEL = "/workspace/zzy/modelscope/Qwen3.8-27B"
SYSTEM = "You are a helpful assistant that can understand images and reason with visual primitives."
MIN_PIXELS = 12544
MAX_PIXELS = 2007040
MAX_LEN = 8192


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-coeff", type=float, default=0.04)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ipc-dir", type=Path, default=None)
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--start-cursor", type=int, default=0)
    p.add_argument("--start-seen", type=int, default=0)
    p.add_argument("--reward", choices=("score", "f1"), default="score")
    return p.parse_args()


def ds_config(world_size: int, total_steps: int, lr: float, warmup: int) -> dict:
    gas = 5
    zero = {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "stage3_gather_16bit_weights_on_model_save": True,
        "overlap_comm": False,
    }
    if world_size == 1:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
        zero["stage3_param_persistence_threshold"] = 0
        zero["stage3_max_live_parameters"] = 2e9
        zero["stage3_prefetch_bucket_size"] = 5e7
    return {
        "train_batch_size": gas * world_size,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": gas,
        "zero_optimization": zero,
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_min_lr": 0.0,
                "warmup_max_lr": lr,
                "warmup_num_steps": warmup,
            },
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 1000,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
    }


def install_empty_skip() -> None:
    import peft.peft_model as peft_model

    orig = peft_model.set_peft_model_state_dict

    def _set(model, state, *args, **kwargs):
        state = {k: v for k, v in state.items() if getattr(v, "numel", lambda: 1)() > 0}
        return orig(model, state, *args, **kwargs)

    peft_model.set_peft_model_state_dict = _set


def zero_lora_dropout(model) -> None:
    import torch.nn as nn

    for mod in model.modules():
        drop = getattr(mod, "lora_dropout", None)
        if isinstance(drop, nn.Dropout):
            drop.p = 0.0
        elif isinstance(drop, nn.ModuleDict):
            for item in drop.values():
                if isinstance(item, nn.Dropout):
                    item.p = 0.0


def load_model(adapter: Path, config: dict, policy: Path | None = None):
    from peft import PeftModel

    install_empty_skip()
    global _DS_CONFIG
    _DS_CONFIG = HfDeepSpeedConfig(config)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    add_special_tokens(processor.tokenizer)
    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = MIN_PIXELS
        ip.max_pixels = MAX_PIXELS
    try:
        from transformers.image_utils import SizeDict

        ip.size = SizeDict(shortest_edge=MIN_PIXELS, longest_edge=MAX_PIXELS)
    except Exception:
        pass
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, trust_remote_code=True
    )
    policy_path = policy if policy is not None else adapter
    model = PeftModel.from_pretrained(base, str(policy_path), is_trainable=True)
    model.load_adapter(str(adapter), adapter_name="ref")
    model.set_adapter("default")
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad = False
    zero_lora_dropout(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_ref = sum(1 for n, _ in model.named_parameters() if ".ref." in n)
    return processor, model, n_train, n_ref


def line_offsets(path: Path) -> list[int]:
    offsets = []
    with path.open("rb") as fin:
        while True:
            pos = fin.tell()
            line = fin.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return offsets


def read_row(path: Path, offset: int) -> dict:
    with path.open("rb") as fin:
        fin.seek(offset)
        return json.loads(fin.readline().decode("utf-8"))


def build_prompt(processor, image_path: Path, question: str, device: torch.device):
    from PIL import Image

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }


def load_reward(name: str = "score"):
    import importlib.util

    path = TVP.parent / "sft_ship_lora" / "grpo_train.py"
    if not path.is_file():
        path = TVP.parent.parent / "sft_ship_lora" / "grpo_train.py"
    spec = importlib.util.spec_from_file_location("ship_grpo_reward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if name == "f1":
        return mod.reward_of_f1
    if name == "score":
        return mod.reward_of
    raise ValueError(f"unknown reward {name}")


def sample_five(engine, inputs, device: torch.device, rank: int) -> list[torch.Tensor]:
    """单卡先把完整权重搬回 NPU 再采样。多卡仍各卡 generate，训练只用 rank 0 的 token。"""
    module = engine.module
    module.set_adapter("default")
    module.eval()
    module.config.use_cache = True
    text_cfg = getattr(module.config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = True
    clear_rope(module)
    prompt_len = int(inputs["input_ids"].shape[1])
    world = dist.get_world_size()
    local: list[torch.Tensor] = []
    gathered = [p for p in module.parameters() if hasattr(p, "ds_id")]
    holder = (
        deepspeed.zero.GatheredParameters(gathered)
        if world == 1 and gathered
        else contextlib.nullcontext()
    )
    if rank == 0 and world == 1:
        print(f"[rollout] gather full weights n={len(gathered)}", flush=True)
    with holder:
        if rank == 0 and world == 1:
            alloc = torch.npu.memory_allocated() / 1024**3
            print(f"[rollout] resident hbm={alloc:.1f}GB", flush=True)
        for i in range(5):
            if rank == 0:
                print(f"[rollout] sample {i + 1}/5", flush=True)
            with torch.no_grad():
                out = module.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    synced_gpus=world > 1,
                )
            local.append(out[0, prompt_len:].detach().to(device=device, dtype=torch.long))
            clear_rope(module)
            del out
    if world == 1:
        rows = [t.view(1, -1) for t in local]
    else:
        if rank == 0:
            lengths = torch.tensor([int(t.shape[0]) for t in local], device=device, dtype=torch.long)
        else:
            lengths = torch.empty(5, device=device, dtype=torch.long)
        dist.broadcast(lengths, src=0)
        rows = []
        for g in range(5):
            n = int(lengths[g].item())
            buf = local[g][:n] if rank == 0 else torch.empty(n, dtype=torch.long, device=device)
            dist.broadcast(buf, src=0)
            rows.append(buf.view(1, -1))
    module.config.use_cache = False
    if text_cfg is not None and hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = False
    module.train()
    return rows


def patch_zero3_conv() -> None:
    """解码时线性注意力直接读 conv1d.weight。ZeRO-3 下它是一维分片，squeeze 会失败。"""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    cls = modeling.Qwen3_5GatedDeltaNet
    if getattr(cls, "_zero3_conv_patched", False):
        return
    orig = cls.forward

    def forward(self, *args, **kwargs):
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        params = [
            p for p in (self.conv1d.weight, self.conv1d.bias)
            if p is not None and hasattr(p, "ds_id")
        ]
        if not params or all(
            getattr(p, "ds_status", None) == ZeroParamStatus.AVAILABLE for p in params
        ):
            return orig(self, *args, **kwargs)
        with deepspeed.zero.GatheredParameters(params):
            return orig(self, *args, **kwargs)

    cls.forward = forward
    cls._zero3_conv_patched = True


def clear_rope(model) -> None:
    for mod in model.modules():
        if getattr(mod, "rope_deltas", None) is not None:
            mod.rope_deltas = None


def nll(engine, inputs, new_ids) -> torch.Tensor:
    prompt = inputs["input_ids"]
    prompt_len = prompt.shape[1]
    full = torch.cat([prompt, new_ids], dim=1)
    labels = full.clone()
    labels[:, :prompt_len] = -100
    attn = torch.ones_like(full)
    fwd = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
    }
    mm = inputs.get("mm_token_type_ids")
    if mm is not None:
        pad = torch.zeros(mm.shape[0], new_ids.shape[1], dtype=mm.dtype, device=mm.device)
        fwd["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
    clear_rope(engine.module)
    out = engine(input_ids=full, attention_mask=attn, labels=labels, **fwd)
    return out.loss


def write_state(path: Path, cursor: int, seen: int, rank: int) -> None:
    if rank != 0:
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"cursor": cursor, "seen": seen}), encoding="utf-8")
    tmp.replace(path)


def fetch_external(
    ipc: Path, request: dict, device: torch.device, rank: int
) -> tuple[list[torch.Tensor] | None, dict]:
    """NPU 0 上的采样进程写回答。两边都等文件，不在等待期间做集合通信。"""
    rid = f"{int(request['id']):06d}"
    resp_path = ipc / f"{rid}.response.json"
    if rank == 0:
        ipc.mkdir(parents=True, exist_ok=True)
        tmp = ipc / f"{rid}.request.tmp"
        tmp.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        tmp.rename(ipc / f"{rid}.request.json")
        print(f"[rollout] request id={rid} adapter={request['adapter']}", flush=True)
    started = time.time()
    announced = 0.0
    while not resp_path.is_file():
        time.sleep(2)
        if rank == 0 and time.time() - announced >= 30:
            announced = time.time()
            print(
                f"[rollout] waiting sampler id={rid} ({time.time() - started:.0f}s)",
                flush=True,
            )
    payload = json.loads(resp_path.read_text(encoding="utf-8"))
    if payload.get("error") or "token_ids" not in payload:
        if rank == 0:
            print(f"[skip] sampler error id={rid} {payload.get('error', 'no tokens')}", flush=True)
        return None, payload
    rows = []
    for ids in payload["token_ids"]:
        if not ids:
            if rank == 0:
                print(f"[skip] empty completion id={rid}", flush=True)
            return None, payload
        rows.append(torch.tensor(ids, dtype=torch.long, device=device).view(1, -1))
    return rows, payload


def save_policy(engine, dest: Path, rank: int) -> None:
    engine.module.set_adapter("default")
    params = [p for n, p in engine.module.named_parameters() if "lora_" in n and p.requires_grad]
    os.makedirs(dest, exist_ok=True)
    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        if rank == 0:
            engine.module.save_pretrained(dest, selected_adapters=["default"])
            print(f"[save] {dest}", flush=True)


def main() -> int:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    deepspeed.init_distributed()
    rank = dist.get_rank()
    is_main = rank == 0

    config = ds_config(world, args.max_steps, args.lr, args.warmup_steps)
    if is_main:
        print(
            f"[init] loading policy={args.policy or args.adapter} ref={args.adapter} "
            f"world={world} resume_cursor={args.start_cursor} resume_seen={args.start_seen} "
            f"reward={args.reward}",
            flush=True,
        )
    processor, model, n_train, n_ref = load_model(args.adapter, config, args.policy)
    if is_main:
        print(f"[init] trainable={n_train/1e6:.1f}M ref_tensors={n_ref}", flush=True)
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    patch_zero3_conv()
    engine.train()

    reward_of = load_reward(args.reward)
    offsets = line_offsets(args.jsonl)
    order = list(range(len(offsets)))
    random.Random(args.seed).shuffle(order)
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.ipc_dir is not None:
            args.ipc_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[data] {len(offsets)} prompts from {args.jsonl} "
            f"online group=5 ipc={args.ipc_dir}",
            flush=True,
        )
    dist.barrier()

    metrics = args.output_dir / "metrics.jsonl"
    state_path = args.output_dir / "state.json"
    seen = args.start_seen
    cursor = args.start_cursor
    current_adapter = args.policy or args.adapter
    while seen < args.max_steps and cursor < len(order):
        row = read_row(args.jsonl, offsets[order[cursor]])
        cursor += 1
        image = (args.image_root / row["image"]).resolve()
        if not image.is_file():
            continue
        gt = int(row["answer"])
        boxes = row.get("boxes") or []
        t0 = time.time()
        if is_main:
            print(f"[rollout] gt={gt} boxes={len(boxes)} image={row['image']}", flush=True)
        inputs = build_prompt(processor, image, row["question"], device)
        sample_meta = {}
        if args.ipc_dir is not None:
            comp, sample_meta = fetch_external(
                args.ipc_dir,
                {
                    "id": cursor,
                    "image": str(image),
                    "question": row["question"],
                    "adapter": str(current_adapter),
                },
                device,
                rank,
            )
            if comp is None:
                del inputs
                write_state(state_path, cursor, seen, rank)
                continue
        else:
            comp = sample_five(engine, inputs, device, rank)
        texts = [
            processor.tokenizer.decode(ids[0], skip_special_tokens=False) for ids in comp
        ]
        scored = [reward_of(text, gt, boxes) for text in texts]
        rewards = torch.tensor([float(item["reward"]) for item in scored], dtype=torch.float32)
        std = float(rewards.std(unbiased=False))
        if std < 1e-6:
            if is_main:
                print(
                    f"[skip] flat reward={float(rewards[0]):.3f} "
                    f"preds={[item['pred'] for item in scored]} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            del inputs, comp
            torch.npu.empty_cache()
            write_state(state_path, cursor, seen, rank)
            continue
        prompt_len = inputs["input_ids"].shape[1]
        if prompt_len + max(int(t.shape[1]) for t in comp) > MAX_LEN:
            if is_main:
                print(f"[skip] long seq image={row['image']} prompt={prompt_len}", flush=True)
            del inputs, comp
            write_state(state_path, cursor, seen, rank)
            continue
        adv = ((rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-8)).tolist()

        seen += 1
        loss_acc = 0.0
        kl_acc = 0.0
        for g in range(5):
            engine.module.set_adapter("ref")
            with torch.no_grad():
                ref_lp = -nll(engine, inputs, comp[g])
            engine.module.set_adapter("default")
            policy_lp = -nll(engine, inputs, comp[g])
            kl = policy_lp - ref_lp.detach()
            loss = -(adv[g] * policy_lp) + args.kl_coeff * kl
            engine.backward(loss)
            engine.step()
            loss_acc += float(loss.detach().float().cpu()) / 5.0
            kl_acc += float(kl.detach().float().cpu()) / 5.0
            del ref_lp, policy_lp, kl, loss
        del inputs, comp
        torch.npu.empty_cache()

        if args.ipc_dir is not None:
            policy_latest = args.output_dir / "policy_latest"
            save_policy(engine, policy_latest, rank)
            current_adapter = policy_latest
        if is_main:
            rec = {
                "step": seen,
                "image": row["image"],
                "gt": gt,
                "preds": [item["pred"] for item in scored],
                "rewards": [float(x) for x in rewards],
                "loss": loss_acc,
                "kl": kl_acc,
                "reward_mean": float(rewards.mean()),
                "std": std,
                "sec": round(time.time() - t0, 1),
                "sample_sec": sample_meta.get("sec"),
            }
            with metrics.open("a", encoding="utf-8") as fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            alloc = torch.npu.memory_allocated() / 1024**3
            print(
                f"[step {seen}] loss={loss_acc:.4f} kl={kl_acc:.4f} "
                f"reward={rec['reward_mean']:.3f} std={std:.3f} "
                f"hbm={alloc:.1f}GB ({rec['sec']}s) gt={gt} "
                f"preds={rec['preds']}",
                flush=True,
            )
        write_state(state_path, cursor, seen, rank)
        if seen % args.save_every == 0:
            save_policy(engine, args.output_dir / f"step_{seen}", rank)
            dist.barrier()

    if seen % args.save_every != 0 and seen > 0:
        save_policy(engine, args.output_dir / f"step_{seen}", rank)
    if is_main:
        print(f"[done] steps={seen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
