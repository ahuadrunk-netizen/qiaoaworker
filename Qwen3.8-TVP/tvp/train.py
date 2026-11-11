"""OmniGuard LoRA SFT。数据、模板渲染、LoRA、训练循环都在本目录。

    torchrun --nproc_per_node=2 train.py --config configs/lora_exp1_llm.yaml --output_dir outputs/lora_exp1
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import argparse
import shutil
from pathlib import Path

# fla 的开关写在 cuda 上。NPU 上先改掉，再 import transformers 的模型代码。
try:
    import fla  # noqa: F401
    import transformers.utils.import_utils as _tfu

    _tfu.is_flash_linear_attention_available = lambda: True
except ImportError:
    pass

import deepspeed
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from collate import ConversationCollator
from dataset import JSONLSFTDataset
from lora import install_lora, load_adapter_state
from tokens import add_special_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=-1)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict):
    specs = cfg["datasets"]
    if len(specs) != 1:
        raise SystemExit("[data] 这个入口只接一份 jsonl")
    ds = specs[0]
    return JSONLSFTDataset(
        jsonl_path=ds["path"],
        image_root=ds.get("image_root"),
        task_type=ds.get("task_type", "counting"),
    )


def build_collator(cfg: dict, tokenizer):
    return ConversationCollator(
        tokenizer=tokenizer,
        max_length=cfg.get("max_length", 8192),
        image_size=cfg.get("image_size", 1024),
        image_min_pixels=cfg.get("image_min_pixels"),
        image_max_pixels=cfg.get("image_max_pixels"),
        processor_model_path=cfg.get("base_model_path") or cfg["model_name_or_path"],
    )


def build_ds_config(cfg: dict, world_size: int, total_steps: int) -> dict:
    micro_bs = cfg.get("batch_size", 1)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    lr = cfg.get("learning_rate", 1e-5)
    warmup = max(1, int(total_steps * cfg.get("warmup_ratio", 0.03)))
    return {
        "train_batch_size": micro_bs * grad_accum * world_size,
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps": grad_accum,
        "zero_optimization": {
            "stage": cfg.get("zero_stage", 3),
            "offload_optimizer": {"device": cfg.get("offload_optimizer", "none")},
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": cfg.get("weight_decay", 0.01),
            },
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
        "gradient_clipping": cfg.get("grad_clip", 1.0),
        "steps_per_print": cfg.get("log_steps", 10),
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
    }


class _SkipSampler(torch.utils.data.Sampler):
    """跳过本轮已经训练过的样本下标，不再把图读进来。"""

    def __init__(self, base, skip: int):
        self.base = base
        self.skip = skip

    def __iter__(self):
        it = iter(self.base)
        for _ in range(self.skip):
            next(it)
        return it

    def __len__(self):
        return max(0, len(self.base) - self.skip)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)


def _adapter_optim_step(path: str) -> int:
    name = Path(path).name
    marker = "step_"
    if marker not in name:
        raise SystemExit(f"[resume] 目录名里没有优化器步数: {path}")
    return int(name.rsplit(marker, 1)[-1])


def _continue_schedule(engine, step: int) -> None:
    sched = getattr(engine, "lr_scheduler", None)
    if sched is None or step <= 0:
        return
    sched.last_batch_iteration = step
    if not hasattr(sched, "get_lr"):
        return
    for group, lr in zip(engine.optimizer.param_groups, sched.get_lr()):
        group["lr"] = lr


def prune_adapters(output_dir: Path, keep: int) -> None:
    dirs = []
    for p in Path(output_dir).glob("epoch_*_step_*"):
        if (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists():
            try:
                step = int(p.name.split("_")[-1])
            except ValueError:
                continue
            dirs.append((step, p))
    dirs.sort()
    if keep > 0:
        for _, p in dirs[:-keep]:
            shutil.rmtree(p, ignore_errors=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    lcfg = cfg.get("lora") or {}
    if lcfg.get("enabled", False):
        install_lora(lcfg)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    deepspeed.init_distributed()

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    tokenizer = add_special_tokens(tokenizer)
    if is_main:
        print(f"[init] tokenizer len={len(tokenizer)}", flush=True)

    dataset = build_dataset(cfg)
    collator = build_collator(cfg, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    resume_optim_steps = 0
    adapter_resume = bool(args.resume) and (Path(args.resume) / "adapter_model.safetensors").is_file()
    if adapter_resume:
        resume_optim_steps = _adapter_optim_step(args.resume)
    skip_batches = resume_optim_steps * cfg.get("gradient_accumulation_steps", 1)
    if skip_batches >= len(sampler):
        if is_main:
            print(
                f"[resume] {args.resume} 已覆盖本轮全部 {len(sampler)} 个 micro-batch，没有剩余样本",
                flush=True,
            )
        return
    if skip_batches:
        sampler = _SkipSampler(sampler, skip_batches)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
    )

    epochs = cfg.get("epochs", 1)
    grad_accum = cfg.get("gradient_accumulation_steps", 1)
    micro_bs = cfg.get("batch_size", 1)
    steps_per_epoch = max(1, len(dataset) // (micro_bs * grad_accum * world_size))
    total_steps = steps_per_epoch * epochs
    ds_config = build_ds_config(cfg, world_size, total_steps)
    if is_main:
        print(
            f"[init] steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
            f"world={world_size} ds={ds_config['zero_optimization']}",
            flush=True,
        )

    dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841
    print(
        f"[rank {local_rank}] before load: dist_init={dist.is_initialized()} "
        f"dist_world={dist.get_world_size() if dist.is_initialized() else 'NA'} "
        f"env_world={world_size} train_bs={ds_config['train_batch_size']}",
        flush=True,
    )
    if is_main:
        print(f"[init] loading {cfg['model_name_or_path']} (ZeRO-3 partition-on-load) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_name_or_path"],
        dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=True,
    )
    vocab = getattr(getattr(model.config, "text_config", None), "vocab_size", None)
    if vocab is not None and len(tokenizer) > vocab:
        raise RuntimeError(
            f"tokenizer ({len(tokenizer)}) exceeds model vocab ({vocab}); "
            "resize_token_embeddings is not supported under partition-on-load"
        )
    if is_main:
        print(f"[init] tokenizer={len(tokenizer)} model_vocab={vocab} (no resize)", flush=True)
    if cfg.get("freeze_vision_tower", False):
        vis = getattr(model, "visual", None) or getattr(getattr(model, "model", None), "visual", None)
        if vis is not None:
            for p in vis.parameters():
                p.requires_grad = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if adapter_resume:
        n_loaded = load_adapter_state(model, args.resume)
        if is_main:
            print(
                f"[resume] loaded {n_loaded} lora tensors from {args.resume} "
                f"optim_step={resume_optim_steps} skip_batches={skip_batches} "
                f"remaining={len(loader)}",
                flush=True,
            )

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    if adapter_resume:
        _continue_schedule(engine, resume_optim_steps)
    if is_main:
        n_train = sum(p.numel() for p in engine.module.parameters() if p.requires_grad)
        lr_now = engine.optimizer.param_groups[0]["lr"]
        print(f"[init] deepspeed ready; trainable≈{n_train} lr={lr_now:.3e}", flush=True)

    start_epoch, global_step = 0, skip_batches
    if args.resume and not adapter_resume:
        _, client_state = engine.load_checkpoint(args.resume)
        if client_state:
            start_epoch = client_state.get("epoch", 0)
            global_step = client_state.get("global_step", 0)
        if is_main:
            print(f"[resume] {args.resume} epoch={start_epoch} step={global_step}", flush=True)

    save_steps = cfg.get("save_steps", 0)
    save_optim_steps = cfg.get("save_optim_steps", 0)
    last_saved_optim_step = resume_optim_steps if adapter_resume else -1
    engine.train()
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main, total=len(loader))
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = engine(**inputs)
            engine.backward(outputs.loss)
            engine.step()
            global_step += 1
            if is_main:
                pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}", "step": global_step})
            if save_steps > 0 and global_step % save_steps == 0:
                tag = f"epoch_{epoch}_step_{global_step}"
                engine.save_checkpoint(
                    str(output_dir), tag=tag,
                    client_state={"epoch": epoch, "global_step": global_step},
                )
                if is_main:
                    print(f"[save] {output_dir}/{tag}", flush=True)
            optim_step = engine.global_steps + resume_optim_steps
            if (save_optim_steps > 0 and optim_step > resume_optim_steps
                    and optim_step % save_optim_steps == 0
                    and optim_step != last_saved_optim_step):
                last_saved_optim_step = optim_step
                ckpt = output_dir / f"epoch_{epoch}_step_{optim_step}"
                try:
                    ok = engine.save_16bit_model(str(ckpt))
                    if is_main:
                        print(f"[save] model-only {ckpt} ok={ok}", flush=True)
                        prune_adapters(output_dir, cfg.get("keep_checkpoints", 2))
                except Exception as e:  # noqa: BLE001
                    if is_main:
                        print(f"[save] model-only FAILED: {type(e).__name__}: {e}", flush=True)
            if args.max_steps > 0 and global_step >= args.max_steps:
                if is_main:
                    print(f"[smoke] reached max_steps={args.max_steps}", flush=True)
                return

    if save_steps > 0:
        engine.save_checkpoint(
            str(output_dir), tag="final",
            client_state={"epoch": epochs, "global_step": global_step},
        )
    final_step = engine.global_steps + resume_optim_steps
    if final_step > resume_optim_steps:
        ckpt = output_dir / f"epoch_{epochs - 1}_step_{final_step}"
        ok = engine.save_16bit_model(str(ckpt))
        if is_main:
            print(f"[save] model-only {ckpt} ok={ok}", flush=True)
            prune_adapters(output_dir, cfg.get("keep_checkpoints", 2))
    if is_main:
        print(f"[done] training complete optim_step={final_step}", flush=True)


if __name__ == "__main__":
    main()
