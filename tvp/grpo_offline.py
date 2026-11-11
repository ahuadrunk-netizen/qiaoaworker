#!/usr/bin/env python3
"""在方案三 SFT LoRA 上，用已经打好分的 5 条一组数据做 GRPO。

组内优势 A_i = (r_i - mean) / std。
loss = -A * mean_logprob(response) + kl_coeff * (policy_logprob - ref_logprob)。
参考模型是最初的 SFT 权重，不随更新改变。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

TVP = Path(__file__).resolve().parent
sys.path.insert(0, str(TVP.parent / "sft_ship_lora"))

from grpo_train import (  # noqa: E402
    _CPUAdam,
    _build_inputs,
    _copy_lora,
    _load,
    _lora_params,
    _set_cache,
    _snapshot_lora,
)


def _response_logprob(model, inputs, new_ids, device: str):
    """和 grpo_train 相同的响应平均 logprob，但 log_softmax 按块重算，避免整段 fp32 logits。"""
    import torch
    import torch.nn.functional as F

    prompt = inputs["input_ids"]
    prompt_len = prompt.shape[1]
    gen = new_ids.to(device).view(1, -1)
    full = torch.cat([prompt, gen], dim=1)
    attn = torch.ones_like(full)
    fwd = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
    }
    mm = inputs.get("mm_token_type_ids")
    if mm is not None:
        pad = torch.zeros(mm.shape[0], gen.shape[1], dtype=mm.dtype, device=mm.device)
        fwd["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
    for mod in model.modules():
        if getattr(mod, "rope_deltas", None) is not None:
            mod.rope_deltas = None
    out = model(input_ids=full, attention_mask=attn, **fwd)
    logits = out.logits[:, :-1, :]
    target = full[:, 1:]
    resp_logits = logits[:, prompt_len - 1 :]
    resp_target = target[:, prompt_len - 1 :]
    if resp_target.numel() == 0:
        return logits.sum() * 0.0

    def chunk_lp(logit_slice, target_slice):
        lp = F.log_softmax(logit_slice.float(), dim=-1)
        return lp.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)

    parts = []
    chunk = 32
    for start in range(0, resp_target.shape[1], chunk):
        end = min(start + chunk, resp_target.shape[1])
        logit_slice = resp_logits[:, start:end, :]
        target_slice = resp_target[:, start:end]
        if logit_slice.requires_grad:
            parts.append(
                torch.utils.checkpoint.checkpoint(
                    chunk_lp, logit_slice, target_slice, use_reentrant=False
                )
            )
        else:
            parts.append(chunk_lp(logit_slice, target_slice))
    return torch.cat(parts, dim=1).mean()


def index_lines(path: Path) -> list[int]:
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-coeff", type=float, default=0.04)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-seq", type=int, default=2560)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def encode_group(tokenizer, responses: list[dict]):
    rows = []
    for resp in responses:
        ids = tokenizer.encode(resp["text"], add_special_tokens=False)
        rows.append(ids)
    return rows


def main() -> int:
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    offsets = index_lines(args.jsonl)
    rng = random.Random(args.seed)
    rng.shuffle(offsets)
    print(f"[data] groups={len(offsets)} from {args.jsonl}", flush=True)

    processor, model, ref_blob = _load(args.adapter, args.device)
    policy_blob = _snapshot_lora(model)
    opt = _CPUAdam([p for _, p in _lora_params(model)], lr=args.lr)
    metrics_path = args.output_dir / "metrics.jsonl"
    fin = args.jsonl.open("rb")

    step = 0
    seen = 0
    skipped = 0
    while True:
        if args.max_steps > 0 and step >= args.max_steps:
            break
        if seen >= len(offsets):
            rng.shuffle(offsets)
            seen = 0
            print("[data] new epoch", flush=True)
        pos = offsets[seen]
        seen += 1
        fin.seek(pos)
        group = json.loads(fin.readline().decode("utf-8"))
        responses = group["responses"]
        rewards_list = [float(item["reward"]) for item in responses]
        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        std = float(rewards.std(unbiased=False))
        if std < 1e-6 or len(responses) < 2:
            skipped += 1
            continue
        image = (args.image_root / group["image"]).resolve()
        if not image.is_file():
            skipped += 1
            print(f"[skip] missing {image}", flush=True)
            continue
        inputs = _build_inputs(processor, image, group["question"], args.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        encoded = encode_group(processor.tokenizer, responses)
        if any(prompt_len + len(ids) > args.max_seq for ids in encoded):
            skipped += 1
            print(
                f"[skip] seq {prompt_len}+{max(len(ids) for ids in encoded)} > {args.max_seq} "
                f"image={group['image']}",
                flush=True,
            )
            del inputs
            torch.npu.empty_cache()
            continue

        step += 1
        t0 = time.time()
        adv = ((rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-8)).tolist()
        opt.zero_grad()
        model.train()
        _set_cache(model, False)
        loss_acc = 0.0
        kl_acc = 0.0
        n = len(responses)
        for g, ids in enumerate(encoded):
            new_ids = torch.tensor(ids, dtype=torch.long)
            _copy_lora(model, ref_blob)
            with torch.no_grad():
                ref_lp = _response_logprob(model, inputs, new_ids, args.device)
            ref_lp = ref_lp.detach()
            torch.npu.empty_cache()
            _copy_lora(model, policy_blob)
            policy_lp = _response_logprob(model, inputs, new_ids, args.device)
            kl = policy_lp - ref_lp.detach()
            loss_i = -(adv[g] * policy_lp) + args.kl_coeff * kl
            (loss_i / n).backward()
            loss_acc += float(loss_i.detach().cpu()) / n
            kl_acc += float(kl.detach().cpu()) / n
            del policy_lp, ref_lp, kl, loss_i, new_ids
            torch.npu.empty_cache()
        torch.nn.utils.clip_grad_norm_([p for _, p in _lora_params(model)], 1.0)
        warm = min(1.0, step / max(args.warmup_steps, 1))
        lr = args.lr * warm
        opt.step(lr)
        opt.zero_grad()
        policy_blob = _snapshot_lora(model)
        torch.npu.empty_cache()
        rec = {
            "step": step,
            "image": group["image"],
            "gt": group["gt"],
            "rewards": rewards_list,
            "std": std,
            "loss": loss_acc,
            "kl": kl_acc,
            "lr": lr,
            "sec": round(time.time() - t0, 1),
            "skipped": skipped,
        }
        with metrics_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[step {step}] loss={loss_acc:.4f} kl={kl_acc:.4f} "
            f"reward={float(rewards.mean()):.3f} std={std:.3f} "
            f"gt={group['gt']} lr={lr:.2e} ({rec['sec']}s) skipped={skipped}",
            flush=True,
        )
        if step % args.save_every == 0:
            model.eval()
            dest = args.output_dir / f"step_{step}"
            dest.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(dest)
            print(f"[save] {dest}", flush=True)
            ckpts = sorted(args.output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[-1]))
            for old in ckpts[: -args.keep_checkpoints] if args.keep_checkpoints > 0 else []:
                for f in old.iterdir():
                    f.unlink()
                old.rmdir()
            model.train()
    fin.close()
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
