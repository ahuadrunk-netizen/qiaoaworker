#!/usr/bin/env python3
"""GRPO on the scheme-3 ship LoRA (Qwen3.8-27B).

One prompt = one aerial image + the English counting question (no gold thinking).
The policy samples ``group_size`` answers (default 5). A rule reward scores each
answer, and the update is the within-group advantage:

    A_i = (r_i - mean(r)) / std(r)

If the five rewards are identical, advantages are 0 and the step is skipped.

Reward, only when ``</think>`` is followed by a bare integer:
    0.3  format
    0.4  Gaussian kernel on |pred - count|  (too many and too few cost the same)
    0.3  box score at IoU 0.5: (matched - extra) / gt
         One unmatched box cancels one true match, so spraying boxes loses
         reward unless the extra box is more likely to hit than to miss.

The frozen reference is the step_1200 adapter, kept on CPU and swapped in for
the KL term so a second copy of the LoRA does not sit on the NPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("GRPO_MODEL", "/workspace/zzy/modelscope/Qwen3.8-27B")
TVP_ROOT = os.environ.get(
    "TVP_ROOT", "/workspace/MYVLLM/lib/Thinking-with-Visual-Primitives-pytorch"
)
SYSTEM = "You are a helpful assistant that can understand images and reason with visual primitives."
MIN_PIXELS = 12544
MAX_PIXELS = 2007040

BOX_RE = re.compile(r"<\|box\|>\s*\[(.*?)\]\s*<\|/box\|>", re.DOTALL)
NUM4_RE = re.compile(
    r"\[\s*(-?[0-9]*\.?[0-9]+)\s*,\s*(-?[0-9]*\.?[0-9]+)\s*,\s*"
    r"(-?[0-9]*\.?[0-9]+)\s*,\s*(-?[0-9]*\.?[0-9]+)\s*\]"
)
CLEAN_COUNT_RE = re.compile(r"\s*(\d+)\s*")
IM_END_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>")


def _count_sigma(gt: int) -> float:
    return (0.3 * abs(gt) + 0.1) / 3.0


def extract_clean_count(response: str) -> int | None:
    """Integer iff it is the only thing after ``</think>``."""
    if not isinstance(response, str) or "</think>" not in response:
        return None
    after = response.split("</think>", 1)[1]
    after = IM_END_RE.sub("", after)
    m = CLEAN_COUNT_RE.fullmatch(after)
    if m is None:
        return None
    return int(m.group(1))


def parse_boxes(text: str) -> list[list[float]]:
    body = text.split("</think>", 1)[0] if "</think>" in text else text
    boxes: list[list[float]] = []
    for m in BOX_RE.finditer(body):
        for g in NUM4_RE.finditer(m.group(1)):
            box = [float(g.group(i)) for i in range(1, 5)]
            if all(0.0 <= v <= 100.0 for v in box) and box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
    return boxes


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_count(gt: list[list[float]], pred: list[list[float]], thr: float = 0.5) -> int:
    pairs = sorted(
        ((_iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)),
        reverse=True,
    )
    used_g, used_p, n = set(), set(), 0
    for score, i, j in pairs:
        if score < thr or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        n += 1
    return n


def box_f1(gt: list[list[float]], pred: list[list[float]], thr: float = 0.5) -> float:
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    n = _match_count(gt, pred, thr)
    prec, rec = n / len(pred), n / len(gt)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def box_score(gt: list[list[float]], pred: list[list[float]], thr: float = 0.5) -> float:
    """(matched - extra) / gt. An unmatched box cancels one true match.

    F1 stays positive when most extra boxes miss, so the policy keeps drawing
    them. This score does not. No boxes on an empty image is still a full score.
    """
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    n = _match_count(gt, pred, thr)
    extra = len(pred) - n
    return (n - extra) / len(gt)


def _reward(text: str, count: int, boxes: list[list[float]], box_fn) -> dict:
    pred = extract_clean_count(text)
    if pred is None:
        return {"reward": 0.0, "format": 0.0, "count": 0.0, "box": 0.0, "pred": None}
    sigma = _count_sigma(count)
    d = float(pred - count)
    count_term = math.exp(-(d * d) / (2.0 * sigma * sigma))
    box_term = box_fn(boxes, parse_boxes(text))
    total = 0.3 + 0.4 * count_term + 0.3 * box_term
    return {
        "reward": total,
        "format": 0.3,
        "count": 0.4 * count_term,
        "box": 0.3 * box_term,
        "pred": pred,
    }


def reward_of(text: str, count: int, boxes: list[list[float]]) -> dict:
    """New box term: one unmatched box cancels one true match."""
    return _reward(text, count, boxes, box_score)


def reward_of_f1(text: str, count: int, boxes: list[list[float]]) -> dict:
    """Old box term: F1. Extra boxes only grow the denominator."""
    return _reward(text, count, boxes, box_f1)


def _check_reward() -> None:
    good = (
        "1. <|ref|>ship<|/ref|><|box|>[[21.48,28.47,23.57,36.53]]<|/box|>.\n"
        "</think>\n\n1"
    )
    gt_box = [[21.48, 28.47, 23.57, 36.53]]
    r = reward_of(good, 1, gt_box)
    assert r["pred"] == 1 and r["reward"] > 0.99, r
    assert reward_of(good + " ships", 1, gt_box)["reward"] == 0.0
    assert extract_clean_count("no close") is None
    neg = "</think>\n\n0<|im_end|>"
    rn = reward_of(neg, 0, [])
    assert rn["pred"] == 0 and rn["reward"] > 0.99, rn
    over = (
        "1. <|ref|>ship<|/ref|><|box|>[[21.48,28.47,23.57,36.53]]<|/box|>.\n"
        "2. <|ref|>ship<|/ref|><|box|>[[1.00,1.00,2.00,2.00]]<|/box|>.\n"
        "</think>\n\n1"
    )
    ro = reward_of(over, 1, gt_box)
    assert ro["box"] == 0.0, ro
    assert ro["reward"] < r["reward"]
    assert abs(reward_of("</think>\n\n2", 1, gt_box)["count"] - reward_of("</think>\n\n0", 1, gt_box)["count"]) < 1e-9
    print("[check] reward ok", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-reward", action="store_true")
    p.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "outputs/lora_exp3/epoch_0_step_1200",
    )
    p.add_argument("--jsonl", type=Path, default=ROOT / "data/jsonl/train.ship.jsonl")
    p.add_argument("--image-root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/grpo_exp3")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--group-size", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-coeff", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class _PromptStream:
    def __init__(self, jsonl: Path, image_root: Path, seed: int):
        self.rows = [
            json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise SystemExit(f"empty jsonl: {jsonl}")
        self.image_root = image_root
        self.rng = random.Random(seed)
        self.order = list(range(len(self.rows)))
        self.rng.shuffle(self.order)
        self.cursor = 0

    def next(self) -> tuple[dict, Path]:
        for _ in range(len(self.rows)):
            if self.cursor % len(self.rows) == 0 and self.cursor > 0:
                self.rng.shuffle(self.order)
            row = self.rows[self.order[self.cursor % len(self.rows)]]
            self.cursor += 1
            path = (self.image_root / row["image"]).resolve()
            if path.is_file():
                return row, path
        raise SystemExit(f"no images found under {self.image_root}")


class _CPUAdam:
    """AdamW states on CPU. LoRA grads are ~2GB; keeping m/v on NPU crowds HBM."""

    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr
        self.state: dict[int, dict] = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, lr: float):
        import torch

        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach().float().cpu()
            st = self.state.get(id(p))
            if st is None:
                st = {"m": torch.zeros_like(g), "v": torch.zeros_like(g), "t": 0}
                self.state[id(p)] = st
            st["t"] += 1
            st["m"].mul_(0.9).add_(g, alpha=0.1)
            st["v"].mul_(0.999).addcmul_(g, g, value=0.001)
            mhat = st["m"] / (1 - 0.9 ** st["t"])
            vhat = st["v"] / (1 - 0.999 ** st["t"])
            upd = (mhat / (vhat.sqrt().add_(1e-8))).to(device=p.device, dtype=p.dtype)
            p.data.add_(upd, alpha=-lr)


def _zero_lora_dropout(model) -> None:
    import torch.nn as nn

    for mod in model.modules():
        drop = getattr(mod, "lora_dropout", None)
        if isinstance(drop, nn.Dropout):
            drop.p = 0.0
        elif isinstance(drop, nn.ModuleDict):
            for d in drop.values():
                if isinstance(d, nn.Dropout):
                    d.p = 0.0


def _lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def _install_empty_skip() -> None:
    import peft.peft_model as peft_model

    orig = peft_model.set_peft_model_state_dict

    def _set(model, state, *a, **k):
        state = {kk: vv for kk, vv in state.items() if getattr(vv, "numel", lambda: 1)() > 0}
        return orig(model, state, *a, **k)

    peft_model.set_peft_model_state_dict = _set


def _load(adapter: Path, device: str):
    import torch

    sys.path.insert(0, TVP_ROOT)
    try:
        import fla  # noqa: F401
        import transformers.utils.import_utils as tfu

        tfu.is_flash_linear_attention_available = lambda: True
        print("[init] flash-linear-attention enabled", flush=True)
    except ImportError:
        print("[init] fla not importable, eager linear attention", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    from tvp.model.special_tokens import add_special_tokens

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

    _install_empty_skip()
    print(f"[init] loading {MODEL}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    print(f"[init] loading adapter {adapter}", flush=True)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    _zero_lora_dropout(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)
    named = _lora_params(model)
    ref = {n: p.detach().float().cpu().clone() for n, p in named}
    n_train = sum(p.numel() for _, p in named)
    print(f"[init] trainable lora params={n_train/1e6:.1f}M tensors={len(named)}", flush=True)
    return processor, model, ref


def _copy_lora(model, blob: dict) -> None:
    """Overwrite live LoRA data from a CPU fp32 snapshot, one tensor at a time."""
    for n, p in _lora_params(model):
        p.data.copy_(blob[n].to(device=p.device, dtype=p.dtype))


def _snapshot_lora(model) -> dict:
    import torch

    return {n: p.detach().float().cpu().clone() for n, p in _lora_params(model)}


def _build_inputs(processor, image_path: Path, question: str, device: str):
    from PIL import Image

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    moved = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            moved[k] = v
            continue
        moved[k] = v.to(device)
    return moved


def _generate(model, inputs, max_new_tokens: int, temperature: float, top_p: float):
    import torch

    model.eval()
    model.config.use_cache = True
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    new_ids = out[0, prompt_len:].detach().cpu().clone()
    del out
    model.train()
    model.config.use_cache = False
    return new_ids


def _response_logprob(model, inputs, new_ids, device: str):
    import torch
    import torch.nn.functional as F

    prompt = inputs["input_ids"]
    prompt_len = prompt.shape[1]
    gen = new_ids.to(device).view(1, -1)
    full = torch.cat([prompt, gen], dim=1)
    attn = torch.ones_like(full)
    # mm_token_type_ids is prompt-length. Generated tokens are text (type 0).
    # Passing the short mask into get_rope_index raises an index error.
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
    logp = F.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    resp = tok[:, prompt_len - 1 :]
    if resp.numel() == 0:
        return tok.sum() * 0.0
    return resp.mean()


def _set_cache(model, enabled: bool) -> None:
    model.config.use_cache = enabled
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = enabled


def _hbm(tag: str) -> None:
    import torch

    alloc = torch.npu.memory_allocated() / 1024**3
    reserved = torch.npu.memory_reserved() / 1024**3
    print(f"[hbm] {tag} allocated={alloc:.1f}GB reserved={reserved:.1f}GB", flush=True)


def _save(model, out_dir: Path, step: int, keep: int) -> None:
    dest = out_dir / f"step_{step}"
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    print(f"[save] {dest}", flush=True)
    ckpts = sorted(out_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[-1]))
    for old in ckpts[:-keep] if keep > 0 else []:
        if old.is_dir():
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
            print(f"[save] removed {old}", flush=True)


def main() -> int:
    args = parse_args()
    if args.check_reward:
        _check_reward()
        return 0

    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import torch
    import torch_npu  # noqa: F401

    if args.group_size < 2:
        raise SystemExit("group-size must be >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    stream = _PromptStream(args.jsonl, args.image_root, args.seed)
    print(
        f"[data] {len(stream.rows)} prompts from {args.jsonl} "
        f"group_size={args.group_size} max_steps={args.max_steps}",
        flush=True,
    )
    processor, model, ref_blob = _load(args.adapter, args.device)
    policy_blob = _snapshot_lora(model)
    _hbm("after load")
    opt = _CPUAdam([p for _, p in _lora_params(model)], lr=args.lr)
    metrics_path = args.output_dir / "metrics.jsonl"
    rollout_path = args.output_dir / "rollouts.jsonl"

    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        row, image = stream.next()
        count = int(row["answer"])
        boxes = row.get("boxes") or []
        question = row["question"]
        print(
            f"[step {step}] image={row['image']} gt={count} boxes={len(boxes)}",
            flush=True,
        )
        inputs = _build_inputs(processor, image, question, args.device)
        texts: list[str] = []
        token_rows: list[torch.Tensor] = []
        parts: list[dict] = []
        for g in range(args.group_size):
            print(f"  [rollout] {g + 1}/{args.group_size}", flush=True)
            new_ids = _generate(
                model, inputs, args.max_new_tokens, args.temperature, args.top_p
            )
            text = processor.tokenizer.decode(new_ids, skip_special_tokens=False)
            texts.append(text)
            token_rows.append(new_ids)
            scored = reward_of(text, count, boxes)
            parts.append(scored)
            print(
                f"  [rollout] pred={scored['pred']} reward={scored['reward']:.3f} "
                f"fmt={scored['format']:.2f} cnt={scored['count']:.3f} box={scored['box']:.3f} "
                f"tokens={new_ids.numel()}",
                flush=True,
            )
        torch.npu.empty_cache()
        _hbm("after rollouts")

        rewards = torch.tensor([p["reward"] for p in parts], dtype=torch.float32)
        std = float(rewards.std(unbiased=False))
        rec = {
            "step": step,
            "image": row["image"],
            "gt": count,
            "n_boxes": len(boxes),
            "rewards": [p["reward"] for p in parts],
            "preds": [p["pred"] for p in parts],
            "std": std,
            "skipped": std < 1e-6,
        }
        if std < 1e-6:
            rec["sec"] = round(time.time() - t0, 1)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[step {step}] flat group reward={float(rewards[0]):.3f} "
                f"std=0 skip update ({rec['sec']}s)",
                flush=True,
            )
            continue

        adv = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-8)
        adv = adv.tolist()
        opt.zero_grad()
        loss_acc = 0.0
        kl_acc = 0.0
        model.train()
        _set_cache(model, False)
        for g in range(args.group_size):
            _copy_lora(model, ref_blob)
            with torch.no_grad():
                ref_lp = _response_logprob(model, inputs, token_rows[g], args.device)
            _copy_lora(model, policy_blob)
            policy_lp = _response_logprob(model, inputs, token_rows[g], args.device)
            kl = policy_lp - ref_lp.detach()
            loss_i = -(adv[g] * policy_lp) + args.kl_coeff * kl
            (loss_i / args.group_size).backward()
            loss_acc += float(loss_i.detach().cpu()) / args.group_size
            kl_acc += float(kl.detach().cpu()) / args.group_size
            del policy_lp, ref_lp, kl, loss_i
            torch.npu.empty_cache()
        torch.nn.utils.clip_grad_norm_([p for _, p in _lora_params(model)], 1.0)
        warm = min(1.0, step / max(args.warmup_steps, 1))
        opt.step(args.lr * warm)
        opt.zero_grad()
        policy_blob = _snapshot_lora(model)
        torch.npu.empty_cache()

        rec.update(
            {
                "loss": loss_acc,
                "kl": kl_acc,
                "lr": args.lr * warm,
                "sec": round(time.time() - t0, 1),
                "mean_reward": float(rewards.mean()),
            }
        )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with rollout_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "image": row["image"],
                        "question": question,
                        "gt": count,
                        "responses": [
                            {"text": t, **p} for t, p in zip(texts, parts)
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(
            f"[step {step}] loss={loss_acc:.4f} kl={kl_acc:.4f} "
            f"reward={float(rewards.mean()):.3f} std={std:.3f} "
            f"lr={args.lr * warm:.2e} ({rec['sec']}s)",
            flush=True,
        )
        if step % args.save_every == 0:
            model.eval()
            _save(model, args.output_dir, step, args.keep_checkpoints)

    if args.max_steps % args.save_every != 0:
        model.eval()
        _save(model, args.output_dir, args.max_steps, args.keep_checkpoints)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
