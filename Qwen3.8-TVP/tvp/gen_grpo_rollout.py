#!/usr/bin/env python3
"""用方案三 SFT 权重，为同一张图独立采样 5 条回答，写成 GRPO 组数据。

采样：temperature 1.0，top-p 0.95，最长 4096 token。
提示只有 system、图像和英文问题。标准答案只用于打分，不进提示。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

TVP = Path(__file__).resolve().parent
SHIP = TVP.parent / "sft_ship_lora"
sys.path.insert(0, str(TVP))
sys.path.insert(0, str(SHIP))

from grpo_train import _build_inputs, _generate, reward_of  # noqa: E402
from tokens import add_special_tokens  # noqa: E402

MODEL = "/workspace/zzy/modelscope/Qwen3.8-27B"
MIN_PIXELS = 12544
MAX_PIXELS = 2007040


def load_model(adapter: str, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    import peft.peft_model as peft_model

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

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    orig = peft_model.set_peft_model_state_dict

    def skip_empty(model, state, *args, **kwargs):
        state = {
            key: value
            for key, value in state.items()
            if getattr(value, "numel", lambda: 1)() > 0
        }
        return orig(model, state, *args, **kwargs)

    peft_model.set_peft_model_state_dict = skip_empty
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.to(device)
    return processor, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mine = [row for row in rows if int(row["manifest_i"]) % args.num_shards == args.shard]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if args.out.is_file():
        done = sum(1 for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip())
    print(
        f"[shard {args.shard}] groups={len(mine)} resume={done} adapter={args.adapter}",
        flush=True,
    )

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed + args.shard)
    processor, model = load_model(str(args.adapter), args.device)
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as fout:
        for n, row in enumerate(mine):
            if n < done:
                continue
            image = (args.image_root / row["image"]).resolve()
            gt = int(row["answer"])
            boxes = row.get("boxes") or []
            responses = []
            if not image.is_file():
                print(f"[shard {args.shard}] missing {image}", flush=True)
            else:
                inputs = _build_inputs(processor, image, row["question"], args.device)
                for g in range(args.group_size):
                    new_ids = _generate(
                        model, inputs, args.max_new_tokens, args.temperature, args.top_p
                    )
                    text = processor.tokenizer.decode(new_ids, skip_special_tokens=False)
                    scored = reward_of(text, gt, boxes)
                    responses.append(
                        {
                            "text": text,
                            "tokens": int(new_ids.numel()),
                            **scored,
                        }
                    )
                    print(
                        f"[shard {args.shard}] {n+1}/{len(mine)} sample {g+1} "
                        f"pred={scored['pred']} reward={scored['reward']:.3f} "
                        f"tokens={int(new_ids.numel())}",
                        flush=True,
                    )
            rewards = [item["reward"] for item in responses]
            rec = {
                "manifest_i": row["manifest_i"],
                "image": row["image"],
                "question": row["question"],
                "gt": gt,
                "boxes": boxes,
                "image_size": row.get("image_size"),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "responses": responses,
                "reward_mean": sum(rewards) / len(rewards) if rewards else None,
                "reward_std": (
                    (sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) ** 0.5
                    if rewards else None
                ),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            dt = time.time() - t0
            seen = n + 1 - done
            rate = seen / dt if dt and seen else 0
            left = (len(mine) - n - 1) / rate if rate else 0
            print(
                f"[shard {args.shard}] group {n+1}/{len(mine)} "
                f"gt={gt} mean={rec['reward_mean']} std={rec['reward_std']} "
                f"eta={left/3600:.1f}h",
                flush=True,
            )
    print(f"[shard {args.shard}] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
