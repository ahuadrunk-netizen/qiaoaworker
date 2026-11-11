#!/usr/bin/env python3
"""用 boxes_px 校验 <|box|>（0–1000）是否与标注一致：画框或算 IoU。

    python3 verify_boxes.py --jsonl data/jsonl/test.jsonl --index 0 --out /tmp/check.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coords import box_space_to_px
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("/tmp/sft_ship_verify.jpg"))
    p.add_argument("--no-draw", action="store_true", help="只算 IoU，不写图")
    p.add_argument("--all", action="store_true", help="遍历整份 jsonl 统计最小 IoU")
    args = p.parse_args()

    lines = args.jsonl.read_text(encoding="utf-8").splitlines()

    if args.all:
        worst = 1.0
        n = 0
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            w, h = row["image_size"]
            space = row.get("coord_space", "percent")
            for box_px, box_n in zip(row["boxes_px"], row["boxes"], strict=True):
                back = box_space_to_px(box_n, w, h, space)
                worst = min(worst, iou([float(v) for v in box_px], back))
                n += 1
        print(f"检查 {n} 个框，最小 IoU = {worst:.4f}")
        return 0

    row = json.loads(lines[args.index])
    w, h = row["image_size"]
    space = row.get("coord_space", "percent")
    img_path = ROOT / row["image"]

    print(f"coord_space={space} image_size={w}x{h}")
    for box_px, box_n in zip(row["boxes_px"], row["boxes"], strict=True):
        back = box_space_to_px(box_n, w, h, space)
        print(
            f"  px={box_px}  norm={box_n}  ->px="
            f"{[round(v, 1) for v in back]}  IoU={iou([float(v) for v in box_px], back):.4f}"
        )

    if args.no_draw:
        return 0

    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for box_px, box_n in zip(row["boxes_px"], row["boxes"], strict=True):
        draw.rectangle([float(v) for v in box_px], outline=(0, 255, 0), width=4)
        draw.rectangle(box_space_to_px(box_n, w, h, space), outline=(255, 0, 0), width=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.out, quality=92)
    print(f"green=boxes_px  red=boxes({space})->px  saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
