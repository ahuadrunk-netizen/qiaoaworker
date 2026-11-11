#!/usr/bin/env python3
"""OmniGuard-68 COCO → 计数 SFT jsonl，格式与 sft_ship_lora 的视觉基元一致。

每张图、每个出现过的类别一条正样本：
  question: How many {plural} are visible in this aerial image?
  thinking: 英文，逐实例 <|ref|>class<|/ref|><|box|>[[x,y,x,y]]<|/box|>，坐标 0–100
  answer:   个数的十进制字符串

负样本与正样本行数 1:1，提问类别分布与正样本一致，且该图确实没有这个类。
单类实例数超过 --max-boxes 的行丢掉，避免超过训练 max_length 后把答案截掉。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/wchen/data/OmniGuard-68")
OUT_DIR = Path(__file__).resolve().parent / "data" / "jsonl"

# 问句用复数；<|ref|> 里仍是类别英文名。
IRREGULAR_PLURAL = {
    "person": "people",
    "people": "people",
    "bus": "buses",
    "fishing": "fishing",
    "swimming": "swimming",
    "smoke": "smoke",
    "construction_equipment": "construction_equipment",
    "construction_garbage": "construction_garbage",
    "road_garbage": "road_garbage",
    "water_floating_debris": "water_floating_debris",
}


def plural(name: str) -> str:
    if name in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[name]
    if name.endswith("s"):
        return name
    return name + "s"


def question(name: str) -> str:
    noun = plural(name).replace("_", " ")
    return f"How many {noun} are visible in this aerial image?"


def fmt_box(box: list[float]) -> str:
    vals = ",".join(f"{v:.2f}" for v in box)
    return f"[[{vals}]]"


def thinking(name: str, boxes: list[list[float]]) -> str:
    n = len(boxes)
    lines = [
        f"The user asks me to count the {name} in this image.",
        "Let me count:",
    ]
    if n == 0:
        lines.append(f"I scan the image but find no {name}.")
    for i, box in enumerate(boxes, 1):
        lines.append(f"{i}. <|ref|>{name}<|/ref|><|box|>{fmt_box(box)}<|/box|>.")
    verb = "is" if n == 1 else "are"
    word = name if n == 1 else plural(name)
    lines.append(f"Counting them up, there {verb} {n} {word} in this image.")
    return "\n".join(lines)


def to_percent(box_xyxy: list[float], w: int, h: int) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    return [
        round(x1 / w * 100, 2),
        round(y1 / h * 100, 2),
        round(x2 / w * 100, 2),
        round(y2 / h * 100, 2),
    ]


def record(image: str, name: str, boxes_px: list[list[int]], wh: tuple[int, int]) -> dict:
    w, h = wh
    boxes = [to_percent(b, w, h) for b in boxes_px]
    return {
        "image": image,
        "image_size": [w, h],
        "coord_space": "percent",
        "dataset": "omniguard",
        "question": question(name),
        "thinking": thinking(name, boxes),
        "answer": str(len(boxes_px)),
        "boxes": boxes,
        "boxes_px": boxes_px,
        "points": [],
    }


def xyxy(bbox: list[float], w: int, h: int) -> list[int] | None:
    x, y, bw, bh = bbox
    x1 = max(0, min(w, int(round(x))))
    y1 = max(0, min(h, int(round(y))))
    x2 = max(0, min(w, int(round(x + bw))))
    y2 = max(0, min(h, int(round(y + bh))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def readable_image(rel: str) -> bool:
    """全零占位文件和缺文件不能进训练，PIL 打不开会让整组 rank 退出。"""
    fp = ROOT / rel
    try:
        with fp.open("rb") as f:
            head = f.read(8)
    except OSError:
        return False
    return head.startswith(b"\xff\xd8") or head.startswith(b"\x89PNG")


def build_split(split: str, max_boxes: int, neg_ratio: float, seed: int) -> tuple[list[dict], dict]:
    ann_path = ROOT / split / "_annotations.coco.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    images = {}
    dropped_unreadable = 0
    for im in ann["images"]:
        rel = f"{split}/{im['file_name']}"
        if readable_image(rel):
            images[im["id"]] = im
        else:
            dropped_unreadable += 1

    by_class: dict[int, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    bad_box = 0
    for a in ann["annotations"]:
        im = images.get(a["image_id"])
        if im is None:
            continue
        box = xyxy(a["bbox"], im["width"], im["height"])
        if box is None:
            bad_box += 1
            continue
        name = cats.get(a["category_id"])
        if not name:
            bad_box += 1
            continue
        by_class[a["image_id"]][name].append(box)

    positives: list[dict] = []
    skipped_long = 0
    has_class: dict[str, set[int]] = defaultdict(set)
    for image_id, im in images.items():
        rel = f"{split}/{im['file_name']}"
        wh = (im["width"], im["height"])
        for name, boxes in by_class.get(image_id, {}).items():
            if len(boxes) > max_boxes:
                skipped_long += 1
                continue
            positives.append(record(rel, name, boxes, wh))
            has_class[name].add(image_id)

    rng = random.Random(f"{seed}:{split}")
    negatives: list[dict] = []
    all_ids = list(images)
    pos_by_name: Counter[str] = Counter()
    for row in positives:
        # question 是复数短语，反查用 thinking 第一行里的类名
        # "The user asks me to count the {name} in this image."
        head = row["thinking"].split("\n", 1)[0]
        name = head.removeprefix("The user asks me to count the ").removesuffix(" in this image.")
        pos_by_name[name] += 1

    recycled = {}
    for name, n_pos in pos_by_name.items():
        want = int(round(neg_ratio * n_pos))
        pool = [i for i in all_ids if i not in has_class[name]]
        if not pool or want == 0:
            recycled[name] = {"want": want, "pool": len(pool), "got": 0}
            continue
        rng.shuffle(pool)
        if want <= len(pool):
            chosen = pool[:want]
            recycled[name] = {"want": want, "pool": len(pool), "got": want, "reuse": False}
        else:
            chosen = [pool[i % len(pool)] for i in range(want)]
            recycled[name] = {"want": want, "pool": len(pool), "got": want, "reuse": True}
        for image_id in chosen:
            im = images[image_id]
            negatives.append(
                record(f"{split}/{im['file_name']}", name, [], (im["width"], im["height"]))
            )

    rows = positives + negatives
    rng.shuffle(rows)
    meta = {
        "split": split,
        "images": len(images),
        "categories": len(cats),
        "positives": len(positives),
        "negatives": len(negatives),
        "rows": len(rows),
        "skipped_long": skipped_long,
        "bad_box": bad_box,
        "dropped_unreadable": dropped_unreadable,
        "max_boxes": max_boxes,
        "neg_ratio": neg_ratio,
        "per_class_neg": recycled,
    }
    return rows, meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-boxes", type=int, default=160)
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, out_name in (("train", "train.jsonl"), ("valid", "val.jsonl")):
        rows, meta = build_split(split, args.max_boxes, args.neg_ratio, args.seed)
        path = OUT_DIR / out_name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[split] = {k: v for k, v in meta.items() if k != "per_class_neg"}
        summary[split]["neg_reuse_classes"] = sum(
            1 for v in meta["per_class_neg"].values() if v.get("reuse")
        )
        print(
            f"[{split}] rows={meta['rows']} pos={meta['positives']} "
            f"neg={meta['negatives']} skipped_long={meta['skipped_long']} -> {path}",
            flush=True,
        )
    (OUT_DIR / "build_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
