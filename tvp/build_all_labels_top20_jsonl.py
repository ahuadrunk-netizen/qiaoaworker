#!/usr/bin/env python3
"""全类别标注里，按框数取前 20 个类别，写成 Qwen TVP 计数验证 jsonl。

只用 COCO valid。一条样本是一张图上的一个类别，字段与
build_omniguard_jsonl.py 的 val.jsonl 相同。坐标是相对原图像素的 0–100。
正样本是这 20 类在 valid 上真实出现的框；负样本与正样本 1:1，
且该图确实没有被问到的类。
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

TVP = Path(__file__).resolve().parent
sys.path.insert(0, str(TVP))
from build_omniguard_jsonl import question, record, xyxy  # noqa: E402

COCO = Path("/data/zzy/python/model_train/train_data/data/all_labels_coco")
OUT_DIR = TVP / "data" / "jsonl"
OUT = OUT_DIR / "val_all_labels_top20.jsonl"
META = OUT_DIR / "val_all_labels_top20.meta.json"
TOP_K = 20
NEG_RATIO = 1.0
SEED = 0


def image_rel(split: str, file_name: str) -> str:
    name = file_name.lstrip("/")
    if name.startswith("images/"):
        return f"{split}/{name}"
    return f"{split}/images/{name}"


def load_split(split: str) -> tuple[dict[int, str], dict[int, dict], list]:
    ann = json.loads((COCO / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    images = {im["id"]: im for im in ann["images"]}
    return cats, images, ann["annotations"]


def main() -> int:
    box_count: Counter[str] = Counter()
    for split in ("train", "valid"):
        cats, _, annotations = load_split(split)
        for a in annotations:
            name = cats.get(a["category_id"])
            if name:
                box_count[name] += 1
    top_names = [name for name, _ in box_count.most_common(TOP_K)]
    top_set = set(top_names)

    cats, images, annotations = load_split("valid")
    by_image: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    bad_box = 0
    valid_boxes: Counter[str] = Counter()
    for a in annotations:
        im = images.get(a["image_id"])
        name = cats.get(a["category_id"])
        if im is None or not name or name not in top_set:
            continue
        box = xyxy(a["bbox"], im["width"], im["height"])
        if box is None:
            bad_box += 1
            continue
        by_image[a["image_id"]][name].append(box)
        valid_boxes[name] += 1

    positives = []
    has_class: dict[str, set[int]] = defaultdict(set)
    missing = []
    for image_id, per_class in by_image.items():
        im = images[image_id]
        rel = image_rel("valid", im["file_name"])
        if not (COCO / rel).is_file():
            missing.append(rel)
        wh = (im["width"], im["height"])
        for name, boxes_px in per_class.items():
            row = record(rel, name, boxes_px, wh)
            row["dataset"] = "all_labels"
            positives.append(row)
            has_class[name].add(image_id)

    rng = random.Random(SEED)
    negatives = []
    all_ids = list(images)
    neg_meta = {}
    for name in top_names:
        n_pos = sum(1 for row in positives if row["question"] == question(name))
        want = int(round(NEG_RATIO * n_pos))
        pool = [i for i in all_ids if i not in has_class[name]]
        rng.shuffle(pool)
        chosen = pool[:want]
        neg_meta[name] = {"positives": n_pos, "want": want, "pool": len(pool), "got": len(chosen)}
        for image_id in chosen:
            im = images[image_id]
            rel = image_rel("valid", im["file_name"])
            row = record(rel, name, [], (im["width"], im["height"]))
            row["dataset"] = "all_labels"
            negatives.append(row)

    rows = positives + negatives
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "image_root": str(COCO),
        "jsonl": str(OUT),
        "split": "valid",
        "rank_by": "bbox_count_train_plus_valid",
        "top_k": TOP_K,
        "neg_ratio": NEG_RATIO,
        "seed": SEED,
        "rows": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "images": len(images),
        "bad_box": bad_box,
        "missing_images": missing[:20],
        "missing_n": len(set(missing)),
        "classes": [
            {
                "rank": i,
                "name": name,
                "boxes_all": box_count[name],
                "boxes_valid": valid_boxes[name],
                "positives": neg_meta[name]["positives"],
                "negatives": neg_meta[name]["got"],
            }
            for i, name in enumerate(top_names, 1)
        ],
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"rows={len(rows)} pos={len(positives)} neg={len(negatives)} "
        f"bad_box={bad_box} missing={len(set(missing))} -> {OUT}",
        flush=True,
    )
    for item in meta["classes"]:
        print(
            f"{item['rank']:2d} {item['name']} boxes={item['boxes_all']} "
            f"valid_boxes={item['boxes_valid']} pos={item['positives']} neg={item['negatives']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
