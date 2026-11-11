#!/usr/bin/env python3
"""将多个 YOLO 数据集的标注转为计数 SFT 用 jsonl（含 thinking 坐标推理）。

数据集在 class_names.py 的 ALL_DATASETS 中注册（ship / construction / vendor / road）。
每张正样本图：出现的每个细类一行 + 宏观类别一行（如「船艇」「工程车辆」）。
负样本（空标注图）：与正样本行数 1:1，按数据集分别采样，answer 为 "0"。

输出：
    data/jsonl/{train,val,test}.jsonl       合并后的总集（训练用）
    data/jsonl/{split}.<dataset>.jsonl       各数据集单独一份（评估用）
    data/images/<dataset>  →  <root>/images  相对软链，image 字段以 data/images/ 开头

    python3 build_jsonl.py --split train val test
    python3 build_jsonl.py --datasets ship road --split test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from class_names import (
    ALL_DATASETS,
    CLASS_NAMES_ZH,
    DATASETS_BY_NAME,
    EN_BY_ZH,
    IRREGULAR_PLURAL_EN,
    DatasetSpec,
)
from coords import COORD_SPACES, DEFAULT_PERCENT_DECIMALS, boxes_px_to_space

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "jsonl"
IMAGES_LINK_DIR = ROOT / "data" / "images"
IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG")
SPLITS = ("train", "val", "test")
# 无 train/val/test 目录的数据集按 hash 划分：前 80% train，10% val，10% test
AUTO_SPLIT_BOUNDS = (80, 90)


# ───────────────────────── YOLO 读取 ─────────────────────────


def load_yaml_names(yolo_root: Path, fallback: tuple[str, ...]) -> list[str]:
    yaml_path = yolo_root / "data.yaml"
    if not yaml_path.is_file():
        return list(fallback)
    names: list[str] = []
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                break
            _, v = line.split(":", 1)
            names.append(v.strip())
    return names or list(fallback)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return w, h


def parse_yolo_label(
    label_path: Path, names: list[str], width: int, height: int
) -> dict[str, list[list[int]]]:
    by_class: dict[str, list[list[int]]] = defaultdict(list)
    if not label_path.is_file():
        return by_class
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        if cid < 0 or cid >= len(names):
            continue
        x1 = int(round((cx - bw / 2) * width))
        y1 = int(round((cy - bh / 2) * height))
        x2 = int(round((cx + bw / 2) * width))
        y2 = int(round((cy + bh / 2) * height))
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        by_class[names[cid]].append([x1, y1, x2, y2])
    return by_class


def _auto_split(rel: str) -> str:
    """对无划分目录的数据集，用路径 hash 做稳定的 80/10/10 划分。"""
    h = int(hashlib.md5(rel.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < AUTO_SPLIT_BOUNDS[0]:
        return "train"
    if h < AUTO_SPLIT_BOUNDS[1]:
        return "val"
    return "test"


def iter_images(spec: DatasetSpec, split: str | None = None):
    """遍历 (image_path, label_path, image_rel, assigned_split)。

    以图片为主键：没有 label 文件 / 空标注 → 负样本。
    split=None 时一次扫完整棵 images 树（无划分目录的数据集避免扫三遍）。
    不 sorted()：NFS 上 2 万张图 materialize+排序会卡住。
    """
    images_root = spec.root / "images"
    labels_root = spec.root / "labels"
    if spec.has_split:
        splits = (split,) if split else SPLITS
        for sp in splits:
            scan_root = images_root / sp
            if not scan_root.is_dir():
                continue
            for image_path in scan_root.rglob("*"):
                if not image_path.is_file() or image_path.suffix not in IMAGE_EXTS:
                    continue
                rel = image_path.relative_to(images_root)
                label_path = labels_root / rel.with_suffix(".txt")
                yield image_path, label_path, rel.as_posix(), sp
        return

    for image_path in images_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix not in IMAGE_EXTS:
            continue
        rel = image_path.relative_to(images_root)
        assigned = _auto_split(rel.as_posix())
        if split is not None and assigned != split:
            continue
        label_path = labels_root / rel.with_suffix(".txt")
        yield image_path, label_path, rel.as_posix(), assigned


def rel_image_field(spec: DatasetSpec, image_rel: str) -> str:
    """jsonl 内 image 字段：相对 sft_ship_lora/ 目录。"""
    return f"data/images/{spec.name}/{image_rel}"


# ───────────────────────── 文本生成 ─────────────────────────


def _ref_en(class_zh: str) -> str:
    return EN_BY_ZH.get(class_zh, class_zh)


def _plural_en(ref: str) -> str:
    if ref in IRREGULAR_PLURAL_EN:
        return IRREGULAR_PLURAL_EN[ref]
    if ref.endswith("s"):
        return ref
    return f"{ref}s"


def _fmt_box(box: list[float], space: str) -> str:
    if space == "percent":
        vals = [f"{float(v):.2f}" for v in box[:4]]
        return f"[[{','.join(vals)}]]"
    vals = [str(int(round(float(v)))) for v in box[:4]]
    return f"[[{','.join(vals)}]]"


def build_thinking(class_zh: str, boxes_norm: list[list[float]], space: str) -> str:
    """TVP collator 会外包 <think>；这里只写内部正文（英文 + ref/box）。

    boxes_norm 为空 → 负样本：不枚举任何 ref/box，收尾写 0。
    结构与正样本保持同构，解析逻辑不用分支。
    """
    ref = _ref_en(class_zh)
    n = len(boxes_norm)
    lines = [
        f"The user asks me to count the {ref} in this image.",
        "Let me count:",
    ]
    if n == 0:
        lines.append(f"I scan the image but find no {ref}.")
    for i, box in enumerate(boxes_norm, 1):
        lines.append(
            f"{i}. <|ref|>{ref}<|/ref|><|box|>{_fmt_box(box, space)}<|/box|>."
        )
    verb = "is" if n == 1 else "are"
    plural = ref if n == 1 else _plural_en(ref)
    lines.append(f"Counting them up, there {verb} {n} {plural} in this image.")
    return "\n".join(lines)


def _question_noun_phrase(class_zh: str) -> str:
    """英文问句里的复数名词短语（如 fishing boats, vessels）。"""
    return _plural_en(_ref_en(class_zh)).replace("_", " ")


def count_question(class_zh: str) -> str:
    noun = _question_noun_phrase(class_zh)
    return f"How many {noun} are visible in this aerial image?"


QUESTION_RE = re.compile(r"How many (.+?) are visible")


def make_record(
    image_field: str,
    class_zh: str,
    boxes_px: list[list[int]],
    image_size: tuple[int, int],
    space: str,
    decimals: int,
    dataset: str,
) -> dict:
    w, h = image_size
    boxes_norm = boxes_px_to_space(boxes_px, w, h, space, decimals)
    return {
        "image": image_field,
        "image_size": [w, h],
        "coord_space": space,
        "dataset": dataset,
        "question": count_question(class_zh),
        "thinking": build_thinking(class_zh, boxes_norm, space),
        "answer": str(len(boxes_px)),
        "boxes": boxes_norm,
        "boxes_px": boxes_px,
        "points": [],
    }


# ───────────────────────── 单数据集构建 ─────────────────────────


def _append_image_records(
    spec: DatasetSpec,
    names: list[str],
    image_path: Path,
    label_path: Path,
    image_rel: str,
    space: str,
    decimals: int,
    positives: list[dict],
    negative_images: list[tuple[str, tuple[int, int]]],
) -> str:
    """处理一张图。返回 'pos' / 'neg' / 'bad'。"""
    try:
        w, h = image_size(image_path)
    except Exception as e:
        print(f"[{spec.name}] 跳过无法读取的图片 {image_path}: {e}", file=sys.stderr)
        return "bad"
    by_class = parse_yolo_label(label_path, names, w, h)
    image_field = rel_image_field(spec, image_rel)
    if not by_class:
        negative_images.append((image_field, (w, h)))
        return "neg"

    for en, boxes in sorted(by_class.items()):
        if not boxes:
            continue
        zh = spec.classes_zh.get(en, CLASS_NAMES_ZH.get(en, en))
        positives.append(
            make_record(image_field, zh, boxes, (w, h), space, decimals, spec.name)
        )
    if spec.macro_zh:
        macro_boxes: list[list[int]] = []
        for en, boxes in by_class.items():
            if en in spec.macro_classes:
                macro_boxes.extend(boxes)
        if macro_boxes:
            positives.append(
                make_record(
                    image_field, spec.macro_zh, macro_boxes, (w, h), space, decimals, spec.name
                )
            )
    return "pos"


def build_dataset_splits(
    spec: DatasetSpec,
    splits: list[str],
    space: str,
    decimals: int,
    neg_ratio: float,
    seed: int,
) -> dict[str, tuple[list[dict], dict]]:
    """一次扫完 images，按 split 分桶后各自配负样本。"""
    names = load_yaml_names(spec.root, spec.classes_en)
    buckets: dict[str, dict] = {
        sp: {"pos": [], "neg_imgs": [], "n_images": 0, "n_bad": 0} for sp in splits
    }
    n_seen = 0
    print(f"[{spec.name}] 扫描 {spec.root / 'images'} …", flush=True)
    for image_path, label_path, image_rel, assigned in iter_images(spec, split=None):
        if assigned not in buckets:
            continue
        b = buckets[assigned]
        b["n_images"] += 1
        n_seen += 1
        if n_seen % 1000 == 0:
            print(f"[{spec.name}] 已扫描 {n_seen} 张", flush=True)
        status = _append_image_records(
            spec, names, image_path, label_path, image_rel, space, decimals, b["pos"], b["neg_imgs"]
        )
        if status == "bad":
            b["n_bad"] += 1
            b["n_images"] -= 1

    out: dict[str, tuple[list[dict], dict]] = {}
    for sp in splits:
        b = buckets[sp]
        negatives: list[dict] = []
        if neg_ratio > 0 and b["neg_imgs"] and b["pos"]:
            negatives = _build_negatives(
                b["pos"], b["neg_imgs"], space, decimals, neg_ratio, seed, f"{spec.name}:{sp}"
            )
        stats = {
            "images": b["n_images"] + b["n_bad"],
            "bad_images": b["n_bad"],
            "positive_images": len({r["image"] for r in b["pos"]}),
            "negative_images_available": len(b["neg_imgs"]),
            "positives": len(b["pos"]),
            "negatives": len(negatives),
        }
        out[sp] = (b["pos"] + negatives, stats)
        print(
            f"[{spec.name}/{sp}] 正 {stats['positives']} / 负 {stats['negatives']} "
            f"（图 {b['n_images']}，空标注 {stats['negative_images_available']}）",
            flush=True,
        )
    return out


def build_dataset_split(
    spec: DatasetSpec,
    split: str,
    space: str,
    decimals: int,
    neg_ratio: float,
    seed: int,
) -> tuple[list[dict], dict]:
    """兼容单 split 调用。"""
    return build_dataset_splits(spec, [split], space, decimals, neg_ratio, seed)[split]


def _build_negatives(
    positives: list[dict],
    negative_images: list[tuple[str, tuple[int, int]]],
    space: str,
    decimals: int,
    neg_ratio: float,
    seed: int,
    dataset: str,
) -> list[dict]:
    """负样本：数量 = neg_ratio × 正样本行数；提问类别按正样本分布抽样。

    负样本图不够时：多类别数据集整轮复用（配不同提问类别）；
    单类别数据集复用只会产生完全重复的行，因此按可用图数截断。
    """
    rng = random.Random(f"{seed}:{dataset}")

    dist: Counter[str] = Counter()
    for rec in positives:
        m = QUESTION_RE.search(rec["question"])
        if not m:
            continue
        phrase = m.group(1)
        zh = _PHRASE_TO_ZH.get(phrase)
        if zh:
            dist[zh] += 1
    if not dist:
        return []

    classes = list(dist)
    weights = [dist[c] for c in classes]

    want = int(round(neg_ratio * len(positives)))
    pool = list(negative_images)
    rng.shuffle(pool)
    if want <= len(pool):
        chosen = pool[:want]
    elif len(classes) == 1:
        chosen = pool
    else:
        chosen = [pool[i % len(pool)] for i in range(want)]

    out = []
    for image_field, (w, h) in chosen:
        class_zh = rng.choices(classes, weights=weights, k=1)[0]
        out.append(make_record(image_field, class_zh, [], (w, h), space, decimals, dataset))
    return out


# 英文复数短语 → 中文类名（细类 + 宏观类）
_PHRASE_TO_ZH: dict[str, str] = {
    _question_noun_phrase(zh): zh for zh in EN_BY_ZH
}


# ───────────────────────── 软链 / 输出 ─────────────────────────


def ensure_image_symlinks(specs: list[DatasetSpec]) -> None:
    """data/images/<name> → <root>/images，使用相对路径以便容器内同样可解析。"""
    IMAGES_LINK_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        link = IMAGES_LINK_DIR / spec.name
        target = spec.root / "images"
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise SystemExit(f"{link} 已存在且不是软链，请手动处理")
        import os

        rel_target = os.path.relpath(target, IMAGES_LINK_DIR)
        link.symlink_to(rel_target, target_is_directory=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="多数据集 YOLO → 计数 SFT jsonl")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[d.name for d in ALL_DATASETS],
        choices=list(DATASETS_BY_NAME),
        help=f"使用哪些数据集（默认全部：{' '.join(DATASETS_BY_NAME)}）",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["test"],
        choices=list(SPLITS),
        help="使用哪些划分（默认仅 test）",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="jsonl 输出目录")
    parser.add_argument(
        "--coord-space",
        choices=COORD_SPACES,
        default="percent",
        help="boxes 使用的坐标空间（默认 percent：0–100 百分比，原点左上角）",
    )
    parser.add_argument(
        "--percent-decimals",
        type=int,
        default=DEFAULT_PERCENT_DECIMALS,
        help=f"percent 模式保留小数位（默认 {DEFAULT_PERCENT_DECIMALS}）",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="每个数据集负样本行数 / 正样本行数，默认 1.0（1:1）；0 = 不加负样本",
    )
    parser.add_argument("--seed", type=int, default=0, help="负样本采样 / 打乱随机种子")
    args = parser.parse_args()

    specs = [DATASETS_BY_NAME[n] for n in args.datasets]
    for spec in specs:
        if not (spec.root / "images").is_dir():
            print(f"未找到 images 目录: {spec.root}", file=sys.stderr)
            return 1

    ensure_image_symlinks(specs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    built: dict[str, dict[str, tuple[list[dict], dict]]] = {}
    for spec in specs:
        built[spec.name] = build_dataset_splits(
            spec,
            list(args.split),
            args.coord_space,
            args.percent_decimals,
            args.neg_ratio,
            args.seed,
        )

    summary: dict = {}
    for split in args.split:
        merged: list[dict] = []
        per_ds: dict = {}
        for spec in specs:
            rows, stats = built[spec.name][split]
            per_path = args.out_dir / f"{split}.{spec.name}.jsonl"
            _write_jsonl(per_path, rows)
            stats = dict(stats)
            stats["path"] = str(per_path)
            per_ds[spec.name] = stats
            merged.extend(rows)
            print(
                f"{split}/{spec.name}: {len(rows)} 条"
                f"（正 {stats['positives']} / 负 {stats['negatives']}，"
                f"图 {stats['images']}，空标注图 {stats['negative_images_available']}）",
                flush=True,
            )

        random.Random(args.seed).shuffle(merged)
        out_path = args.out_dir / f"{split}.jsonl"
        _write_jsonl(out_path, merged)
        n_neg = sum(1 for r in merged if r["answer"] == "0")
        summary[split] = {
            "samples": len(merged),
            "positives": len(merged) - n_neg,
            "negatives": n_neg,
            "path": str(out_path),
            "datasets": per_ds,
        }
        print(
            f"{split}: 合计 {len(merged)} 条（正 {len(merged) - n_neg} / 负 {n_neg}）-> {out_path}",
            flush=True,
        )

    meta_path = args.out_dir / "build_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "datasets": {s.name: str(s.root) for s in specs},
                "splits": summary,
                "coord_space": args.coord_space,
                "percent_decimals": args.percent_decimals,
                "neg_ratio": args.neg_ratio,
                "seed": args.seed,
                "auto_split_bounds": AUTO_SPLIT_BOUNDS,
                "coord_note": (
                    "boxes are in coord_space relative to the ORIGINAL image "
                    "(percent: 0-100 from top-left); boxes_px = pixel GT for eval"
                ),
                "record_keys": [
                    "image",
                    "image_size",
                    "coord_space",
                    "dataset",
                    "question",
                    "thinking",
                    "answer",
                    "boxes",
                    "boxes_px",
                    "points",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
