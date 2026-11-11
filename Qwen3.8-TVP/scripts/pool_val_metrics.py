#!/usr/bin/env python3
"""Pool OmniGuard val shards into the comparison table.

Hits are ``box_recall_iou50 * gt_boxes``. Count rates and format are weighted
by ``n``. Empty-image and positive-image accuracy are weighted by ``neg_n``
and ``pos_n``. Mean matched IoU is weighted by hits.

The size-bucket precision/recall in README.md is not in these shard files.
Only ``small_recall_iou50`` is.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw_metrics"

GROUPS = (
    ("sft_exp1", "SFT scheme 1", ("exp1_s0", "exp1_s1", "exp1_s2", "exp1_s3")),
    ("sft_exp3", "SFT scheme 3", ("exp3_s0", "exp3_s1", "exp3_s2", "exp3_s3")),
    ("grpo_f1_428", "GRPO F1 428", ("grpo428_s0", "grpo428_s1", "grpo428_s2", "grpo428_s3")),
    ("grpo_f1_1000", "GRPO F1 1000", ("grpo1000_s0", "grpo1000_s1", "grpo1000_s2", "grpo1000_s3")),
    ("grpo_boxpen_1000", "GRPO new box 1000", ("boxpen1000_s0", "boxpen1000_s1", "boxpen1000_s2")),
    ("grpo_boxpen2_partial", "GRPO new box +1000, partial", ("boxpen2_1000_s0", "boxpen2_1000_s1")),
)


def load(tag: str) -> dict:
    path = RAW / f"{tag}.metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def pool(tags: tuple[str, ...]) -> dict:
    acc = {
        "n": 0.0,
        "exact": 0.0,
        "mae": 0.0,
        "fmt": 0.0,
        "pred": 0.0,
        "gt": 0.0,
        "hits": 0.0,
        "iou": 0.0,
        "neg_n": 0.0,
        "neg_c": 0.0,
        "pos_n": 0.0,
        "pos_c": 0.0,
        "small_gt": 0.0,
        "small_hit": 0.0,
    }
    for tag in tags:
        row = load(tag)
        n = float(row["n"])
        hits = float(row["box_recall_iou50"]) * float(row["gt_boxes"])
        acc["n"] += n
        acc["exact"] += float(row["count_exact"]) * n
        acc["mae"] += float(row["count_mae"]) * n
        acc["fmt"] += float(row["format_ok"]) * n
        acc["pred"] += float(row["pred_boxes"])
        acc["gt"] += float(row["gt_boxes"])
        acc["hits"] += hits
        acc["iou"] += float(row["mean_matched_iou"]) * hits
        acc["neg_n"] += float(row["neg_n"])
        acc["neg_c"] += float(row["neg_acc"]) * float(row["neg_n"])
        acc["pos_n"] += float(row["pos_n"])
        acc["pos_c"] += float(row["pos_acc"]) * float(row["pos_n"])
        acc["small_gt"] += float(row["small_gt"])
        acc["small_hit"] += float(row["small_recall_iou50"]) * float(row["small_gt"])
    n = acc["n"]
    hits = acc["hits"]
    precision = hits / acc["pred"]
    recall = hits / acc["gt"]
    return {
        "n": int(n),
        "count_exact": acc["exact"] / n,
        "count_mae": acc["mae"] / n,
        "neg_acc": acc["neg_c"] / acc["neg_n"],
        "pos_acc": acc["pos_c"] / acc["pos_n"],
        "format_ok": acc["fmt"] / n,
        "pred_boxes": int(acc["pred"]),
        "gt_boxes": int(acc["gt"]),
        "hits": int(round(hits)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "mean_matched_iou": acc["iou"] / hits,
        "small_recall": acc["small_hit"] / acc["small_gt"],
    }


def main() -> None:
    header = (
        "| method | n | exact | MAE | empty | pos | P | R | F1 | pred | hits | IoU | small R |"
    )
    rule = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    print(header)
    print(rule)
    for _key, name, tags in GROUPS:
        m = pool(tags)
        print(
            "| {name} | {n} | {exact:.2%} | {mae:.3f} | {neg:.2%} | {pos:.2%} | "
            "{p:.2%} | {r:.2%} | {f1:.2%} | {pred:,} | {hits:,} | {iou:.3f} | {sr:.2%} |".format(
                name=name,
                n=m["n"],
                exact=m["count_exact"],
                mae=m["count_mae"],
                neg=m["neg_acc"],
                pos=m["pos_acc"],
                p=m["precision"],
                r=m["recall"],
                f1=m["f1"],
                pred=m["pred_boxes"],
                hits=m["hits"],
                iou=m["mean_matched_iou"],
                sr=m["small_recall"],
            )
        )


if __name__ == "__main__":
    main()
