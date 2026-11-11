#!/usr/bin/env python3
"""用标准答案上下浮动个数，做成 5 条一组的 GRPO 数据。

每组五个个数：真值-2、-1、0、+1、+2。小于 0 的改成还没出现过的更大整数，
保证五个数不同。框的条数和这个数一致：少了就删掉后面的标准框，多了就把
已有框平移到和标注对不上的位置。文本格式与模型采样相同。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TVP = Path(__file__).resolve().parent
sys.path.insert(0, str(TVP))
sys.path.insert(0, str(TVP.parent / "sft_ship_lora"))

from build_omniguard_jsonl import thinking  # noqa: E402
from grpo_train import box_f1, reward_of  # noqa: E402

NAME_RE = re.compile(r"count the (\S+) in this image")


def target_counts(gt: int) -> list[int]:
    picked: list[int] = []
    for delta in (-2, -1, 0, 1, 2):
        value = gt + delta
        if value >= 0 and value not in picked:
            picked.append(value)
    extra = (max(picked) if picked else gt) + 1
    while len(picked) < 5:
        picked.append(extra)
        extra += 1
    return picked


def _clamp_box(box: list[float]) -> list[float] | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > 100:
        x1 -= x2 - 100
        x2 = 100.0
    if y2 > 100:
        y1 -= y2 - 100
        y2 = 100.0
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(100.0, x2), min(100.0, y2)
    if x2 - x1 < 0.2 or y2 - y1 < 0.2:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def displaced(box: list[float], gt_boxes: list[list[float]], attempt: int) -> list[float]:
    w = max(0.4, box[2] - box[0])
    h = max(0.4, box[3] - box[1])
    sign_x = 1 if attempt % 2 == 0 else -1
    sign_y = 1 if (attempt // 2) % 2 == 0 else -1
    for scale in range(1, 6):
        dx = sign_x * (w + 4.0) * scale
        dy = sign_y * (h + 4.0) * scale
        moved = _clamp_box([box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy])
        if moved is None:
            continue
        if box_f1([box], [moved]) < 0.5 and box_f1(gt_boxes, [moved]) == 0.0:
            return moved
    # 大框平移后仍重叠时，放到左上角的一小块，至少不和原框重合。
    slot = attempt % 8
    x = 2.0 + (slot % 4) * 8.0
    y = 2.0 + (slot // 4) * 8.0
    return [x, y, round(x + min(w, 3.0), 2), round(y + min(h, 3.0), 2)]


def boxes_for_count(gt_boxes: list[list[float]], count: int) -> list[list[float]]:
    if count <= 0:
        return []
    if count <= len(gt_boxes):
        return [list(box) for box in gt_boxes[:count]]
    out = [list(box) for box in gt_boxes]
    seed = gt_boxes[-1] if gt_boxes else [8.0, 8.0, 12.0, 11.0]
    while len(out) < count:
        out.append(displaced(seed, gt_boxes, len(out)))
    return out


def class_name(row: dict) -> str:
    match = NAME_RE.search(row.get("thinking") or "")
    if not match:
        raise ValueError(f"no class name in thinking: {row.get('question')}")
    return match.group(1)


def response_text(name: str, boxes: list[list[float]], count: int) -> str:
    body = thinking(name, boxes)
    return f"{body}\n</think>\n\n{count}<|im_end|>"


def build_group(row: dict) -> dict:
    gt = int(row["answer"])
    gt_boxes = row.get("boxes") or []
    name = class_name(row)
    responses = []
    for count in target_counts(gt):
        pred_boxes = boxes_for_count(gt_boxes, count)
        text = response_text(name, pred_boxes, count)
        scored = reward_of(text, gt, gt_boxes)
        responses.append({"text": text, "pred_boxes": pred_boxes, **scored})
    rewards = [item["reward"] for item in responses]
    mean = sum(rewards) / len(rewards)
    var = sum((item - mean) ** 2 for item in rewards) / len(rewards)
    return {
        "manifest_i": row.get("manifest_i"),
        "source_i": row.get("source_i"),
        "image": row["image"],
        "question": row["question"],
        "gt": gt,
        "boxes": gt_boxes,
        "image_size": row.get("image_size"),
        "source": "count_shift",
        "responses": responses,
        "reward_mean": mean,
        "reward_std": var ** 0.5,
    }


def main() -> int:
    src = TVP / "data" / "jsonl" / "train.jsonl"
    out = TVP / "grpo_data" / "exp3" / "shifted_groups.jsonl"
    n = flat = skipped = 0
    under = over = paired = 0
    under_sum = over_sum = 0.0
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for source_i, line in enumerate(fin):
            if not line.strip():
                continue
            row = json.loads(line)
            row["source_i"] = source_i
            row["manifest_i"] = source_i
            try:
                group = build_group(row)
            except ValueError:
                skipped += 1
                continue
            rewards = {item["pred"]: item["reward"] for item in group["responses"]}
            gt = group["gt"]
            if gt - 1 in rewards and gt + 1 in rewards:
                paired += 1
                under_sum += rewards[gt - 1]
                over_sum += rewards[gt + 1]
                if rewards[gt - 1] < rewards[gt + 1]:
                    under += 1
                elif rewards[gt + 1] < rewards[gt - 1]:
                    over += 1
            if group["reward_std"] < 1e-6:
                flat += 1
            fout.write(json.dumps(group, ensure_ascii=False) + "\n")
            n += 1
    print(
        f"groups={n} flat={flat} skipped={skipped} out={out} "
        f"paired_pm1={paired} under_lower={under} over_lower={over} "
        f"mean_reward_gt-1={under_sum / paired if paired else 0:.4f} "
        f"mean_reward_gt+1={over_sum / paired if paired else 0:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
