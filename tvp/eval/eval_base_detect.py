#!/usr/bin/env python3
"""原始 Qwen3.8：每张图只报每个目标的类别和框，再按验证集的类对齐打分。

坐标按 Qwen 的 0–1000 转成和 SFT 一样的 0–100 百分比。计数是该类框的个数。
框的精确率、召回、F1、IoU 与 eval_lora_val.Meter 相同。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

QWEN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(QWEN / "sft_ship_lora"))
import eval_lora_val as ev  # noqa: E402

MODEL = "/workspace/zzy/modelscope/Qwen3.8-27B"
MIN_PIXELS = 12544
MAX_PIXELS = 2007040
PROMPT = (
    "List every object in this aerial image.\n"
    "For each object write exactly one line:\n"
    "<|object_ref_start|>category<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
    "category is a short English name. "
    "x1, y1, x2, y2 are integers from 0 to 1000. "
    "Origin is the top-left corner, x to the right, y downward. "
    "Do not explain."
)

REF_BOX_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*"
    r"<\|box_start\|>(.*?)<\|box_end\|>",
    re.DOTALL,
)
BOX_ONLY_RE = re.compile(r"<\|box_start\|>(.*?)<\|box_end\|>", re.DOTALL)
NUMS_RE = re.compile(r"-?\d+(?:\.\d+)?")
CLASS_RE = re.compile(r"count the (\S+) in this image")


def norm_label(text: str) -> str:
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text == "buses":
        return "bus"
    if text.endswith("s") and not text.endswith("ss") and text not in {"people"}:
        text = text[:-1]
    return text


def same_class(pred: str, gt_name: str) -> bool:
    return norm_label(pred) == norm_label(gt_name.replace("_", " "))


def to_percent(vals: list[float], wh: list[int]) -> list[float] | None:
    if len(vals) != 4:
        return None
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    peak = max(abs(x1), abs(y1), abs(x2), abs(y2))
    w, h = wh
    if peak <= 100:
        box = [x1, y1, x2, y2]
    elif peak <= 1000:
        box = [v / 10.0 for v in (x1, y1, x2, y2)]
    elif w and h:
        box = [x1 / w * 100, y1 / h * 100, x2 / w * 100, y2 / h * 100]
    else:
        return None
    return [min(100.0, max(0.0, v)) for v in box]


def _pairs_from_obj(obj: dict) -> tuple[str, list[float]] | None:
    if not isinstance(obj, dict):
        return None
    label = obj.get("label") or obj.get("category") or obj.get("class") or ""
    raw = obj.get("bbox_2d") or obj.get("bbox") or obj.get("box")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        vals = [float(v) for v in raw[:4]]
    except (TypeError, ValueError):
        return None
    return str(label), vals


def _json_items(text: str) -> list[tuple[str, list[float]]] | None:
    start = 0
    while True:
        begin = text.find("[", start)
        if begin < 0:
            return None
        depth = 0
        end = None
        for i in range(begin, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            return None
        try:
            data = json.loads(text[begin : end + 1])
        except json.JSONDecodeError:
            start = begin + 1
            continue
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            items = []
            for obj in data:
                pair = _pairs_from_obj(obj)
                if pair:
                    items.append(pair)
            return items
        start = end + 1


def _nums(chunk: str) -> list[float] | None:
    found = [float(x) for x in NUMS_RE.findall(chunk)]
    if len(found) < 4:
        return None
    return found[:4]


def parse_detections(text: str, wh: list[int]) -> tuple[list[tuple[str, list[float]]], bool]:
    """返回 ([(label, percent box)], 是否解析出结构)。"""
    body = text.split("</think>")[-1] if "</think>" in text else text
    raw: list[tuple[str, list[float]]] = []
    for label, chunk in REF_BOX_RE.findall(body):
        vals = _nums(chunk)
        if vals:
            raw.append((label.strip(), vals))
    parsed = bool(raw)
    if not raw:
        found = _json_items(body)
        if found is not None:
            raw = found
            parsed = True
        else:
            parsed = False
    out = []
    for label, vals in raw:
        box = to_percent(vals, wh)
        if box is None:
            continue
        if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
            continue
        out.append((label, box))
    if raw and not out:
        parsed = True
    return out, parsed or bool(out)


def class_of(row: dict) -> str:
    m = CLASS_RE.search(row.get("thinking") or "")
    if m:
        return m.group(1)
    return ""


def load_base(device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
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
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    model.to(device)
    return processor, model


def generate(processor, model, image_path: Path, device: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "You locate objects and report a category and a box for each one."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    try:
        text = processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        text = processor.apply_chat_template(messages, **kwargs)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen, skip_special_tokens=False)


def group_images(path: Path) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = row["image"]
        if image not in groups:
            order.append(image)
        groups[image].append(row)
    return [(image, groups[image]) for image in order]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    args = p.parse_args()

    groups = [
        item for i, item in enumerate(group_images(args.jsonl)) if i % args.num_shards == args.shard
    ]
    if args.limit:
        groups = groups[: args.limit]
    done_images = set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_images.add(json.loads(line)["image"])
    print(
        f"[base-det s{args.shard}] images={len(groups)} resume_images={len(done_images)}",
        flush=True,
    )
    processor, model = load_base(args.device)
    meter = ev.Meter()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                meter.add(json.loads(line))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seen = 0
    with args.out.open("a", encoding="utf-8") as fout:
        for image_rel, rows in groups:
            if image_rel in done_images:
                continue
            image = (args.image_root / image_rel).resolve()
            wh = rows[0]["image_size"]
            if not image.is_file():
                text, dets, parsed = "", [], False
            else:
                text = generate(processor, model, image, args.device, args.max_new_tokens)
                dets, parsed = parse_detections(text, wh)
            for row in rows:
                cname = class_of(row)
                kept = [box for label, box in dets if same_class(label, cname)]
                named = any(same_class(label, cname) for label, _ in dets)
                gt = int(row["answer"])
                rec = {
                    "image": image_rel,
                    "question": row["question"],
                    "gt": gt,
                    "pred": len(kept) if parsed else None,
                    "n_gt_boxes": len(row.get("boxes") or []),
                    "n_pred_boxes": len(kept),
                    "format_ok": parsed,
                    "ref_ok": (not named) if gt == 0 else named,
                    "ref_expected": cname,
                    "gt_boxes": row.get("boxes") or [],
                    "pred_boxes": kept,
                    "image_size": wh,
                    "text": text if row is rows[0] else "",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                meter.add(rec)
            fout.flush()
            seen += 1
            if seen % 5 == 0 or seen == 1:
                m = meter.as_dict()
                rate = seen / (time.time() - t0)
                left = (len(groups) - len(done_images) - seen) / rate if rate else 0
                print(
                    f"[base-det s{args.shard}] {seen}/{len(groups)} "
                    f"exact={m['count_exact']:.3f} mae={m['count_mae']:.3f} "
                    f"f1={m['box_f1_iou50']:.3f} parsed={m['format_ok']:.3f} "
                    f"boxes={len(dets)} {rate:.3f} img/s eta={left/3600:.1f}h "
                    f"sample={text[:180].replace(chr(10), ' ')}",
                    flush=True,
                )
    summary = meter.as_dict()
    summary["tag"] = args.out.stem
    summary["adapter"] = "base-detect"
    metrics = args.out.with_suffix(".metrics.json")
    metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
