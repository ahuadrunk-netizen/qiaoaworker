#!/usr/bin/env python3
"""在 val 上评测一个 LoRA adapter（计数 + 框）。

三个 step_1200 adapter 是在船舶 jsonl 上训的，默认评 data/jsonl/val.ship.jsonl。
prompt / 图像像素预算与训练一致（system prompt、percent 框、max_pixels=2007040）。

    python3 eval_lora_val.py \
        --adapter outputs/lora_exp1/epoch_0_step_1200 \
        --tag exp1 --device npu:0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL = "/workspace/zzy/modelscope/Qwen3.8-27B"
SYSTEM = "You are a helpful assistant that can understand images and reason with visual primitives."
MIN_PIXELS = 12544
MAX_PIXELS = 2007040
SMALL_AREA_PX = 32 * 32

BOX_RE = re.compile(
    r"<\|box\|>\s*\[(.*?)\]\s*<\|/box\|>",
    re.DOTALL,
)
NUM4_RE = re.compile(
    r"\[\s*(-?[0-9]*\.?[0-9]+)\s*,\s*(-?[0-9]*\.?[0-9]+)\s*,\s*"
    r"(-?[0-9]*\.?[0-9]+)\s*,\s*(-?[0-9]*\.?[0-9]+)\s*\]"
)
REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
COUNT_TAIL_RE = re.compile(r"-?\d+")
ASKED_RE = re.compile(r"(?i)how many (.+?) are visible")


def parse_count(text: str) -> int | None:
    tail = text.split("</think>")[-1] if "</think>" in text else text
    m = COUNT_TAIL_RE.search(tail)
    if m:
        return int(m.group())
    m = re.search(r"there (?:is|are) (\d+)", text)
    return int(m.group(1)) if m else None


def parse_boxes(text: str) -> tuple[list[list[float]], bool]:
    """返回 (boxes, format_ok)。format_ok：每个 <|box|> 里都能解析出 4 个数。"""
    boxes: list[list[float]] = []
    tags = list(BOX_RE.finditer(text))
    if not tags:
        return boxes, "<|box|>" not in text
    ok = True
    for m in tags:
        found = list(NUM4_RE.finditer(m.group(1)))
        if not found:
            ok = False
            continue
        for g in found:
            box = [float(g.group(i)) for i in range(1, 5)]
            if not all(0.0 <= v <= 100.0 for v in box):
                ok = False
            boxes.append(box)
    return boxes, ok


def expected_ref(question: str, gt_thinking: str) -> str | None:
    refs = REF_RE.findall(gt_thinking or "")
    if refs:
        return refs[0].strip()
    m = ASKED_RE.search(question or "")
    if not m:
        return None
    phrase = m.group(1).strip().lower().replace(" ", "_")
    if phrase.endswith("s") and not phrase.endswith("ss"):
        phrase = phrase[:-1]
    return phrase


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt: list[list[float]], pred: list[list[float]], thr: float = 0.5):
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            pairs.append((iou(g, p), i, j))
    pairs.sort(reverse=True)
    used_g, used_p = set(), set()
    matches = []
    for score, i, j in pairs:
        if score < thr or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matches.append((i, j, score))
    return matches


def box_area_px(box_pct: list[float], wh: list[int]) -> float:
    w, h = wh
    return abs(box_pct[2] - box_pct[0]) / 100.0 * w * abs(box_pct[3] - box_pct[1]) / 100.0 * h


class Meter:
    def __init__(self) -> None:
        self.n = 0
        self.count_hit = 0
        self.count_abs = 0.0
        self.count_parsed = 0
        self.neg = 0
        self.neg_hit = 0
        self.pos = 0
        self.pos_hit = 0
        self.format_ok = 0
        self.ref_ok = 0
        self.ref_n = 0
        self.gt_boxes = 0
        self.pred_boxes = 0
        self.matched = 0
        self.iou_sum = 0.0
        self.mae_sum = 0.0
        self.center_sum = 0.0
        self.small_gt = 0
        self.small_hit = 0

    def add(self, row: dict) -> None:
        self.n += 1
        gt = int(row["gt"])
        pred = row["pred"]
        if pred is not None:
            self.count_parsed += 1
            self.count_abs += abs(pred - gt)
            if pred == gt:
                self.count_hit += 1
        if gt == 0:
            self.neg += 1
            if pred == 0:
                self.neg_hit += 1
        else:
            self.pos += 1
            if pred == gt:
                self.pos_hit += 1
        if row["format_ok"]:
            self.format_ok += 1
        if row["ref_expected"] is not None:
            self.ref_n += 1
            if row["ref_ok"]:
                self.ref_ok += 1
        gt_boxes = row["gt_boxes"]
        pred_boxes = row["pred_boxes"]
        self.gt_boxes += len(gt_boxes)
        self.pred_boxes += len(pred_boxes)
        matches = match_boxes(gt_boxes, pred_boxes)
        self.matched += len(matches)
        wh = row["image_size"]
        small_idx = {i for i, b in enumerate(gt_boxes) if box_area_px(b, wh) < SMALL_AREA_PX}
        self.small_gt += len(small_idx)
        hit_g = {i for i, _, _ in matches}
        self.small_hit += len(small_idx & hit_g)
        for i, j, score in matches:
            self.iou_sum += score
            g, p = gt_boxes[i], pred_boxes[j]
            self.mae_sum += sum(abs(g[k] - p[k]) for k in range(4)) / 4.0
            gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
            pc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            self.center_sum += ((gc[0] - pc[0]) ** 2 + (gc[1] - pc[1]) ** 2) ** 0.5

    def as_dict(self) -> dict:
        n = max(1, self.n)
        prec = self.matched / self.pred_boxes if self.pred_boxes else 0.0
        rec = self.matched / self.gt_boxes if self.gt_boxes else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {
            "n": self.n,
            "count_exact": self.count_hit / n,
            "count_mae": self.count_abs / n,
            "count_parsed": self.count_parsed / n,
            "neg_n": self.neg,
            "neg_acc": self.neg_hit / self.neg if self.neg else None,
            "pos_n": self.pos,
            "pos_acc": self.pos_hit / self.pos if self.pos else None,
            "format_ok": self.format_ok / n,
            "ref_ok": self.ref_ok / self.ref_n if self.ref_n else None,
            "gt_boxes": self.gt_boxes,
            "pred_boxes": self.pred_boxes,
            "box_precision_iou50": prec,
            "box_recall_iou50": rec,
            "box_f1_iou50": f1,
            "mean_matched_iou": self.iou_sum / self.matched if self.matched else None,
            "coord_mae_percent": self.mae_sum / self.matched if self.matched else None,
            "center_err_percent": self.center_sum / self.matched if self.matched else None,
            "small_gt": self.small_gt,
            "small_recall_iou50": self.small_hit / self.small_gt if self.small_gt else None,
        }


def load_model(adapter: str | None, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sys.path.insert(0, "/workspace/MYVLLM/lib/Thinking-with-Visual-Primitives-pytorch")
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

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel
        # ZeRO-3 存 adapter 时 lm_head.base_layer.weight 被写成 shape [0]，丢掉即可。
        import peft.peft_model as _peft_model

        _orig_set = _peft_model.set_peft_model_state_dict

        def _set_skip_empty(model, peft_model_state_dict, *a, **k):
            peft_model_state_dict = {
                kk: vv
                for kk, vv in peft_model_state_dict.items()
                if getattr(vv, "numel", lambda: 1)() > 0
            }
            return _orig_set(model, peft_model_state_dict, *a, **k)

        _peft_model.set_peft_model_state_dict = _set_skip_empty
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.to(device)
    return processor, model


DETECT_SYSTEM = "You are a helpful assistant."
QWEN_BOX_RE = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*"
    r"<\|box_start\|>\s*\(?\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)?\s*,\s*"
    r"\(?\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)?\s*<\|box_end\|>",
    re.DOTALL,
)


def asked_name(question: str, thinking: str) -> str:
    head = (thinking or "").split("\n", 1)[0]
    prefix = "The user asks me to count the "
    suffix = " in this image."
    if head.startswith(prefix) and head.endswith(suffix):
        return head[len(prefix) : -len(suffix)]
    m = ASKED_RE.search(question or "")
    if not m:
        return ""
    return m.group(1).strip().lower().replace(" ", "_")


def detect_prompt(name: str) -> str:
    phrase = name.replace("_", " ")
    return (
        f"Locate every {phrase} in this image. "
        "For each object, give its category and bounding box. "
        "Reply with only a JSON list. Each item must be "
        '{"label": "<category>", "bbox_2d": [x1, y1, x2, y2]}. '
        "Use integer coordinates from 0 to 1000, relative to image width and height. "
        "If there are none, reply []."
    )


def _norm_label(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text.lower()).strip()


def label_matches(label: str, name: str) -> bool:
    if not label.strip():
        return True
    a, b = _norm_label(label), _norm_label(name)
    if a == b or a.rstrip("s") == b.rstrip("s"):
        return True
    irregular = {"person": "people", "people": "person"}
    return irregular.get(a) == b or irregular.get(b) == a


def _coord_scale(vals: list[float]) -> float:
    peak = max((abs(v) for v in vals), default=0.0)
    if peak <= 1.5:
        return 100.0
    if peak <= 100.0:
        return 1.0
    return 0.1


def parse_detections(text: str) -> list[tuple[str, list[float]]] | None:
    body = text.split("</think>")[-1] if "</think>" in text else text
    native = list(QWEN_BOX_RE.finditer(body))
    if native:
        return [
            (m.group(1).strip(), [float(m.group(i)) for i in range(2, 6)])
            for m in native
        ]
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", body, re.DOTALL)
    if fenced:
        blob = fenced.group(1)
    else:
        start, end = body.find("["), body.rfind("]")
        if start < 0 or end <= start:
            if re.search(r"\bno objects\b|\bthere are no\b|\bnone\b", body, re.IGNORECASE):
                return []
            return None
        blob = body[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    if len(data) == 0:
        return []
    items: list[tuple[str, list[float]]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        box = obj.get("bbox_2d", obj.get("bbox", obj.get("box")))
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        label = str(obj.get("label") or obj.get("category") or "")
        items.append((label, [float(box[i]) for i in range(4)]))
    return items


def score_detect(question: str, thinking: str, answer: str, boxes, image_size, text: str) -> dict:
    name = asked_name(question, thinking)
    found = parse_detections(text)
    pred_boxes: list[list[float]] = []
    pred = None
    if found is not None:
        kept = [(label, box) for label, box in found if label_matches(label, name)]
        flat = [v for _, box in kept for v in box]
        scale = _coord_scale(flat)
        pred_boxes = [[v * scale for v in box] for _, box in kept]
        pred = len(pred_boxes)
    return {
        "gt": int(answer),
        "pred": pred,
        "gt_boxes": boxes or [],
        "pred_boxes": pred_boxes,
        "format_ok": found is not None,
        "ref_expected": name or None,
        "ref_ok": found is not None and all(label_matches(label, name) for label, _ in found),
        "image_size": image_size,
        "text": text,
    }


def generate_one(
    processor,
    model,
    image_path: Path,
    question: str,
    device: str,
    max_new_tokens: int,
    thinking: bool = True,
    system: str = SYSTEM,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen = out[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen, skip_special_tokens=False)


def score_text(question: str, thinking: str, answer: str, boxes, image_size, text: str) -> dict:
    pred = parse_count(text)
    pred_boxes, format_ok = parse_boxes(text)
    if "</think>" not in text or pred is None:
        format_ok = False
    exp = expected_ref(question, thinking)
    refs = [r.strip() for r in REF_RE.findall(text)]
    ref_ok = exp is None or (all(r == exp for r in refs) if refs else int(answer) == 0)
    return {
        "gt": int(answer),
        "pred": pred,
        "gt_boxes": boxes or [],
        "pred_boxes": pred_boxes,
        "format_ok": format_ok,
        "ref_expected": exp,
        "ref_ok": ref_ok,
        "image_size": image_size,
        "text": text,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--jsonl", type=Path, default=ROOT / "data/jsonl/val.ship.jsonl")
    p.add_argument("--image-root", type=Path, default=ROOT)
    p.add_argument("--device", type=str, default="npu:0")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--task", choices=("count", "detect"), default="count")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "eval_val")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / f"{args.tag}.jsonl"
    done = 0
    if pred_path.is_file():
        done = sum(1 for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip())
    adapter = str(args.adapter) if args.adapter else None
    print(f"[{args.tag}] samples={len(rows)} resume={done} adapter={adapter or 'base'}", flush=True)

    processor, model = load_model(adapter, args.device)
    meter = Meter()
    if done:
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                meter.add(json.loads(line))

    t0 = time.time()
    with pred_path.open("a", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            if i < done:
                continue
            image = (args.image_root / row["image"]).resolve()
            if not image.is_file():
                print(f"[{args.tag}] missing {image}", flush=True)
                text = ""
            else:
                if args.task == "detect":
                    question = detect_prompt(asked_name(row["question"], row.get("thinking", "")))
                    text = generate_one(
                        processor,
                        model,
                        image,
                        question,
                        args.device,
                        args.max_new_tokens,
                        thinking=False,
                        system=DETECT_SYSTEM,
                    )
                else:
                    question = row["question"]
                    text = generate_one(
                        processor, model, image, question, args.device, args.max_new_tokens
                    )
            scorer = score_detect if args.task == "detect" else score_text
            scored = scorer(
                row["question"],
                row.get("thinking", ""),
                row["answer"],
                row.get("boxes") or [],
                row["image_size"],
                text,
            )
            rec = {
                "i": i,
                "image": row["image"],
                "question": row["question"],
                "gt": scored["gt"],
                "pred": scored["pred"],
                "n_gt_boxes": len(scored["gt_boxes"]),
                "n_pred_boxes": len(scored["pred_boxes"]),
                "format_ok": scored["format_ok"],
                "ref_ok": scored["ref_ok"],
                "ref_expected": scored["ref_expected"],
                "gt_boxes": scored["gt_boxes"],
                "pred_boxes": scored["pred_boxes"],
                "image_size": scored["image_size"],
                "text": scored["text"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            meter.add(scored)
            if (i + 1) % 10 == 0 or i + 1 == len(rows):
                dt = time.time() - t0
                seen = i + 1 - done
                rate = seen / dt if dt and seen else 0
                left = (len(rows) - i - 1) / rate if rate else 0
                m = meter.as_dict()
                print(
                    f"[{args.tag}] {i+1}/{len(rows)} exact={m['count_exact']:.3f} "
                    f"mae={m['count_mae']:.3f} f1={m['box_f1_iou50']:.3f} "
                    f"{rate:.2f} samp/s eta={left/3600:.1f}h",
                    flush=True,
                )

    summary = meter.as_dict()
    summary["tag"] = args.tag
    summary["adapter"] = adapter or "base"
    summary["task"] = args.task
    summary["jsonl"] = str(args.jsonl)
    out_metrics = args.out_dir / f"{args.tag}.metrics.json"
    out_metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
