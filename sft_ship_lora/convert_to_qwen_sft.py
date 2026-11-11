#!/usr/bin/env python3
"""中间 jsonl → Qwen3.8(qwen3_5) SFT 对话格式。

关键：本模型 chat_template.jinja 的 assistant 分支是

    '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content

即 **思维链必须放在 assistant.reasoning_content 字段**，不能内联写 <think> 到 content，
否则渲染出两组 <think>。图像必须用 list content（{"type":"image"}），模板才会插入
<|vision_start|><|image_pad|><|vision_end|>；纯文本写 "[image: x.jpg]" 模型看不到图。

    python3 convert_to_qwen_sft.py --input data/jsonl/train.jsonl
    python3 convert_to_qwen_sft.py --input data/jsonl/train.jsonl --chat-format llamafactory
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from class_names import CLASS_NAMES_ZH, EN_BY_ZH, IRREGULAR_PLURAL_EN
from coords import DEFAULT_PERCENT_DECIMALS

ROOT = Path(__file__).resolve().parent

DEFAULT_SYSTEM = (
    "You are a helpful assistant that can understand images and reason with visual primitives."
)

# 中文问法里的类别 → <|ref|> 内英文标签（含所有数据集的细类 + 宏观类）
ZH_TO_REF_EN: dict[str, str] = dict(EN_BY_ZH)

QUESTION_LABEL_RE_ZH = re.compile(r"有多少(?:个|艘)(.+?)[？?]")
QUESTION_LABEL_RE_EN = re.compile(
    r"(?i)how many (.+?) are visible in this aerial image\??"
)


def _plural_en(ref: str) -> str:
    if ref in IRREGULAR_PLURAL_EN:
        return IRREGULAR_PLURAL_EN[ref]
    return ref if ref.endswith("s") else f"{ref}s"


# 英文问句里的复数短语 → 中文类别名（供 ref_tag 使用）
_EN_PHRASE_TO_ZH: dict[str, str] = {}
for _en, _zh in CLASS_NAMES_ZH.items():
    _EN_PHRASE_TO_ZH[_plural_en(_en).replace("_", " ")] = _zh
    _EN_PHRASE_TO_ZH[_en.replace("_", " ")] = _zh


def parse_label_from_question(question: str) -> str:
    m = QUESTION_LABEL_RE_EN.search(question)
    if m:
        phrase = m.group(1).strip().lower()
        return _EN_PHRASE_TO_ZH.get(phrase, phrase)
    m = QUESTION_LABEL_RE_ZH.search(question)
    return m.group(1).strip() if m else "object"


def ref_tag(label_zh: str, ref_lang: str) -> str:
    if ref_lang == "zh":
        return label_zh
    return ZH_TO_REF_EN.get(label_zh, label_zh.replace(" ", "_"))


def format_box(box: list[float], coord_space: str, decimals: int) -> str:
    """percent → 固定小数位（0–100，左上角原点）；qwen1000 → 整数。"""
    if coord_space == "percent":
        vals = [f"{float(v):.{decimals}f}" for v in box[:4]]
    else:
        vals = [str(int(round(float(v)))) for v in box[:4]]
    return f"[[{','.join(vals)}]]"


def plural(ref: str, n: int) -> str:
    return ref if n == 1 else _plural_en(ref)


def build_reasoning(
    question: str,
    boxes: list[list[float]],
    ref_lang: str,
    coord_space: str,
    decimals: int,
) -> str:
    """<think> 内部内容（不含标签）：开场 + 逐个目标坐标 + 计数收尾。"""
    ref = ref_tag(parse_label_from_question(question), ref_lang)
    n = len(boxes)

    lines = [
        f"The user asks me to count the {ref} in this image.",
        "Let me count:",
    ]
    if n == 0:
        lines.append(f"I scan the image but find no {ref}.")
    for i, box in enumerate(boxes, 1):
        lines.append(
            f"{i}. <|ref|>{ref}<|/ref|>"
            f"<|box|>{format_box(box, coord_space, decimals)}<|/box|>."
        )
    verb = "is" if n == 1 else "are"
    lines.append(
        f"Counting them up, there {verb} {n} {plural(ref, n)} in this image."
    )
    return "\n".join(lines)


def convert_row(
    row: dict,
    *,
    system: str,
    ref_lang: str,
    chat_format: str,
    image_root: Path,
    decimals: int,
) -> dict:
    image_rel = row["image"]
    image_abs = (image_root / image_rel).resolve()
    if not image_abs.is_file():
        image_abs = (ROOT / image_rel).resolve()

    coord_space = row.get("coord_space", "percent")
    question = row["question"]
    answer = str(row.get("answer", "0"))
    reasoning = build_reasoning(
        question=question,
        boxes=row.get("boxes") or [],
        ref_lang=ref_lang,
        coord_space=coord_space,
        decimals=decimals,
    )

    if chat_format == "qwen_native":
        # 与 chat_template.jinja 对齐：图像走 list content，思维链走 reasoning_content
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_abs)},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": answer,
            },
        ]
    else:
        # LLaMA-Factory sharegpt 风格：<image> 占位符 + 内联 think
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"<image>\n{question}"},
            {
                "role": "assistant",
                "content": f"<think>\n{reasoning}\n</think>\n\n{answer}",
            },
        ]

    return {"messages": messages, "images": [str(image_abs)]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ship jsonl to Qwen SFT messages")
    parser.add_argument("--input", type=Path, required=True, help="中间 jsonl（build_jsonl 输出）")
    parser.add_argument(
        "--output", type=Path, default=None, help="默认 data/qwen_sft/<input_stem>.jsonl"
    )
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    parser.add_argument("--ref-lang", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--chat-format",
        choices=("qwen_native", "llamafactory"),
        default="qwen_native",
        help="qwen_native=reasoning_content+图像list（默认，已对齐本模型模板）",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT,
        help="解析 images/ 相对路径的根目录（默认 sft_ship_lora）",
    )
    parser.add_argument(
        "--percent-decimals",
        type=int,
        default=DEFAULT_PERCENT_DECIMALS,
        help=f"percent 坐标写入 <|box|> 的小数位（默认 {DEFAULT_PERCENT_DECIMALS}）",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        print(f"找不到输入: {inp}", file=sys.stderr)
        return 1

    out = args.output or ROOT / "data" / "qwen_sft" / f"{inp.stem}.jsonl"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with inp.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = convert_row(
                json.loads(line),
                system=args.system,
                ref_lang=args.ref_lang,
                chat_format=args.chat_format,
                image_root=args.image_root,
                decimals=args.percent_decimals,
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"写入 {n} 条 ({args.chat_format}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
