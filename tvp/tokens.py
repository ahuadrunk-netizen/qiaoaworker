"""视觉基元特殊 token。Qwen3.8 词表里已经有这几个，add 是幂等的。"""

REF_START = "<|ref|>"
REF_END = "<|/ref|>"
BOX_START = "<|box|>"
BOX_END = "<|/box|>"
POINT_START = "<|point|>"
POINT_END = "<|/point|>"

SPECIAL_TOKENS = [
    REF_START, REF_END,
    BOX_START, BOX_END,
    POINT_START, POINT_END,
]


def add_special_tokens(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer
