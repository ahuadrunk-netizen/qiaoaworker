"""框坐标空间转换。

- percent   : 相对原图宽高的百分比 0–100，原点为图像左上角（本项目默认）
- qwen1000  : Qwen3.8 原生的 0–1000 归一化（qwen38_vision_box.parse_model_boxes 用）
- px        : 原图像素

百分比与 0–1000 只差 10 倍刻度，互转不丢信息（受小数位限制）。
"""

from __future__ import annotations

QWEN_NORM = 1000
PERCENT_NORM = 100
COORD_SPACES = ("percent", "qwen1000")
DEFAULT_PERCENT_DECIMALS = 2


def _clampf(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ---------- 像素 → 归一化 ----------


def xyxy_px_to_percent(
    box: list[int], width: int, height: int, decimals: int = DEFAULT_PERCENT_DECIMALS
) -> list[float]:
    """原图像素 [x1,y1,x2,y2] → 百分比 0–100（左上角为原点）。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [
        round(_clampf(x1 / width * PERCENT_NORM, 0.0, PERCENT_NORM), decimals),
        round(_clampf(y1 / height * PERCENT_NORM, 0.0, PERCENT_NORM), decimals),
        round(_clampf(x2 / width * PERCENT_NORM, 0.0, PERCENT_NORM), decimals),
        round(_clampf(y2 / height * PERCENT_NORM, 0.0, PERCENT_NORM), decimals),
    ]


def xyxy_px_to_qwen1000(box: list[int], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    return [
        clamp(int(round(x1 / width * QWEN_NORM)), 0, QWEN_NORM),
        clamp(int(round(y1 / height * QWEN_NORM)), 0, QWEN_NORM),
        clamp(int(round(x2 / width * QWEN_NORM)), 0, QWEN_NORM),
        clamp(int(round(y2 / height * QWEN_NORM)), 0, QWEN_NORM),
    ]


# ---------- 归一化 → 像素 ----------


def xyxy_percent_to_px(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [
        x1 / PERCENT_NORM * width,
        y1 / PERCENT_NORM * height,
        x2 / PERCENT_NORM * width,
        y2 / PERCENT_NORM * height,
    ]


def xyxy_qwen1000_to_px(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [
        x1 / QWEN_NORM * width,
        y1 / QWEN_NORM * height,
        x2 / QWEN_NORM * width,
        y2 / QWEN_NORM * height,
    ]


# ---------- 按坐标空间分发 ----------


def boxes_px_to_space(
    boxes: list[list[int]],
    width: int,
    height: int,
    space: str,
    decimals: int = DEFAULT_PERCENT_DECIMALS,
) -> list[list[float]] | list[list[int]]:
    if space == "percent":
        return [xyxy_px_to_percent(b, width, height, decimals) for b in boxes]
    if space == "qwen1000":
        return [xyxy_px_to_qwen1000(b, width, height) for b in boxes]
    raise ValueError(f"unknown coord space: {space}")


def box_space_to_px(box: list[float], width: int, height: int, space: str) -> list[float]:
    if space == "percent":
        return xyxy_percent_to_px(box, width, height)
    if space == "qwen1000":
        return xyxy_qwen1000_to_px(box, width, height)
    raise ValueError(f"unknown coord space: {space}")
