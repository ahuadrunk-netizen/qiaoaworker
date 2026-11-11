"""jsonl 样本 → Qwen 对话文本 → processor 输入。

thinking 放进 assistant.reasoning_content，由模型自带的 chat_template.jinja
包成 <think>…</think>。answer 是 </think> 后面的整数。
loss 只算 assistant 这一段。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

SYSTEM_PROMPT = (
    "You are a helpful assistant that can understand images, reason with visual primitives, and excel at image detection, object recognition, precise localization, bounding box prediction, and visual analysis."
)


def build_conversation(sample: dict) -> list[dict]:
    """一条 jsonl 样本 → system / user / assistant。"""
    user_content = []
    image_path = sample.get("image_path")
    if image_path:
        user_content.append({"type": "image", "image": str(image_path)})
    user_content.append({"type": "text", "text": sample.get("question") or "Describe what you see."})

    assistant = {"role": "assistant", "content": (sample.get("answer") or "").strip() or "0"}
    thinking = (sample.get("thinking") or "").strip()
    if thinking:
        assistant["reasoning_content"] = thinking
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        assistant,
    ]


def render_conversation(processor, messages: list[dict]) -> str:
    """套模型目录里的 chat_template.jinja，得到整段训练文本。"""
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


class ConversationCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 8192,
        image_size: int = 1024,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        processor_model_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        if not processor_model_path:
            raise SystemExit("[collate] 缺少 base_model_path，无法加载 processor")
        self.processor = AutoProcessor.from_pretrained(processor_model_path, trust_remote_code=True)
        self._align_image_processor()

    def _align_image_processor(self):
        if self.image_min_pixels is None:
            return
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            return
        from transformers.image_utils import SizeDict

        image_processor.size = SizeDict(
            shortest_edge=self.image_min_pixels,
            longest_edge=self.image_max_pixels,
        )

    def __call__(self, batch_samples: List[dict]) -> Dict[str, torch.Tensor]:
        texts, images, metadata_list = [], [], []
        for sample in batch_samples:
            texts.append(render_conversation(self.processor, build_conversation(sample)))
            metadata_list.append(sample.get("metadata", {}))
            img_path = sample.get("image_path")
            img = None
            if img_path and Path(img_path).exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                except (OSError, Image.UnidentifiedImageError) as e:
                    print(f"[collate] unreadable image {img_path}: {e}", flush=True)
            if img is None:
                img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
            images.append(img)

        try:
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        except ValueError as e:
            if "Mismatch" in str(e) and "token" in str(e):
                inputs = self.processor(
                    text=texts, images=images, return_tensors="pt", padding=True
                )
            else:
                raise

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_token_ids = self.tokenizer.encode("assistant", add_special_tokens=False)
        newline_id = self.tokenizer.convert_tokens_to_ids("\n")

        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            seq_len = len(seq)
            mask = [True] * seq_len
            i = 0
            while i < seq_len:
                if seq[i] == im_start_id:
                    role_end = min(i + 1 + len(assistant_token_ids), seq_len)
                    if seq[i + 1:role_end] == assistant_token_ids:
                        content_start = role_end
                        while content_start < seq_len and seq[content_start] in (newline_id, 10):
                            content_start += 1
                        content_end = content_start
                        while content_end < seq_len and seq[content_end] != im_end_id:
                            content_end += 1
                        for j in range(content_start, min(content_end + 1, seq_len)):
                            mask[j] = False
                i += 1
            for j in range(seq_len):
                if mask[j]:
                    labels[b, j] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100
        result = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "metadata": metadata_list,
        }
        for key in ("pixel_values", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            if key in inputs:
                result[key] = inputs[key]
        return result
