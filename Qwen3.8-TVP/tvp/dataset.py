"""JSONL → 一条训练样本。字段原样取出，不改问句。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


class JSONLSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: Optional[str] = None, task_type: str = "counting"):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root else self.jsonl_path.parent
        self.task_type = task_type
        self.samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_rel = item.get("image", "")
        image_path = self.image_root / image_rel
        metadata = {"task_type": self.task_type, "image": image_rel}
        for key in ("count", "answer", "boxes", "points"):
            if key in item:
                metadata[key] = item[key]
        return {
            "image_path": str(image_path) if image_path.exists() else None,
            "question": item.get("question", ""),
            "thinking": item.get("thinking", ""),
            "answer": item.get("answer", ""),
            "metadata": metadata,
        }
