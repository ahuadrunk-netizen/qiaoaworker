#!/usr/bin/env python3
"""ship_yolo SFT 入口：复用 TVP 的 ZeRO-3 训练，另加两处修正 + LoRA 支持。

1) <think> 包装
   Qwen3.8 的 chat_template.jinja 对 assistant 消息 **无条件** 输出
       '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content
   而 TVP 的 ConversationCollator 把 '<think>...</think>' 内联进了 content，
   reasoning_content 留空 → 渲染出一个空 think 壳 + 一个真 think，共两组。
   这里把 assistant 改为 reasoning_content(思维正文) + content(最终答案)。

2) LoRA（可选，config 里的 `lora:` 段）
   TVP 的 train_sft_deepspeed.py 只做全参，没有 PEFT。这里通过 patch
   AutoModelForImageTextToText.from_pretrained，在 ZeRO-3 partition-on-load
   之后、deepspeed.initialize 之前注入 LoRA。

   Qwen3.8 是混合注意力：64 层里 16 层 full_attention(q/k/v/o_proj)，
   48 层 linear_attention(in_proj_*/out_proj)，两种都要覆盖。
   `mtp.*`（投机解码头）必须排除，否则按后缀匹配会误伤。

3) checkpoint
   LoRA 模式下把 save_16bit_model 换成只存 adapter（~0.7GB），
   而不是完整 51GB 权重。

均不修改 /data/wchen 下的共享代码。启动见 run_sft_4npu.sh / run_lora.sh。
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

TVP_ROOT = Path(
    os.environ.get(
        "TVP_ROOT",
        "/workspace/MYVLLM/lib/Thinking-with-Visual-Primitives-pytorch",
    )
)
if not (TVP_ROOT / "tvp").is_dir():
    raise SystemExit(f"TVP_ROOT 不存在或结构不对: {TVP_ROOT}")
sys.path.insert(0, str(TVP_ROOT))

from tvp.dataset.collators import ConversationCollator, TRAINING_SYSTEM_PROMPT  # noqa: E402


# ───────────────────────── 1. <think> 修正 ─────────────────────────


def _build_conversation(self, sample: dict) -> list[dict]:
    """与 TVP 原版一致，但 thinking 走 reasoning_content，不内联标签。"""
    image_path = sample.get("image_path")
    question = sample.get("question", "")
    thinking = sample.get("thinking", "")
    answer = sample.get("answer", "")
    label = sample.get("label", "")
    boxes = sample.get("boxes", [])
    points = sample.get("points", [])

    user_content = []
    if image_path:
        user_content.append({"type": "image", "image": str(image_path)})
    if question:
        text = question
    elif label:
        text = f"Locate the {label} in the image."
    else:
        text = "Describe what you see."
    user_content.append({"type": "text", "text": text})

    # content = 思维块之后的最终回答
    content = answer
    if not content and boxes:
        from tvp.model.special_tokens import REF_END, REF_START, format_box_token

        if label:
            content = f"{REF_START}{label}{REF_END}{format_box_token(boxes)}"
        else:
            content = format_box_token(boxes)
    if not content and points:
        from tvp.model.special_tokens import format_point_token

        content = format_point_token(points)
    if not content:
        content = "I see the object."

    assistant = {"role": "assistant", "content": content.strip()}
    if thinking:
        # 模板会包 <think>…</think>，这里只给正文
        assistant["reasoning_content"] = thinking.strip()

    return [
        {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        assistant,
    ]


ConversationCollator._build_conversation = _build_conversation


# ───────────────────────── 2. LoRA 注入 ─────────────────────────

# full_attention 层（64 层里的 16 层）
ATTN_FULL = ("q_proj", "k_proj", "v_proj", "o_proj")
# linear_attention / GatedDeltaNet 层（另外 48 层）
ATTN_LINEAR = ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj")
LLM_MLP = ("gate_proj", "up_proj", "down_proj")
VISION_BLOCK = ("qkv", "proj", "linear_fc1", "linear_fc2")
MERGER = ("linear_fc1", "linear_fc2")


def _read_cfg() -> dict:
    """从 sys.argv 里取 --config 并读 yaml。"""
    import yaml

    argv = sys.argv
    if "--config" not in argv:
        return {}
    path = argv[argv.index("--config") + 1]
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _is_mtp(name: str) -> bool:
    return name == "mtp" or name.startswith("mtp.") or ".mtp." in name


def _vision_block_idx(name: str) -> int | None:
    marker = "visual.blocks."
    if marker not in name:
        return None
    try:
        return int(name.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def _collect_targets(model, lcfg: dict) -> list[str]:
    """枚举真实模块名，返回 LoRA 要挂的**全名**列表（避免后缀误匹配 mtp）。"""
    import torch.nn as nn

    want_llm_attn = lcfg.get("llm_attn", True)
    want_llm_mlp = lcfg.get("llm_mlp", True)
    want_lm_head = lcfg.get("lm_head", True)
    want_merger = lcfg.get("merger", False)
    vision_last_n = int(lcfg.get("vision_last_n", 0) or 0)

    n_vision_blocks = 0
    for name, _ in model.named_modules():
        idx = _vision_block_idx(name)
        if idx is not None:
            n_vision_blocks = max(n_vision_blocks, idx + 1)
    vision_from = n_vision_blocks - vision_last_n if vision_last_n > 0 else None

    targets: list[str] = []
    for name, mod in model.named_modules():
        if _is_mtp(name) or not isinstance(mod, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]

        if name == "lm_head" or name.endswith(".lm_head"):
            if want_lm_head:
                targets.append(name)
            continue

        if "visual.merger" in name:
            if want_merger and leaf in MERGER:
                targets.append(name)
            continue

        vidx = _vision_block_idx(name)
        if vidx is not None:
            if vision_from is not None and vidx >= vision_from and leaf in VISION_BLOCK:
                targets.append(name)
            continue

        if "language_model" in name:
            if want_llm_attn and leaf in ATTN_FULL + ATTN_LINEAR:
                targets.append(name)
            elif want_llm_mlp and leaf in LLM_MLP:
                targets.append(name)
    return targets


def _wrap_lora(model, lcfg: dict):
    from peft import LoraConfig, get_peft_model

    targets = _collect_targets(model, lcfg)
    if not targets:
        raise SystemExit("[lora] target_modules 为空，检查 config 的 lora: 段")

    rank = int(lcfg.get("r", 64))
    peft_cfg = LoraConfig(
        r=rank,
        lora_alpha=int(lcfg.get("alpha", 2 * rank)),
        lora_dropout=float(lcfg.get("dropout", 0.05)),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    # gradient checkpointing + PEFT：不打开输入 requires_grad，反传会在
    # checkpoint 边界断掉（loss 不降也不报错）。
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True))

    if int(os.environ.get("RANK", "0")) == 0:
        import collections

        kinds = collections.Counter()
        for t in targets:
            if t.endswith("lm_head"):
                kinds["lm_head"] += 1
            elif "visual.merger" in t:
                kinds["visual.merger"] += 1
            elif "visual.blocks" in t:
                kinds["visual.blocks"] += 1
            elif t.rsplit(".", 1)[-1] in ATTN_FULL:
                kinds["llm.self_attn"] += 1
            elif t.rsplit(".", 1)[-1] in ATTN_LINEAR:
                kinds["llm.linear_attn"] += 1
            else:
                kinds["llm.mlp"] += 1
        n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        print(
            f"[lora] r={rank} alpha={peft_cfg.lora_alpha} "
            f"modules={len(targets)} {dict(kinds)} lora_params={n_lora/1e6:.1f}M",
            flush=True,
        )
    return model


def _install_lora_patches(lcfg: dict) -> None:
    from transformers import AutoModelForImageTextToText

    _orig_from_pretrained = AutoModelForImageTextToText.from_pretrained

    def _patched(*args, **kwargs):
        model = _orig_from_pretrained(*args, **kwargs)
        return _wrap_lora(model, lcfg)

    AutoModelForImageTextToText.from_pretrained = _patched

    # LoRA 模式下只存 adapter（~0.7GB），不存 51GB 全量权重
    from deepspeed.runtime.engine import DeepSpeedEngine

    _orig_save16 = DeepSpeedEngine.save_16bit_model

    def _save_adapter(self, save_dir, save_filename="pytorch_model.bin",
                      exclude_frozen_parameters=False):
        module = self.module
        if not hasattr(module, "peft_config"):
            return _orig_save16(self, save_dir, save_filename, exclude_frozen_parameters)
        import deepspeed as _ds

        params = [p for n, p in module.named_parameters() if "lora_" in n]
        os.makedirs(save_dir, exist_ok=True)
        with _ds.zero.GatheredParameters(params, modifier_rank=0):
            if self.global_rank == 0:
                module.save_pretrained(save_dir, safe_serialization=True)
                print(f"[save] lora adapter -> {save_dir}", flush=True)
        return True

    DeepSpeedEngine.save_16bit_model = _save_adapter


_LORA_CFG = (_read_cfg().get("lora") or {})
if _LORA_CFG.get("enabled", False):
    _install_lora_patches(_LORA_CFG)


# ───────────────────────── 3. 交给 TVP 的 main() ─────────────────────────

if __name__ == "__main__":
    train_script = TVP_ROOT / "tools" / "sft" / "train_sft_deepspeed.py"
    if not train_script.is_file():
        raise SystemExit(f"找不到训练脚本: {train_script}")
    runpy.run_path(str(train_script), run_name="__main__")
