"""按模块全名挂 LoRA，并让 ZeRO-3 只存 adapter。"""
from __future__ import annotations

import os

import torch.nn as nn

ATTN_FULL = ("q_proj", "k_proj", "v_proj", "o_proj")
ATTN_LINEAR = ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj")
LLM_MLP = ("gate_proj", "up_proj", "down_proj")
VISION_BLOCK = ("qkv", "proj", "linear_fc1", "linear_fc2")
MERGER = ("linear_fc1", "linear_fc2")


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


def collect_targets(model, lcfg: dict) -> list[str]:
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


def wrap_lora(model, lcfg: dict):
    from peft import LoraConfig, get_peft_model

    targets = collect_targets(model, lcfg)
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


def load_adapter_state(model, adapter_dir: str) -> int:
    """把已保存的 LoRA 权重写回还没被 ZeRO 切分的适配器。

    存盘时 lm_head.base_layer 在 ZeRO-3 下是空张量，不能写回去。
    """
    from safetensors.torch import load_file
    from peft.utils.save_and_load import set_peft_model_state_dict

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    state = {
        key: value
        for key, value in load_file(path).items()
        if "lora_" in key and value.numel() > 0
    }
    result = set_peft_model_state_dict(model, state)
    missing = [key for key in result.missing_keys if "lora_" in key]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"adapter {adapter_dir} did not match this LoRA: "
            f"missing={missing[:8]} unexpected={list(result.unexpected_keys)[:8]}"
        )
    return len(state)


def install_lora(lcfg: dict) -> None:
    """在 from_pretrained 返回后挂 LoRA；存盘时只收集 lora_ 参数。"""
    from transformers import AutoModelForImageTextToText

    _orig = AutoModelForImageTextToText.from_pretrained

    def _patched(*args, **kwargs):
        return wrap_lora(_orig(*args, **kwargs), lcfg)

    AutoModelForImageTextToText.from_pretrained = _patched

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
