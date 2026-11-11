#!/usr/bin/env python3
"""在一张 NPU 上用完整权重为 GRPO 抽 5 条回答。

训练进程把请求写进 ipc 目录。这里生成 token，再把 LoRA 换成刚更新的那份。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from peft import PeftModel
from safetensors.torch import load_file
from peft.utils.save_and_load import set_peft_model_state_dict
from transformers import AutoModelForImageTextToText, AutoProcessor

import grpo_train as g


def load_sampler(adapter: Path):
    g.install_empty_skip()
    processor = AutoProcessor.from_pretrained(g.MODEL, trust_remote_code=True)
    g.add_special_tokens(processor.tokenizer)
    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = g.MIN_PIXELS
        ip.max_pixels = g.MAX_PIXELS
    try:
        from transformers.image_utils import SizeDict

        ip.size = SizeDict(shortest_edge=g.MIN_PIXELS, longest_edge=g.MAX_PIXELS)
    except Exception:
        pass
    base = AutoModelForImageTextToText.from_pretrained(
        g.MODEL, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()
    model.to("npu:0")
    model.config.use_cache = True
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "use_cache"):
        text_cfg.use_cache = True
    return processor, model


def reload_adapter(model, adapter: Path) -> int:
    path = adapter / "adapter_model.safetensors"
    state = {
        key: value
        for key, value in load_file(str(path)).items()
        if "lora_" in key and value.numel() > 0
    }
    model.set_adapter("default")
    set_peft_model_state_dict(model, state)
    return len(state)


def generate_five(processor, model, image: Path, question: str) -> list[list[int]]:
    device = torch.device("npu:0")
    inputs = g.build_prompt(processor, image, question, device)
    prompt_len = int(inputs["input_ids"].shape[1])
    g.clear_rope(model)
    rows: list[list[int]] = []
    for i in range(5):
        t0 = time.time()
        print(f"[sampler] sample {i + 1}/5", flush=True)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
            )
        new_ids = out[0, prompt_len:].detach().to("cpu", dtype=torch.long)
        rows.append([int(x) for x in new_ids.tolist()])
        print(
            f"[sampler] sample {i + 1}/5 tokens={len(rows[-1])} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        g.clear_rope(model)
        del out
    del inputs
    torch.npu.empty_cache()
    return rows


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ipc-dir", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    args = p.parse_args()
    torch.npu.set_device(0)
    args.ipc_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sampler] loading {args.adapter}", flush=True)
    processor, model = load_sampler(args.adapter)
    loaded = str(args.adapter.resolve())
    loaded_mtime = (args.adapter / "adapter_model.safetensors").stat().st_mtime
    alloc = torch.npu.memory_allocated() / 1024**3
    print(f"[sampler] ready hbm={alloc:.1f}GB", flush=True)

    while True:
        pending = sorted(args.ipc_dir.glob("*.request.json"))
        chosen = None
        for path in pending:
            rid = path.name.split(".", 1)[0]
            if not (args.ipc_dir / f"{rid}.response.json").is_file():
                chosen = path
                break
        if chosen is None:
            time.sleep(2)
            continue
        rid = chosen.name.split(".", 1)[0]
        request = json.loads(chosen.read_text(encoding="utf-8"))
        adapter = Path(request["adapter"])
        stamp = adapter / "adapter_model.safetensors"
        mtime = stamp.stat().st_mtime if stamp.is_file() else loaded_mtime
        resolved = str(adapter.resolve()) if adapter.exists() else str(adapter)
        if resolved != loaded or mtime != loaded_mtime:
            n = reload_adapter(model, adapter)
            loaded = resolved
            loaded_mtime = mtime
            print(f"[sampler] reloaded {adapter} tensors={n}", flush=True)
        t0 = time.time()
        resp = args.ipc_dir / f"{rid}.response.json"
        tmp = args.ipc_dir / f"{rid}.response.tmp"
        try:
            token_ids = generate_five(
                processor, model, Path(request["image"]), request["question"]
            )
            payload = {
                "id": request["id"],
                "token_ids": token_ids,
                "sec": round(time.time() - t0, 1),
                "hbm": round(torch.npu.memory_allocated() / 1024**3, 2),
            }
        except Exception as exc:
            payload = {"id": request["id"], "error": f"{type(exc).__name__}: {exc}"}
            print(f"[sampler] failed id={rid} {payload['error']}", flush=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.rename(resp)
        print(f"[sampler] response id={rid} sec={payload.get('sec')}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
