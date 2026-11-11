#!/usr/bin/env bash
# GRPO on scheme-3 LoRA (epoch_0_step_1200), group_size=5, one NPU.
#
#   DEVICES=0 bash run_grpo.sh
# Phy 8 is left alone: exp1 val eval is on that card.
set -euo pipefail

DEVICES="${DEVICES:-0}"
CONTAINER="${CONTAINER:-zzy-ship-sft}"
SFT_DIR="/workspace/zzy/python/model_train/Qwen3.8/sft_ship_lora"
LOG="/data/zzy/python/model_train/Qwen3.8/logs/grpo_exp3.log"
ADAPTER="${ADAPTER:-$SFT_DIR/outputs/lora_exp3/epoch_0_step_1200}"

echo "=== GRPO exp3  device=$DEVICES  group=5  adapter=$ADAPTER ==="
echo "log=$LOG"

mkdir -p "$(dirname "$LOG")" /data/zzy/tmp/{home,hf,triton,cache}

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "容器 $CONTAINER 未运行" >&2
  exit 1
fi

docker exec -e ASCEND_RT_VISIBLE_DEVICES="$DEVICES" \
  -e HOME=/workspace/zzy/tmp/home \
  -e TMPDIR=/workspace/zzy/tmp \
  -e HF_HOME=/workspace/zzy/tmp/hf \
  -e TRITON_CACHE_DIR=/workspace/zzy/tmp/triton_grpo \
  -e XDG_CACHE_HOME=/workspace/zzy/tmp/cache \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e TVP_ROOT=/workspace/MYVLLM/lib/Thinking-with-Visual-Primitives-pytorch \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -lc "
    set -eo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1} 2>/dev/null || true
    export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    mkdir -p /workspace/zzy/tmp/triton_grpo
    cd $SFT_DIR
    exec python -u grpo_train.py \
      --adapter $ADAPTER \
      --group-size 5 \
      --max-steps ${MAX_STEPS:-200} \
      --save-every ${SAVE_EVERY:-20} \
      --device npu:0 \
      --output-dir $SFT_DIR/outputs/grpo_exp3
  " 2>&1 | tee -a "$LOG"
