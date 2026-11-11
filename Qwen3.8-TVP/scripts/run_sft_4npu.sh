#!/usr/bin/env bash
# Qwen3.8-27B 全参 SFT (DeepSpeed ZeRO-3) on ship_yolo counting — NPU Phy 2,3,4,5
#
# 走本目录的 train_entry.py（修正了 Qwen3.8 的 double-<think>），而不是
# TVP 的 scripts/run_sft_deepspeed.sh。
set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3,4,5}"
CONTAINER="${CONTAINER:-zzy-ship-sft}"
MASTER_PORT="${MASTER_PORT:-29511}"

# 容器内路径（/data/zzy -> /workspace/zzy, MYVLLM -> /workspace/MYVLLM）
TVP_ROOT="/workspace/MYVLLM/lib/Thinking-with-Visual-Primitives-pytorch"
SFT_DIR="/workspace/zzy/python/model_train/Qwen3.8/sft_ship_lora"
CONFIG="${CONFIG:-$SFT_DIR/configs/sft_ship_qwen38_4npu.yaml}"
OUT="${OUT:-$SFT_DIR/outputs/sft_ship_4npu_v2}"

LOG="/data/zzy/python/model_train/Qwen3.8/logs/sft_ship_4npu_v2.log"
NPUS=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')

echo "=== ship SFT ZeRO-3  devices=$ASCEND_RT_VISIBLE_DEVICES (n=$NPUS) ==="
echo "config=$CONFIG"
echo "out=$OUT"
echo "log=$LOG"

mkdir -p "$(dirname "$LOG")" /data/zzy/tmp/{home,hf,triton,cache}

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "容器 $CONTAINER 未运行，请先: bash $(dirname "$0")/launch_train_container.sh" >&2
  exit 1
fi

docker exec -e ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
  -e HOME=/workspace/zzy/tmp/home \
  -e TMPDIR=/workspace/zzy/tmp \
  -e HF_HOME=/workspace/zzy/tmp/hf \
  -e TRITON_CACHE_DIR=/workspace/zzy/tmp/triton \
  -e XDG_CACHE_HOME=/workspace/zzy/tmp/cache \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e TVP_ROOT="$TVP_ROOT" \
  "$CONTAINER" bash -lc "
    set -eo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1} 2>/dev/null || true
    export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    cd $TVP_ROOT
    exec torchrun --nproc_per_node=$NPUS --master_port=$MASTER_PORT \
      $SFT_DIR/train_entry.py \
      --config $CONFIG --output_dir $OUT ${EXTRA:-}
  " 2>&1 | tee -a "$LOG"
