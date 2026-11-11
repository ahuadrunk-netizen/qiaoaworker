#!/usr/bin/env bash
# Qwen3.8-27B LoRA SFT (DeepSpeed ZeRO-3) on ship_yolo counting
#
#   EXP=exp1 DEVICES=8,9   PORT=29521 bash run_lora.sh
#   EXP=exp2 DEVICES=10,11 PORT=29522 bash run_lora.sh
set -euo pipefail

EXP="${EXP:-exp1}"
case "$EXP" in
  exp1) CFG_NAME="lora_exp1_llm.yaml" ;;
  exp2) CFG_NAME="lora_exp2_merger.yaml" ;;
  exp3) CFG_NAME="lora_exp3_vision.yaml" ;;
  *)    echo "未知 EXP=$EXP（exp1|exp2|exp3）" >&2; exit 1 ;;
esac

DEVICES="${DEVICES:-8,9}"
PORT="${PORT:-29521}"
CONTAINER="${CONTAINER:-zzy-ship-sft}"
NPUS=$(echo "$DEVICES" | tr ',' '\n' | grep -c '[0-9]')

TVP_ROOT="/workspace/MYVLLM/lib/Thinking-with-Visual-Primitives-pytorch"
SFT_DIR="/workspace/zzy/python/model_train/Qwen3.8/sft_ship_lora"
CONFIG="$SFT_DIR/configs/$CFG_NAME"
OUT="$SFT_DIR/outputs/lora_$EXP"
LOG="/data/zzy/python/model_train/Qwen3.8/logs/lora_$EXP.log"

echo "=== LoRA SFT $EXP  devices=$DEVICES (n=$NPUS) port=$PORT ==="
echo "config=$CONFIG"
echo "out=$OUT"
echo "log=$LOG"

mkdir -p "$(dirname "$LOG")" /data/zzy/tmp/{home,hf,triton,cache}

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "容器 $CONTAINER 未运行，请先: bash $(dirname "$0")/launch_train_container.sh" >&2
  exit 1
fi

docker exec -e ASCEND_RT_VISIBLE_DEVICES="$DEVICES" \
  -e HOME=/workspace/zzy/tmp/home \
  -e TMPDIR=/workspace/zzy/tmp \
  -e HF_HOME=/workspace/zzy/tmp/hf \
  -e TRITON_CACHE_DIR="/workspace/zzy/tmp/triton_$EXP" \
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
    mkdir -p /workspace/zzy/tmp/triton_$EXP
    cd $TVP_ROOT
    exec torchrun --nproc_per_node=$NPUS --master_port=$PORT \
      $SFT_DIR/train_entry.py \
      --config $CONFIG --output_dir $OUT ${EXTRA:-}
  " 2>&1 | tee -a "$LOG"
