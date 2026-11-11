#!/usr/bin/env bash
# 原始 Qwen3.8-27B，无 LoRA。按类别要框和类名，再算计数 / IoU / F1。
# NPU 4-11 各评验证集 1/8。不碰 12-15（vLLM）。
set -euo pipefail

BASE="/workspace/zzy/python/model_train/Qwen3.8"
EVAL="$BASE/sft_ship_lora/eval_lora_val.py"
OUT="$BASE/TVP/eval_omniguard"
LOG="/data/zzy/python/model_train/Qwen3.8/logs"
CONTAINER="${CONTAINER:-zzy-ship-sft}"

launch() {
  local dev="$1" shard="$2"
  local log="$LOG/eval_omniguard_basedet_s${shard}.log"
  echo "start base shard=$shard npu=$dev log=$log"
  setsid docker exec \
    -e ASCEND_RT_VISIBLE_DEVICES="$dev" \
    -e HOME=/workspace/zzy/tmp/home \
    -e TMPDIR=/workspace/zzy/tmp \
    -e HF_HOME=/workspace/zzy/tmp/hf \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    "$CONTAINER" bash -lc "
      set -eo pipefail
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
      source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1} 2>/dev/null || true
      export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1
      python3 $EVAL \
        --tag basedet_s${shard} \
        --task detect \
        --jsonl $OUT/shards/val8.shard${shard}.jsonl \
        --image-root /data/wchen/data/OmniGuard-68 \
        --device npu:0 \
        --max-new-tokens 4096 \
        --out-dir $OUT
    " >"$log" 2>&1 < /dev/null &
}

for pair in "4 0" "5 1" "6 2" "7 3" "8 4" "9 5" "10 6" "11 7"; do
  set -- $pair
  launch "$1" "$2"
  sleep 20
done
echo "launched"
