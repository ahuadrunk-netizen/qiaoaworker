#!/usr/bin/env bash
# GRPO step 500：与 step 1000 同一套 val 四分片，NPU 6、7、8、9。
set -euo pipefail

BASE="/workspace/zzy/python/model_train/Qwen3.8"
EVAL="$BASE/sft_ship_lora/eval_lora_val.py"
OUT="$BASE/TVP/eval_omniguard"
ADAPTER="$BASE/TVP/outputs/grpo_exp3_online/step_500"
LOG="$BASE/logs"
CONTAINER="${CONTAINER:-zzy-ship-sft}"

launch() {
  local dev="$1" shard="$2"
  local log="$LOG/eval_omniguard_grpo500_s${shard}.log"
  echo "start grpo500 shard=$shard npu=$dev"
  docker exec -d \
    -e ASCEND_RT_VISIBLE_DEVICES="$dev" \
    -e HOME=/workspace/zzy/tmp/home \
    -e TMPDIR=/workspace/zzy/tmp \
    -e HF_HOME=/workspace/zzy/tmp/hf \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    -e PYTHONUNBUFFERED=1 \
    "$CONTAINER" bash -lc "
      set -eo pipefail
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
      source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1} 2>/dev/null || true
      export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1
      python3 $EVAL \
        --adapter $ADAPTER \
        --tag grpo500_s${shard} \
        --jsonl $OUT/shards/val.shard${shard}.jsonl \
        --image-root /data/wchen/data/OmniGuard-68 \
        --device npu:0 \
        --max-new-tokens 4096 \
        --out-dir $OUT \
        > $log 2>&1
    "
}

launch 6 0
sleep 15
launch 7 1
sleep 15
launch 8 2
sleep 15
launch 9 3
echo "launched"
