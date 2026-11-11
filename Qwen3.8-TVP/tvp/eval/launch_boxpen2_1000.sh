#!/usr/bin/env bash
# boxpen 再训 1000 步（grpo_exp3_boxpen2/step_1000）对照 boxpen step 1000。
# 同一套 OmniGuard val 三片。NPU 9/10/11 在跑 f1cont，这里用空闲的 1 和 5。
# shard 2 已改到单独的 NPU 上跑，不再接在 shard 0 后面。
set -euo pipefail

BASE="/workspace/zzy/python/model_train/Qwen3.8"
EVAL="$BASE/sft_ship_lora/eval_lora_val.py"
SHARDS="$BASE/TVP/eval_omniguard_boxpen/shards"
OUT="$BASE/TVP/eval_omniguard_boxpen2"
ADAPTER="$BASE/TVP/outputs/grpo_exp3_boxpen2/step_1000"
LOG="$BASE/logs"
CONTAINER="${CONTAINER:-zzy-ship-sft}"

launch() {
  local dev="$1"
  shift
  local body=""
  local shard
  for shard in "$@"; do
    body+="
      echo start shard=${shard}
      python3 $EVAL \\
        --adapter $ADAPTER \\
        --tag boxpen2_1000_s${shard} \\
        --jsonl $SHARDS/shard${shard}.jsonl \\
        --image-root /data/wchen/data/OmniGuard-68 \\
        --device npu:0 \\
        --max-new-tokens 4096 \\
        --task count \\
        --out-dir $OUT \\
        > $LOG/eval_omniguard_boxpen2_1000_s${shard}.log 2>&1
    "
  done
  echo "start boxpen2_1000 npu=$dev shards=$*"
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
      $body
    "
}

mkdir -p /data/zzy/python/model_train/Qwen3.8/TVP/eval_omniguard_boxpen2
launch 1 0
sleep 25
launch 5 1
echo "launched"
