#!/usr/bin/env bash
# 新奖赏 GRPO step 1000，OmniGuard val 三片。NPU 4、6、7。
# 不占用 0（旧采样）、1/5/8/9（前 20 类验证）、10–15（其他任务）。
set -euo pipefail

BASE="/workspace/zzy/python/model_train/Qwen3.8"
EVAL="$BASE/sft_ship_lora/eval_lora_val.py"
OUT="$BASE/TVP/eval_omniguard_boxpen"
ADAPTER="$BASE/TVP/outputs/grpo_exp3_boxpen/step_1000"
LOG="$BASE/logs"
CONTAINER="${CONTAINER:-zzy-ship-sft}"

launch() {
  local dev="$1" shard="$2"
  local log="$LOG/eval_omniguard_boxpen1000_s${shard}.log"
  echo "start boxpen1000 shard=$shard npu=$dev"
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
        --tag boxpen1000_s${shard} \
        --jsonl $OUT/shards/shard${shard}.jsonl \
        --image-root /data/wchen/data/OmniGuard-68 \
        --device npu:0 \
        --max-new-tokens 4096 \
        --task count \
        --out-dir $OUT \
        > $log 2>&1
    "
}

launch 6 0
sleep 20
launch 7 1
sleep 20
launch 4 2
echo "launched"
