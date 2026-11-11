#!/usr/bin/env bash
# F1 续训 GRPO step 1000，与 boxpen1000 同一套 OmniGuard val 三片。
# 空闲卡：NPU 9、10、11。
set -euo pipefail

BASE="/workspace/zzy/python/model_train/Qwen3.8"
EVAL="$BASE/sft_ship_lora/eval_lora_val.py"
SHARDS="$BASE/TVP/eval_omniguard_boxpen/shards"
OUT="$BASE/TVP/eval_omniguard_f1cont"
ADAPTER="$BASE/TVP/outputs/grpo_exp3_f1cont/step_1000"
LOG="$BASE/logs"
CONTAINER="${CONTAINER:-zzy-ship-sft}"

launch() {
  local dev="$1" shard="$2"
  local log="$LOG/eval_omniguard_f1cont1000_s${shard}.log"
  echo "start f1cont1000 shard=$shard npu=$dev"
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
        --tag f1cont1000_s${shard} \
        --jsonl $SHARDS/shard${shard}.jsonl \
        --image-root /data/wchen/data/OmniGuard-68 \
        --device npu:0 \
        --max-new-tokens 4096 \
        --task count \
        --out-dir $OUT \
        > $log 2>&1
    "
}

launch 9 0
sleep 20
launch 10 1
sleep 20
launch 11 2
echo "launched"
