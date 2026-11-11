#!/usr/bin/env bash
# 全类别标注框数前 20 类的 valid 计数集。方案三与 GRPO step 1000 各评两片。
# 卡：方案三用 8、9，GRPO 1000 用 1、5。不占用 0（旧采样）、4（boxpen 采样）、6/7（boxpen 更新）。
set -euo pipefail

BASE="/workspace/zzy/python/model_train/Qwen3.8"
EVAL="$BASE/sft_ship_lora/eval_lora_val.py"
OUT="$BASE/TVP/eval_all_labels_cls20"
LOG="$BASE/logs"
ROOT="/workspace/zzy/python/model_train/train_data/data/all_labels_coco"
CONTAINER="${CONTAINER:-zzy-ship-sft}"

launch() {
  local name="$1" adapter="$2" dev="$3" shard="$4"
  local log="$LOG/eval_all_labels_cls20_${name}_s${shard}.log"
  echo "start ${name} shard=${shard} npu=${dev}"
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
        --adapter $adapter \
        --tag ${name}_s${shard} \
        --jsonl $OUT/shards/shard${shard}.jsonl \
        --image-root $ROOT \
        --device npu:0 \
        --max-new-tokens 4096 \
        --task count \
        --out-dir $OUT \
        > $log 2>&1
    "
}

EXP3="$BASE/TVP/outputs/lora_exp3/epoch_0_step_2819"
GRPO="$BASE/TVP/outputs/grpo_exp3_online/step_1000"
launch exp3 "$EXP3" 8 0
sleep 20
launch exp3 "$EXP3" 9 1
sleep 20
launch grpo1000 "$GRPO" 1 0
sleep 20
launch grpo1000 "$GRPO" 5 1
echo "launched"
