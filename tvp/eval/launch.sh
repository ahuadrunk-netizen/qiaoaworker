#!/usr/bin/env bash
# OmniGuard val：方案 1 用 NPU 4-7，方案 3 用 NPU 8-11。每张卡评 1/4。
set -euo pipefail

BASE="/workspace/zzy/python/model_train/Qwen3.8"
EVAL="$BASE/sft_ship_lora/eval_lora_val.py"
OUT="$BASE/TVP/eval_omniguard"
LOG="/data/zzy/python/model_train/Qwen3.8/logs"
CONTAINER="${CONTAINER:-zzy-ship-sft}"

launch() {
  local exp="$1" dev="$2" shard="$3"
  local log="$LOG/eval_omniguard_${exp}_s${shard}.log"
  echo "start $exp shard=$shard npu=$dev log=$log"
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
        --adapter $BASE/TVP/outputs/lora_${exp}/epoch_0_step_2819 \
        --tag ${exp}_s${shard} \
        --jsonl $OUT/shards/val.shard${shard}.jsonl \
        --image-root /data/wchen/data/OmniGuard-68 \
        --device npu:0 \
        --max-new-tokens 4096 \
        --out-dir $OUT
    " >"$log" 2>&1 < /dev/null &
}

launch exp1 4 0
sleep 20
launch exp1 5 1
sleep 20
launch exp1 6 2
sleep 20
launch exp1 7 3
sleep 20
launch exp3 8 0
sleep 20
launch exp3 9 1
sleep 20
launch exp3 10 2
sleep 20
launch exp3 11 3
echo "launched"
