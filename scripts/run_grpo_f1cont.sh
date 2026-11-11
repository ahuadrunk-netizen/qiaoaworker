#!/usr/bin/env bash
# 对照：从旧 GRPO step 1000 出发，框奖励仍用 F1，再训 1000 步。
# 数据顺序、参考模型和步数与 grpo_exp3_boxpen 的第一轮相同，只换奖励。
# 采样 NPU 8，更新 NPU 10、11。不占用 4/6/7（新奖赏续训）。
set -euo pipefail

SAMPLE_DEVICE="${SAMPLE_DEVICE:-8}"
DEVICES="${DEVICES:-10,11}"
PORT="${PORT:-29544}"
CONTAINER="${CONTAINER:-zzy-ship-sft}"
MAX_STEPS="${MAX_STEPS:-1000}"
BASE="/workspace/zzy/python/model_train/Qwen3.8"
ADAPTER="$BASE/TVP/outputs/lora_exp3/epoch_0_step_2819"
POLICY="$BASE/TVP/outputs/grpo_exp3_f1cont/step_50"
OUT="$BASE/TVP/outputs/grpo_exp3_f1cont"
IPC="$OUT/ipc"
LOG="$BASE/logs/grpo_omniguard_f1cont.log"
SAMPLE_LOG="$BASE/logs/grpo_omniguard_f1cont_sample.log"
NPUS=$(echo "$DEVICES" | tr ',' '\n' | grep -c '[0-9]')

echo "=== GRPO f1cont sample=NPU$SAMPLE_DEVICE update=NPU$DEVICES steps=$MAX_STEPS ==="
echo "policy=$POLICY ref=$ADAPTER reward=f1"
mkdir -p /data/zzy/tmp/{home,hf,triton,cache} /data/zzy/python/model_train/Qwen3.8/logs

ENV_COMMON="
  set -eo pipefail
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
  source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=\${ATB_CXX_ABI:-1} 2>/dev/null || true
  export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  cd $BASE/TVP
"

docker exec "$CONTAINER" bash -lc "rm -rf '$IPC' && mkdir -p '$IPC' '$OUT'"
docker exec "$CONTAINER" bash -lc "printf '\n===== f1cont resume %s policy=f1cont/step_50 reward=f1 cursor=156 seen=50 =====\n' \"\$(date '+%F %T')\" >> '$LOG'"
docker exec "$CONTAINER" bash -lc "printf '\n===== f1cont start %s =====\n' \"\$(date '+%F %T')\" >> '$SAMPLE_LOG'"

docker exec -d -e ASCEND_RT_VISIBLE_DEVICES="$SAMPLE_DEVICE" \
  -e HOME=/workspace/zzy/tmp/home \
  -e TMPDIR=/workspace/zzy/tmp \
  -e HF_HOME=/workspace/zzy/tmp/hf \
  -e TRITON_CACHE_DIR=/workspace/zzy/tmp/triton_grpo_f1cont_sample \
  -e XDG_CACHE_HOME=/workspace/zzy/tmp/cache \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -lc "
    $ENV_COMMON
    mkdir -p /workspace/zzy/tmp/triton_grpo_f1cont_sample
    exec python3 grpo_sample.py \
      --ipc-dir '$IPC' \
      --adapter '$POLICY' \
      >> '$SAMPLE_LOG' 2>&1
  "

docker exec -d -e ASCEND_RT_VISIBLE_DEVICES="$DEVICES" \
  -e HOME=/workspace/zzy/tmp/home \
  -e TMPDIR=/workspace/zzy/tmp \
  -e HF_HOME=/workspace/zzy/tmp/hf \
  -e TRITON_CACHE_DIR=/workspace/zzy/tmp/triton_grpo_f1cont \
  -e XDG_CACHE_HOME=/workspace/zzy/tmp/cache \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -lc "
    $ENV_COMMON
    mkdir -p /workspace/zzy/tmp/triton_grpo_f1cont
    exec torchrun --nproc_per_node=$NPUS --master_port=$PORT \
      grpo_train.py \
      --adapter '$ADAPTER' \
      --policy '$POLICY' \
      --reward f1 \
      --start-cursor 156 \
      --start-seen 50 \
      --jsonl $BASE/TVP/data/jsonl/train.jsonl \
      --image-root /data/wchen/data/OmniGuard-68 \
      --output-dir '$OUT' \
      --ipc-dir '$IPC' \
      --max-steps $MAX_STEPS \
      --save-every 50 \
      >> '$LOG' 2>&1
  "

echo "detached sampler and trainer"
