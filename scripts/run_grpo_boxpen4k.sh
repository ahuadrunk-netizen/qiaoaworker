#!/usr/bin/env bash
# 新奖赏从 boxpen2/step_1000（累计 2000 步）再训 2000 步，累计 4000 步。
# 奖励仍是 score，KL 参考仍是方案三 SFT。数据顺序接着 cursor 5998。
# 不写回 grpo_exp3_boxpen2，避免覆盖正在评测的 step_1000。
# 采样 NPU 10，更新 NPU 11（单卡 ZeRO-3 参数卸载）。
set -euo pipefail

SAMPLE_DEVICE="${SAMPLE_DEVICE:-10}"
DEVICES="${DEVICES:-11}"
PORT="${PORT:-29545}"
CONTAINER="${CONTAINER:-zzy-ship-sft}"
MAX_STEPS="${MAX_STEPS:-2000}"
BASE="/workspace/zzy/python/model_train/Qwen3.8"
ADAPTER="$BASE/TVP/outputs/lora_exp3/epoch_0_step_2819"
POLICY="$BASE/TVP/outputs/grpo_exp3_boxpen2/step_1000"
OUT="$BASE/TVP/outputs/grpo_exp3_boxpen4k"
IPC="$OUT/ipc"
LOG="$BASE/logs/grpo_omniguard_boxpen4k.log"
SAMPLE_LOG="$BASE/logs/grpo_omniguard_boxpen4k_sample.log"
NPUS=$(echo "$DEVICES" | tr ',' '\n' | grep -c '[0-9]')

echo "=== GRPO boxpen4k sample=NPU$SAMPLE_DEVICE update=NPU$DEVICES steps=$MAX_STEPS ==="
echo "policy=$POLICY ref=$ADAPTER"
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
docker exec "$CONTAINER" bash -lc "printf '\n===== boxpen4k start %s policy=boxpen2/step_1000 cursor=5998 steps=2000 total=4000 =====\n' \"\$(date '+%F %T')\" >> '$LOG'"
docker exec "$CONTAINER" bash -lc "printf '\n===== boxpen4k start %s =====\n' \"\$(date '+%F %T')\" >> '$SAMPLE_LOG'"

docker exec -d -e ASCEND_RT_VISIBLE_DEVICES="$SAMPLE_DEVICE" \
  -e HOME=/workspace/zzy/tmp/home \
  -e TMPDIR=/workspace/zzy/tmp \
  -e HF_HOME=/workspace/zzy/tmp/hf \
  -e TRITON_CACHE_DIR=/workspace/zzy/tmp/triton_grpo_boxpen4k_sample \
  -e XDG_CACHE_HOME=/workspace/zzy/tmp/cache \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -lc "
    $ENV_COMMON
    mkdir -p /workspace/zzy/tmp/triton_grpo_boxpen4k_sample
    exec python3 grpo_sample.py \
      --ipc-dir '$IPC' \
      --adapter '$POLICY' \
      >> '$SAMPLE_LOG' 2>&1
  "

docker exec -d -e ASCEND_RT_VISIBLE_DEVICES="$DEVICES" \
  -e HOME=/workspace/zzy/tmp/home \
  -e TMPDIR=/workspace/zzy/tmp \
  -e HF_HOME=/workspace/zzy/tmp/hf \
  -e TRITON_CACHE_DIR=/workspace/zzy/tmp/triton_grpo_boxpen4k \
  -e XDG_CACHE_HOME=/workspace/zzy/tmp/cache \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -lc "
    $ENV_COMMON
    mkdir -p /workspace/zzy/tmp/triton_grpo_boxpen4k
    exec torchrun --nproc_per_node=$NPUS --master_port=$PORT \
      grpo_train.py \
      --adapter '$ADAPTER' \
      --policy '$POLICY' \
      --reward score \
      --start-cursor 5998 \
      --start-seen 0 \
      --jsonl $BASE/TVP/data/jsonl/train.jsonl \
      --image-root /data/wchen/data/OmniGuard-68 \
      --output-dir '$OUT' \
      --ipc-dir '$IPC' \
      --max-steps $MAX_STEPS \
      --save-every 50 \
      >> '$LOG' 2>&1
  "

echo "detached sampler and trainer"
