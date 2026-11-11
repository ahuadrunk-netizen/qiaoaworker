# Cluster notes

These launchers assume:

- Docker container `zzy-ship-sft`
- Host `/data/zzy` → `/workspace/zzy`
- OmniGuard mounted at `/data/wchen/data/OmniGuard-68`
- Qwen3.8-27B at `/workspace/zzy/modelscope/Qwen3.8-27B`
- Do not occupy NPU 0 if an old leftover sampler is still there

NPU map used for the comparison runs:

| Job | Sample | Update | Port |
| --- | --- | --- | --- |
| boxpen | 4 | 6,7 | 29542 |
| boxpen2 | 4 | 6,7 | 29543 |
| f1cont | 8 | 10,11 | 29544 |

KL reference for every GRPO run: `outputs/lora_exp3/epoch_0_step_2819`.

Do not overwrite `grpo_exp3_online/step_1000` or `grpo_exp3_boxpen/step_1000`.
