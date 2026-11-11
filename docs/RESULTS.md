# OmniGuard-68 validation

Protocol: 20,878 images, 50,425 GT boxes, greedy IoU 0.5 match, `max_new_tokens=4096`, counting prompt. Size uses original-image area of percent boxes. Precision-by-size assigns unmatched predictions to the predicted box’s own size.

## Headline

New box penalty (from old GRPO 1000, another 1000 steps, same prompt order) is the best finished checkpoint: highest box F1 and precision, lowest count MAE, fewer predicted boxes than SFT, still higher recall than SFT.

Old F1 GRPO finds the most true boxes and also dumps the most false boxes. SFT is the most conservative on empty images.

The new-reward gain vs old GRPO 1000 is **not** a pure reward ablation: it is extra training. The matched control is `f1cont` (same start, old F1, same seed-0 order). That eval is still running. Direction of the new run (fewer boxes, higher P, slightly lower R) matches the new penalty.

## Pooled table

| | SFT scheme 3 | GRPO F1 1000 | GRPO boxpen 1000 |
| --- | ---: | ---: | ---: |
| Count exact | 72.98% | 72.32% | **73.12%** |
| Count MAE | 1.507 | 1.428 | **1.218** |
| Empty-image acc | **95.74%** | 92.75% | 95.51% |
| Positive-image acc | 50.23% | **51.89%** | 50.72% |
| Format | 98.66% | 97.69% | **99.28%** |
| Pred boxes | 55,658 | 78,848 | **48,613** |
| Hits | 22,190 | **28,341** | 26,170 |
| Precision | 39.87% | 35.94% | **53.83%** |
| Recall | 44.01% | **56.20%** | 51.90% |
| F1 | 41.84% | 43.85% | **52.85%** |
| Mean matched IoU | 0.810 | 0.809 | **0.815** |
| Over-count / image | 0.37 | 0.51 | **0.29** |
| Under-count / image | 1.14 | **0.92** | 0.93 |

## Size buckets

GT: small 3,153 / medium 18,218 / large 29,054.

| Size | Metric | SFT | GRPO F1 | GRPO boxpen |
| --- | --- | ---: | ---: | ---: |
| Small | P / R / F1 | 16.82 / 18.52 / 17.63 | 13.72 / **38.03** / 20.17 | **35.53** / 33.33 / **34.40** |
| Medium | P / R / F1 | 26.37 / 28.36 / 27.33 | 30.40 / **52.16** / 38.41 | **47.25** / 46.57 / **46.91** |
| Large | P / R / F1 | 50.44 / 56.58 / 53.34 | 45.41 / **60.71** / 51.96 | **60.05** / 57.25 / **58.62** |

## Data identity (control)

`train.jsonl` has 180,433 prompts, 39,117 unique images, shuffled with `random.Random(0)`.

- Old GRPO online finished at cursor 2568 / seen 1000.
- First boxpen replayed that queue from cursor 0; the first 2568 rollout images match the old log exactly, then 285 extra prompts.
- `f1cont` replays from cursor 0 with `--reward f1`.
- `boxpen2` continues from boxpen cursor 2853 on later prompts (~7% of upcoming images were seen earlier under another class).

## Incomplete follow-ups

| Run | Status | Partial signal |
| --- | --- | --- |
| boxpen2 1000 (more new-reward steps) | 2/3 val shards | n=13,919, F1 56.28%, P 60.23%, MAE 1.17 |
| f1cont 1000 (old F1 control) | val still running | needed before attributing the 52.85 F1 solely to the new formula |

Raw shard metrics live in `results/raw_metrics/`.
