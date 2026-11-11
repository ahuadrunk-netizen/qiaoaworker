# OmniGuard-68 validation

Protocol: 20,878 images, 50,425 GT boxes, greedy IoU 0.5 match, `max_new_tokens=4096`, counting prompt. Size uses original-image area of percent boxes. Precision-by-size assigns unmatched predictions to the predicted box’s own size.

## Headline

New box penalty (from old GRPO 1000, another 1000 steps, same prompt order) is the best finished checkpoint: highest box F1 and precision, lowest count MAE, fewer predicted boxes than SFT, still higher recall than SFT.

Old F1 GRPO finds the most true boxes and also dumps the most false boxes. SFT is the most conservative on empty images. Scheme 1 (LLM-only LoRA) and scheme 3 differ by 0.19 points of exact match and 0.16 points of box F1, so the later GRPO gap is not a vision-tower effect.

The F1 incentive shows up before the run finishes. Step 428 already predicts 72,723 boxes at 34.79% precision and 93.54% empty-image accuracy. Step 1000 goes to 78,848 boxes, 56.20% recall, and 92.75% empty-image accuracy.

The new-reward gain vs old GRPO 1000 includes extra training. The matched control is `f1cont` (same start, old F1, same seed-0 order). That eval is not in this repo. Direction of the new run (fewer boxes, higher P, slightly lower R) matches the new penalty.

Scheme 1 and step 428 are pooled from `results/raw_metrics/` with `scripts/pool_val_metrics.py` (hits = recall × gt boxes). The scheme 3, F1 1000, and boxpen columns match that same pooling.

## Pooled table

| | SFT scheme 1 | SFT scheme 3 | GRPO F1 428 | GRPO F1 1000 | GRPO boxpen 1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Count exact | 72.79% | 72.98% | 72.72% | 72.32% | **73.12%** |
| Count MAE | 1.512 | 1.507 | 1.485 | 1.428 | **1.218** |
| Empty-image acc | 95.61% | **95.74%** | 93.54% | 92.75% | 95.51% |
| Positive-image acc | 49.98% | 50.23% | **51.90%** | 51.89% | 50.72% |
| Format | 98.69% | 98.66% | 97.95% | 97.69% | **99.28%** |
| Pred boxes | 55,097 | 55,658 | 72,723 | 78,848 | **48,613** |
| Hits | 21,992 | 22,190 | 25,303 | **28,341** | 26,170 |
| Precision | 39.92% | 39.87% | 34.79% | 35.94% | **53.83%** |
| Recall | 43.61% | 44.01% | 50.18% | **56.20%** | 51.90% |
| F1 | 41.68% | 41.84% | 41.09% | 43.85% | **52.85%** |
| Mean matched IoU | 0.812 | 0.810 | 0.809 | 0.809 | **0.816** |
| Over-count / image |  | 0.37 |  | 0.51 | **0.29** |
| Under-count / image |  | 1.14 |  | **0.92** | 0.93 |

Over-count and under-count are kept from the earlier aggregate. They are not in the shard json, so scheme 1 and step 428 leave those two rows blank. Mean IoU for boxpen is 0.8155 when hits-weighted from the shards; the table rounds it to 0.816.

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
