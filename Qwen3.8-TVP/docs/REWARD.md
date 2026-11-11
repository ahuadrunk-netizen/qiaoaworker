# GRPO reward

A sample is scored only if the text after `</think>` is a bare integer. Otherwise the reward is 0.

\[
R = 0.3 + 0.4\cdot \exp\!\left(-\frac{(N-\mathrm{gt})^2}{2\sigma^2}\right) + 0.3\cdot S_{\mathrm{box}}
\]

\[
\sigma = \frac{0.3\,|\mathrm{gt}| + 0.1}{3}
\]

Count error is symmetric: predicting 0 or 2 when gt=1 costs the same.

## Box term

Boxes are percent xyxy. Matches are greedy one-to-one at IoU 0.5. Let \(n\) be matches, \(N\) predicted boxes, \(M\) ground-truth boxes.

| Name | Formula | Flag |
| --- | --- | --- |
| Old F1 | \(S = 2n / (N+M)\) | `--reward f1` |
| New score | \(S = (n - (N-n)) / M = (2n-N)/M\) | `--reward score` (default) |

Empty image and no predicted boxes: \(S=1\). Either side empty otherwise: \(S=0\). The new score can be negative.

F1 barely punishes extra boxes: one false box only grows the denominator. Break-even hit rate is about \(F1/2\). The new term makes one unmatched box cancel one true match, so spraying boxes loses reward unless the extra box is more likely to hit than to miss.

## Loss (this trainer)

This is not paper-GRPO clipped ratio. For each of 5 group samples:

\[
\mathcal{L} = -A\cdot \overline{\log\pi} + 0.04\,(\overline{\log\pi}-\overline{\log\pi_{\mathrm{ref}}})
\]

Advantage \(A\) is group z-score of \(R\). The step is skipped only when reward std \(< 10^{-6}\) (all five rewards identical), not when count accuracy is 0 or 1.

The KL reference stays the SFT adapter (`lora_exp3/epoch_0_step_2819`), not the GRPO checkpoint.
