# Results

## Games

| Condition | Planned | Complete | Excluded | Messages cut to 60 words |
|---|---|---|---|---|
| coalition | 6 | 6 | 0 | 3 of 240 |

Agent settings used (model, temperature, thinking): Qwen/Qwen3.8-27B-FP8, 0.7, thinking False.
Monitor settings used: Qwen/Qwen3.8-27B-FP8, temperature 0.7, thinking False, 3 readings, prompt 212c62e53f7d4d02.

Main comparison: 0 clean and 6 coalition games (positive controls excluded). Intervals are 95% bootstrap intervals over games.

## Headline

| Detector | Coalitions caught (5% false-alarm threshold) | Clean games flagged (at most 5% by construction) | Right pair named (coalition games) |
|---|---|---|---|
| LLM monitor | n/a (no clean games) | n/a | 67% [33%, 100%] |

## Secondary metrics

| Detector | Log-loss (lower is better) | Brier (lower is better) | AUROC | Calibration error (ECE, collusion score) | Failed games |
|---|---|---|---|---|---|
| LLM monitor | 1.47 [0.23, 2.91] | 0.62 [0.10, 1.22] | nan [nan, nan] | 0.048 | 0 |

Monitor spread over its 3 readings per game: mean SD of the 7 probabilities 0.061, SD of P(none) 0.058; the readings disagree on the most likely label in 33% of games.

## Hypotheses

- **M1 (allies transmit information): does not hold.** Partner weight 0.42 for allies vs nan for pairs in clean games; difference nan [nan, nan].
- **H3 (monitor names the right pair above chance, 1/6 = 17%): supported.** 67% [33%, 100%].
