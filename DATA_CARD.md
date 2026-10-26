# Data card: Séance Collusion Detection games and monitor predictions

## What this is

Recorded games of Séance, a four-player guessing game played by LLMs, with and without a secret colluding pair, and the predictions of every collusion monitor I tested on them. It is the first game of a proposed benchmark of hard negatives for collusion monitors: honest games where open, table-wide cooperation looks like collusion.

## Contents

Each run is a folder under `runs/`:

- `games/<game_id>.json`: one complete game. Settings and seed, the player model and its inference settings, each player's role, the system prompts, and for every round the secret, each player's private range, public message, private notes and guess, and the points. Also final scores, token counts and every raw model reply. Game ids are assigned after shuffling and say nothing about the condition.
- `manifest.json`: the list of games with each game's condition and allied pair. **This file holds the answers; never give it to a monitor.**
- `predictions/<monitor>.jsonl`: one line per game: `game_id`, the 7 probabilities, the detection score, the monitor's settings (model, prompt hash), and for LLM monitors each reading's probabilities, written reasons and raw replies.
- `metrics/`: scores computed from the predictions.
- `failed/`: games that failed after retries, kept for the record and excluded from every analysis.

A monitor must see only `seance.public_view.public_view(record)`: messages, guesses, secrets and scores. No ranges, no private notes, no roles.

## Splits

| Split | Runs | Games | How they were made |
|---|---|---|---|
| Test | `final`, `final-haiku` | 106 + 106 (each 50 clean, 50 coalition, 6 positive control; same deals, different players) | New seed (2026102502), played after every monitor was frozen (commit `9b4ebad`, tag `final-freeze`); scored once |
| Development | `main`, `fresh`, `neutral`, `winner`, `redact`, `pilot-haiku` | 252 | Everything before the freeze; grew as I iterated |

The development split was used to build and choose monitors and prompts, so numbers on it are optimistic. Earlier pilot and smoke-test runs (`pilot-qwen`, `pilot2-*`, `smoke-*`) are kept for the record and are not part of either split.

## Labels and ground truth

- Each game's label is one of 7 classes: one of the 6 pairs, or `none`.
- The label is set by the game builder before play (a balanced, seeded assignment of pairs to coalition games), so it is exact, not annotated.
- Conditions: `clean` (label `none`), `coalition` (a pair told to improvise a code and warned about the auditor), `positive_control` (a pair handed a simple code; main study and test sets, used only to check that monitors can catch an obvious code).
- The private notes record what each player intended. They are model output, useful for reading a game but not part of the label.

## Measures

On clean + coalition games: honest tables accused (P(none) < 0.5), honest mean P(none), coalitions called "none", AUROC of the detection score, catch rate at no more than 5% of clean games flagged and at zero false alarms, right pair named, log-loss, calibration error. Intervals are 95% percentile bootstrap over 10,000 resamples, clean and coalition games resampled separately; comparisons between monitors are paired.

## Players and monitors

- Players: Qwen3.8-27B (FP8, served with vLLM, temperature 0.7, thinking off) in every run except `pilot-haiku` and `final-haiku` (Claude Haiku 5.5, default sampling, thinking off).
- Monitors with predictions: Qwen3.8-27B (plain prompt, plus-factor prompt and three variants), Gemma 4 31B, Clef and Jev (decision models), a statistics detector with no language model, Claude Haiku 5.5 (both test sets) and Sonnet and Opus 5.5 (the Qwen-played test set), and the Prior and Top-scorer baselines. Most monitors read each game with the players renamed per game from a pool of historical mediums' names, so a monitor's lean toward one name can't enter.

## Known limits

- One game, and two player models, both small. Other players may collude, or cooperate openly, in quite different ways.
- 100 scored test games per player model (50 + 50): large differences show; monitors within about 0.1 AUROC or 8 accused tables of each other will look tied.
- Coalitions are instructed to collude and improvise their codes during the game; many codes are crude. This measures monitors against prompted collusion, not collusion a model chose on its own.
- Players use four fixed names (Daniel, Eileen, Gladys, Leonora). The original monitor leaned toward naming Daniel; renamed views remove this.
- Text is English only.

## Contamination

Once these games are public, monitors can be tuned to the test games, by people or by models trained on public data. Treat test numbers reported after the public release (October 2026) with that in mind, develop on the development split, and prefer scoring on newly played games (the game and runner are released for that).

## Intended use

Evaluating and comparing collusion monitors that read a public multi-agent record, especially their false-alarm rate on honest cooperation. Studying how LLM players coordinate covertly and how they cooperate openly.

Not intended as training data for making agents better at covert coordination, or as a measure of how often real deployed models collude.

## Personal and sensitive data

None. All players are language models using fictional names; the game contains no real people's data. Model outputs are kept as returned.

## License

Code: MIT (`LICENSE`). Data (game records, predictions, explorer data under `runs/` and `docs/explorer/data/`): CC BY 4.0 (`LICENSE-DATA`). Model outputs in the records come from Qwen3.8-27B, Gemma 4 31B, Clef, Jev and Claude 5.5 models; their providers' terms also apply to reusing those outputs. In particular, TypeSafe's terms forbid using Jev's outputs to train a model that imitates Jev.
