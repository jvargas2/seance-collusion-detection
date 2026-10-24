# Séance: project instructions

Jorge's Apart AI Collusion Research Sprint project. Spec, decisions and progress live in `personal-brain/Career/Apart collusion sprint.md`; the paper is written through `personal-brain/Career/Apart sprint - Paper dictation.md`.

## Running experiments

- **Every new test or paid run starts in its own T3 thread** (Jorge, 2026-10-24), launched with `t3_thread_launch` in this project's main checkout, so the planning thread stays free for conversation while the run goes. The run thread does the code change if one is needed, deploys, runs, watches with Monitor, stops the Modal app, commits, writes the results to the brain notes, and reports back with the numbers. Give it the full context it needs in the launch message; it has none of this thread's history.
- **No spending without Jorge's explicit yes for that specific run,** with a cost estimate (Modal credit or any paid API).
- **Stop the Modal app when a run ends** (`uv run modal app stop <app> -y`) and check `uv run modal billing summary`.
- Each run writes its own files (e.g. `runs/main/predictions/llm_monitor_<tag>.jsonl`, `runs/<new run>/`), so parallel threads don't collide; never overwrite the main study's records or predictions.
- Never choose models, prompts or thresholds by looking at detector results on the games they are scored on; anything decided after seeing results is reported as a follow-up (listed in the paper's pre-registration appendix).
