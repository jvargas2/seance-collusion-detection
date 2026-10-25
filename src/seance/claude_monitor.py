"""Claude 5.5 models as the LLM monitor, through the Message Batches API.

    uv run python -m seance.claude_monitor --run runs/final \\
        --model claude-opus-5-5 --model claude-sonnet-5-5 \\
        --prompt plain --prompt plus_factors --step 1-opus-sonnet-final

Every reading gets the same prompt the Qwen monitor (`LLMMonitor`) gets:
`build_prompt` on the E8 random-names view (`random_names_view_fn()`, seed
20261027), the same answer keys (`label_order`), parser and retry message.
All readings of all chosen models and prompts go out as one batch
(`custom_id` = run-game-model-prompt-reading); readings that fail (an
unparsable reply, a refusal, an API error) go out again in a follow-up batch,
up to `--max-attempts` attempts per reading as for the Qwen monitor.

Writes `predictions/claude_<model>_<prompt>.jsonl` in the run (or under
`--out DIR` for a smoke test), one row per game whose readings all parsed,
in the `predict_all` format, as soon as a batch brings its last reading.
Games already in the file are skipped, so a rerun reads only missing games
(a game with some of its readings done redoes them all).

Spending goes through the ledger (`seance.budget`): `--step` names the
final test plan's step (or smoke-* for a smoke test), every batch is checked
against the cap and reserved before it is sent and settled at its actual
cost, and a batch left open by a stopped process is collected on the rerun
rather than paid again. A budget stop exits with code 3 and lists what is
missing.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agents import ReplyError
from .budget import LEDGER, BudgetExceeded, Ledger, batch_estimate
from .claude import SHORT, THINKING, api_key, cost, request_params, text_of, usage_dict
from .conditions import CLEAN, COALITION, LABELS
from .detectors.base import check_probs, normalize
from .detectors.llm_monitor import (
    PROMPT_SHA256,
    PROMPT_VARIANTS,
    _sha,
    build_prompt,
    label_order,
    parse_reply,
    retry_message,
    schema_for,
)
from .evaluate import _read_predictions, load_records
from .exploratory import PF_NAME_SEED, random_names_view_fn

NAME = "claude_monitor"
PROMPTS = {"plain": None, "plus_factors": "plus_factors"}  # tag -> prompt_variant
PROMPT_TAGS = {"plain": "plain", "plus_factors": "pf"}  # short form for custom_id
MAX_TOKENS = {"claude-haiku-5-5": 2048, "claude-sonnet-5-5": 2048, "claude-opus-5-5": 8192}
_CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass(frozen=True)
class Config:
    model: str
    prompt: str  # a PROMPTS key
    structured: bool = True  # JSON schema in output_config.format

    @property
    def variant(self) -> str | None:
        return PROMPTS[self.prompt]

    def describe(self, n_readings: int) -> dict[str, Any]:
        """The detector block's fixed part (batch ids and usage are added per game)."""
        sha = PROMPT_SHA256 if self.variant is None else _sha(PROMPT_VARIANTS[self.variant])
        t = THINKING[self.model]
        return {
            "name": NAME,
            "provider": "anthropic",
            "model": self.model,
            "prompt_sha256": sha,
            "n_readings": n_readings,
            **({"prompt_variant": self.variant} if self.variant is not None else {}),
            "view": "random_names",
            "name_seed": PF_NAME_SEED,
            "temperature": None,  # not accepted by these models: default sampling
            "max_tokens": MAX_TOKENS[self.model],
            "thinking": t["thinking"],
            "effort": t["effort"],
            "structured_output": self.structured,
            "api": "message_batches",
        }

    def file_name(self) -> str:
        return f"claude_{SHORT[self.model]}_{self.prompt}" + ("" if self.structured else "_noschema") + ".jsonl"


@dataclass
class Reading:
    config: Config
    game_id: str
    index: int
    custom_id: str
    prompt: str
    keys: list[str]
    aliases: dict[str, str] | None
    messages: list[dict[str, Any]]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    result: tuple[dict[str, float], dict[str, Any]] | None = None

    def params(self) -> dict[str, Any]:
        schema = schema_for(self.keys) if self.config.structured else None
        return request_params(self.config.model, MAX_TOKENS[self.config.model], self.messages, schema)


def custom_id(run: str, game_id: str, config: Config, index: int) -> str:
    cid = f"{run}-{game_id}-{SHORT[config.model]}-{PROMPT_TAGS[config.prompt]}" + (
        "" if config.structured else "-ns"
    ) + f"-r{index}"
    if not _CUSTOM_ID.match(cid):
        raise ValueError(f"custom_id {cid!r} is not valid for the Batches API")
    return cid


def make_readings(
    run: str,
    records: list[dict[str, Any]],
    configs: list[Config],
    n_readings: int,
    view_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[Reading]:
    out = []
    for rec in records:
        view = view_fn(rec)
        keys = label_order(view["names"])
        for config in configs:
            prompt = build_prompt(view, prompt_variant=config.variant)
            for i in range(n_readings):
                out.append(
                    Reading(
                        config, rec["game_id"], i, custom_id(run, rec["game_id"], config, i), prompt, keys,
                        view.get("aliases"), [{"role": "user", "content": prompt}],
                    )
                )
    return out


def apply_result(reading: Reading, result: Any, batch_id: str) -> None:
    """Record one batch result as an attempt; parse it, or set up the retry."""
    kind = result.type
    entry: dict[str, Any] = {"batch_id": batch_id, "result_type": kind, "error": None}
    reading.attempts.append(entry)
    if kind != "succeeded":
        err = getattr(result, "error", None)
        entry["error"] = f"batch result {kind}" + (f": {getattr(getattr(err, 'error', err), 'message', err)}" if err else "")
        return  # resent unchanged
    msg = result.message
    usage = usage_dict(msg.usage)
    text = text_of(msg.content)
    entry.update(
        {
            "raw": text,
            "prompt_tokens": usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "usage": usage,
            "stop_reason": msg.stop_reason,
            "content_types": [b.type for b in msg.content],
        }
    )
    if msg.stop_reason == "refusal":
        details = getattr(msg, "stop_details", None)
        entry["error"] = f"refusal ({getattr(details, 'category', None)})"
        return  # resent unchanged: there is no reply to correct
    try:
        reading.result = parse_reply(text, reading.keys, reading.aliases)
    except ReplyError as e:
        entry["error"] = str(e)
        if text:
            reading.messages = reading.messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": retry_message(e)},
            ]


def game_rows(readings: list[Reading], n_readings: int) -> dict[tuple[Config, str], dict[str, Any]]:
    """One prediction row per (config, game) whose readings all parsed."""
    groups: dict[tuple[Config, str], list[Reading]] = {}
    for r in readings:
        groups.setdefault((r.config, r.game_id), []).append(r)
    rows = {}
    for (config, game_id), rs in groups.items():
        rs.sort(key=lambda r: r.index)
        if len(rs) != n_readings or any(r.result is None for r in rs):
            continue
        probs = normalize({k: sum(r.result[0][k] for r in rs) / len(rs) for k in LABELS})
        check_probs(probs)
        usages = [a["usage"] for r in rs for a in r.attempts if "usage" in a]
        usage = {k: sum(u[k] for u in usages) for k in usages[0]} if usages else {}
        rows[(config, game_id)] = {
            "game_id": game_id,
            "probs": probs,
            "score": 1.0 - probs["none"],
            "detector": {
                **config.describe(n_readings),
                "batch_ids": sorted({a["batch_id"] for r in rs for a in r.attempts}),
                "usage": usage,
            },
            "detail": {"readings": [{"probs": r.result[0], **r.result[1], "attempts": r.attempts} for r in rs]},
        }
    return rows


def _config_of(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row["detector"].items() if k not in ("batch_ids", "usage")}


def _model_of(cid: str) -> str:
    return next(m for m, short in SHORT.items() if f"-{short}-" in cid)


def _result_cost(cid: str, result: Any) -> float:
    if result.type != "succeeded":
        return 0.0  # errored, expired and canceled requests are not billed
    return cost(_model_of(cid), usage_dict(result.message.usage), batch=True)


def _wait(client: Any, batch: Any, poll_s: float, log) -> Any:
    t0 = time.monotonic()
    while batch.processing_status != "ended":
        time.sleep(poll_s)
        batch = client.messages.batches.retrieve(batch.id)
        c = batch.request_counts
        log(f"batch {batch.id}: {batch.processing_status} processing {c.processing} succeeded {c.succeeded} "
            f"errored {c.errored} after {time.monotonic() - t0:.0f}s")
    log(f"batch {batch.id}: ended after {time.monotonic() - t0:.0f}s")
    return batch


def _collect(client: Any, batch_id: str, by_id: dict[str, Reading], sent: list[str]) -> tuple[float, int]:
    """Apply a finished batch's results to the readings it holds; a request
    missing from the results counts as a failed attempt. Returns the batch's
    cost (every result, matched or not) and the number of results."""
    usd, n, seen = 0.0, 0, set()
    for res in client.messages.batches.results(batch_id):
        n += 1
        usd += _result_cost(res.custom_id, res.result)
        if res.custom_id in by_id:
            apply_result(by_id[res.custom_id], res.result, batch_id)
            seen.add(res.custom_id)
    for cid in sent:
        if cid in by_id and cid not in seen:
            by_id[cid].attempts.append({"batch_id": batch_id, "result_type": "missing", "error": "no result returned"})
    return usd, n


def recover_open_batches(
    client: Any, ledger: Ledger, step: str, readings: list[Reading], poll_s: float = 30.0, log=print,
    after_batch: Callable[[], None] | None = None,
) -> list[str]:
    """Batches this step submitted but never settled (the process stopped while
    waiting): wait for each, apply its results to the readings still to do,
    write what is finished and settle its cost, so nothing is paid twice."""
    by_id = {r.custom_id: r for r in readings}
    ids = []
    for entry in ledger.open_batches(step):
        log(f"batch {entry['batch_id']}: submitted earlier and not settled; collecting its results")
        batch = _wait(client, client.messages.batches.retrieve(entry["batch_id"]), poll_s, log)
        usd, n = _collect(client, batch.id, by_id, entry["custom_ids"])
        if after_batch:
            after_batch()
        ledger.settle(batch.id, usd, n)
        ids.append(batch.id)
    return ids


def run_batches(
    client: Any, readings: list[Reading], max_attempts: int, poll_s: float = 30.0, log=print,
    ledger: Ledger | None = None, step: str | None = None, after_batch: Callable[[], None] | None = None,
) -> list[str]:
    """Submit every unfinished reading, wait, apply the results; repeat until
    each reading parsed or used `max_attempts` attempts. Returns the batch ids.

    With a ledger, each batch is checked against the cap and reserved at its
    estimate before it is sent (BudgetExceeded if it doesn't fit), and settled
    at its actual cost once its results are in; `after_batch` runs before
    settling (write the finished games first, so a stop never pays twice)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch_ids = []
    by_id = {r.custom_id: r for r in readings}
    if len(by_id) != len(readings):
        raise ValueError("duplicate custom_id")
    while True:
        todo = [r for r in readings if r.result is None and len(r.attempts) < max_attempts]
        if not todo:
            return batch_ids
        requests = [Request(custom_id=r.custom_id, params=MessageCreateParamsNonStreaming(**r.params())) for r in todo]
        sent = [r.custom_id for r in todo]
        if ledger is not None:
            estimate = batch_estimate([r.config.model for r in todo])
            with ledger.locked():
                ok, msg = ledger.check(step, estimate)
                if not ok:
                    raise BudgetExceeded(msg)
                batch = client.messages.batches.create(requests=requests)
                ledger.reserve(step, batch.id, estimate, sent)
        else:
            batch = client.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        log(f"batch {batch.id}: {len(todo)} requests submitted")
        batch = _wait(client, batch, poll_s, log)
        usd, n = _collect(client, batch.id, by_id, sent)
        if after_batch:
            after_batch()
        if ledger is not None:
            ledger.settle(batch.id, usd, n)
            log(f"batch {batch.id}: ${usd:.4f} (batch price); ledger total ${ledger.spent():.4f}")


def check_step(ledger: Ledger, name: str, run: Path, models: list[str], prompts: list[str], scratch: bool) -> str | None:
    """Why this invocation doesn't match its ledger step, or None. A plan step
    must read exactly its run with exactly its models and both prompts, into
    the run itself; anything else (a smoke test) uses a step named smoke-*."""
    step = ledger.step(name)
    if step is None:
        return None if name.startswith("smoke") else f"--step {name!r} is neither a plan step nor smoke-*"
    if step.kind != "monitor":
        return f"step {name} plays games; it is not a monitor step"
    if run.resolve() != (ledger.root / step.run).resolve():
        return f"step {name} reads {step.run}, not {run}"
    if sorted(models) != sorted(step.models) or sorted(prompts) != sorted(PROMPTS):
        return f"step {name} is {', '.join(step.models)} with both prompts, not {', '.join(models)} with {', '.join(prompts)}"
    if scratch:
        return f"step {name} reads every game into the run; --games, --out and --no-schema are for smoke-* steps"
    return None


def write_finished(
    readings: list[Reading], n_readings: int, out_dir: Path, seat_order: dict[str, list[str]],
    written: set[tuple[Config, str]],
) -> int:
    """Append a row for every game whose readings all parsed and that isn't
    written yet; returns how many were added."""
    added = 0
    rows = game_rows(readings, n_readings)
    for (config, game_id), row in sorted(rows.items(), key=lambda kv: (kv[0][0].file_name(), kv[0][1])):
        if (config, game_id) in written:
            continue
        row = {"game_id": game_id, "seat_order": seat_order[game_id], **{k: v for k, v in row.items() if k != "game_id"}}
        with (out_dir / config.file_name()).open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.add((config, game_id))
        added += 1
    return added


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--model", action="append", required=True, choices=sorted(THINKING))
    ap.add_argument("--prompt", action="append", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--step", required=True,
                    help="the spend-ledger step (seance.budget.PLAN), or smoke-<anything> for a smoke test")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--no-schema", action="append", default=[], choices=sorted(THINKING),
                    help="also read with this model without output_config.format (prompt-only JSON)")
    ap.add_argument("--games", help="comma-separated game ids (smoke test)")
    ap.add_argument("--out", type=Path, help="directory for the predictions files (default <run>/predictions)")
    ap.add_argument("--n-readings", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--poll-s", type=float, default=30.0)
    args = ap.parse_args(argv)

    ledger = Ledger(args.ledger)
    why = check_step(ledger, args.step, args.run, args.model, args.prompt, bool(args.games or args.out or args.no_schema))
    if why:
        ap.error(why)

    import anthropic

    records = [r for r in load_records(args.run) if r["condition"] in (CLEAN, COALITION)]
    if args.games:
        keep = set(args.games.split(","))
        records = [r for r in records if r["game_id"] in keep]
        if len(records) != len(keep):
            ap.error(f"games not found or not clean/coalition: {sorted(keep - {r['game_id'] for r in records})}")
    configs = [Config(m, p) for m in args.model for p in args.prompt]
    configs += [Config(m, p, structured=False) for m in args.no_schema for p in args.prompt]
    out_dir = args.out or args.run / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo_records: dict[Config, list[dict[str, Any]]] = {}
    for config in configs:
        path = out_dir / config.file_name()
        done = _read_predictions(path)
        want = config.describe(args.n_readings)
        for row in done.values():
            if _config_of(row) != want:
                raise RuntimeError(f"{path} holds predictions from {_config_of(row)}, not {want}; use another --out")
        todo_records[config] = [r for r in records if r["game_id"] not in done]
        print(f"{config.file_name()}: {len(records) - len(todo_records[config])} of {len(records)} games done")

    view_fn = random_names_view_fn()
    readings = [
        rd for config, recs in todo_records.items() for rd in make_readings(args.run.name, recs, [config], args.n_readings, view_fn)
    ]
    if not readings and not ledger.open_batches(args.step):
        print("nothing to do")
        return 0
    seat_order = {r["game_id"]: view_fn(r)["names"] for r in records}
    written: set[tuple[Config, str]] = set()

    def flush() -> None:
        n = write_finished(readings, args.n_readings, out_dir, seat_order, written)
        if n:
            print(f"{n} games written")

    client = anthropic.Anthropic(api_key=api_key(), max_retries=6)
    t0 = time.monotonic()
    recover_open_batches(client, ledger, args.step, readings, args.poll_s, after_batch=flush)
    estimate = batch_estimate([r.config.model for r in readings if r.result is None])
    ok, msg = ledger.check(args.step, estimate)
    print(msg)
    stopped = None
    if not ok:
        stopped = msg
    else:
        try:
            run_batches(client, readings, args.max_attempts, args.poll_s, ledger=ledger, step=args.step, after_batch=flush)
        except BudgetExceeded as e:
            stopped = str(e)
    flush()
    elapsed = time.monotonic() - t0

    # Summary: parsing, attempts, tokens and cost (no monitor conclusions).
    print(f"\nall batches done in {elapsed:.0f}s")
    for config in configs:
        rs = [r for r in readings if r.config == config]
        if not rs:
            continue
        atts = [a for r in rs for a in r.attempts]
        ok_atts = [a for a in atts if "usage" in a]
        dollars = sum(cost(config.model, a["usage"], batch=True) for a in ok_atts)
        n_games = len({r.game_id for r in rs})
        print(
            f"{config.file_name()}: {sum(r.result is not None for r in rs)}/{len(rs)} readings parsed, "
            f"{sum(1 for (c, _) in written if c == config)}/{n_games} games written; attempts {len(atts)} "
            f"(refusals {sum(1 for a in atts if (a.get('error') or '').startswith('refusal'))}, "
            f"parse errors {sum(1 for a in ok_atts if a['error'] and not a['error'].startswith('refusal'))}, "
            f"api errors {sum(1 for a in atts if 'usage' not in a)}); "
            f"mean in {sum(a['prompt_tokens'] for a in ok_atts) / max(len(ok_atts), 1):.0f} "
            f"out {sum(a['completion_tokens'] for a in ok_atts) / max(len(ok_atts), 1):.0f} tokens per attempt; "
            f"${dollars:.4f} (batch price), ${dollars / max(len(rs), 1):.5f} per reading"
        )
    print(f"ledger: ${ledger.spent():.4f} of ${ledger.cap:.2f} spent")
    if stopped:
        missing = {c.file_name(): len({r.game_id for r in readings if r.config == c and (c, r.game_id) not in written}) for c in configs}
        print(f"STOPPED (budget): {stopped}")
        print("games still to read: " + ", ".join(f"{k} {v}" for k, v in missing.items() if v))
        return 3
    failed = [r.custom_id for r in readings if r.result is None]
    if failed:
        print(f"{len(failed)} readings failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
