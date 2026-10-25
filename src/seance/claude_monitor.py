"""Claude 5.5 models as the LLM monitor, through the Message Batches API.

    uv run python -m seance.claude_monitor --run runs/final \\
        --model claude-haiku-5-5 --model claude-sonnet-5-5 --model claude-opus-5-5 \\
        --prompt plain --prompt plus_factors

Every reading gets the same prompt the Qwen monitor (`LLMMonitor`) gets:
`build_prompt` on the E8 random-names view (`random_names_view_fn()`, seed
20261027), the same answer keys (`label_order`), parser and retry message.
All readings of all chosen models and prompts go out as one batch
(`custom_id` = run-game-model-prompt-reading); readings that fail (an
unparsable reply, a refusal, an API error) go out again in a follow-up batch,
up to `--max-attempts` attempts per reading as for the Qwen monitor.

Writes `predictions/claude_<model>_<prompt>.jsonl` in the run (or under
`--out DIR` for a smoke test), one row per game whose readings all parsed,
in the `predict_all` format. Games already in the file are skipped.
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


def run_batches(client: Any, readings: list[Reading], max_attempts: int, poll_s: float = 30.0, log=print) -> list[str]:
    """Submit every unfinished reading, wait, apply the results; repeat until
    each reading parsed or used `max_attempts` attempts. Returns the batch ids."""
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
        batch = client.messages.batches.create(
            requests=[Request(custom_id=r.custom_id, params=MessageCreateParamsNonStreaming(**r.params())) for r in todo]
        )
        batch_ids.append(batch.id)
        t0 = time.monotonic()
        log(f"batch {batch.id}: {len(todo)} requests submitted")
        while batch.processing_status != "ended":
            time.sleep(poll_s)
            batch = client.messages.batches.retrieve(batch.id)
            c = batch.request_counts
            log(f"batch {batch.id}: {batch.processing_status} processing {c.processing} succeeded {c.succeeded} "
                f"errored {c.errored} after {time.monotonic() - t0:.0f}s")
        log(f"batch {batch.id}: ended after {time.monotonic() - t0:.0f}s")
        seen = set()
        for res in client.messages.batches.results(batch.id):
            apply_result(by_id[res.custom_id], res.result, batch.id)
            seen.add(res.custom_id)
        for r in todo:  # a request missing from the results counts as a failed attempt
            if r.custom_id not in seen:
                r.attempts.append({"batch_id": batch.id, "result_type": "missing", "error": "no result returned"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--model", action="append", required=True, choices=sorted(THINKING))
    ap.add_argument("--prompt", action="append", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--no-schema", action="append", default=[], choices=sorted(THINKING),
                    help="also read with this model without output_config.format (prompt-only JSON)")
    ap.add_argument("--games", help="comma-separated game ids (smoke test)")
    ap.add_argument("--out", type=Path, help="directory for the predictions files (default <run>/predictions)")
    ap.add_argument("--n-readings", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--poll-s", type=float, default=30.0)
    args = ap.parse_args(argv)

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
    if not readings:
        print("nothing to do")
        return 0
    client = anthropic.Anthropic(api_key=api_key(), max_retries=6)
    t0 = time.monotonic()
    run_batches(client, readings, args.max_attempts, args.poll_s)
    elapsed = time.monotonic() - t0

    rows = game_rows(readings, args.n_readings)
    seat_order = {r["game_id"]: view_fn(r)["names"] for r in records}
    for (config, game_id), row in sorted(rows.items(), key=lambda kv: (kv[0][0].file_name(), kv[0][1])):
        row = {"game_id": game_id, "seat_order": seat_order[game_id], **{k: v for k, v in row.items() if k != "game_id"}}
        with (out_dir / config.file_name()).open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary: parsing, attempts, tokens and cost (no monitor conclusions).
    print(f"\nall batches done in {elapsed:.0f}s")
    for config in configs:
        rs = [r for r in readings if r.config == config]
        if not rs:
            continue
        atts = [a for r in rs for a in r.attempts]
        ok = [a for a in atts if "usage" in a]
        dollars = sum(cost(config.model, a["usage"], batch=True) for a in ok)
        n_games = len({r.game_id for r in rs})
        print(
            f"{config.file_name()}: {sum(r.result is not None for r in rs)}/{len(rs)} readings parsed, "
            f"{sum(1 for (c, _) in rows if c == config)}/{n_games} games written; attempts {len(atts)} "
            f"(refusals {sum(1 for a in atts if (a.get('error') or '').startswith('refusal'))}, "
            f"parse errors {sum(1 for a in ok if a['error'] and not a['error'].startswith('refusal'))}, "
            f"api errors {sum(1 for a in atts if 'usage' not in a)}); "
            f"mean in {sum(a['prompt_tokens'] for a in ok) / max(len(ok), 1):.0f} "
            f"out {sum(a['completion_tokens'] for a in ok) / max(len(ok), 1):.0f} tokens per attempt; "
            f"${dollars:.4f} (batch price), ${dollars / max(len(rs), 1):.5f} per reading"
        )
    failed = [r.custom_id for r in readings if r.result is None]
    if failed:
        print(f"{len(failed)} readings failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
