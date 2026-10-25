"""The Anthropic API spend ledger: actual cost from each response's usage, a
hard cap for the whole sprint, and reservations so the final test set's
Claude steps run in priority order.

    uv run python -m seance.budget --init      # once: the spend before the ledger existed
    uv run python -m seance.budget --status    # spent so far, each step's estimate, what may start now

The ledger is `runs/api_spend.jsonl`, one JSON object per line, append-only:

- `spend`: dollars actually billed (a player call, or the spend before the
  ledger), from the response's token counts at list price (`claude.cost`);
- `reserve`: a submitted Message Batch, held at its estimate until settled;
- `settle`: the batch's actual cost, from the usage of every result.

Spent so far = every `spend` and `settle`, plus the estimate of every batch
still open. A step starts only if spent so far + its estimate + the
estimates of every higher-priority step not yet done fit under `CAP_USD`;
the same check runs again before each batch and each player call, so a step
that would pass the cap stops cleanly and says what is missing. A step not in
`PLAN` (a smoke test) ranks below every plan step. Estimates come from the
measured costs below; a step's estimate is only what it still has to do (the
games not yet played, the games not yet in a predictions file), so a rerun
after adding credit asks only for the rest.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .claude import SHORT, cost

LEDGER = Path("runs/api_spend.jsonl")
# $20 loaded (Jorge, 2026-10-25); the cap leaves $0.70 for estimate error.
CAP_USD = 19.30
N_READINGS = 3
PROMPTS = ("plain", "plus_factors")

# Batch-price dollars per reading, the mean of both prompts in the smoke test
# (runs/smoke-claude: fresh p000 + p001, 6 readings per model and prompt).
READING_COST = {"claude-haiku-5-5": 0.000415, "claude-sonnet-5-5": 0.00883, "claude-opus-5-5": 0.01657}
# Dollars per Haiku 5.5 player game: the mean of the 6 pilot games (runs/pilot-haiku).
PLAYER_GAME_COST = 0.0213
# Calls per game in the pilot (480 calls, 497 attempts for 6 games), for the per-call check.
PLAYER_CALLS_PER_GAME = 83


@dataclass(frozen=True)
class Step:
    name: str
    kind: str  # "monitor" or "players"
    run: str  # run directory, relative to the repository root
    models: tuple[str, ...]
    n_games: int  # clean + coalition games read (monitor), or games played (players)


# Priority order (Final test plan, "API budget"): Sonnet and Opus never read the Haiku games.
PLAN: tuple[Step, ...] = (
    Step("1-opus-sonnet-final", "monitor", "runs/final", ("claude-opus-5-5", "claude-sonnet-5-5"), 100),
    Step("2-haiku-monitor-final", "monitor", "runs/final", ("claude-haiku-5-5",), 100),
    Step("3-haiku-players", "players", "runs/final-haiku", ("claude-haiku-5-5",), 106),
    Step("4-haiku-monitor-final-haiku", "monitor", "runs/final-haiku", ("claude-haiku-5-5",), 100),
)


class BudgetExceeded(RuntimeError):
    """A step (or one more batch or call of it) would pass the cap."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def predictions_file(run: Path, model: str, prompt: str) -> Path:
    return run / "predictions" / f"claude_{SHORT[model]}_{prompt}.jsonl"


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text().splitlines():
        try:
            json.loads(line)
            n += 1
        except json.JSONDecodeError:
            continue
    return n


def step_remaining(step: Step, root: Path) -> float:
    """Estimated dollars still to spend on `step`, from what its files hold."""
    run = root / step.run
    if step.kind == "players":
        done = len(list((run / "games").glob("*.json"))) if (run / "games").exists() else 0
        return max(step.n_games - done, 0) * PLAYER_GAME_COST
    total = 0.0
    for model in step.models:
        for prompt in PROMPTS:
            left = max(step.n_games - _count_rows(predictions_file(run, model, prompt)), 0)
            total += left * N_READINGS * READING_COST[model]
    return total


def batch_estimate(models: list[str]) -> float:
    """Estimated dollars for one batch with one request per entry of `models`."""
    return sum(READING_COST[m] for m in models)


class Ledger:
    def __init__(
        self, path: Path = LEDGER, cap: float | None = None, plan: tuple[Step, ...] | None = None,
        root: Path | None = None, remaining_ttl_s: float = 30.0,
    ):
        self.path = Path(path)
        self.cap = CAP_USD if cap is None else cap
        self.plan = PLAN if plan is None else plan
        self.root = root or Path.cwd()
        self.remaining_ttl_s = remaining_ttl_s
        self._offset = 0
        self._spent = 0.0
        self._open: dict[str, dict[str, Any]] = {}  # batch id -> reserve entry
        self._remaining_cache: dict[str, tuple[float, float]] = {}

    # --- reading ---------------------------------------------------------------------

    def _refresh(self) -> None:
        """Read the lines appended since the last read (a partial last line waits)."""
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n") + 1
        for line in data[:end].decode().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e["type"] == "spend":
                self._spent += e["usd"]
            elif e["type"] == "reserve":
                self._open[e["batch_id"]] = e
            elif e["type"] == "settle":
                self._open.pop(e["batch_id"], None)
                self._spent += e["usd"]
            else:
                raise ValueError(f"{self.path}: unknown entry type {e['type']!r}")
        self._offset += end

    def spent(self) -> float:
        """Billed so far plus every open batch at its estimate."""
        self._refresh()
        return self._spent + sum(e["usd"] for e in self._open.values())

    def open_batches(self, step: str) -> list[dict[str, Any]]:
        self._refresh()
        return [e for e in self._open.values() if e["step"] == step]

    def step(self, name: str) -> Step | None:
        return next((s for s in self.plan if s.name == name), None)

    def _remaining(self, step: Step) -> float:
        now = time.monotonic()
        hit = self._remaining_cache.get(step.name)
        if hit and now - hit[0] < self.remaining_ttl_s:
            return hit[1]
        value = step_remaining(step, self.root)
        self._remaining_cache[step.name] = (now, value)
        return value

    def ahead_of(self, name: str) -> list[Step]:
        """The plan steps ranked above `name` (all of them for a step not in the plan)."""
        names = [s.name for s in self.plan]
        return list(self.plan[: names.index(name)] if name in names else self.plan)

    def check(self, name: str, estimate: float) -> tuple[bool, str]:
        """May `name` spend `estimate` more now? The message says why not."""
        spent = self.spent()
        # An open batch is already in `spent` at its estimate, so its step holds only the rest.
        held = {}
        for e in self._open.values():
            held[e["step"]] = held.get(e["step"], 0.0) + e["usd"]
        ahead = [(s.name, max(self._remaining(s) - held.get(s.name, 0.0), 0.0)) for s in self.ahead_of(name)]
        ahead = [(n, v) for n, v in ahead if v > 0]
        need = spent + estimate + sum(v for _, v in ahead)
        if need <= self.cap + 1e-9:
            return True, f"ok: ${spent:.2f} spent + ${estimate:.2f} for {name}" + "".join(
                f" + ${v:.2f} held for {n}" for n, v in ahead) + f" = ${need:.2f} of ${self.cap:.2f}"
        return False, (
            f"{name} stopped before spending: it needs ${estimate:.2f} more, but ${spent:.2f} is spent"
            + "".join(f" and ${v:.2f} is held for {n}" for n, v in ahead)
            + f"; ${need:.2f} would pass the ${self.cap:.2f} cap by ${need - self.cap:.2f}"
        )

    # --- writing ---------------------------------------------------------------------

    @contextmanager
    def locked(self) -> Iterator[None]:
        """One process at a time between a check and the entry it allows."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"at": _now(), **entry}) + "\n")

    def spend(self, step: str, usd: float, **detail: Any) -> None:
        with self.locked():
            self._append({"type": "spend", "step": step, "usd": usd, **detail})

    def reserve(self, step: str, batch_id: str, usd: float, custom_ids: list[str]) -> None:
        self._append({"type": "reserve", "step": step, "batch_id": batch_id, "usd": usd, "custom_ids": custom_ids})

    def settle(self, batch_id: str, usd: float, n_results: int) -> None:
        with self.locked():
            self._refresh()
            if batch_id not in self._open:
                raise ValueError(f"batch {batch_id} is not open in {self.path}")
            step = self._open[batch_id]["step"]
            self._append({"type": "settle", "step": step, "batch_id": batch_id, "usd": usd, "n_results": n_results})


class LedgerClient:
    """A `ChatClient` wrapper for the Haiku players: checks the ledger before
    each call and records each call's actual cost after it."""

    def __init__(self, inner: Any, ledger: Ledger, step: str, run: str):
        self.inner, self.ledger, self.step, self.run = inner, ledger, step, run
        self.model = inner.model
        self.per_call = PLAYER_GAME_COST / PLAYER_CALLS_PER_GAME

    async def complete(self, messages, schema_name, schema):
        ok, msg = self.ledger.check(self.step, self.per_call)
        if not ok:
            raise BudgetExceeded(msg)
        comp = await self.inner.complete(messages, schema_name, schema)
        if comp.usage:
            self.ledger.spend(self.step, cost(self.model, comp.usage), what="player_call", run=self.run,
                              model=self.model, usage=comp.usage)
        return comp


# --- the spend before the ledger ------------------------------------------------------


def prior_spend(root: Path) -> list[dict[str, Any]]:
    """The Haiku pilot and the monitor smoke test from their own records, plus
    the setup probes (about $0.01, from the sprint notes)."""
    games = [json.loads(p.read_text()) for p in sorted((root / "runs/pilot-haiku/games").glob("*.json"))]
    pilot = sum(cost(g["model"], g["tokens"]["usage"]) for g in games)
    smoke = 0.0
    for p in sorted((root / "runs/smoke-claude/predictions").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            smoke += cost(row["detector"]["model"], row["detector"]["usage"], batch=True)
    return [
        {"type": "spend", "step": "0-before-ledger", "usd": pilot, "what": "Haiku player pilot (runs/pilot-haiku), from the records"},
        {"type": "spend", "step": "0-before-ledger", "usd": smoke, "what": "Claude monitor smoke test (runs/smoke-claude), from the records"},
        {"type": "spend", "step": "0-before-ledger", "usd": 0.01, "what": "API setup probes, estimated in the sprint notes"},
    ]


def status(ledger: Ledger) -> str:
    lines = [f"spent ${ledger.spent():.4f} of ${ledger.cap:.2f} (open batches: {len(ledger._open)})"]
    for s in ledger.plan:
        left = step_remaining(s, ledger.root)
        ok, msg = ledger.check(s.name, left) if left > 0 else (True, "done")
        lines.append(f"  {s.name}: ${left:.2f} still to spend; {'may start' if ok else 'blocked'}: {msg}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--init", action="store_true", help="write the spend before the ledger (refuses if it exists)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    ledger = Ledger(args.ledger)
    if args.init:
        if args.ledger.exists():
            ap.error(f"{args.ledger} exists; it is append-only")
        for e in prior_spend(Path.cwd()):
            ledger._append(e)
    print(status(ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
