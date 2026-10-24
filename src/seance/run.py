"""Async game runner: plays GameSpecs and writes one JSON record per game.

    uv run python -m seance.run --out runs/smoke --fake --clean 4 --coalition 4 --positive 2

Resumable: the run directory holds `manifest.json` (the specs, written once),
`games/<game_id>.json` per finished game and `failed/<game_id>.json` per
failed attempt. Re-running the same command skips finished games and retries
failed ones. The manifest names each game's condition and allies, so never
hand this directory to a detector; detectors get `public_view` output only.

Concurrency: `--concurrency` caps in-flight model calls across all games
(the number that matters for a single Modal GPU container). Within a game,
speaking is sequential (each medium sees earlier messages) and the four
guesses go out together.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import game as G
from .agents import GUESS, SPEAK, AgentCallFailed, CallLog, call_medium, system_prompt
from .conditions import GameSpec, build_main_study, build_specs
from .inference import ChatClient, LimitedClient
from .skins import get_skin

SCHEMA_VERSION = 1
MANIFEST = "manifest.json"
GAMES_DIR = "games"
FAILED_DIR = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@functools.cache
def code_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=Path(__file__).parent, capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _settings(skin: str, max_attempts: int, posting_phases: int) -> dict[str, Any]:
    return {
        "n_mediums": G.N_MEDIUMS,
        "n_rounds": G.N_ROUNDS,
        "secret_range": [G.SECRET_MIN, G.SECRET_MAX],
        "vision_width": G.VISION_WIDTH,
        "points_by_place": list(G.POINTS_BY_PLACE),
        "tie_rule": G.TIE_RULE,
        "posting_mode": G.POSTING_MODE,
        "posting_phases": posting_phases,
        "message_word_cap": G.MAX_MESSAGE_WORDS,
        "skin": skin,
        "max_attempts_per_call": max_attempts,
    }


class GameFailed(RuntimeError):
    """A game that could not finish, with what it had spent so far."""

    def __init__(self, cause: BaseException, rounds_done: int, calls: list[CallLog]):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.rounds_done = rounds_done
        self.calls = calls


async def play_game(
    spec: GameSpec,
    client: ChatClient,
    max_attempts: int = 3,
    inference_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Play one game to the end and return its full record (hidden fields included).

    Raises GameFailed (wrapping the cause) if any call or step fails."""
    skin = get_skin(spec.skin)
    names = G.MEDIUM_NAMES
    setup = G.deal(spec.seed)
    game = G.Game(setup, allies=spec.allies, posting_phases=spec.posting_phases)
    seats = range(setup.n_mediums)
    systems = [system_prompt(skin, s, spec.role(s), spec.partner(s), spec.posting_phases) for s in seats]
    notes: list[list[dict[str, Any]]] = [[] for _ in seats]  # per seat, per round
    calls: list[CallLog] = []
    rounds_out: list[dict[str, Any]] = []
    started = _now()

    try:
        for rnd in setup.rounds:
            round_notes: list[dict[str, Any]] = [{SPEAK: []} for _ in seats]
            for s in seats:  # filled in as the round goes; each seat sees only its own
                notes[s].append(round_notes[s])

            async def all_at_once(phase: str) -> list[dict]:
                # Every prompt is built before anyone's reply is recorded, so no
                # medium sees another's message (or note) from this phase.
                results = await asyncio.gather(
                    *(
                        call_medium(
                            client, skin, systems[s], game, s, phase, notes[s], spec.partner(s), max_attempts
                        )
                        for s in seats
                    ),
                    return_exceptions=True,
                )
                errors = [r for r in results if isinstance(r, BaseException)]
                calls.extend(r.log for r in errors if isinstance(r, AgentCallFailed))
                calls.extend(r[1] for r in results if not isinstance(r, BaseException))
                if errors:
                    raise errors[0]
                return [out for out, _ in results]

            while game.posting_phase is not None:
                outs = await all_at_once(SPEAK)
                for s, out in zip(seats, outs):
                    round_notes[s][SPEAK].append(out["private_note"])
                game.submit_posts([out["public_message"] for out in outs])

            outs = await all_at_once(GUESS)
            for s, out in zip(seats, outs):
                round_notes[s][GUESS] = out["private_note"]
            res = game.submit_guesses([out["guess"] for out in outs])

            rounds_out.append(
                {
                    "round": rnd.index + 1,
                    "secret": rnd.secret,
                    "visions": {names[s]: [rnd.visions[s].lo, rnd.visions[s].hi] for s in seats},
                    "messages": [
                        {"phase": k + 1, "speaker": names[s], "text": t}
                        for k, texts in enumerate(res.posts)
                        for s, t in enumerate(texts)
                    ],
                    "guesses": {names[s]: res.guesses[s] for s in seats},
                    "points": {names[s]: res.points[s] for s in seats},
                    "private_notes": {names[s]: round_notes[s] for s in seats},
                }
            )
    except AgentCallFailed as e:
        if e.log not in calls:
            calls.append(e.log)
        raise GameFailed(e, len(game.results), calls) from e
    except Exception as e:
        raise GameFailed(e, len(game.results), calls) from e

    totals, pay = game.totals(), game.payoffs()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "game_id": spec.game_id,
        "spec": spec.to_dict(),
        "seed": spec.seed,
        "condition": spec.condition,
        "label": spec.label,
        "allies": [names[s] for s in spec.allies] if spec.allies else None,
        "model": client.model,
        "inference": inference_settings or {},
        "enable_thinking": (inference_settings or {}).get("enable_thinking"),
        "settings": _settings(spec.skin, max_attempts, spec.posting_phases),
        "names": list(names),
        "roles": {names[s]: spec.role(s) for s in seats},
        "system_prompts": {names[s]: systems[s] for s in seats},
        "rounds": rounds_out,
        "scores": {
            "individual": {names[s]: totals[s] for s in seats},
            "payoff": {names[s]: pay[s] for s in seats},
        },
        "tokens": {
            "calls": len(calls),
            "attempts": sum(len(c.attempts) for c in calls),
            "prompt": sum(c.prompt_tokens for c in calls),
            "completion": sum(c.completion_tokens for c in calls),
            "truncated_messages": sum(1 for c in calls if c.truncated),
            # Non-zero means thinking was on despite the setting: check the endpoint.
            "attempts_with_reasoning": sum(
                1 for c in calls for a in c.attempts if a.get("reasoning_chars", 0) > 0
            ),
        },
        "calls": [c.to_dict() for c in calls],
        "started_at": started,
        "finished_at": _now(),
        "code_version": code_version(),
    }


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


def load_or_create_manifest(out_dir: Path, specs: list[GameSpec], meta: dict[str, Any]) -> list[GameSpec]:
    """The first run fixes the specs; later runs reuse them so resuming is exact."""
    path = out_dir / MANIFEST
    if path.exists():
        data = json.loads(path.read_text())
        if not data["games"]:
            raise ValueError(f"{path} lists no games; delete it and start again")
        return [GameSpec.from_dict(d) for d in data["games"]]
    if not specs:
        raise ValueError("refusing to create a manifest with no games")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"created_at": _now(), **meta, "games": [s.to_dict() for s in specs]})
    return specs


def game_path(out_dir: Path, game_id: str) -> Path:
    return out_dir / GAMES_DIR / f"{game_id}.json"


def failed_path(out_dir: Path, game_id: str) -> Path:
    return out_dir / FAILED_DIR / f"{game_id}.json"


def pending(out_dir: Path, specs: list[GameSpec]) -> list[GameSpec]:
    return [s for s in specs if not game_path(out_dir, s.game_id).exists()]


async def run_games(
    specs: list[GameSpec],
    client: ChatClient,
    out_dir: Path,
    max_attempts: int = 3,
    max_games_in_flight: int = 8,
    inference_settings: dict[str, Any] | None = None,
    log=print,
) -> dict[str, int]:
    (out_dir / GAMES_DIR).mkdir(parents=True, exist_ok=True)
    (out_dir / FAILED_DIR).mkdir(parents=True, exist_ok=True)
    todo = pending(out_dir, specs)
    log(f"{len(specs) - len(todo)} of {len(specs)} games already done; playing {len(todo)}")
    gate = asyncio.Semaphore(max_games_in_flight)
    counts = {"complete": 0, "failed": 0}
    failed_by: Counter[str] = Counter()

    async def one(spec: GameSpec) -> None:
        async with gate:
            t0 = time.monotonic()
            fpath = failed_path(out_dir, spec.game_id)
            try:
                record = await play_game(spec, client, max_attempts, inference_settings)
            except GameFailed as e:
                counts["failed"] += 1
                failed_by[f"{spec.condition} {spec.label}"] += 1
                _write_json(
                    fpath,
                    {
                        "status": "failed",
                        "game_id": spec.game_id,
                        "error": str(e),
                        "traceback": "".join(traceback.format_exception(e.cause)),
                        "rounds_done": e.rounds_done,
                        "tokens": {
                            "calls": len(e.calls),
                            "prompt": sum(c.prompt_tokens for c in e.calls),
                            "completion": sum(c.completion_tokens for c in e.calls),
                        },
                        "calls": [c.to_dict() for c in e.calls],
                        "at": _now(),
                    },
                )
                log(f"{spec.game_id}: FAILED after {e.rounds_done} rounds ({e})")
                return
            _write_json(game_path(out_dir, spec.game_id), record)
            fpath.unlink(missing_ok=True)
            counts["complete"] += 1
            log(f"{spec.game_id}: done in {time.monotonic() - t0:.1f}s")

    await asyncio.gather(*(one(s) for s in todo))
    if failed_by:
        log("failed games by condition and pair: " + ", ".join(f"{k}: {v}" for k, v in sorted(failed_by.items())))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="run directory")
    ap.add_argument("--main-study", action="store_true", help="the pre-registered 50/50/6 games (fixed seeds)")
    ap.add_argument("--posting-phases", type=int, default=G.DEFAULT_POSTING_PHASES, choices=(1, 2),
                    help="simultaneous posting phases per round (2 = post, read, post again)")
    ap.add_argument("--clean", type=int, default=0)
    ap.add_argument("--coalition", type=int, default=0)
    ap.add_argument("--positive", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="seed for building the game list")
    ap.add_argument("--skin", default="seance")
    ap.add_argument("--fake", action="store_true", help="use the scripted fake model (offline)")
    ap.add_argument("--concurrency", type=int, default=8, help="max in-flight model calls")
    ap.add_argument("--games-in-flight", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args(argv)

    if args.fake:
        from .fake import ScriptedModel

        inner: ChatClient = ScriptedModel(seed=args.seed)
        inf = {"model": inner.model}
    else:
        from .inference import OpenAICompatClient, load_settings

        settings = load_settings(args.env_file)
        inner = OpenAICompatClient(settings)
        inf = settings.describe()

    if args.main_study:
        if args.clean or args.coalition or args.positive:
            print("--main-study fixes the game list; drop --clean/--coalition/--positive", file=sys.stderr)
            return 2
        specs = build_main_study(args.skin, args.posting_phases)
    else:
        specs = build_specs(args.clean, args.coalition, args.positive, args.seed, args.skin,
                            posting_phases=args.posting_phases)
    build_keys = ("main_study", "clean", "coalition", "positive", "seed", "skin", "posting_phases")
    meta = {"build_args": {k: getattr(args, k) for k in build_keys}, "inference": inf}
    if (args.out / MANIFEST).exists():
        prior = json.loads((args.out / MANIFEST).read_text()).get("inference", {})
        if prior != inf:
            print(
                f"run directory was started with inference settings {prior}, now {inf}; use a new --out",
                file=sys.stderr,
            )
            return 2
        print(f"resuming from {args.out / MANIFEST}; game-list arguments are ignored", file=sys.stderr)
    elif not specs:
        print("no games: pass --clean/--coalition/--positive", file=sys.stderr)
        return 2
    specs = load_or_create_manifest(args.out, specs, meta)
    client = LimitedClient(inner, args.concurrency)
    counts = asyncio.run(
        run_games(specs, client, args.out, args.max_attempts, args.games_in_flight, inf)
    )
    print(f"complete {counts['complete']}, failed {counts['failed']}, peak in-flight calls {client.peak_in_flight}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
