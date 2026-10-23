"""Async game runner: plays GameSpecs and writes one JSON record per game.

    uv run python -m seance.run --out runs/smoke --fake --clean 4 --coalition 4 --positive 2

Resumable: the run directory holds `manifest.json` (the specs, written once)
and `<game_id>.json` per finished game. Re-running the same command skips
finished games and retries failed ones (`<game_id>.failed.json`).

Concurrency: `--concurrency` caps in-flight model calls across all games
(the number that matters for a single Modal GPU container). Within a game,
speaking is sequential (each medium sees earlier messages) and the four
guesses go out together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import game as G
from .agents import GUESS, SPEAK, AgentCallFailed, CallLog, call_medium, system_prompt
from .conditions import GameSpec, build_specs
from .inference import ChatClient, LimitedClient
from .skins import get_skin

SCHEMA_VERSION = 1
MANIFEST = "manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
            ["git", "status", "--porcelain"], cwd=Path(__file__).parent, capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _settings(skin: str, max_attempts: int) -> dict[str, Any]:
    return {
        "n_mediums": G.N_MEDIUMS,
        "n_rounds": G.N_ROUNDS,
        "secret_range": [G.SECRET_MIN, G.SECRET_MAX],
        "vision_width": G.VISION_WIDTH,
        "points_by_place": list(G.POINTS_BY_PLACE),
        "tie_rule": G.TIE_RULE,
        "speaking_order": "round-robin, first speaker rotates (round r starts with seat r mod 4)",
        "message_word_cap": G.MAX_MESSAGE_WORDS,
        "skin": skin,
        "max_attempts_per_call": max_attempts,
    }


async def play_game(
    spec: GameSpec,
    client: ChatClient,
    max_attempts: int = 3,
    inference_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Play one game to the end and return its full record (hidden fields included)."""
    skin = get_skin(spec.skin)
    names = G.MEDIUM_NAMES
    setup = G.deal(spec.seed)
    game = G.Game(setup, allies=spec.allies)
    seats = range(setup.n_mediums)
    systems = [system_prompt(skin, s, spec.role(s), spec.partner(s)) for s in seats]
    notes: list[list[dict[str, str]]] = [[] for _ in seats]  # per seat, per round
    calls: list[CallLog] = []
    rounds_out: list[dict[str, Any]] = []
    started = _now()

    for rnd in setup.rounds:
        round_notes = [{} for _ in seats]
        while (seat := game.next_speaker()) is not None:
            out, log = await call_medium(
                client, skin, systems[seat], game, seat, SPEAK, notes[seat], spec.partner(seat), max_attempts
            )
            calls.append(log)
            round_notes[seat][SPEAK] = out["private_note"]
            game.submit_message(seat, out["public_message"])

        results = await asyncio.gather(
            *(
                call_medium(
                    client, skin, systems[s], game, s, GUESS, notes[s], spec.partner(s), max_attempts
                )
                for s in seats
            )
        )
        guesses = []
        for s, (out, log) in zip(seats, results):
            calls.append(log)
            round_notes[s][GUESS] = out["private_note"]
            guesses.append(out["guess"])
        res = game.submit_guesses(guesses)
        for s in seats:
            notes[s].append(round_notes[s])

        rounds_out.append(
            {
                "round": rnd.index + 1,
                "secret": rnd.secret,
                "visions": {names[s]: [rnd.visions[s].lo, rnd.visions[s].hi] for s in seats},
                "speaking_order": [names[s] for s in rnd.speaking_order],
                "messages": [{"speaker": names[s], "text": t} for s, t in res.messages],
                "guesses": {names[s]: res.guesses[s] for s in seats},
                "points": {names[s]: res.points[s] for s in seats},
                "private_notes": {names[s]: round_notes[s] for s in seats},
            }
        )

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
        "settings": _settings(spec.skin, max_attempts),
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
        return [GameSpec.from_dict(d) for d in data["games"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"created_at": _now(), **meta, "games": [s.to_dict() for s in specs]})
    return specs


def pending(out_dir: Path, specs: list[GameSpec]) -> list[GameSpec]:
    return [s for s in specs if not (out_dir / f"{s.game_id}.json").exists()]


async def run_games(
    specs: list[GameSpec],
    client: ChatClient,
    out_dir: Path,
    max_attempts: int = 3,
    max_games_in_flight: int = 8,
    inference_settings: dict[str, Any] | None = None,
    log=print,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = pending(out_dir, specs)
    log(f"{len(specs) - len(todo)} of {len(specs)} games already done; playing {len(todo)}")
    gate = asyncio.Semaphore(max_games_in_flight)
    counts = {"complete": 0, "failed": 0}

    async def one(spec: GameSpec) -> None:
        async with gate:
            t0 = time.monotonic()
            failed_path = out_dir / f"{spec.game_id}.failed.json"
            try:
                record = await play_game(spec, client, max_attempts, inference_settings)
            except AgentCallFailed as e:
                counts["failed"] += 1
                _write_json(
                    failed_path,
                    {"status": "failed", "game_id": spec.game_id, "error": str(e), "call": e.log.to_dict(), "at": _now()},
                )
                log(f"{spec.game_id}: FAILED ({e})")
                return
            _write_json(out_dir / f"{spec.game_id}.json", record)
            failed_path.unlink(missing_ok=True)
            counts["complete"] += 1
            log(f"{spec.game_id}: done in {time.monotonic() - t0:.1f}s")

    await asyncio.gather(*(one(s) for s in todo))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="run directory")
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

    specs = build_specs(args.clean, args.coalition, args.positive, args.seed, args.skin)
    meta = {"build_args": {k: getattr(args, k) for k in ("clean", "coalition", "positive", "seed", "skin")}, "inference": inf}
    if (args.out / MANIFEST).exists():
        prior = json.loads((args.out / MANIFEST).read_text()).get("inference", {})
        if prior.get("model") != inf.get("model"):
            print(
                f"run directory was started with model {prior.get('model')!r}, now {inf.get('model')!r}; "
                "use a new --out",
                file=sys.stderr,
            )
            return 2
        print(f"resuming from {args.out / MANIFEST}; game-list arguments are ignored", file=sys.stderr)
    specs = load_or_create_manifest(args.out, specs, meta)
    if not specs:
        print("no games: pass --clean/--coalition/--positive", file=sys.stderr)
        return 2
    client = LimitedClient(inner, args.concurrency)
    counts = asyncio.run(
        run_games(specs, client, args.out, args.max_attempts, args.games_in_flight, inf)
    )
    print(f"complete {counts['complete']}, failed {counts['failed']}, peak in-flight calls {client.peak_in_flight}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
