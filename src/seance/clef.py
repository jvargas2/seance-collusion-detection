"""E13: score clean and coalition games with Clef (deployed by
`modal_clef.py`), every game read with the E8 random medium names so the
readings pair with `llm_monitor_pf_standard` / `llm_monitor_pf_plus_factors`.

    uv run python -m seance.clef --run runs/main --variant plain --variant plus_factors

Writes `predictions/clef_<variant>.jsonl` (resumable; `--out` for a smoke test)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .conditions import CLEAN, COALITION
from .detectors.decision_model import VARIANTS, DecisionModelMonitor
from .evaluate import load_records, predict_all
from .exploratory import random_names_view_fn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--games", help="comma-separated game ids (smoke test)")
    ap.add_argument("--out", type=Path, help="predictions file (one variant only); default predictions/clef_<variant>.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)

    import modal

    from .modal_clef import APP_NAME, MODEL, REVISION

    clef = modal.Cls.from_name(APP_NAME, "Clef")()

    async def scorer(request):
        response = await clef.answer.remote.aio(request)
        if response["revision"] != REVISION:
            raise RuntimeError(f"deployed revision {response['revision']} is not {REVISION}")
        return response

    records = [r for r in load_records(args.run) if r["condition"] in (CLEAN, COALITION)]
    if args.games:
        keep = set(args.games.split(","))
        records = [r for r in records if r["game_id"] in keep]
    if args.out and len(args.variant) > 1:
        ap.error("--out takes one variant")
    for variant in args.variant:
        det = DecisionModelMonitor(scorer, MODEL, REVISION, variant)
        path = args.out or args.run / "predictions" / f"clef_{variant}.jsonl"
        done = asyncio.run(predict_all(det, records, path, args.concurrency, view_fn=random_names_view_fn()))
        print(f"E13 clef {variant} {args.run}: {len(done)} of {len(records)} games scored", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
