"""E13: score clean and coalition games with Clef (deployed by
`modal_clef.py`), every game read with the E8 random medium names so the
readings pair with `llm_monitor_pf_standard` / `llm_monitor_pf_plus_factors`.

    uv run python -m seance.clef --run runs/main --variant plain --variant plus_factors

Writes `predictions/clef_<variant>.jsonl` (resumable; `--out` for a smoke test).
`--compare` pairs both Clef versions with both E8 monitors, per run and with
the held-out runs (fresh, neutral) pooled, into runs/main/metrics/exploratory_clef.json."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .conditions import CLEAN, COALITION
from .detectors.decision_model import VARIANTS, DecisionModelMonitor
from .evaluate import _read_predictions, load_records, predict_all
from .exploratory import PF_MEASURES, compare_monitors, random_names_view_fn

MONITORS = {
    "clef_plain": "clef_plain",
    "clef_plus": "clef_plus_factors",
    "llm_standard": "llm_monitor_pf_standard",
    "llm_plus": "llm_monitor_pf_plus_factors",
}
PAIRS = [("clef_plain", "llm_standard"), ("clef_plain", "llm_plus"), ("clef_plus", "llm_standard"), ("clef_plus", "llm_plus"), ("clef_plus", "clef_plain")]
SAMPLES = {"held_out": ("fresh", "neutral"), "fresh": ("fresh",), "neutral": ("neutral",), "main": ("main",)}


def compare(runs_dir: Path, n_boot: int, monitors: dict[str, str] = MONITORS, pairs: list[tuple[str, str]] = PAIRS) -> dict:
    """Paired differences (a minus b) over each sample; game ids get their run as a prefix."""
    out = {}
    for sample, runs in SAMPLES.items():
        records, preds = [], {m: {} for m in monitors}
        for run in runs:
            for r in load_records(runs_dir / run):
                if r["condition"] in (CLEAN, COALITION):
                    records.append({**r, "game_id": f"{run}/{r['game_id']}"})
            for m, tag in monitors.items():
                for g, row in _read_predictions(runs_dir / run / "predictions" / f"{tag}.jsonl").items():
                    preds[m][f"{run}/{g}"] = row
        out[sample] = {f"{a}_minus_{b}": compare_monitors(records, preds[a], preds[b], n_boot) for a, b in pairs}
    return out


def print_comparison(out: dict) -> None:
    for sample, pairs in out.items():
        for key, c in pairs.items():
            print(f"\n{sample} {key}: {c['n_games']} games ({c['n_clean']} clean, {c['n_excluded']} excluded)")
            for m in PF_MEASURES:
                d = c[m]
                print(f"  {m:26s} {d['a_value']:7.3f} vs {d['b_value']:7.3f}: {d['difference']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    ap.add_argument("--compare", type=Path, metavar="RUNS_DIR")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--games", help="comma-separated game ids (smoke test)")
    ap.add_argument("--out", type=Path, help="predictions file (one variant only); default predictions/clef_<variant>.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)
    if args.compare:
        out = compare(args.compare, args.n_boot)
        (args.compare / "main" / "metrics" / "exploratory_clef.json").write_text(json.dumps(out, indent=1))
        print_comparison(out)
        return 0
    if not (args.run and args.variant):
        ap.error("--run and --variant are required unless --compare")

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
