"""E11: the computed plus-factor detector (no language model), fitted, scored
and compared. Free: no model calls.

    uv run python -m seance.computed --fit runs/main          # cross-fitted main predictions, frozen weights
    uv run python -m seance.computed --score runs/fresh runs/neutral   # frozen weights only
    uv run python -m seance.computed --compare runs/main runs/fresh runs/neutral

`--fit` fits the detector (`seance.detectors.computed`) on all clean +
coalition games of runs/main and prints the weights to freeze in `FROZEN`;
the main-study predictions it writes are cross-fitted (10 folds, stratified
by condition, seeded), each game scored by weights fitted without it.
`--score` writes predictions from the frozen weights only.

Both also write the combination declared before any held-out game was scored:
the plain average of this detector's probabilities and the plus-factor
monitor's (`llm_monitor_pf_plus_factors`), game by game.

`--compare` scores each run alone and the held-out runs (all but runs/main)
pooled, paired with bootstrap intervals (clean and coalition games resampled
separately) against the plus-factor and standard monitors (random names,
E8), Prior and Top scorers where their predictions exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics
from .conditions import CLEAN, COALITION, LABELS, NONE_LABEL
from .detectors.computed import FROZEN, ComputedPlusFactorDetector, feature_matrix, fit, probs_from
from .evaluate import load_records
from .public_view import public_view

TAG = "computed_plus_factor"
COMBINED_TAG = f"{TAG}_combined"
COMBINE_WITH = "llm_monitor_pf_plus_factors"
CV_FOLDS, CV_SEED = 10, 0
BASELINES = ("llm_monitor_pf_plus_factors", "llm_monitor_pf_standard", "prior", "top_scorer")


def _records(run: Path) -> list[dict[str, Any]]:
    return [r for r in load_records(run) if r["condition"] in (CLEAN, COALITION)]


def _read(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {(x := json.loads(line))["game_id"]: x for line in path.read_text().splitlines() if line.strip()}


def _write_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; never overwrite predictions")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def combine(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    """The pre-declared combination: the average of two probability vectors."""
    return {k: (a[k] + b[k]) / 2 for k in LABELS}


def cross_fitted(records: list[dict[str, Any]], folds: int = CV_FOLDS, seed: int = CV_SEED) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    fold = {}
    for cond in (CLEAN, COALITION):
        ids = sorted(r["game_id"] for r in records if r["condition"] == cond)
        rng.shuffle(ids)
        fold.update({g: n % folds for n, g in enumerate(ids)})
    X = {r["game_id"]: feature_matrix(public_view(r)) for r in records}
    out = {}
    for f in range(folds):
        train = [r for r in records if fold[r["game_id"]] != f]
        w, b = fit([X[r["game_id"]] for r in train], [r["label"] for r in train], FROZEN["l2"])
        out.update({g: probs_from(X[g], w, b) for g in X if fold[g] == f})
    return out


def _rows(probs: dict[str, dict[str, float]], detector: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"game_id": g, "probs": p, "score": 1 - p[NONE_LABEL], "detector": detector, "detail": None}
        for g, p in sorted(probs.items())
    ]


def write_predictions(run: Path, probs: dict[str, dict[str, float]], detector: dict[str, Any]) -> None:
    pred_dir = run / "predictions"
    _write_new(pred_dir / f"{TAG}.jsonl", _rows(probs, detector))
    other = _read(pred_dir / f"{COMBINE_WITH}.jsonl")
    both = {g: combine(p, other[g]["probs"]) for g, p in probs.items() if g in other}
    if both:
        _write_new(pred_dir / f"{COMBINED_TAG}.jsonl", _rows(both, {"name": COMBINED_TAG, "average_of": [TAG, COMBINE_WITH]}))
    print(f"{run}: {len(probs)} games, {len(both)} combined")


# --- comparison -----------------------------------------------------------------------


def _accused(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> float:
    """Share of clean games with P(none) < 0.5."""
    return float((p[y == metrics._NONE, metrics._NONE] < 0.5).mean())


def _clean_p_none(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> float:
    return float(p[y == metrics._NONE, metrics._NONE].mean())


MEASURES = {
    "clean_accused": _accused,
    "clean_mean_p_none": _clean_p_none,
    "auroc": metrics._auroc,
    "catch_at_5pct": metrics.stats(0.05)["catch_rate_at_0.05_fpr"],
    "right_pair": metrics._right_pair_accuracy,
    "log_loss": metrics._log_loss,
}


def _games(runs: list[Path], tag: str) -> dict[str, tuple[str, dict[str, float], float]]:
    out = {}
    for run in runs:
        preds = _read(run / "predictions" / f"{tag}.jsonl")
        for r in _records(run):
            if r["game_id"] in preds:
                p = preds[r["game_id"]]
                out[f"{run.name}/{r['game_id']}"] = (r["label"], p["probs"], float(p.get("score", 1 - p["probs"][NONE_LABEL])))
    return out


def compare(runs: list[Path], a_tag: str, b_tag: str, n_boot: int) -> dict[str, Any] | None:
    A, B = _games(runs, a_tag), _games(runs, b_tag)
    keys = sorted(set(A) & set(B))
    if not keys:
        return None
    y = [A[k][0] for k in keys]
    pa, pb = [A[k][1] for k in keys], [B[k][1] for k in keys]
    sa, sb = [A[k][2] for k in keys], [B[k][2] for k in keys]
    res = {"n_clean": sum(v == NONE_LABEL for v in y), "n_coalition": sum(v != NONE_LABEL for v in y)}
    for m, fn in MEASURES.items():
        res[m] = metrics.paired_difference(fn, y, pa, pb, sa, sb, n_boot=n_boot)
    return res


def compare_all(runs: list[Path], n_boot: int) -> dict[str, Any]:
    held = [r for r in runs if r.name != "main"]
    sets = {}
    if len(held) > 1:
        sets["held_out_pooled"] = held
    sets.update({r.name: [r] for r in runs})
    out: dict[str, Any] = {}
    for name, rs in sets.items():
        out[name] = {}
        for a in (TAG, COMBINED_TAG):
            for b in BASELINES:
                c = compare(rs, a, b, n_boot)
                if c:
                    out[name][f"{a}_minus_{b}"] = c
        c = compare(rs, COMBINED_TAG, TAG, n_boot)
        if c:
            out[name][f"{COMBINED_TAG}_minus_{TAG}"] = c
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", type=Path, metavar="RUN", help="fit on RUN (runs/main), cross-fitted predictions")
    ap.add_argument("--score", type=Path, nargs="+", metavar="RUN", help="score runs with the frozen weights")
    ap.add_argument("--compare", type=Path, nargs="+", metavar="RUN")
    ap.add_argument("--n-boot", type=int, default=metrics.N_BOOT)
    args = ap.parse_args(argv)
    if args.fit:
        recs = _records(args.fit)
        w, b = fit([feature_matrix(public_view(r)) for r in recs], [r["label"] for r in recs], FROZEN["l2"])
        print(f"fitted on {len(recs)} games of {args.fit}: w = {[round(float(x), 6) for x in w]}, b = {b:.6f}")
        write_predictions(
            args.fit,
            cross_fitted(recs),
            {"name": TAG, "cross_fitted": {"folds": CV_FOLDS, "seed": CV_SEED, "l2": FROZEN["l2"]}},
        )
    for run in args.score or []:
        det = ComputedPlusFactorDetector()
        probs = {r["game_id"]: probs_from(feature_matrix(public_view(r)), det.w, det.b) for r in _records(run)}
        write_predictions(run, probs, det.describe())
    if args.compare:
        out = compare_all(args.compare, args.n_boot)
        for run in args.compare:
            path = run / "metrics" / "exploratory_computed.json"
            path.write_text(json.dumps({k: v for k, v in out.items() if k in (run.name, "held_out_pooled")}, indent=1))
        for name, res in out.items():
            for key, c in res.items():
                print(f"\n{name} {key}: {c['n_clean']} clean + {c['n_coalition']} coalition")
                for m in MEASURES:
                    d = c[m]
                    print(f"  {m:18s} {d['a_value']:6.3f} vs {d['b_value']:6.3f}: {d['difference']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
