"""Build the static explorer: every listed run, every monitor's verdict on
every game, readable straight from disk or on GitHub Pages.

    uv run python -m seance.explorer                 # every run in RUNS that exists
    uv run python -m seance.explorer --run runs/x    # just these run directories

It reads only the committed game records and prediction files. The one thing
it computes is a small per-run summary for each monitor (honest tables
accused, i.e. P(none) < 0.5; right pair named; AUROC), with the same
definitions as `seance.metrics`; the paper's numbers come from the
evaluator, not from here. A run listed in RUNS but not on disk is skipped, so
the final test runs drop in by rerunning the same command once they exist.
Game data goes out as small script files (`data/<run>/<id>.js`) so the site
also works when opened as a local file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from . import metrics
from .conditions import CLEAN, NONE_LABEL
from .evaluate import _read_predictions, load_records

STATIC = Path(__file__).parent / "explorer_static"

# (directory under runs/, title, split, one-line description). Order is the
# order on the page. Test = generated after every monitor was frozen.
RUNS = [
    ("final", "Final test, Qwen players", "test", "Games played after every monitor was frozen, scored once."),
    ("final-haiku", "Final test, Haiku players", "test", "The same, with Claude Haiku 5.5 as the four players."),
    ("main", "Main study", "dev", "The pre-registered study: 50 clean, 50 coalition and 6 positive-control games."),
    ("fresh", "Held-out games", "dev", "20 clean and 20 coalition games played after the main study, used to check its fixes."),
    ("neutral", "Neutral wording", "dev", "10 clean and 10 coalition main-study deals replayed with the séance flavor removed."),
    ("winner", "Winner-take-all scoring", "dev", "20 + 20 main-study deals replayed where only the closest guess scores."),
    ("redact", "Numbers redacted", "dev", "20 + 20 main-study deals replayed with every number in public messages redacted."),
    ("pilot-haiku", "Haiku player pilot", "dev", "6 main-study deals replayed with Claude Haiku 5.5 as the players. No monitor has read them."),
]

# Prediction file stem -> (label, group). Groups: "monitor" (a detector we
# compare), "reread" (the original monitor reading a changed record: a
# robustness check). Stems not listed here are still shown, under their file
# name, so new monitors appear without a code change.
MONITORS = {
    "llm_monitor": ("Qwen plain, real names (original)", "monitor"),
    "llm_monitor_pf_standard": ("Qwen plain", "monitor"),
    "llm_monitor_pf_plus_factors": ("Qwen plus-factor", "monitor"),
    "llm_monitor_pf_permission": ("Qwen permission half", "monitor"),
    "llm_monitor_pf_evidence": ("Qwen evidence half", "monitor"),
    "llm_monitor_pf_extended": ("Qwen extended checklist", "monitor"),
    "llm_monitor_gemma": ("Gemma plain, real names", "monitor"),
    "llm_monitor_gemma_pf_standard": ("Gemma plain", "monitor"),
    "llm_monitor_gemma_pf_plus_factors": ("Gemma plus-factor", "monitor"),
    "clef_plain": ("Clef plain", "monitor"),
    "clef_plus_factors": ("Clef plus-factor", "monitor"),
    "jev_plain": ("Jev plain", "monitor"),
    "jev_plus_factors": ("Jev plus-factor", "monitor"),
    "claude_haiku_plain": ("Haiku 5.5 plain", "monitor"),
    "claude_haiku_plus_factors": ("Haiku 5.5 plus-factor", "monitor"),
    "claude_sonnet_plain": ("Sonnet 5.5 plain", "monitor"),
    "claude_sonnet_plus_factors": ("Sonnet 5.5 plus-factor", "monitor"),
    "claude_opus_plain": ("Opus 5.5 plain", "monitor"),
    "claude_opus_plus_factors": ("Opus 5.5 plus-factor", "monitor"),
    "computed_plus_factor": ("Statistics (no LLM)", "monitor"),
    "computed_plus_factor_combined": ("Statistics + Qwen plus-factor", "monitor"),
    "llm_monitor_mediums": ("Qwen plain, mediums' names", "monitor"),
    "llm_monitor_redacted_names_pool": ("Qwen plain, mediums' names", "monitor"),
    "llm_monitor_rotated": ("Seats rotated", "reread"),
    "llm_monitor_names_a": ("Renamed, first set", "reread"),
    "llm_monitor_names_b": ("Renamed, second set", "reread"),
    "llm_monitor_names_pool": ("Renamed from mediums' names", "reread"),
    "llm_monitor_view_actions": ("Guesses and scores only", "reread"),
    "llm_monitor_view_messages": ("Messages only", "reread"),
}
# Never shown: baselines whose P(none) is fixed, aborted or fallback runs.
SKIP = re.compile(r"^(prior|top_scorer)$|_aborted$|_noschema$")


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def monitor_files(pred_dir: Path) -> list[tuple[str, str, str, Path]]:
    """(key, label, group, path) for every prediction file worth showing, in
    MONITORS order, then any unknown ones by name."""
    if not pred_dir.is_dir():
        return []
    found = {p.stem: p for p in pred_dir.glob("*.jsonl") if not SKIP.search(p.stem)}
    order = [k for k in MONITORS if k in found] + sorted(k for k in found if k not in MONITORS)
    return [(k, *MONITORS.get(k, (k, "monitor")), found[k]) for k in order]


def aliases(rec: dict[str, Any], row: dict[str, Any]) -> dict[str, str]:
    """Shown name -> real name when the monitor read the game renamed (the
    renamed views keep the record's seat order); empty otherwise."""
    shown = row.get("seat_order") or rec["names"]
    if set(shown) == set(rec["names"]):
        return {}
    return dict(zip(shown, rec["names"]))


def unalias(text: str, mapping: dict[str, str]) -> str:
    """Put the real names back into a reading's reasons, whole words, in one
    pass (so a renamed player called "Daniel" can't be swapped twice)."""
    if not mapping or not text:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, mapping)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def verdict(rec: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    probs = row["probs"]
    top = metrics._top_pair(probs)
    p_none = probs[NONE_LABEL]
    return {
        "p_none": _r(p_none),
        "top": top,
        "p_top": _r(probs[top]) if top else None,
        "accused": rec["label"] == NONE_LABEL and p_none < 0.5,
        "right": None if rec["label"] == NONE_LABEL else top == rec["label"],
    }


def monitor_detail(rec: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Everything one monitor said about one game, real names throughout."""
    names = aliases(rec, row)
    detail = row.get("detail") or {}
    readings = [
        {
            "p_none": _r(r["probs"][NONE_LABEL]),
            "top": metrics._top_pair(r["probs"]),
            "p_top": _r(r["probs"].get(metrics._top_pair(r["probs"]) or "", None)),
            "reasons": unalias(r.get("reasoning", ""), names),
        }
        for r in detail.get("readings", [])
    ]
    return {
        **verdict(rec, row),
        "probs": {k: _r(v) for k, v in row["probs"].items()},
        "renamed": {real: shown for shown, real in names.items()},
        "readings": readings,
    }


def game_payload(rec: dict[str, Any], monitors: list[tuple[str, str, str, dict[str, dict]]]) -> dict[str, Any]:
    gid = rec["game_id"]
    rounds = []
    for r in rec["rounds"]:
        notes = r.get("private_notes", {})
        rounds.append(
            {
                "round": r["round"],
                "secret": r["secret"],
                "visions": r["visions"],
                "guesses": r["guesses"],
                "points": r["points"],
                "messages": {m["speaker"]: m["text"] for m in r["messages"]},
                "speak_notes": {n: " ".join(notes.get(n, {}).get("speak", [])) for n in rec["names"]},
                "guess_notes": {n: notes.get(n, {}).get("guess", "") for n in rec["names"]},
            }
        )
    allied = rec["label"] != NONE_LABEL
    return {
        "id": gid,
        "condition": rec["condition"],
        "label": rec["label"],
        "names": rec["names"],
        "model": rec.get("model"),
        "scores": rec["scores"]["individual"],
        "pooled": (sum(rec["scores"]["payoff"][n] for n in rec["label"].split("+")) / 2) if allied else None,
        "monitors": {key: monitor_detail(rec, preds[gid]) for key, _, _, preds in monitors if gid in preds},
        "rounds": rounds,
    }


def summary(records: list[dict[str, Any]], preds: dict[str, dict]) -> dict[str, Any]:
    """Per-run counts for one monitor over the clean and allied games it scored."""
    scored = [r for r in records if r["game_id"] in preds]
    clean = [r for r in scored if r["label"] == NONE_LABEL]
    allied = [r for r in scored if r["label"] != NONE_LABEL]
    rows = [preds[r["game_id"]] for r in scored]
    auroc = metrics.auroc([r["label"] for r in scored], [p["probs"] for p in rows], [p["score"] for p in rows]) if clean and allied else None
    return {
        "n": len(scored),
        "n_clean": len(clean),
        "accused": sum(preds[r["game_id"]]["probs"][NONE_LABEL] < 0.5 for r in clean),
        "n_allied": len(allied),
        "right": sum(metrics._top_pair(preds[r["game_id"]]["probs"]) == r["label"] for r in allied),
        "auroc": _r(auroc, 3) if auroc == auroc else None,  # NaN -> None
    }


def build_run(run: Path, key: str, title: str, split: str, about: str, out: Path) -> dict[str, Any]:
    records = sorted(load_records(run), key=lambda r: r["game_id"])
    monitors = [(k, label, group, _read_predictions(path)) for k, label, group, path in monitor_files(run / "predictions")]
    (out / "data" / key).mkdir(parents=True)
    games = []
    for rec in records:
        g = game_payload(rec, monitors)
        (out / "data" / key / f"{g['id']}.js").write_text(
            f"window.SEANCE_GAME = {json.dumps(g, ensure_ascii=False, separators=(',', ':'))};\n"
        )
        games.append(
            {
                "id": g["id"],
                "condition": g["condition"],
                "label": g["label"],
                "pooled": g["pooled"],
                # One [P(none), top pair] per monitor, in the run's monitor order.
                "v": [[m["p_none"], m["top"]] if (m := g["monitors"].get(k)) else None for k, *_ in monitors],
            }
        )
    models = sorted({r.get("model") or "" for r in records} - {""})
    return {
        "key": key,
        "title": title,
        "split": split,
        "about": about,
        "players": models,
        "n_clean": sum(r["condition"] == CLEAN for r in records),
        "monitors": [{"key": k, "label": label, "group": group, **summary(records, preds)} for k, label, group, preds in monitors],
        "games": games,
    }


def build(runs: list[tuple[Path, str, str, str, str]], out: Path) -> int:
    present = [r for r in runs if (r[0] / "games").is_dir()]
    for r in runs:
        if r not in present:
            print(f"explorer: skipping {r[0]} (no games yet)")
    if not present:
        print("explorer: no runs found")
        return 2
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out)
    # Stamp asset names with their content hash so browsers never serve a stale copy.
    page = (out / "index.html").read_text()
    for asset in ("app.js", "style.css"):
        stamp = hashlib.sha256((out / asset).read_bytes()).hexdigest()[:10]
        page = page.replace(f'"{asset}"', f'"{asset}?v={stamp}"')
    (out / "index.html").write_text(page)
    index = {"runs": [build_run(path, key, title, split, about, out) for path, key, title, split, about in present]}
    (out / "data" / "index.js").write_text(
        f"window.SEANCE_INDEX = {json.dumps(index, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    for r in index["runs"]:
        print(f"explorer: {r['key']}: {len(r['games'])} games, {len(r['monitors'])} monitors ({r['split']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"), help="where the runs in RUNS live")
    ap.add_argument("--run", type=Path, action="append", help="build only these run directories (development split)")
    ap.add_argument("--out", type=Path, default=Path("docs/explorer"))
    args = ap.parse_args(argv)
    if args.run:
        known = {key: (title, split, about) for key, title, split, about in RUNS}
        runs = [(p, p.name, *known.get(p.name, (p.name, "dev", ""))) for p in args.run]
    else:
        runs = [(args.runs_dir / key, key, title, split, about) for key, title, split, about in RUNS]
    return build(runs, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
