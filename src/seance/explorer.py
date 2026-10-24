"""Build the static explorer: a two-page site (all games, one game) from a
scored run, readable straight from disk or on GitHub Pages.

    uv run python -m seance.explorer --run runs/main --out docs/explorer

It reads only what the evaluator and the exploratory analyses already
wrote (game records, predictions, metrics) and computes nothing new except
which pair each reading put on top. Game data goes out as small script files
(`data/<id>.js`) so the site also works when opened as a local file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from . import metrics
from .conditions import CLEAN, COALITION, NONE_LABEL, POSITIVE_CONTROL
from .evaluate import _read_predictions, load_records

STATIC = Path(__file__).parent / "explorer_static"
REREADS = {
    "llm_monitor_rotated": "Seats in a different order",
    "llm_monitor_names_a": "Everyone renamed, first set",
    "llm_monitor_names_b": "Everyone renamed, second set",
    "top_scorer": "Whoever scored most",
}


def _verdict(row: dict[str, Any]) -> dict[str, Any]:
    probs = row["probs"]
    top = metrics._top_pair(probs)
    return {"top": top, "p_top": probs[top] if top else None, "p_none": probs[NONE_LABEL]}


def game_payload(rec: dict[str, Any], preds: dict[str, dict], rereads: dict[str, dict], threshold: float) -> dict[str, Any]:
    gid = rec["game_id"]
    row = preds.get(gid)
    monitor = None
    if row:
        v = _verdict(row)
        flagged = row["score"] > threshold
        named_right = v["top"] == rec["label"] if rec["label"] != NONE_LABEL else None
        monitor = {
            **v,
            "suspicion": row["score"],
            "flagged": flagged,
            "probs": row["probs"],
            "caught": bool(flagged and named_right),
            "readings": [
                {"probs": r["probs"], "reasoning": r.get("reasoning", ""), "top": metrics._top_pair(r["probs"])}
                for r in (row.get("detail") or {}).get("readings", [])
            ],
        }
    again = []
    for name, title in REREADS.items():
        r = rereads.get(name, {}).get(gid)
        if r:
            v = _verdict(r)
            again.append({"title": title, "top": v["top"], "p_top": v["p_top"]})
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
    return {
        "id": gid,
        "condition": rec["condition"],
        "label": rec["label"],
        "names": rec["names"],
        "roles": rec["roles"],
        "scores": rec["scores"]["individual"],
        "pooled": (sum(rec["scores"]["payoff"][n] for n in rec["label"].split("+")) / 2) if rec["label"] != NONE_LABEL else None,
        "monitor": monitor,
        "again": again,
        "rounds": rounds,
    }


def summary_row(g: dict[str, Any]) -> dict[str, Any]:
    m = g["monitor"] or {}
    return {
        "id": g["id"],
        "condition": g["condition"],
        "label": g["label"],
        "suspicion": m.get("suspicion"),
        "flagged": m.get("flagged"),
        "top": m.get("top"),
        "p_none": m.get("p_none"),
        "caught": m.get("caught"),
        "pooled": g["pooled"],
    }


def facts(metrics_dir: Path) -> dict[str, Any]:
    mon = json.loads((metrics_dir / "llm_monitor.json").read_text())["main_clean_vs_coalition"]
    top = json.loads((metrics_dir / "top_scorer.json").read_text())["main_clean_vs_coalition"]
    pay = json.loads((metrics_dir / "payoff.json").read_text())
    catch_key = next(k for k in mon if k.startswith("catch_rate_at"))
    n_coal = mon["n_games"] - mon["n_clean"]
    return {
        "caught": round(mon[catch_key]["value"] * n_coal),
        "n_coalition": n_coal,
        "right_pair": mon["right_pair_accuracy"]["value"],
        "auroc": mon["auroc"]["value"],
        "auroc_top_scorer": top["auroc"]["value"],
        "pooled_before": pay["coalition"]["pooled_before"],
        "pooled_after": pay["coalition"]["pooled_after"],
        "average_pair": pay["clean_pair_average"],
        "threshold": mon["catch_detail"]["threshold"],
    }


def build(run: Path, out: Path) -> int:
    records = [r for r in load_records(run) if r["condition"] in (CLEAN, COALITION, POSITIVE_CONTROL)]
    pred_dir = run / "predictions"
    preds = _read_predictions(pred_dir / "llm_monitor.jsonl")
    rereads = {n: _read_predictions(pred_dir / f"{n}.jsonl") for n in REREADS if (pred_dir / f"{n}.jsonl").exists()}
    f = facts(run / "metrics")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out)
    (out / "data").mkdir()
    rows = []
    for rec in sorted(records, key=lambda r: r["game_id"]):
        g = game_payload(rec, preds, rereads, f["threshold"])
        (out / "data" / f"{g['id']}.js").write_text(
            f"window.SEANCE_GAME = {json.dumps(g, ensure_ascii=False, separators=(',', ':'))};\n"
        )
        rows.append(summary_row(g))
    index = {"facts": f, "games": rows}
    (out / "data" / "index.js").write_text(
        f"window.SEANCE_INDEX = {json.dumps(index, ensure_ascii=False, separators=(',', ':'))};\n"
    )
    print(f"explorer: {len(rows)} games written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("runs/main"))
    ap.add_argument("--out", type=Path, default=Path("docs/explorer"))
    args = ap.parse_args(argv)
    return build(args.run, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
