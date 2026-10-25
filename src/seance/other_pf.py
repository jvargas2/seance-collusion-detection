"""E15 analysis: another monitor model (Gemma 4 31B, tag "gemma") read with
the frozen prompt and with the plus-factor paragraph, E8's random names,
against itself and against the matching Qwen monitors (E8), held-out runs
pooled first, then per run.

    uv run python -m seance.other_pf --compare runs --tag gemma

Writes runs/main/metrics/exploratory_<tag>_pf.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clef import compare, print_comparison


def monitors(tag: str) -> dict[str, str]:
    return {
        f"{tag}_plain": f"llm_monitor_{tag}_pf_standard",
        f"{tag}_plus": f"llm_monitor_{tag}_pf_plus_factors",
        "llm_standard": "llm_monitor_pf_standard",
        "llm_plus": "llm_monitor_pf_plus_factors",
    }


def pairs(tag: str) -> list[tuple[str, str]]:
    return [(f"{tag}_plus", f"{tag}_plain"), (f"{tag}_plain", "llm_standard"), (f"{tag}_plus", "llm_plus")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=Path, required=True, metavar="RUNS_DIR")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    out = compare(args.compare, args.n_boot, monitors(args.tag), pairs(args.tag))
    (args.compare / "main" / "metrics" / f"exploratory_{args.tag}_pf.json").write_text(json.dumps(out, indent=1))
    print_comparison(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
