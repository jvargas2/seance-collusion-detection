"""Figure 1 for the paper: for each monitor model, the plain prompt against
the plus-factor prompt, on honest tables accused and AUROC, with 95%
bootstrap intervals. Reads the `final_table.json` that `seance.final` writes
(no model calls, no new numbers); a model with neither prompt scored is left
out, so the same command works before and after every file exists.

    uv run --group report python -m seance.paper_figures \\
        --table runs/final/metrics/final_table.json --label "Qwen players" \\
        --table runs/final-haiku/metrics/final_table.json --label "Haiku players" \\
        --out runs/final/metrics/paper_figure1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (row label, plain row key, plus-factor row key), top to bottom
MODELS = (
    ("Qwen3.8 27B", "qwen_plain", "qwen_pf"),
    ("Gemma 4 31B", "gemma_plain", "gemma_pf"),
    ("Clef (decision model)", "clef_plain", "clef_pf"),
    ("Jev (decision model)", "jev_plain", "jev_pf"),
    ("Claude Haiku 5.5", "haiku_plain", "haiku_pf"),
    ("Claude Sonnet 5.5", "sonnet_plain", "sonnet_pf"),
    ("Claude Opus 5.5", "opus_plain", "opus_pf"),
)
PANELS = (("clean_accused", "Honest tables accused"), ("auroc", "AUROC, coalition vs honest"))

# Validated pair (dataviz reference palette, light): plain = orange, plus-factor = blue.
PLAIN, PF = "#eb6834", "#2a78d6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _measure(table: dict, key: str, measure: str) -> tuple[float, float, float] | None:
    row = table["rows"].get(key)
    if not row or not row.get("scored"):
        return None
    m = row["measures"][measure]
    scale = 1 / table["n_clean"] if measure == "clean_accused" else 1.0  # counts -> share
    return m["value"] * scale, m["lo"] * scale, m["hi"] * scale


def draw(tables: list[dict], labels: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in MODELS if any(_measure(t, k, "auroc") for t in tables for k in m[1:])]
    n_sets = len(tables)
    fig, axes = plt.subplots(
        n_sets, len(PANELS), figsize=(9.5, 0.42 * len(models) * n_sets + 1.6), squeeze=False, sharey="row",
        facecolor=SURFACE,
    )
    for si, (table, label) in enumerate(zip(tables, labels)):
        for pi, (measure, title) in enumerate(PANELS):
            ax = axes[si][pi]
            ax.set_facecolor(SURFACE)
            for y, (_, plain_key, pf_key) in enumerate(models):
                a, b = _measure(table, plain_key, measure), _measure(table, pf_key, measure)
                if a and b:
                    ax.plot([a[0], b[0]], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
                for v, color, dy, marker, face in ((a, PLAIN, -0.13, "o", SURFACE), (b, PF, 0.13, "o", PF)):
                    if v is None:
                        continue
                    ax.plot([v[1], v[2]], [y + dy] * 2, color=color, lw=2, zorder=2, solid_capstyle="round")
                    ax.plot(v[0], y + dy, marker=marker, ms=8, mfc=face, mec=color, mew=2, zorder=3, ls="none")
            ax.set_yticks(range(len(models)), [m[0] for m in models], color=INK, fontsize=9)
            if pi == 0:  # the row's axes share y, so invert once
                ax.invert_yaxis()
            ax.set_xlim(-0.03, 1.03) if measure == "clean_accused" else ax.set_xlim(0.4, 1.01)
            if measure == "clean_accused":
                ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            else:
                ax.axvline(0.5, color=MUTED, lw=1, ls=":", zorder=0)
                ax.text(0.505, len(models) - 0.55, "chance", color=MUTED, fontsize=8, va="top")
            ax.grid(axis="x", color=GRID, lw=1)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            head = title if n_sets == 1 else f"{title}\n{label}"
            ax.set_title(head, loc="left", fontsize=10, color=INK)
    handles = [
        plt.Line2D([], [], marker="o", ms=8, mfc=SURFACE, mec=PLAIN, mew=2, color=PLAIN, lw=2, label="Plain prompt"),
        plt.Line2D([], [], marker="o", ms=8, mfc=PF, mec=PF, mew=2, color=PF, lw=2, label="Plus-factor paragraph"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, action="append", required=True, help="final_table.json (repeat for a second player set)")
    ap.add_argument("--label", action="append", help="one per --table, e.g. 'Qwen players'")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    labels = args.label or [str(t.parent.parent.name) for t in args.table]
    if len(labels) != len(args.table):
        ap.error("give one --label per --table")
    draw([json.loads(t.read_text()) for t in args.table], labels, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
