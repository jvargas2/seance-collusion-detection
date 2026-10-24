"""Exactly what a monitor may see: the public record of one game.

Built by copying a whitelist of public fields out of a game record, never by
deleting hidden ones, so a field added to records later stays hidden unless
it is added here on purpose. The monitor sees, per round: the
public messages (with their posting phase), every guess, the secret and
each medium's points; and each medium's individual running total.

It never sees visions, private notes, roles, the condition or label, the
allies, pooled payoffs, prompts, raw model output or the seed (the seed would
let anyone re-deal the visions).

`tests/test_public_view.py` fails if any hidden value leaks in any form.
"""

from __future__ import annotations

import json
from typing import Any

PUBLIC_VIEW_VERSION = 2


def public_view(record: dict[str, Any]) -> dict[str, Any]:
    names = [str(n) for n in record["names"]]
    rounds = []
    for r in record["rounds"]:
        rounds.append(
            {
                "round": int(r["round"]),
                "messages": [
                    {"phase": int(m["phase"]), "speaker": str(m["speaker"]), "text": str(m["text"])}
                    for m in r["messages"]
                ],
                "guesses": {n: int(r["guesses"][n]) for n in names},
                "secret": int(r["secret"]),
                "points": {n: float(r["points"][n]) for n in names},
            }
        )
    totals = {n: sum(r["points"][n] for r in rounds) for n in names}
    return {
        "public_view_version": PUBLIC_VIEW_VERSION,
        "game_id": str(record["game_id"]),
        "names": names,
        "rounds": rounds,
        "totals": totals,
    }


def render_text(view: dict[str, Any]) -> str:
    """A plain-text transcript of a public view, for an LLM monitor. A partial
    view (exploratory) may leave out the messages, or the guesses and points."""
    lines = [f"Game {view['game_id']}. Players: {', '.join(view['names'])}."]
    for r in view["rounds"]:
        lines.append("")
        lines.append(f"Round {r['round']}:")
        if "messages" in r:
            phases = sorted({m["phase"] for m in r["messages"]})
            for k in phases:
                if len(phases) > 1:
                    lines.append(f"  Messages, posting {k} of {len(phases)}:")
                for m in r["messages"]:
                    if m["phase"] == k:
                        lines.append(f"  {m['speaker']}: {json.dumps(m['text'], ensure_ascii=False)}")
        if "guesses" in r:
            lines.append("  Guesses: " + ", ".join(f"{n} {g}" for n, g in r["guesses"].items()) + ".")
        if "points" in r:
            lines.append(
                f"  Secret: {r['secret']}. Points: "
                + ", ".join(f"{n} {p:g}" for n, p in r["points"].items())
                + "."
            )
        else:
            lines.append(f"  Secret: {r['secret']}.")
    if "totals" in view:
        lines.append("")
        lines.append("Totals: " + ", ".join(f"{n} {t:g}" for n, t in view["totals"].items()) + ".")
    return "\n".join(lines)
