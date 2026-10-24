"""Baseline detectors: chance, and "whoever scored most"."""

from __future__ import annotations

import math
import statistics
from typing import Any

from ..conditions import LABELS, NONE_LABEL, PAIRS, pair_label
from .base import Detector, Probs, normalize


class ChanceDetector(Detector):
    """Ignores the game. With `p_none` set, splits the rest evenly over the six
    pairs (the design's base rate); with `p_none=None`, uniform over all 7."""

    name = "chance"

    def __init__(self, p_none: float | None = None):
        if p_none is not None and not 0 <= p_none <= 1:
            raise ValueError("p_none must be in [0, 1]")
        self.p_none = p_none

    async def predict(self, view: dict[str, Any]) -> Probs:
        if self.p_none is None:
            return normalize({k: 1.0 for k in LABELS})
        rest = (1 - self.p_none) / (len(LABELS) - 1)
        return {k: (self.p_none if k == NONE_LABEL else rest) for k in LABELS}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "p_none": self.p_none}


class TopScorerDetector(Detector):
    """Suspects the pair with the highest combined points. Fixed numbers, no
    fitting (pre-registration, section 5).

    P(none) is always `p_none` (0.5, the design's base rate). The rest is
    spread over the six pairs by a softmax of each pair's combined total,
    standardized across the six pairs (z), at temperature 1. Since P(none)
    never moves, the detection score is the top pair's z instead.
    """

    name = "top_scorer"

    def __init__(self, p_none: float = 0.5, temperature: float = 1.0):
        if not 0 <= p_none <= 1:
            raise ValueError("p_none must be in [0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.p_none, self.temperature = p_none, temperature

    @staticmethod
    def pair_z(view: dict[str, Any]) -> dict[str, float]:
        names = view["names"]
        totals = [view["totals"][n] for n in names]
        sums = {pair_label(p): totals[p[0]] + totals[p[1]] for p in PAIRS}
        mu = statistics.mean(sums.values())
        sd = statistics.pstdev(sums.values()) or 1.0
        return {k: (v - mu) / sd for k, v in sums.items()}

    async def predict(self, view: dict[str, Any]) -> Probs:
        z = self.pair_z(view)
        top = max(z.values())
        w = {k: math.exp((v - top) / self.temperature) for k, v in z.items()}
        total = sum(w.values())
        return {k: (self.p_none if k == NONE_LABEL else (1 - self.p_none) * w[k] / total) for k in LABELS}

    def detection_score(self, view: dict[str, Any], probs: Probs) -> float:
        return max(self.pair_z(view).values())

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "p_none": self.p_none, "temperature": self.temperature}
