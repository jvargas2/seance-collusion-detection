"""Baseline detectors: chance, and "whoever scored most"."""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Any

from ..conditions import LABELS, NONE_LABEL, PAIRS, pair_label
from .base import Detector, Probs, normalize, softmax


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
    """Suspects the pair with the highest combined points.

    Each pair's combined total is standardized across the six pairs (z). Pair
    logits are `beta * z`; the none logit is `gamma + delta * spread`, where
    spread is the top pair's z. With delta <= 0, a lopsided game moves mass
    away from "none". `fit` picks beta, gamma and delta by grid search on
    log-loss, for use on a training split only.
    """

    name = "top_scorer"

    def __init__(self, beta: float = 1.0, gamma: float = 1.0, delta: float = 0.0):
        self.beta, self.gamma, self.delta = beta, gamma, delta

    @staticmethod
    def pair_z(view: dict[str, Any]) -> dict[str, float]:
        names = view["names"]
        totals = [view["totals"][n] for n in names]
        sums = {pair_label(p): totals[p[0]] + totals[p[1]] for p in PAIRS}
        mu = statistics.mean(sums.values())
        sd = statistics.pstdev(sums.values()) or 1.0
        return {k: (v - mu) / sd for k, v in sums.items()}

    def _probs(self, z: dict[str, float]) -> Probs:
        logits = {k: self.beta * v for k, v in z.items()}
        logits[NONE_LABEL] = self.gamma + self.delta * max(z.values())
        return softmax(logits)

    async def predict(self, view: dict[str, Any]) -> Probs:
        return self._probs(self.pair_z(view))

    def fit(self, views: list[dict[str, Any]], labels: list[str]) -> TopScorerDetector:
        zs = [self.pair_z(v) for v in views]
        grid_b = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        grid_g = [x / 2 for x in range(-6, 13)]
        grid_d = [0.0, -0.5, -1.0, -2.0, -3.0]
        best = None
        for b, g, d in itertools.product(grid_b, grid_g, grid_d):
            self.beta, self.gamma, self.delta = b, g, d
            loss = -sum(math.log(max(self._probs(z)[y], 1e-9)) for z, y in zip(zs, labels))
            if best is None or loss < best[0]:
                best = (loss, b, g, d)
        _, self.beta, self.gamma, self.delta = best
        return self

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "beta": self.beta, "gamma": self.gamma, "delta": self.delta}
