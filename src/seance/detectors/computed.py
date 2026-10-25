"""A computed plus-factor detector (E11): no language model, only the numbers.

Per pair and round it asks how close the pair's two guesses are, and whether
the openly shared information explains that closeness: honest tables that all
converge on the same shared estimate are close to everyone, so closeness only
counts when it singles the pair out from the two other players, and when the
common guess is not a number someone stated in public that round. It reads
only the public view (messages, guesses, secrets, points), never visions,
roles or labels.

Pair features, each averaged over rounds:

- `excess`: log(1 + mean distance from the pair's guesses to the other two
  players' guesses) - log(1 + distance between the pair's guesses).
- `unexplained`: within 1 of each other, more than 3 from the outsiders on
  average, and their midpoint not within 1 of any number (digits) in that
  round's public messages.
- `closest`: the pair's distance is strictly the smallest of the six pairs.

The set (and the L2 penalty) was chosen by 10-fold cross-fitted log-loss on
runs/main only, from candidates that also included plain closeness (within 1)
and the pair's accuracy relative to the outsiders; neither helped.

Probabilities: a conditional logit over the 7 labels, pair logit `w . f`,
"none" logit `b`. The weights are fitted on runs/main (`fit`) and frozen in
`FROZEN`; held-out games are scored with the frozen numbers only.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from ..conditions import LABELS, NONE_LABEL, PAIRS, pair_label
from .base import Detector, Probs

FEATURES = ("excess", "unexplained", "closest")
_NUM = re.compile(r"\d+(?:\.\d+)?")
PAIR_LABELS = tuple(pair_label(p) for p in PAIRS)
DEFAULT_L2 = 1.0


def pair_features(view: dict[str, Any]) -> dict[str, np.ndarray]:
    """Label -> feature vector (FEATURES order) for each of the six pairs."""
    names = view["names"]
    out = {}
    for a, b in PAIRS:
        i, j = names[a], names[b]
        others = [n for n in names if n not in (i, j)]
        acc = np.zeros(len(FEATURES))
        for rd in view["rounds"]:
            g = rd["guesses"]
            d = abs(g[i] - g[j])
            out_d = float(np.mean([abs(g[x] - g[k]) for x in (i, j) for k in others]))
            stated = [float(x) for m in rd["messages"] for x in _NUM.findall(m["text"])]
            mid = (g[i] + g[j]) / 2
            explained = min((abs(mid - x) for x in stated), default=math.inf) <= 1
            rest = [abs(g[names[p]] - g[names[q]]) for p, q in PAIRS if (p, q) != (a, b)]
            acc += [
                math.log1p(out_d) - math.log1p(d),
                d <= 1 and out_d > 3 and not explained,
                d < min(rest),
            ]
        out[pair_label((a, b))] = acc / len(view["rounds"])
    return out


def feature_matrix(view: dict[str, Any]) -> np.ndarray:
    """6 x len(FEATURES), rows in PAIR_LABELS order."""
    f = pair_features(view)
    return np.array([f[k] for k in PAIR_LABELS])


def probs_from(X: np.ndarray, w: np.ndarray, b: float) -> Probs:
    logits = np.append(X @ w, b)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return dict(zip(PAIR_LABELS + (NONE_LABEL,), map(float, p)))


def fit(Xs: list[np.ndarray], labels: list[str], l2: float = DEFAULT_L2, iters: int = 100) -> tuple[np.ndarray, float]:
    """Penalized maximum likelihood (L2 on w, not on b) by Newton's method."""
    k = Xs[0].shape[1]
    theta = np.zeros(k + 1)  # w..., b
    # Design per game: 7 x (k+1); pair rows [f, 0], none row [0, 1].
    D = [np.vstack([np.hstack([X, np.zeros((6, 1))]), np.append(np.zeros(k), 1.0)]) for X in Xs]
    Y = [(PAIR_LABELS + (NONE_LABEL,)).index(y) for y in labels]
    pen = np.diag(np.append(np.full(k, l2), 0.0))
    for _ in range(iters):
        grad = pen @ theta
        hess = pen.copy()
        for Di, yi in zip(D, Y):
            z = Di @ theta
            p = np.exp(z - z.max())
            p /= p.sum()
            mu = p @ Di
            grad -= Di[yi] - mu
            hess += Di.T @ (p[:, None] * Di) - np.outer(mu, mu)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if np.abs(step).max() < 1e-10:
            break
    return theta[:k], float(theta[k])


# Fitted on all 100 clean + coalition games of runs/main (`python -m
# seance.exploratory --run runs/main --computed-fit`), l2 = DEFAULT_L2, and
# frozen before any other run was scored.
FROZEN: dict[str, Any] = {
    "w": (1.722168, 1.529297, 1.397267),
    "b": 3.140173,
    "l2": DEFAULT_L2,
    "fitted_on": "runs/main",
}


class ComputedPlusFactorDetector(Detector):
    name = "computed_plus_factor"

    def __init__(self, w: Any = None, b: float | None = None):
        w = FROZEN["w"] if w is None else w
        b = FROZEN["b"] if b is None else b
        if w is None or b is None:
            raise ValueError("no frozen weights; fit on runs/main first")
        self.w, self.b = np.asarray(w, dtype=float), float(b)

    async def predict(self, view: dict[str, Any]) -> Probs:
        return probs_from(feature_matrix(view), self.w, self.b)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "features": list(FEATURES), "w": [float(x) for x in self.w], "b": self.b}


assert set(PAIR_LABELS + (NONE_LABEL,)) == set(LABELS)
