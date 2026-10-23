"""The one detector interface: a public view in, 7 probabilities out.

A detector only ever receives `seance.public_view.public_view(record)`, never
the record itself. Its output maps each of the 7 labels (6 pairs + "none",
`seance.conditions.LABELS`) to a probability; the values sum to 1.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from ..conditions import LABELS, NONE_LABEL

Probs = dict[str, float]


def normalize(weights: dict[str, float]) -> Probs:
    """Turn non-negative weights over all 7 labels into probabilities."""
    missing = set(LABELS) - set(weights)
    extra = set(weights) - set(LABELS)
    if missing or extra:
        raise ValueError(f"labels mismatch: missing {sorted(missing)}, extra {sorted(extra)}")
    vals = {k: float(weights[k]) for k in LABELS}
    if any(not math.isfinite(v) or v < 0 for v in vals.values()):
        raise ValueError(f"weights must be finite and non-negative: {vals}")
    total = sum(vals.values())
    if total <= 0:
        raise ValueError("weights sum to zero")
    return {k: v / total for k, v in vals.items()}


def softmax(logits: dict[str, float]) -> Probs:
    top = max(logits.values())
    return normalize({k: math.exp(v - top) for k, v in logits.items()})


def check_probs(p: Probs, tol: float = 1e-6) -> None:
    if set(p) != set(LABELS):
        raise ValueError("a prediction must cover exactly the 7 labels")
    if any(v < 0 for v in p.values()) or abs(sum(p.values()) - 1) > tol:
        raise ValueError(f"not a probability distribution: {p}")


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    async def predict(self, view: dict[str, Any]) -> Probs:
        """Probabilities over LABELS for one game's public view."""

    async def predict_with_detail(self, view: dict[str, Any]) -> tuple[Probs, dict[str, Any] | None]:
        """Probabilities plus any extra output worth logging (e.g. an LLM
        monitor's reasoning and raw replies)."""
        return await self.predict(view), None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


__all__ = ["Detector", "Probs", "normalize", "softmax", "check_probs", "LABELS", "NONE_LABEL"]
