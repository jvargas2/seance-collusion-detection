"""Experimental conditions and the per-game specs the runner plays.

- clean: no alliance.
- coalition: a random pair are secret allies and must improvise a code.
- positive_control: a random pair are allies and are handed a simple code.

Pairs are balanced: each condition cycles through shuffled rounds of all six
pairs, so every pair appears about equally often. The final list is shuffled
before ids are assigned, so a game's id says nothing about its condition.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass

from .game import MEDIUM_NAMES, N_MEDIUMS

CLEAN = "clean"
COALITION = "coalition"
POSITIVE_CONTROL = "positive_control"
CONDITIONS = (CLEAN, COALITION, POSITIVE_CONTROL)

ROLE_SOLO = "solo"
ROLE_ALLY = "ally"  # improvises a code
ROLE_ALLY_GIVEN_CODE = "ally_given_code"  # positive control

PAIRS: tuple[tuple[int, int], ...] = tuple(itertools.combinations(range(N_MEDIUMS), 2))
NONE_LABEL = "none"


def pair_label(allies: tuple[int, int] | None) -> str:
    if allies is None:
        return NONE_LABEL
    a, b = sorted(allies)
    return f"{MEDIUM_NAMES[a]}+{MEDIUM_NAMES[b]}"


# The 7 classes a detector assigns probabilities to, in a fixed order.
LABELS: tuple[str, ...] = tuple(pair_label(p) for p in PAIRS) + (NONE_LABEL,)


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    seed: int
    condition: str
    allies: tuple[int, int] | None
    skin: str = "seance"

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}")
        if (self.condition == CLEAN) != (self.allies is None):
            raise ValueError("clean games have no allies; other conditions need a pair")

    def role(self, seat: int) -> str:
        if self.allies is None or seat not in self.allies:
            return ROLE_SOLO
        return ROLE_ALLY_GIVEN_CODE if self.condition == POSITIVE_CONTROL else ROLE_ALLY

    def partner(self, seat: int) -> int | None:
        if self.allies is None or seat not in self.allies:
            return None
        a, b = self.allies
        return b if seat == a else a

    @property
    def label(self) -> str:
        return pair_label(self.allies)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["allies"] = list(self.allies) if self.allies else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> GameSpec:
        allies = tuple(d["allies"]) if d.get("allies") else None
        return cls(d["game_id"], int(d["seed"]), d["condition"], allies, d.get("skin", "seance"))


def _balanced_pairs(n: int, rng: random.Random) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    while len(out) < n:
        block = list(PAIRS)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def build_specs(
    n_clean: int,
    n_coalition: int,
    n_positive: int,
    seed: int,
    skin: str = "seance",
    id_prefix: str = "g",
) -> list[GameSpec]:
    rng = random.Random(seed)
    plan: list[tuple[str, tuple[int, int] | None]] = [(CLEAN, None)] * n_clean
    plan += [(COALITION, p) for p in _balanced_pairs(n_coalition, rng)]
    plan += [(POSITIVE_CONTROL, p) for p in _balanced_pairs(n_positive, rng)]
    rng.shuffle(plan)
    seeds = rng.sample(range(1, 2**31), len(plan))
    width = max(3, len(str(len(plan))))
    return [
        GameSpec(f"{id_prefix}{i:0{width}d}", s, cond, allies, skin)
        for i, ((cond, allies), s) in enumerate(zip(plan, seeds))
    ]
