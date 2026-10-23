"""Wording packs. Each pack is a set of flavor nouns over shared templates."""

from __future__ import annotations

from dataclasses import dataclass

from . import neutral, seance
from .templates import NOUN_SLOTS


@dataclass(frozen=True)
class Skin:
    name: str
    nouns: dict[str, str]

    def render(self, template: str, **values: object) -> str:
        overlap = set(values) & set(self.nouns)
        if overlap:
            raise ValueError(f"values may not override flavor nouns: {sorted(overlap)}")
        return template.format(**self.nouns, **values)


def _load(module) -> Skin:
    missing = set(NOUN_SLOTS) ^ set(module.NOUNS)
    if missing:
        raise ValueError(f"skin {module.NAME} slot mismatch: {sorted(missing)}")
    return Skin(module.NAME, dict(module.NOUNS))


SKINS: dict[str, Skin] = {s.name: s for s in (_load(seance), _load(neutral))}


def get_skin(name: str) -> Skin:
    try:
        return SKINS[name]
    except KeyError:
        raise ValueError(f"unknown skin {name!r}; have {sorted(SKINS)}") from None
