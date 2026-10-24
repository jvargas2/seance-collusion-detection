"""The rule sentences must be word-for-word identical across wording packs;
only the flavor nouns may differ."""

import re
import string

import pytest

from seance.agents import GUESS, SPEAK, system_prompt, user_prompt
from seance.conditions import ROLE_ALLY, ROLE_ALLY_GIVEN_CODE, ROLE_SOLO
from seance.skins import SKINS
from seance.skins import templates as T
from tests.helpers import game_mid_round, notes_for

SEANCE_WORDS = ("ghost", "séance", "seance", "medium", "vision", "midnight", "candle", "spirit", "knock", "haunt")


def _all_templates() -> dict[str, str]:
    return {k: v for k, v in vars(T).items() if k.isupper() and isinstance(v, str)}


def _normalize(text: str, nouns: dict[str, str]) -> str:
    """Replace each noun value with its slot name, in one pass, longest first."""
    by_value = {v: k for k, v in nouns.items()}
    pattern = re.compile("|".join(re.escape(v) for v in sorted(by_value, key=len, reverse=True)))
    return pattern.sub(lambda m: f"<{by_value[m.group(0)]}>", text)


def _rendered_prompts(skin):
    out = []
    for n_phases in (1, 2):
        for seat, role, partner in [(1, ROLE_SOLO, None), (0, ROLE_ALLY, 2), (2, ROLE_ALLY_GIVEN_CODE, 0)]:
            out.append(system_prompt(skin, seat, role, partner, n_phases))
            for phase, posted in [(SPEAK, k) for k in range(n_phases)] + [(GUESS, n_phases)]:
                game = game_mid_round(posted=posted, posting_phases=n_phases)
                notes = notes_for(seat, 2, n_phases) + [{"speak": [f"now {k}" for k in range(posted)]}]
                out.append(user_prompt(skin, game, seat, phase, notes, partner))
    first = game_mid_round(rounds_done=0, posted=0)
    out.append(user_prompt(skin, first, 0, SPEAK, [], None))
    out.append(skin.render(T.RETRY, error="bad"))
    return out


def test_every_pack_defines_exactly_the_noun_slots():
    for skin in SKINS.values():
        assert set(skin.nouns) == set(T.NOUN_SLOTS)


def test_templates_use_only_known_placeholders():
    allowed_values = {
        "me", "names", "n_mediums", "n_rounds", "secret_min", "secret_max", "width", "max_words",
        "partner", "round", "lo", "hi", "mid", "speaker", "text", "guesses", "secret",
        "points", "speak_note", "guess_note", "totals", "pooled", "error", "posting_rule", "phase", "n_phases",
    }
    for name, tmpl in _all_templates().items():
        fields = {f for _, f, _, _ in string.Formatter().parse(tmpl) if f}
        assert fields <= set(T.NOUN_SLOTS) | allowed_values, name


def test_noun_values_never_appear_in_the_fixed_template_text():
    # Otherwise reversing the substitution below could hide a real difference.
    for skin in SKINS.values():
        for name, tmpl in _all_templates().items():
            fixed = re.sub(r"\{[a-z_]+\}", "", tmpl)
            for value in skin.nouns.values():
                assert value not in fixed, (skin.name, name, value)


def test_rendered_prompts_differ_only_in_flavor_nouns():
    packs = list(SKINS.values())
    assert len(packs) >= 2
    base = [_normalize(t, packs[0].nouns) for t in _rendered_prompts(packs[0])]
    for other in packs[1:]:
        normalized = [_normalize(t, other.nouns) for t in _rendered_prompts(other)]
        for a, b in zip(base, normalized):
            assert a == b


def test_packs_actually_differ_and_neutral_has_no_seance_flavor():
    seance, neutral = SKINS["seance"], SKINS["neutral"]
    assert _rendered_prompts(seance) != _rendered_prompts(neutral)
    for text in _rendered_prompts(neutral):
        low = text.lower()
        for word in SEANCE_WORDS:
            assert word not in low, word


@pytest.mark.parametrize("skin", list(SKINS.values()), ids=lambda s: s.name)
def test_flavor_nouns_are_short_and_not_gory(skin):
    gory = ("blood", "gore", "corpse", "murder", "kill", "dead body", "severed", "scream")
    for value in skin.nouns.values():
        assert len(value.split()) <= 10
        assert not any(g in value.lower() for g in gory)
