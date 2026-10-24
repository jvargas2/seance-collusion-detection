"""The monitor's view must never contain visions, private notes or anything
else hidden, in any form."""

import copy
import json
import re

import pytest

from seance.conditions import build_specs
from seance.fake import ScriptedModel
from seance.inference import Completion
from seance.public_view import public_view, render_text
from seance.run import play_game

CANARY = "ZZLEAK"
CANARY_NUM = 987654


class CanaryNotes(ScriptedModel):
    """Scripted model whose private notes carry a unique canary per call."""

    async def complete(self, messages, schema_name, schema):
        comp = await super().complete(messages, schema_name, schema)
        obj = json.loads(comp.text)
        obj["private_note"] = f"{CANARY}-note-{self.calls} {obj['private_note']}"
        return Completion(text=json.dumps(obj), model=comp.model)


async def _records(n_each=2, seed=3):
    specs = build_specs(n_each, n_each, n_each, seed=seed)
    return [await play_game(s, CanaryNotes(seed=seed)) for s in specs]


def _dump(view) -> str:
    return json.dumps(view, ensure_ascii=False) + "\n" + render_text(view)


def _paths(obj, prefix=""):
    """Every leaf path, with dict keys of medium names collapsed to <name>."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = "<name>" if k in ("Ada", "Bram", "Cora", "Dov") else k
            yield from _paths(v, f"{prefix}.{key}")
    elif isinstance(obj, list):
        for v in obj:
            yield from _paths(v, f"{prefix}[]")
    else:
        yield prefix


ALLOWED_PATHS = {
    ".public_view_version",
    ".game_id",
    ".names[]",
    ".rounds[].round",
    ".rounds[].messages[].phase",
    ".rounds[].messages[].speaker",
    ".rounds[].messages[].text",
    ".rounds[].guesses.<name>",
    ".rounds[].secret",
    ".rounds[].points.<name>",
    ".totals.<name>",
}


async def test_view_has_only_whitelisted_fields():
    for rec in await _records():
        assert set(_paths(public_view(rec))) == ALLOWED_PATHS


async def test_no_private_notes_visions_roles_or_seed_in_view():
    for rec in await _records():
        view = public_view(rec)
        text = _dump(view)
        assert CANARY not in text
        for r in rec["rounds"]:
            for notes in r["private_notes"].values():
                for note in [*notes["speak"], notes["guess"]]:
                    assert note not in text
        assert str(rec["seed"]) not in text
        gid = view["game_id"].lower()
        assert re.fullmatch(r"[gp]\d+", gid)
        assert not any(x in gid for x in ("clean", "coalition", "positive", "none", "ada", "bram", "cora", "dov"))
        for word in ("vision", "private", "role", "ally", "allies", "solo", "condition", "label",
                     "payoff", "pooled", "coalition", "positive", "clean", "seed", "prompt", "raw"):
            assert not re.search(rf"\b{word}", json.dumps(view).lower()), word
        # No [lo, hi] pair survives anywhere as a structure.
        visions = {tuple(v) for r in rec["rounds"] for v in r["visions"].values()}
        def walk(o):
            if isinstance(o, list):
                if len(o) == 2 and all(isinstance(x, int) for x in o):
                    assert tuple(o) not in visions
                for x in o:
                    walk(x)
            elif isinstance(o, dict):
                for x in o.values():
                    walk(x)
        walk(view)


async def test_view_matches_the_public_record():
    rec = (await _records(1))[0]
    view = public_view(rec)
    for rv, rr in zip(view["rounds"], rec["rounds"]):
        assert rv["messages"] == rr["messages"]
        assert rv["guesses"] == rr["guesses"]
        assert rv["points"] == rr["points"]
        assert rv["secret"] == rr["secret"]
    assert view["totals"] == rec["scores"]["individual"]


def _poison(record):
    """Put a canary into every hidden field and add unknown fields at every level."""
    rec = copy.deepcopy(record)
    public_round_keys = {"round", "messages", "guesses", "secret", "points"}
    public_top = {"game_id", "names", "rounds"}

    def poison_value(v):
        if isinstance(v, dict):
            return {k: poison_value(x) for k, x in v.items()} | {f"{CANARY}_k": CANARY}
        if isinstance(v, list):
            return [poison_value(x) for x in v] + [CANARY]
        if isinstance(v, bool) or v is None:
            return CANARY
        if isinstance(v, (int, float)):
            return CANARY_NUM
        return f"{CANARY} {v}"

    for k in list(rec):
        if k not in public_top:
            rec[k] = poison_value(rec[k])
    rec[f"{CANARY}_top"] = CANARY
    for r in rec["rounds"]:
        for k in list(r):
            if k not in public_round_keys:
                r[k] = poison_value(r[k])
        r["visions"] = {n: [CANARY_NUM, CANARY_NUM + 39] for n in rec["names"]}
        r[f"{CANARY}_round"] = CANARY
        for m in r["messages"]:
            m["hidden"] = CANARY
            m["private_note"] = CANARY
        r["guesses"][f"{CANARY}_extra"] = CANARY_NUM
        r["points"][f"{CANARY}_extra"] = CANARY_NUM
    return rec


async def test_poisoned_hidden_fields_never_reach_the_view():
    for rec in await _records(1):
        view = public_view(_poison(rec))
        text = _dump(view)
        assert CANARY not in text
        assert str(CANARY_NUM) not in text
        assert set(_paths(view)) == ALLOWED_PATHS
        # Poisoning hidden fields changes nothing public.
        assert view == public_view(rec)


def test_view_requires_public_fields_rather_than_guessing():
    with pytest.raises((KeyError, TypeError)):
        public_view({"game_id": "g1", "names": ["Ada"], "rounds": [{"round": 1}]})
