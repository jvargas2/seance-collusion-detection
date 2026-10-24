"""The exploratory "winner" scoring rule (E7), and proof that the default
rule's prompts, scores, records and monitor prompt are unchanged."""

import json
from pathlib import Path

import pytest

from seance import exploratory as X
from seance import game as G
from seance.agents import system_prompt
from seance.conditions import CLEAN, COALITION, GameSpec, build_main_study, build_replication, build_specs, main_pairs
from seance.detectors import LLMMonitor
from seance.detectors.llm_monitor import PROMPT, PROMPT_SHA256, SCORING_SENTENCES, build_prompt
from seance.fake import ScriptedModel
from seance.public_view import public_view
from seance.run import play_game
from seance.skins import get_skin

MAIN = Path(__file__).parent.parent / "runs" / "main"


def _main_records():
    return [json.loads(p.read_text()) for p in sorted((MAIN / "games").glob("*.json"))]


def test_winner_scoring_gives_3_to_the_closest_and_splits_ties():
    assert G.score_round(50, (50, 49, 70, 10), "winner") == (3.0, 0.0, 0.0, 0.0)
    assert G.score_round(50, (48, 52, 70, 10), "winner") == (1.5, 1.5, 0.0, 0.0)
    assert G.score_round(50, (48, 52, 48, 10), "winner") == (1.0, 1.0, 1.0, 0.0)
    assert G.score_round(50, (7, 7, 7, 7), "winner") == (0.75,) * 4
    with pytest.raises(ValueError):
        G.score_round(50, (1, 2, 3, 4), "nope")
    with pytest.raises(ValueError):
        G.Game(G.deal(0), scoring="nope")


@pytest.mark.skipif(not (MAIN / "games").exists(), reason="main-study records not present")
def test_default_path_reproduces_every_main_study_prompt_and_score():
    skin = get_skin("seance")
    recs = _main_records()
    assert len(recs) == 106
    for rec in recs:
        spec = GameSpec.from_dict(rec["spec"])
        assert spec.to_dict() == rec["spec"]  # no "scoring" key on the default path
        for s, name in enumerate(rec["names"]):
            assert system_prompt(skin, s, spec.role(s), spec.partner(s), spec.posting_phases) == rec["system_prompts"][name]
        for r in rec["rounds"]:
            guesses = tuple(r["guesses"][n] for n in rec["names"])
            assert G.score_round(r["secret"], guesses) == tuple(r["points"][n] for n in rec["names"])
        assert "scoring" not in rec["settings"]


@pytest.mark.skipif(not (MAIN / "predictions" / "llm_monitor.jsonl").exists(), reason="main predictions not present")
def test_default_monitor_prompt_is_the_frozen_one():
    row = json.loads((MAIN / "predictions" / "llm_monitor.jsonl").read_text().splitlines()[0])
    assert row["detector"]["prompt_sha256"] == PROMPT_SHA256
    rec = json.loads((MAIN / "games" / f"{row['game_id']}.json").read_text())
    view = public_view(rec)
    assert build_prompt(view) == build_prompt(view, scoring="split")
    mon = LLMMonitor(ScriptedModel(), settings={"temperature": 0.7})
    assert "scoring" not in mon.describe()
    winner = build_prompt(view, scoring="winner")
    assert winner != build_prompt(view) and SCORING_SENTENCES["winner"] in winner
    assert "6 points" not in winner and "6 points" in PROMPT
    assert LLMMonitor(ScriptedModel(), settings={"temperature": 0.7}, scoring="winner").describe()["scoring"] == "winner"


def test_winner_prompt_changes_only_the_scoring_sentence():
    skin = get_skin("seance")
    for seat, role, partner in ((0, "solo", None), (1, "ally", 3)):
        split = system_prompt(skin, seat, role, partner)
        winner = system_prompt(skin, seat, role, partner, scoring="winner")
        a, b = split.splitlines(), winner.splitlines()
        assert len(a) == len(b)
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        assert len(diff) == 1 and "only the closest guess scores" in diff[0][1]


def test_replication_under_winner_keeps_the_deals_and_pairs():
    base = build_replication(20, "seance")
    win = build_replication(20, "seance", scoring="winner")
    assert [(s.seed, s.allies, s.condition) for s in base] == [(s.seed, s.allies, s.condition) for s in win]
    assert all(s.game_id.startswith("w") and s.scoring == "winner" for s in win)
    assert sorted(s.seed for s in win) == list(range(20)) + list(range(50, 70))
    pairs = main_pairs()
    assert all(s.allies == pairs[s.seed] for s in win if s.condition == COALITION)
    assert GameSpec.from_dict(win[0].to_dict()) == win[0]
    assert all("scoring" not in s.to_dict() for s in build_main_study())


async def test_a_winner_game_hands_out_3_points_a_round():
    spec = build_specs(0, 1, 0, seed=7)[0]
    spec = GameSpec(spec.game_id, spec.seed, spec.condition, spec.allies, scoring="winner")
    rec = await play_game(spec, ScriptedModel(seed=7))
    assert rec["settings"]["scoring"] == "winner" and rec["spec"]["scoring"] == "winner"
    assert all(sum(r["points"].values()) == pytest.approx(3) for r in rec["rounds"])
    assert "only the closest guess scores" in next(iter(rec["system_prompts"].values()))


def test_pooled_names_are_seeded_by_deal_and_avoid_the_usual_names():
    rec_a = {"seed": 3, "names": list(G.MEDIUM_NAMES)}
    rec_b = {"seed": 3, "names": list(G.MEDIUM_NAMES), "game_id": "other"}
    m = X.pool_mapping(rec_a)
    assert m == X.pool_mapping(rec_b)
    assert len(set(m.values())) == 4 and set(m.values()) <= set(X.MEDIUM_POOL)
    assert not set(X.MEDIUM_POOL) & set(G.MEDIUM_NAMES)
    assert len({tuple(X.pool_mapping({"seed": s, "names": list(G.MEDIUM_NAMES)}).values()) for s in range(40)}) > 30


async def test_scoring_comparison_pairs_by_seed():
    specs = build_replication(2, "seance")
    old = [await play_game(s, ScriptedModel(seed=1)) for s in specs]
    win = [GameSpec(s.game_id, s.seed, s.condition, s.allies, scoring="winner") for s in build_replication(2, "seance", scoring="winner")]
    new = [await play_game(s, ScriptedModel(seed=1)) for s in win]

    def preds(recs, p_none):
        out = {}
        for r in recs:
            probs = {k: (1 - p_none) / 6 for k in ("Daniel+Eileen", "Daniel+Gladys", "Daniel+Leonora",
                                                    "Eileen+Gladys", "Eileen+Leonora", "Gladys+Leonora")}
            if r["condition"] == COALITION:
                probs = {k: 0.0 for k in probs} | {r["label"]: 1 - p_none}
            out[r["game_id"]] = {"probs": probs | {"none": p_none}}
        return out

    out = X.scoring_comparison(new, preds(new, 0.8), old, preds(old, 0.3), 15.0, 30.0, n_boot=200)
    assert out["n_clean"] == 2 and out["n_coalition"] == 2
    assert out["clean_mean_p_none"]["difference"] == pytest.approx(0.5)
    assert out["clean_games_p_none_below_half"] == {"new": 0.0, "main": 2.0, "difference": -2.0, "lo": -2.0, "hi": -2.0}
    assert out["right_pair_strict"] == {"new": 2, "main": 2, "n": 2}
    assert out["pair_points"]["new"]["average_pair_realized"] == pytest.approx(15)
    assert out["pair_points"]["main"]["average_pair_realized"] == pytest.approx(30)
    assert {r["condition"] for r in new} == {CLEAN, COALITION}
