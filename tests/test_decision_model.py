import json
import math

import pytest

from seance import exploratory as X
from seance.conditions import LABELS, NONE_LABEL, build_specs
from seance.detectors import llm_monitor as M
from seance.detectors.decision_model import (
    QUESTION_ID,
    DecisionModelMonitor,
    build_request,
    parse_answer,
    question_sha256,
)
from seance.evaluate import predict_all
from seance.fake import ScriptedModel
from seance.public_view import public_view, render_text
from seance.run import play_game


@pytest.fixture(scope="module")
async def record():
    return await play_game(build_specs(0, 1, 0, seed=51)[0], ScriptedModel(seed=51))


def test_frozen_monitor_prompt_is_unchanged(record):
    assert M.PROMPT_SHA256 == "212c62e53f7d4d02"
    view = public_view(record)
    assert M.build_prompt(view) == M.PROMPT.format(
        names=", ".join(view["names"][:-1]) + " and " + view["names"][-1],
        n_rounds=M.N_ROUNDS, lo=M.SECRET_MIN, hi=M.SECRET_MAX, width=M.VISION_WIDTH, max_words=M.MAX_MESSAGE_WORDS,
        label_keys=", ".join(f'"{k}": <probability>' for k in LABELS), record=render_text(view),
    )


def test_request_carries_the_frozen_rules_and_the_record(record):
    view = X.random_names_view_fn()(record)
    req = build_request(view, "plain")
    q = req["questions"][QUESTION_ID]
    assert req["state"] == render_text(view)
    assert M.build_prompt(view).startswith(q["instructions"])
    assert M._REPLY_SENTENCE not in q["instructions"] and "The record:" not in q["instructions"]
    assert M.PLUS_FACTORS_PARAGRAPH not in q["instructions"]
    assert q["type"] == "choice" and list(q["criteria"]) == M.label_order(view["names"])
    assert all(n in q["instructions"] for n in view["names"])


def test_plus_factor_version_differs_only_by_the_paragraph(record):
    view = X.random_names_view_fn()(record)
    plain, plus = build_request(view, "plain"), build_request(view, "plus_factors")
    qa, qb = plain["questions"][QUESTION_ID], plus["questions"][QUESTION_ID]
    assert qb["instructions"] == qa["instructions"] + "\n\n" + M.PLUS_FACTORS_PARAGRAPH
    assert qa["criteria"] == qb["criteria"] and plain["state"] == plus["state"]
    assert question_sha256("plain") != question_sha256("plus_factors")


def test_answer_maps_shown_names_back_to_labels(record):
    view = X.random_names_view_fn()(record)
    keys = M.label_order(view["names"])
    raw = {k: (0.4 if i == 0 else 0.1) for i, k in enumerate(keys)}
    p = parse_answer({"answers": {QUESTION_ID: {"probabilities": raw}}}, view)
    assert set(p) == set(LABELS) and math.isclose(sum(p.values()), 1)
    assert p[M._canonical(keys[0], view["aliases"])] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        parse_answer({"answers": {QUESTION_ID: {"probabilities": {**raw, "extra": 0.0}}}}, view)


async def test_predictions_file(record, tmp_path):
    seen = []

    async def scorer(req):
        seen.append(req)
        crit = req["questions"][QUESTION_ID]["criteria"]
        return {"answers": {QUESTION_ID: {"probabilities": {k: (0.7 if k == NONE_LABEL else 0.05) for k in crit}}}}

    det = DecisionModelMonitor(scorer, "Cloudflare/clef", "abc", "plus_factors")
    path = tmp_path / "clef.jsonl"
    await predict_all(det, [record], path, view_fn=X.random_names_view_fn())
    row = json.loads(path.read_text())
    assert row["probs"][NONE_LABEL] == pytest.approx(0.7)
    assert row["detector"] == {
        "name": "decision_model_monitor", "model": "Cloudflare/clef", "revision": "abc",
        "variant": "plus_factors", "question_sha256": question_sha256("plus_factors"), "n_readings": 1,
    }
    assert row["seat_order"] == list(X.random_names(record).values())
    assert len(seen) == 1
