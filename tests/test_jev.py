import json

import pytest

from seance import exploratory as X
from seance import jev as J
from seance.conditions import NONE_LABEL, build_specs
from seance.detectors.decision_model import QUESTION_ID, DecisionModelMonitor, build_request, question_sha256
from seance.evaluate import predict_all
from seance.fake import ScriptedModel
from seance.run import play_game


@pytest.fixture(scope="module")
async def record():
    return await play_game(build_specs(0, 1, 0, seed=51)[0], ScriptedModel(seed=51))


def answer_for(req, version="jev-1.14.0", tokens=4000):
    crit = req["questions"][QUESTION_ID]["criteria"] if QUESTION_ID in req["questions"] else None
    if crit is None:
        return {"model": version, "answers": {"probe": {"type": "noul", "noul": 0.99}}, "usage": {"input_tokens": 20, "output_tokens": 1}}
    probs = {k: (0.7 if k == NONE_LABEL else 0.05) for k in crit}
    return {"model": version, "answers": {QUESTION_ID: {"type": "choice", "choice": NONE_LABEL, "confidence": 0.6, "probabilities": probs}},
            "usage": {"input_tokens": tokens, "output_tokens": 90}}


def test_key_from_env_then_file(tmp_path):
    f = tmp_path / "api_key"
    f.write_text("  file-key\n")
    assert J.api_key({J.KEY_ENV: " env-key \n"}, f) == "env-key"
    assert J.api_key({}, f) == "file-key"
    with pytest.raises(SystemExit) as e:
        J.api_key({}, tmp_path / "missing")
    assert "file-key" not in str(e.value)


async def test_scorer_retries_pins_version_and_counts_spend(record, monkeypatch):
    monkeypatch.setattr(J.asyncio, "sleep", _no_sleep)
    calls = []

    def post(url, body, key):
        calls.append((url, key))
        if len(calls) == 2:
            return 429, {"detail": "slow down"}, 0.0
        return 200, answer_for(body), None

    s = J.JevScorer("k", post=post)
    assert await s.probe() == "jev-1.14.0"
    req = build_request(X.random_names_view_fn()(record), "plain", J.MODEL)
    await s(req)
    assert len(calls) == 3 and all(c == (J.API_URL, "k") for c in calls)
    assert s.input_tokens == 4020 and s.spent_usd == pytest.approx(4020 * 0.042 / 1e6)


async def test_scorer_refuses_another_version_another_model_and_the_cap(record):
    req = build_request(X.random_names_view_fn()(record), "plain", J.MODEL)
    versions = iter(["jev-1.14.0", "jev-1.15.0"])
    s = J.JevScorer("k", post=lambda u, b, k: (200, answer_for(b, next(versions)), None))
    await s(req)
    with pytest.raises(RuntimeError, match="pinned"):
        await s(req)
    with pytest.raises(ValueError):
        await s({**req, "model": "jev-preview"})
    capped = J.JevScorer("k", post=lambda u, b, k: (200, answer_for(b, tokens=20_000_000), None), cap_usd=0.5)
    await capped(req)
    with pytest.raises(J.CostCapReached):
        await capped(req)
    bad = J.JevScorer("secret-key", post=lambda u, b, k: (401, {"detail": "bad key"}, None))
    with pytest.raises(RuntimeError) as e:
        await bad(req)
    assert "secret-key" not in str(e.value)


async def test_predictions_record_the_version(record, tmp_path):
    s = J.JevScorer("k", post=lambda u, b, k: (200, answer_for(b), None))
    version = await s.probe()
    det = DecisionModelMonitor(s, J.MODEL, version, "plus_factors")
    path = tmp_path / "jev.jsonl"
    await predict_all(det, [record], path, view_fn=X.random_names_view_fn())
    row = json.loads(path.read_text())
    assert row["detector"] == {
        "name": "decision_model_monitor", "model": "jev-latest", "revision": "jev-1.14.0",
        "variant": "plus_factors", "question_sha256": question_sha256("plus_factors"), "n_readings": 1,
    }
    assert row["detail"]["response_model"] == "jev-1.14.0"
    assert row["probs"][NONE_LABEL] == pytest.approx(0.7)


def test_compare_pairs_name_known_monitors():
    assert all(a in J.MONITORS and b in J.MONITORS for a, b in J.PAIRS)
    assert J.MONITORS["jev_plain"] == "jev_plain" and J.MONITORS["llm_plus"] == "llm_monitor_pf_plus_factors"


async def _no_sleep(_):
    return None
