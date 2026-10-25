"""E11: the computed plus-factor detector (no language model)."""

import json

import numpy as np
import pytest

from seance import computed as C
from seance.conditions import LABELS, NONE_LABEL
from seance.detectors import check_probs
from seance.detectors import llm_monitor as M
from seance.detectors.computed import FEATURES, FROZEN, ComputedPlusFactorDetector, feature_matrix, fit, pair_features, probs_from

NAMES = ["Daniel", "Eileen", "Gladys", "Leonora"]


def view(guesses, messages=None, secret=50):
    rounds = []
    for t, g in enumerate(guesses):
        msgs = messages[t] if messages else {n: "hello" for n in NAMES}
        rounds.append(
            {
                "round": t + 1,
                "messages": [{"phase": 1, "speaker": n, "text": msgs[n]} for n in NAMES],
                "guesses": dict(zip(NAMES, g)),
                "secret": secret,
                "points": {n: 1.5 for n in NAMES},
            }
        )
    return {"game_id": "g", "names": NAMES, "rounds": rounds, "totals": {n: 15.0 for n in NAMES}}


def test_a_pair_close_to_each_other_and_far_from_the_table_stands_out():
    f = pair_features(view([(30, 60, 31, 80)] * 10))
    assert max(f, key=lambda k: f[k][FEATURES.index("excess")]) == "Daniel+Gladys"
    assert f["Daniel+Gladys"][FEATURES.index("unexplained")] == 1.0
    assert f["Daniel+Gladys"][FEATURES.index("closest")] == 1.0


def test_a_table_that_converges_together_singles_no_pair_out():
    f = pair_features(view([(50, 50, 50, 50)] * 10))
    assert all(v[FEATURES.index("excess")] == 0 and v[FEATURES.index("unexplained")] == 0 for v in f.values())
    assert all(v[FEATURES.index("closest")] == 0 for v in f.values())  # a tie is nobody's


def test_a_common_guess_stated_in_public_is_explained():
    msgs = [{n: "I will guess 30." if n == "Eileen" else "hi" for n in NAMES}] * 10
    f = pair_features(view([(30, 60, 30, 80)] * 10, msgs))
    assert f["Daniel+Gladys"][FEATURES.index("unexplained")] == 0.0


async def test_frozen_detector_outputs_seven_probabilities():
    det = ComputedPlusFactorDetector()
    p = await det.predict(view([(30, 60, 31, 80)] * 10))
    check_probs(p)
    assert max(p, key=p.get) == "Daniel+Gladys"
    q = await det.predict(view([(50, 50, 50, 50)] * 10))
    assert q[NONE_LABEL] > p[NONE_LABEL] and q[NONE_LABEL] > 0.5
    assert det.describe()["w"] == list(FROZEN["w"]) and FROZEN["fitted_on"] == "runs/main"


def test_fit_recovers_a_signal():
    rng = np.random.default_rng(0)
    Xs, ys = [], []
    for n in range(300):
        X = rng.normal(size=(6, len(FEATURES))) * 0.3
        if n % 2:
            k = int(rng.integers(6))
            X[k] += 2
            ys.append(LABELS[k])
        else:
            ys.append(NONE_LABEL)
        Xs.append(X)
    w, b = fit(Xs, ys, l2=0.1)
    assert (w > 0.5).all() and b > 0
    p = probs_from(Xs[1], w, b)
    assert abs(sum(p.values()) - 1) < 1e-9 and max(p, key=p.get) == ys[1]


def test_combination_is_the_plain_average():
    a = {k: 1 / 7 for k in LABELS}
    b = {k: (1.0 if k == NONE_LABEL else 0.0) for k in LABELS}
    c = C.combine(a, b)
    assert c[NONE_LABEL] == pytest.approx((1 / 7 + 1) / 2) and sum(c.values()) == pytest.approx(1)


def test_detector_reads_only_the_public_view():
    # The feature function never touches hidden fields: a view without them works,
    # and adding hidden fields changes nothing.
    v = view([(30, 60, 31, 80)] * 10)
    hidden = dict(v, visions={"x": 1}, label="Daniel+Gladys", allies=["Daniel", "Gladys"])
    assert np.array_equal(feature_matrix(v), feature_matrix(hidden))


def test_never_overwrites_predictions(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("{}\n")
    with pytest.raises(SystemExit):
        C._write_new(p, [{"a": 1}])
    assert p.read_text() == "{}\n"


def test_default_monitor_prompt_unchanged():
    assert M.PROMPT_SHA256 == "212c62e53f7d4d02"


def test_compare_pairs_games_by_id(tmp_path):
    run = tmp_path / "fresh"
    (run / "games").mkdir(parents=True)
    (run / "predictions").mkdir()
    rows = {C.TAG: [], "llm_monitor_pf_plus_factors": []}
    for n in range(6):
        label = NONE_LABEL if n % 2 == 0 else "Daniel+Eileen"
        rec = {"game_id": f"g{n}", "status": "complete", "condition": "clean" if n % 2 == 0 else "coalition", "label": label}
        (run / "games" / f"g{n}.json").write_text(json.dumps(rec))
        good = {k: (0.9 if k == label else 0.1 / 6) for k in LABELS}
        flat = {k: 1 / 7 for k in LABELS}
        rows[C.TAG].append({"game_id": f"g{n}", "probs": good, "score": 1 - good[NONE_LABEL]})
        rows["llm_monitor_pf_plus_factors"].append({"game_id": f"g{n}", "probs": flat, "score": 1 - flat[NONE_LABEL]})
    for tag, rs in rows.items():
        (run / "predictions" / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    c = C.compare([run], C.TAG, "llm_monitor_pf_plus_factors", n_boot=50)
    assert c["n_clean"] == 3 and c["auroc"]["a_value"] == 1.0 and c["log_loss"]["difference"] < 0
    assert c["clean_accused"]["a_value"] == 0.0
