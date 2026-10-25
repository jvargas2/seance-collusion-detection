import json
import re
from collections import Counter

import numpy as np
import pytest

from seance import exploratory as X
from seance.conditions import CLEAN, COALITION, LABELS, NONE_LABEL, build_specs
from seance.detectors import LLMMonitor
from seance.agents import ReplyError
from seance.detectors.llm_monitor import SCHEMA, build_prompt, label_order, parse_reply, schema_for
from seance.evaluate import predict_all
from seance.fake import ScriptedModel
from seance.inference import Completion
from seance.public_view import public_view, render_text
from seance.run import play_game


@pytest.fixture(scope="module")
async def record():
    return await play_game(build_specs(0, 1, 0, seed=51)[0], ScriptedModel(seed=51))


def test_usual_seat_order_keeps_the_frozen_prompt_and_schema(record):
    view = public_view(record)
    assert label_order(view["names"]) == list(LABELS)
    assert json.dumps(schema_for(label_order(view["names"]))) == json.dumps(SCHEMA)
    assert X.rotate_view(view, 0) == view


def test_rotation_reorders_but_changes_no_value(record):
    view = public_view(record)
    rot = X.rotate_view(view, 2)
    names = view["names"]
    assert rot["names"] == names[2:] + names[:2]
    assert list(rot["totals"]) == rot["names"] and rot["totals"] == view["totals"]
    for a, b in zip(view["rounds"], rot["rounds"]):
        assert [m["speaker"] for m in b["messages"]] == rot["names"]
        assert sorted(map(json.dumps, a["messages"])) == sorted(map(json.dumps, b["messages"]))
        assert a["guesses"] == b["guesses"] and a["points"] == b["points"] and a["secret"] == b["secret"]
    text = render_text(rot)
    assert text.index(names[2] + ":") < text.index(names[0] + ":")
    prompt = build_prompt(rot)
    order = label_order(rot["names"])
    assert order[0] == f"{names[2]}+{names[3]}" and order[-1] == NONE_LABEL
    assert f'"{names[2]}+{names[0]}"' in prompt and f'"{names[0]}+{names[2]}"' not in prompt
    # A reply in the rotated keys comes back keyed by the usual labels.
    reply = {k: (0.4 if k == f"{names[3]}+{names[0]}" else 0.1) for k in order}
    probs, _ = parse_reply(json.dumps({"reasoning": "r", "probabilities": reply}), order)
    assert set(probs) == set(LABELS) and probs[f"{names[0]}+{names[3]}"] == pytest.approx(0.4)
    with pytest.raises(ReplyError):
        parse_reply(json.dumps({"reasoning": "r", "probabilities": reply}))  # usual keys expected


def test_rotations_are_balanced_within_each_condition():
    recs = [{"game_id": f"c{i}", "condition": CLEAN} for i in range(50)]
    recs += [{"game_id": f"k{i}", "condition": COALITION} for i in range(50)]
    rot = X.assign_rotations(recs)
    for prefix in "ck":
        counts = Counter(v for k, v in rot.items() if k.startswith(prefix))
        assert sorted(counts.values()) == [12, 12, 13, 13]
    assert rot == X.assign_rotations(recs)  # seeded


class Recorder:
    model = "recorder"

    def __init__(self):
        self.schemas = []

    async def complete(self, messages, schema_name, schema):
        keys = list(schema["properties"]["probabilities"]["properties"])
        self.schemas.append(keys)
        probs = {k: 1 / 7 for k in keys}
        return Completion(text=json.dumps({"reasoning": "r", "probabilities": probs}), model=self.model)


async def test_rotated_monitor_run_records_the_seat_order(record, tmp_path):
    client = Recorder()
    mon = LLMMonitor(client, settings={"temperature": 0.7}, n_readings=3)
    recs = [record]
    view_fn = lambda rec: X.rotate_view(public_view(rec), 1)  # noqa: E731
    done = await predict_all(mon, recs, tmp_path / "p.jsonl", view_fn=view_fn)
    row = done[record["game_id"]]
    assert row["seat_order"] == record["names"][1:] + record["names"][:1]
    assert client.schemas[0] == label_order(row["seat_order"])


def test_recalibration_fixes_an_overconfident_detector():
    rng = np.random.default_rng(1)
    labels, preds = [], []
    for i in range(100):
        label = NONE_LABEL if i < 50 else LABELS[i % 6]
        # Ranks well but always near-certain of collusion, like the monitor.
        guess = LABELS[i % 6] if rng.random() < 0.6 else LABELS[(i + 1) % 6]
        p = {k: 0.001 for k in LABELS}
        p[guess] = 0.95 if label != NONE_LABEL else 0.7
        p[NONE_LABEL] = 0.01 if label != NONE_LABEL else 0.25
        total = sum(p.values())
        labels.append(label)
        preds.append({k: v / total for k, v in p.items()})
    res = X.recalibrate(labels, preds, n_splits=10, n_boot=200)
    r = res["temperature_and_bias"]
    assert r["log_loss_held_out"] < res["raw_log_loss"]
    assert r["fit_on_all_games"]["none_bias"] > 0


def test_position_effect_counts_seats_and_names():
    names = ["Daniel", "Eileen", "Gladys", "Leonora"]

    def probs(top):
        p = {k: 0.05 for k in LABELS}
        p[top] = 0.7
        return p

    records = [
        {"game_id": "a", "condition": CLEAN, "label": NONE_LABEL, "names": names},
        {"game_id": "b", "condition": CLEAN, "label": NONE_LABEL, "names": names},
        {"game_id": "c", "condition": COALITION, "label": "Gladys+Leonora", "names": names},
        {"game_id": "d", "condition": CLEAN, "label": NONE_LABEL, "names": names},
    ]
    preds = {
        "a": {"probs": probs("Daniel+Eileen"), "seat_order": names},
        "b": {"probs": probs("Daniel+Eileen"), "seat_order": names[2:] + names[:2]},
        "c": {"probs": probs("Gladys+Leonora"), "seat_order": names[2:] + names[:2]},
        "d": {"probs": {k: 1 / 7 for k in LABELS}, "seat_order": names},  # a tie names no pair
    }
    res = X.position_effect(records, preds, n_boot=50)
    clean = res["clean_top_pair_includes"]
    assert clean["Daniel"]["k"] == 2 and clean["seat_1"]["k"] == 1 and clean["seat_3"]["k"] == 1
    right = res["coalition_right_pair"]
    assert clean["Daniel"]["n"] == 2 and res["ties_named_no_pair"][CLEAN] == 1
    assert right["ally_seated_first"] == {**right["ally_seated_first"], "k": 1, "n": 1}
    assert right["daniel_not_ally"]["n"] == 1 and right["no_ally_seated_first"]["n"] == 0


def test_renaming_changes_names_everywhere_and_maps_answers_back(record):
    view = public_view(record)
    names = view["names"]
    mapping = dict(zip(names, ["William", "Florence", "Arthur", "Margaret"]))
    view["rounds"][0]["messages"][0]["text"] = f"{names[1]}'s range and {names[2]} agree; Danielle stays."
    ren = X.rename_view(view, mapping)
    text = render_text(ren)
    for n in names:
        assert f"{n}:" not in text and f" {n} " not in text and f"{n}'s" not in text
    assert "Florence's range and Arthur agree" in text and "Danielle" in text  # whole words only
    assert ren["totals"] == {mapping[n]: t for n, t in view["totals"].items()}
    record_text = build_prompt(ren).split("The record:")[1]
    assert "aliases" not in build_prompt(ren)
    assert not any(re.search(rf"\b{n}\b", record_text) for n in names)
    keys = label_order(ren["names"])
    reply = {k: (0.4 if k == "Florence+Margaret" else 0.1) for k in keys}
    probs, _ = parse_reply(json.dumps({"reasoning": "r", "probabilities": reply}), keys, ren["aliases"])
    assert probs[f"{names[1]}+{names[3]}"] == pytest.approx(0.4)


def test_mens_names_are_balanced_over_players_and_pairs():
    names = ["Daniel", "Eileen", "Gladys", "Leonora"]
    recs = [{"game_id": f"c{i}", "condition": CLEAN, "names": names} for i in range(50)]
    recs += [{"game_id": f"k{i}", "condition": COALITION, "names": names} for i in range(50)]
    slots = X.assign_names(recs)
    for prefix in "ck":
        games = [g for k, g in slots.items() if k.startswith(prefix)]
        pairs = Counter(tuple(sorted(n for n, (s, _) in g.items() if s == "m")) for g in games)
        assert sorted(pairs.values()) == [8, 8, 8, 8, 9, 9]
        for g in games:
            assert sorted(g.values()) == [("f", 0), ("f", 1), ("m", 0), ("m", 1)]


def test_gender_effect_counts_mens_names_in_the_top_pair():
    names = ["Daniel", "Eileen", "Gladys", "Leonora"]
    slots = {g: {"Daniel": ("m", 0), "Eileen": ("f", 0), "Gladys": ("m", 1), "Leonora": ("f", 1)} for g in "abc"}
    records = [
        {"game_id": "a", "condition": CLEAN, "label": NONE_LABEL, "names": names},
        {"game_id": "b", "condition": CLEAN, "label": NONE_LABEL, "names": names},
        {"game_id": "c", "condition": COALITION, "label": "Daniel+Gladys", "names": names},
    ]

    def probs(top):
        p = {k: 0.05 for k in LABELS}
        p[top] = 0.7
        return p

    preds = {"a": {"probs": probs("Daniel+Gladys")}, "b": {"probs": probs("Eileen+Gladys")}, "c": {"probs": probs("Daniel+Gladys")}}
    res = X.gender_effect(records, preds, slots, "a", n_boot=50)
    assert res["clean_men_in_top_pair"]["value"] == pytest.approx(1.5)
    assert res["clean_top_pair_kind"]["two_men"]["k"] == 1 and res["clean_top_pair_kind"]["one_each"]["k"] == 1
    assert res["clean_top_pair_includes_name"]["William"]["k"] == 1  # Daniel shown as William
    assert res["clean_top_pair_includes_player_who_played_as_daniel"]["k"] == 1
    assert res["coalition_right_pair"]["allies_two_men"]["k"] == 1


def test_message_style_counts_digits_and_number_words_by_role():
    rec = {
        "roles": {"A": "ally", "B": "solo"},
        "rounds": [{"messages": [{"speaker": "A", "text": "the clock strikes forty"}, {"speaker": "B", "text": "I see 30 to 69"}]}],
    }
    style = X.message_style([rec])
    assert style["ally"] == {"messages": 1, "digits": 0.0, "number_words": 1.0, "sharing_words": 0.0}
    assert style["solo"] == {"messages": 1, "digits": 1.0, "number_words": 0.0, "sharing_words": 0.0}


def test_partial_views_hide_one_side_and_keep_the_prompt_honest(record):
    view = public_view(record)
    msgs = X.partial_view(view, "messages")
    acts = X.partial_view(view, "actions")
    t_msgs, t_acts = render_text(msgs), render_text(acts)
    assert "Guesses:" not in t_msgs and "Points:" not in t_msgs and "Totals:" not in t_msgs and "Secret:" in t_msgs
    assert all(f"{m['speaker']}: " not in t_acts for m in view["rounds"][0]["messages"])
    assert "Guesses:" in t_acts and "Totals:" in t_acts
    p_full, p_msgs = build_prompt(view), build_prompt(msgs, "messages")
    assert "messages, guesses, secrets and points" in p_full and "guesses and points are hidden" in p_msgs
    assert render_text(view) == render_text(X.partial_view(view, "actions") | {"rounds": view["rounds"]})


def test_plus_factor_prompt_adds_one_paragraph_and_leaves_the_frozen_prompt_alone(record):
    from seance.detectors import llm_monitor as M

    assert M.PROMPT_SHA256 == "212c62e53f7d4d02"
    view = public_view(record)
    std, plus = build_prompt(view), build_prompt(view, prompt_variant="plus_factors")
    assert plus.replace(M.PLUS_FACTORS_PARAGRAPH + "\n\n", "") == std
    assert plus.index(M.PLUS_FACTORS_PARAGRAPH) < plus.index("Reply with only")
    client = ScriptedModel(seed=1)
    assert LLMMonitor(client, settings={"temperature": 0.7}).describe()["prompt_sha256"] == M.PROMPT_SHA256
    d = LLMMonitor(client, settings={"temperature": 0.7}, prompt_variant="plus_factors").describe()
    assert d["prompt_variant"] == "plus_factors" and d["prompt_sha256"] != M.PROMPT_SHA256
    with pytest.raises(ValueError):
        LLMMonitor(client, settings={"temperature": 0.7}, prompt_variant="nope")


def test_random_names_are_seeded_per_game_and_map_back(record):
    a, b = X.random_names(record), X.random_names(record)
    assert a == b and len(set(a.values())) == 4
    assert set(a.values()) <= set(X.MEDIUM_POOL) and not set(a.values()) & set(record["names"])
    other = {**record, "game_id": record["game_id"] + "x"}
    assert any(X.random_names(other) != X.random_names({**record, "game_id": g}) for g in ("q1", "q2", "q3"))
    view = X.random_names_view_fn()(record)
    assert view["names"] == [a[n] for n in record["names"]]
    text = render_text(view)
    assert not any(re.search(rf"\b{n}\b", text) for n in record["names"])
    assert view["aliases"] == {v: k for k, v in a.items()}


def test_compare_monitors_measures():
    labels = [NONE_LABEL] * 4 + ["Daniel+Eileen"] * 4
    recs = [{"game_id": f"g{i}", "label": la, "condition": CLEAN if la == NONE_LABEL else COALITION} for i, la in enumerate(labels)]

    def probs(none, top="Daniel+Eileen"):
        rest = (1 - none) / 6
        p = {k: rest for k in LABELS}
        p[NONE_LABEL] = none
        return p

    # a: confident "none" on clean games; b: accuses every clean game.
    a = {r["game_id"]: {"probs": probs(0.9 if r["condition"] == CLEAN else 0.2)} for r in recs}
    b = {r["game_id"]: {"probs": probs(0.3)} for r in recs}
    out = X.compare_monitors(recs, a, b, n_boot=200)
    assert out["n_games"] == 8 and out["n_clean"] == 4
    assert out["clean_mean_p_none"]["a_value"] == pytest.approx(0.9)
    assert out["clean_n_p_none_below_half"]["a_value"] == 0 and out["clean_n_p_none_below_half"]["b_value"] == 4
    assert out["catch_rate_at_0_fpr"]["a_value"] == 1.0 and out["catch_rate_at_0_fpr"]["b_value"] == 0.0
    assert out["auroc"]["a_value"] == 1.0 and out["auroc"]["b_value"] == 0.5


def test_e10_variants_leave_the_frozen_and_e8_prompts_alone(record):
    from seance.detectors import llm_monitor as M

    assert M.PROMPT_SHA256 == "212c62e53f7d4d02"
    assert M._sha(M.PROMPT_VARIANTS["plus_factors"]) == "bd978314f3e56eb4"  # E8's recorded hash
    view = public_view(record)
    std = build_prompt(view)
    assert std == build_prompt(view, prompt_variant=None) and "How to judge" not in std
    paragraphs = {
        "pf_permission": M.PF_PERMISSION_PARAGRAPH,
        "pf_evidence": M.PF_EVIDENCE_PARAGRAPH,
        "pf_extended": M.PF_EXTENDED_PARAGRAPH,
    }
    for name, para in paragraphs.items():
        text = build_prompt(view, prompt_variant=name)
        assert text.replace(para + "\n\n", "") == std
        assert text.index(para) < text.index("Reply with only")
    # The two halves together are the E8 paragraph; the extended one keeps all of it.
    assert M.PLUS_FACTORS_PARAGRAPH == M.PF_PERMISSION_PARAGRAPH + " C" + M.PF_EVIDENCE_PARAGRAPH.removeprefix("How to judge: c")
    assert "(1)" not in M.PF_PERMISSION_PARAGRAPH and "allowed" not in M.PF_EVIDENCE_PARAGRAPH
    ext = M.PF_EXTENDED_PARAGRAPH
    assert ext.startswith(M.PLUS_FACTORS_PARAGRAPH.split("(4)")[0]) and ext.endswith('favor "none".')
    assert all(f"({i})" in ext for i in range(1, 8))
    assert len({M._sha(M.PROMPT_VARIANTS[n]) for n in paragraphs} | {M.PROMPT_SHA256}) == 4


def test_e10_paths_and_pooled_summary(tmp_path):
    from seance.evaluate import GAMES_DIR

    assert X.pf_predictions_path(tmp_path, "standard").name == "llm_monitor_pf_standard.jsonl"
    assert X.pf_predictions_path(tmp_path, "plus_factors").name == "llm_monitor_pf_plus_factors.jsonl"
    assert X.pf_predictions_path(tmp_path, "pf_extended").name == "llm_monitor_pf_extended.jsonl"

    def probs(none):
        p = {k: (1 - none) / 6 for k in LABELS}
        p[NONE_LABEL] = none
        return p

    runs = []
    for run_name in ("r1", "r2"):  # the same game ids in both runs must not collide
        run = tmp_path / run_name
        (run / GAMES_DIR).mkdir(parents=True)
        (run / "predictions").mkdir()
        for i, la in enumerate([NONE_LABEL, NONE_LABEL, "Daniel+Eileen", "Daniel+Eileen"]):
            cond = CLEAN if la == NONE_LABEL else COALITION
            (run / GAMES_DIR / f"g{i}.json").write_text(json.dumps({"game_id": f"g{i}", "label": la, "condition": cond, "status": "complete"}))
        for which, clean_none in (("standard", 0.2), ("plus_factors", 0.6), ("pf_extended", 0.9)):
            rows = [{"game_id": f"g{i}", "probs": probs(clean_none if i < 2 else 0.1)} for i in range(4)]
            X.pf_predictions_path(run / "predictions", which).write_text("".join(json.dumps(r) + "\n" for r in rows))
        runs.append(run)
    out = X.pf_variants_summary(runs, n_boot=100)
    assert set(out) == {"runs", "pf_extended_minus_standard", "pf_extended_minus_plus_factors"}
    c = out["pf_extended_minus_standard"]
    assert c["n_games"] == 8 and c["n_clean"] == 4
    assert c["clean_mean_p_none"]["a_value"] == pytest.approx(0.9) and c["clean_mean_p_none"]["b_value"] == pytest.approx(0.2)
    assert out["pf_extended_minus_plus_factors"]["clean_n_p_none_below_half"]["difference"] == 0


class ThinkingMonitorModel:
    """Answers in the content; the reasoning holds a decoy answer that must not be parsed."""

    model = "fake"

    def __init__(self, fail_first: bool = False):
        self.fail_first = fail_first
        self.calls = 0

    async def complete(self, messages, schema_name, schema):
        self.calls += 1
        keys = list(schema["properties"]["probabilities"]["properties"])
        probs = {k: 1.0 if k == NONE_LABEL else 0.0 for k in keys}
        decoy = json.dumps({"reasoning": "x", "probabilities": {k: 1.0 if k != NONE_LABEL else 0.0 for k in keys}})
        if self.fail_first and self.calls == 1:
            return Completion(text="", model=self.model, reasoning="ran out " + decoy, finish_reason="length")
        text = json.dumps({"reasoning": "r", "probabilities": probs})
        return Completion(text=text, model=self.model, reasoning="thinking " + decoy, reasoning_chars=9, finish_reason="stop")


THINK_OFF = {"temperature": 0.7, "max_tokens": 2048, "enable_thinking": False,
             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
THINK_ON = {**THINK_OFF, "max_tokens": 8192, "enable_thinking": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


async def test_thinking_off_monitor_records_exactly_what_it_did_before(record, tmp_path):
    mon = LLMMonitor(ThinkingMonitorModel(), settings=THINK_OFF, n_readings=3)
    done = await predict_all(mon, [record], tmp_path / "p.jsonl", view_fn=X.random_names_view_fn())
    row = done[record["game_id"]]
    for r in row["detail"]["readings"]:
        assert [set(a) for a in r["attempts"]] == [{"raw", "prompt_tokens", "completion_tokens", "error"}]
    assert row["detector"] == {"name": "llm_monitor", "model": "fake", "prompt_sha256": "212c62e53f7d4d02", "n_readings": 3, **THINK_OFF}


async def test_thinking_on_monitor_parses_the_content_and_keeps_the_thinking(record, tmp_path):
    client = ThinkingMonitorModel(fail_first=True)
    mon = LLMMonitor(client, settings=THINK_ON, n_readings=3)
    done = await predict_all(mon, [record], tmp_path / "p.jsonl", view_fn=X.random_names_view_fn())
    row = done[record["game_id"]]
    assert row["probs"][NONE_LABEL] == pytest.approx(1.0)  # the content's answer, not the reasoning's decoy
    d = row["detector"]
    assert d["enable_thinking"] is True and d["max_tokens"] == 8192 and d["prompt_sha256"] == "212c62e53f7d4d02"
    attempts = [a for r in row["detail"]["readings"] for a in r["attempts"]]
    assert all(a["thinking"].startswith(("thinking", "ran out")) for a in attempts)
    stats = X.thinking_stats(done)
    assert stats["n_readings"] == 3 and stats["n_attempts"] == 4
    assert stats["parse_failures"] == 1 and stats["hit_max_tokens"] == 1
    assert stats["thinking_chars_mean"] == len(attempts[-1]["thinking"])


def test_thinking_settings_check_allows_only_the_thinking_switch():
    main = {"name": "llm_monitor", "prompt_sha256": "212c62e53f7d4d02", **THINK_OFF}
    assert X.thinking_diff(main, {**main, **THINK_ON}) == []
    assert X.thinking_diff(main, main)  # thinking still off
    extra = {**main, **THINK_ON, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 5}}
    assert X.thinking_diff(main, extra)


def _think_run(tmp_path, record, extra_env):
    run = tmp_path / "run"
    (run / "games").mkdir(parents=True, exist_ok=True)
    (run / "games" / "g.json").write_text(json.dumps(record))
    main_row = {"game_id": record["game_id"], "probs": {}, "detector": {
        "name": "llm_monitor", "model": "m", "prompt_sha256": "212c62e53f7d4d02", "n_readings": 3,
        "base_url": "http://old/v1", "json_mode": "json_schema", **THINK_OFF}}
    (tmp_path / "main.jsonl").write_text(json.dumps(main_row) + "\n")
    env = tmp_path / ".env"
    env.write_text("SEANCE_BASE_URL=http://x/v1\nSEANCE_API_KEY=k\nSEANCE_MODEL=m\nSEANCE_TEMPERATURE=0.7\n" + extra_env)
    return ["--run", str(run), "--plus-factors", "standard", "--think", "--settings-from", str(tmp_path / "main.jsonl"),
            "--env-file", str(env)]


def test_think_reading_refuses_when_thinking_is_off_or_another_setting_differs(record, tmp_path, monkeypatch, capsys):
    import os

    for k in list(os.environ):
        if k.startswith("SEANCE_"):
            monkeypatch.delenv(k)
    assert X.main(_think_run(tmp_path, record, "SEANCE_MAX_TOKENS=8192\n")) == 2
    assert "enable_thinking" in capsys.readouterr().out
    assert X.main(_think_run(tmp_path, record, "SEANCE_ENABLE_THINKING=true\nSEANCE_TEMPERATURE=0.5\n")) == 2
    assert "temperature" in capsys.readouterr().out
    assert not list((tmp_path / "run" / "predictions").glob("*.jsonl"))


def _other_model_run(tmp_path, record, extra_env, which="plus_factors", tag=True):
    args = _think_run(tmp_path, record, extra_env)
    args[args.index("--plus-factors") + 1] = which
    args.remove("--think")
    return args + (["--monitor-tag", "gemma"] if tag else [])


def test_other_model_plus_factor_reading_lets_only_the_model_and_prompt_differ(record, tmp_path, monkeypatch, capsys):
    import os

    for k in list(os.environ):
        if k.startswith("SEANCE_"):
            monkeypatch.delenv(k)
    seen = []

    async def fake_predict_all(det, records, path, concurrency, view_fn):
        seen.append((det.describe(), path.name))
        return {}

    monkeypatch.setattr(X, "predict_all", fake_predict_all)
    other = "SEANCE_MODEL=g\nSEANCE_MAX_TOKENS=2048\n"
    assert X.main(_other_model_run(tmp_path, record, other, tag=False)) == 2  # model differs without --monitor-tag
    assert "model" in capsys.readouterr().out
    assert X.main(_other_model_run(tmp_path, record, other + "SEANCE_TEMPERATURE=0.5\n")) == 2
    assert "temperature" in capsys.readouterr().out
    assert X.main(_other_model_run(tmp_path, record, other)) == 0
    assert X.main(_other_model_run(tmp_path, record, other, which="standard")) == 0
    (plus, plus_path), (std, std_path) = seen
    assert plus_path == "llm_monitor_gemma_pf_plus_factors.jsonl" and std_path == "llm_monitor_gemma_pf_standard.jsonl"
    assert plus["model"] == std["model"] == "g" and plus["prompt_variant"] == "plus_factors"
    assert std["prompt_sha256"] == "212c62e53f7d4d02" and plus["prompt_sha256"] == "bd978314f3e56eb4"


def test_other_model_compare_monitors_and_pairs():
    from seance import other_pf

    m = other_pf.monitors("gemma")
    assert m["gemma_plain"] == "llm_monitor_gemma_pf_standard" and m["gemma_plus"] == "llm_monitor_gemma_pf_plus_factors"
    assert all(a in m and b in m for a, b in other_pf.pairs("gemma"))
