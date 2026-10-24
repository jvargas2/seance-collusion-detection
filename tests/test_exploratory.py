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
