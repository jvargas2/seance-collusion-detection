"""Whole games end to end with the scripted fake model, offline."""

import asyncio
import json
import statistics

from seance import game as G
from seance.conditions import (
    CLEAN,
    COALITION,
    LABELS,
    PAIRS,
    POSITIVE_CONTROL,
    ROLE_ALLY,
    ROLE_ALLY_GIVEN_CODE,
    ROLE_SOLO,
    build_specs,
)
from seance.fake import ScriptedModel
from seance.inference import Completion, LimitedClient
from seance.run import MANIFEST, GameFailed, failed_path, game_path, load_or_create_manifest, main, play_game, run_games


def _strip_times(rec):
    rec = dict(rec)
    for k in ("started_at", "finished_at", "code_version"):
        rec.pop(k)
    for c in rec["calls"]:
        for a in c["attempts"]:
            a.pop("latency_s")
    return rec


def test_build_specs_counts_roles_and_balance():
    specs = build_specs(12, 12, 6, seed=1)
    assert len(specs) == 30 and len({s.game_id for s in specs}) == 30
    assert len({s.seed for s in specs}) == 30
    by = {c: [s for s in specs if s.condition == c] for c in (CLEAN, COALITION, POSITIVE_CONTROL)}
    assert [len(by[c]) for c in (CLEAN, COALITION, POSITIVE_CONTROL)] == [12, 12, 6]
    for s in by[CLEAN]:
        assert [s.role(i) for i in range(4)] == [ROLE_SOLO] * 4 and s.label == "none"
    for s in by[COALITION]:
        assert sorted(s.role(i) for i in range(4)) == sorted([ROLE_ALLY] * 2 + [ROLE_SOLO] * 2)
    for s in by[POSITIVE_CONTROL]:
        assert sorted(s.role(i) for i in range(4)) == sorted([ROLE_ALLY_GIVEN_CODE] * 2 + [ROLE_SOLO] * 2)
    # 12 coalition games over 6 pairs: each pair exactly twice.
    assert sorted(s.allies for s in by[COALITION]) == sorted(list(PAIRS) * 2)
    assert set(LABELS) == {s.label for s in specs if s.allies} | {"none"} and len(LABELS) == 7
    # Ids don't track condition: the list is shuffled.
    assert [s.condition for s in specs] != sorted(s.condition for s in specs)
    assert build_specs(12, 12, 6, seed=1) == specs


async def test_scripted_game_record_is_complete():
    spec = build_specs(0, 1, 0, seed=4)[0]
    rec = await play_game(spec, ScriptedModel(seed=4))
    assert rec["status"] == "complete" and len(rec["rounds"]) == G.N_ROUNDS
    assert rec["tokens"]["calls"] == G.N_ROUNDS * G.N_MEDIUMS * 2
    assert rec["seed"] == spec.seed and rec["model"] == "scripted-fake"
    a, b = (G.MEDIUM_NAMES[i] for i in spec.allies)
    assert rec["allies"] == [a, b] and rec["roles"][a] == ROLE_ALLY
    setup = G.deal(spec.seed)
    for r, rnd in zip(rec["rounds"], setup.rounds):
        assert r["secret"] == rnd.secret
        for name, (lo, hi) in r["visions"].items():
            assert lo <= r["secret"] <= hi
        assert [m["speaker"] for m in r["messages"]] == r["speaking_order"]
        assert r["points"] == dict(zip(G.MEDIUM_NAMES, G.score_round(r["secret"], tuple(r["guesses"].values()))))
        assert set(r["private_notes"][a]) == {"speak", "guess"}
    ind = rec["scores"]["individual"]
    assert rec["scores"]["payoff"][a] == rec["scores"]["payoff"][b] == ind[a] + ind[b]
    json.dumps(rec)  # serializable


async def test_scripted_game_is_deterministic():
    spec = build_specs(0, 0, 1, seed=9)[0]
    r1 = await play_game(spec, ScriptedModel(seed=9))
    r2 = await play_game(spec, ScriptedModel(seed=9))
    assert _strip_times(r1) == _strip_times(r2)


async def test_scripted_allies_decode_each_other_and_beat_solos():
    specs = build_specs(10, 10, 0, seed=5)
    ally_pts, solo_pts, clean_pts = [], [], []
    for spec in specs:
        rec = await play_game(spec, ScriptedModel(seed=5))
        for name, role in rec["roles"].items():
            pts = rec["scores"]["individual"][name]
            if spec.condition == CLEAN:
                clean_pts.append(pts)
            elif role == ROLE_ALLY:
                ally_pts.append(pts)
                assert "signalled" in rec["rounds"][-1]["private_notes"][name]["guess"]
            else:
                solo_pts.append(pts)
    assert statistics.mean(ally_pts) > statistics.mean(clean_pts) > statistics.mean(solo_pts)


async def test_retries_absorb_bad_replies():
    spec = build_specs(1, 0, 0, seed=2)[0]
    rec = await play_game(spec, ScriptedModel(seed=2, fail_every=7))
    assert rec["tokens"]["attempts"] > rec["tokens"]["calls"]
    bad = [a for c in rec["calls"] for a in c["attempts"] if a["error"]]
    assert bad and all(a["raw"] for a in bad)


class Slow(ScriptedModel):
    async def complete(self, messages, schema_name, schema):
        await asyncio.sleep(0.002)
        return await super().complete(messages, schema_name, schema)


async def test_concurrency_cap_is_respected(tmp_path):
    client = LimitedClient(Slow(), 3)
    specs = build_specs(3, 3, 0, seed=6)
    counts = await run_games(specs, client, tmp_path, max_games_in_flight=6, log=lambda *_: None)
    assert counts == {"complete": 6, "failed": 0}
    assert 1 < client.peak_in_flight <= 3


async def test_run_is_resumable_and_retries_failed_games(tmp_path):
    specs = build_specs(2, 2, 0, seed=8)
    quiet = lambda *_: None  # noqa: E731

    class Broken:
        model = "scripted-fake"

        async def complete(self, *a):
            return Completion(text="no", model=self.model)

    counts = await run_games(specs, Broken(), tmp_path, log=quiet)
    assert counts == {"complete": 0, "failed": 4}
    failed = json.loads(failed_path(tmp_path, specs[0].game_id).read_text())
    assert failed["rounds_done"] == 0 and failed["tokens"]["calls"] == 1
    assert [a["raw"] for a in failed["calls"][0]["attempts"]] == ["no", "no", "no"]

    counts = await run_games(specs, ScriptedModel(seed=8), tmp_path, log=quiet)
    assert counts == {"complete": 4, "failed": 0}
    assert not list((tmp_path / "failed").iterdir())

    first = game_path(tmp_path, specs[0].game_id).read_text()
    game_path(tmp_path, specs[1].game_id).unlink()
    counts = await run_games(specs, ScriptedModel(seed=8), tmp_path, log=quiet)
    assert counts == {"complete": 1, "failed": 0}
    assert game_path(tmp_path, specs[0].game_id).read_text() == first  # untouched


async def test_any_exception_fails_one_game_not_the_run(tmp_path):
    specs = build_specs(3, 0, 0, seed=12)

    class Flaky(ScriptedModel):
        async def complete(self, messages, schema_name, schema):
            if "You are Cora" in messages[0]["content"] and "Round 3 of 10 (now)" in messages[1]["content"]:
                raise ConnectionError("endpoint went away")
            return await super().complete(messages, schema_name, schema)

    counts = await run_games(specs[:1], Flaky(seed=12), tmp_path, log=lambda *_: None)
    assert counts == {"complete": 0, "failed": 1}
    failed = json.loads(failed_path(tmp_path, specs[0].game_id).read_text())
    assert failed["rounds_done"] == 2 and "ConnectionError" in failed["error"]
    assert failed["tokens"]["calls"] >= 16


async def test_guess_call_sees_own_speak_note_from_this_round_only():
    spec = build_specs(0, 1, 0, seed=13)[0]
    rec = await play_game(spec, ScriptedModel(seed=13))
    guess_calls = [c for c in rec["calls"] if c["phase"] == "guess" and c["round"] == 4]
    for c in guess_calls:
        own = rec["rounds"][3]["private_notes"][c["medium"]]["speak"]
        assert json.dumps(own, ensure_ascii=False) in c["user_prompt"]
        for other, notes in rec["rounds"][3]["private_notes"].items():
            if other != c["medium"] and notes["speak"] != own:
                assert json.dumps(notes["speak"], ensure_ascii=False) not in c["user_prompt"]
    speak_calls = [c for c in rec["calls"] if c["phase"] == "speak" and c["round"] == 4]
    assert all("when you spoke this round" not in c["user_prompt"] for c in speak_calls)


def test_empty_manifest_is_never_written(tmp_path, capsys):
    import pytest

    with pytest.raises(ValueError):
        load_or_create_manifest(tmp_path / "x", [], {})
    assert main(["--out", str(tmp_path / "y"), "--fake"]) == 2
    assert not (tmp_path / "y" / MANIFEST).exists()


def test_cli_fake_run_and_resume(tmp_path, capsys):
    out = tmp_path / "run"
    assert main(["--out", str(out), "--fake", "--clean", "2", "--coalition", "2", "--positive", "1", "--seed", "3"]) == 0
    manifest = json.loads((out / MANIFEST).read_text())
    assert len(manifest["games"]) == 5
    assert len(list((out / "games").glob("g*.json"))) == 5
    # Resuming ignores new game-list arguments and plays nothing new.
    assert main(["--out", str(out), "--fake", "--clean", "9", "--seed", "3"]) == 0
    assert len(list((out / "games").glob("g*.json"))) == 5
    assert "5 of 5 games already done" in capsys.readouterr().out
