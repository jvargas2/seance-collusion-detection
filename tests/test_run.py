"""Whole games end to end with the scripted fake model, offline."""

import asyncio
import json
import statistics

from seance import game as G
import pytest

from seance.conditions import (
    MAIN_SEED_LIMIT,
    PILOT_SEED_MIN,
    pair_from_seed,
    build_main_study,
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
        assert [(m["phase"], m["speaker"]) for m in r["messages"]] == [(1, n) for n in G.MEDIUM_NAMES]
        assert "speaking_order" not in r
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
    assert failed["rounds_done"] == 0 and failed["tokens"]["calls"] == 4  # four simultaneous posts
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
        own = rec["rounds"][3]["private_notes"][c["medium"]]["speak"][0]
        assert json.dumps(own, ensure_ascii=False) in c["user_prompt"]
        for other, notes in rec["rounds"][3]["private_notes"].items():
            if other != c["medium"] and notes["speak"][0] != own:
                assert json.dumps(notes["speak"][0], ensure_ascii=False) not in c["user_prompt"]
    speak_calls = [c for c in rec["calls"] if c["phase"] == "speak" and c["round"] == 4]
    assert all("from posting this round" not in c["user_prompt"] for c in speak_calls)


@pytest.mark.parametrize("phases", [1, 2])
async def test_posting_is_simultaneous(phases):
    spec = build_specs(0, 1, 0, seed=14, posting_phases=phases)[0]
    rec = await play_game(spec, ScriptedModel(seed=14))
    assert rec["settings"]["posting_mode"] == "simultaneous"
    assert rec["settings"]["posting_phases"] == phases
    assert rec["tokens"]["calls"] == G.N_ROUNDS * G.N_MEDIUMS * (phases + 1)
    for r in rec["rounds"]:
        assert [m["phase"] for m in r["messages"]] == [k for k in range(1, phases + 1) for _ in range(4)]
        for k in range(1, phases + 1):
            speak = [c for c in rec["calls"] if c["round"] == r["round"] and c["posting_phase"] == k]
            assert len(speak) == 4
            for c in speak:
                now = c["user_prompt"].split("(now):", 1)[1]
                # Nobody sees a message from the phase they are writing in.
                assert f"Messages, posting {k} of" not in now
                if k == 1:
                    assert "Messages" not in now and "everyone writes at the same time" in now
                else:
                    assert f"Messages, posting {k - 1} of {phases}:" in now
                    for m in r["messages"]:
                        if m["phase"] == k - 1:
                            assert f'{m["speaker"]}: {json.dumps(m["text"], ensure_ascii=False)}' in now
        assert len(r["private_notes"]["Ada"]["speak"]) == phases


def test_main_study_specs_are_fixed():
    specs = build_main_study()
    assert len(specs) == 106 and len({s.game_id for s in specs}) == 106
    by = {c: sorted(s.seed for s in specs if s.condition == c) for c in (CLEAN, COALITION, POSITIVE_CONTROL)}
    assert by[CLEAN] == list(range(0, 50))
    assert by[COALITION] == list(range(50, 100))
    assert by[POSITIVE_CONTROL] == list(range(100, 106))
    for s in specs:
        assert s.posting_phases == 1
        if s.allies is not None:
            assert s.allies == pair_from_seed(s.seed)
    assert build_main_study() == specs  # same ids, pairs and seeds every time
    assert [s.condition for s in specs] != sorted(s.condition for s in specs)  # ids shuffled
    assert len({s.allies for s in specs if s.allies}) == 6  # every pair appears


def test_pilot_seeds_avoid_the_main_study():
    for seed in range(20):
        for s in build_specs(5, 5, 2, seed=seed):
            assert s.seed >= PILOT_SEED_MIN > MAIN_SEED_LIMIT
            assert s.game_id.startswith("p")


def test_cli_main_study_refuses_extra_counts(tmp_path):
    assert main(["--out", str(tmp_path / "m"), "--fake", "--main-study", "--clean", "3"]) == 2


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
    assert len(list((out / "games").glob("p*.json"))) == 5
    # Resuming ignores new game-list arguments and plays nothing new.
    assert main(["--out", str(out), "--fake", "--clean", "9", "--seed", "3"]) == 0
    assert len(list((out / "games").glob("p*.json"))) == 5
    assert "5 of 5 games already done" in capsys.readouterr().out


class OutOfOrder(ScriptedModel):
    """Replies arrive in reverse seat order (Dov first); logs starts and ends."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.events = []

    async def complete(self, messages, schema_name, schema):
        me = messages[0]["content"].split("You are ", 1)[1].split(",", 1)[0]
        seat = G.MEDIUM_NAMES.index(me)
        self.events.append(("start", seat))
        batch = (len(self.events) - 1) // 8 * 8  # this call's batch of 4 starts + 4 ends
        # Wait for all four to start, then for every higher seat to finish
        # (explicit ordering, not sleeps, so timer jitter can't reorder them).
        while sum(e == "start" for e, _ in self.events[batch:]) < 4 or not {
            ("end", s) for s in range(seat + 1, 4)
        } <= set(self.events[batch:]):
            await asyncio.sleep(0)
        out = await super().complete(messages, schema_name, schema)
        self.events.append(("end", seat))
        return out


async def test_phase_calls_overlap_and_reply_order_changes_nothing():
    spec = build_specs(0, 1, 0, seed=15)[0]
    model = OutOfOrder(seed=15)
    rec = await play_game(spec, model)
    # Calls come in batches of four: all four start before any of them ends.
    for i in range(0, len(model.events), 8):
        batch = model.events[i : i + 8]
        assert [e for e, _ in batch[:4]] == ["start"] * 4
        assert [e for e, _ in batch[4:]] == ["end"] * 4
        assert [s for _, s in batch[4:]] == [3, 2, 1, 0]  # replies out of seat order
    plain = await play_game(spec, ScriptedModel(seed=15))
    assert _strip_times(rec)["rounds"] == _strip_times(plain)["rounds"]
    assert [c["user_prompt"] for c in rec["calls"]] != [] and sorted(
        c["user_prompt"] for c in rec["calls"]
    ) == sorted(c["user_prompt"] for c in plain["calls"])


class FirstTryJunkFor(ScriptedModel):
    def __init__(self, name, *a, **kw):
        super().__init__(*a, **kw)
        self.name = name

    async def complete(self, messages, schema_name, schema):
        if f"You are {self.name}," in messages[0]["content"] and len(messages) == 2:
            return Completion(text="junk", model=self.model)
        return await super().complete(messages, schema_name, schema)


async def test_one_mediums_retries_leave_other_prompts_unchanged():
    spec = build_specs(1, 0, 0, seed=16)[0]
    retried = await play_game(spec, FirstTryJunkFor("Bram", seed=16))
    plain = await play_game(spec, ScriptedModel(seed=16))
    assert _strip_times(retried)["rounds"] == _strip_times(plain)["rounds"]
    for a, b in zip(retried["calls"], plain["calls"]):
        assert a["user_prompt"] == b["user_prompt"]
        assert (len(a["attempts"]) == 2) == (a["medium"] == "Bram")


async def test_failure_inside_a_posting_phase(tmp_path):
    spec = build_specs(1, 0, 0, seed=17)[0]

    class CoraBreaksRound3(ScriptedModel):
        async def complete(self, messages, schema_name, schema):
            if "You are Cora," in messages[0]["content"] and "Round 3 of 10 (now)" in messages[1]["content"]:
                return Completion(text="junk", model=self.model)
            return await super().complete(messages, schema_name, schema)

    counts = await run_games([spec], CoraBreaksRound3(seed=17), tmp_path, log=lambda *_: None)
    assert counts == {"complete": 0, "failed": 1}
    failed = json.loads(failed_path(tmp_path, spec.game_id).read_text())
    assert failed["rounds_done"] == 2
    round3 = [c for c in failed["calls"] if c["round"] == 3]
    assert sorted(c["medium"] for c in round3) == list(G.MEDIUM_NAMES)  # the other three are logged too
    cora = next(c for c in round3 if c["medium"] == "Cora")
    assert len(cora["attempts"]) == 3 and cora["result"] is None
    assert failed["tokens"]["calls"] == 2 * 8 + 4
