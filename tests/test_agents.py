import json
import re

import pytest

from seance import game as G
from seance.agents import (
    GUESS,
    SPEAK,
    AgentCallFailed,
    ReplyError,
    call_medium,
    parse_guess,
    parse_speak,
    system_prompt,
    user_prompt,
)
from seance.conditions import ROLE_ALLY, ROLE_ALLY_GIVEN_CODE, ROLE_SOLO
from seance.inference import Completion
from seance.skins import SKINS
from tests.helpers import game_mid_round, notes_for

ALLIANCE_WORDS = ("ally", "allies", "alliance", "partner", "pooled", "code", "together", "team", "coordinat", "collu")


class Canned:
    """A client that returns queued replies and records what it was sent."""

    model = "canned"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.sent = []

    async def complete(self, messages, schema_name, schema):
        self.sent.append(messages)
        return Completion(text=self.texts.pop(0), model=self.model, prompt_tokens=10, completion_tokens=5)


def test_parse_speak_variants():
    good = {"private_note": "n", "public_message": "  hello \n there "}
    assert parse_speak(json.dumps(good)) == {"private_note": "n", "public_message": "hello there"}
    assert parse_speak("```json\n" + json.dumps(good) + "\n```")["public_message"] == "hello there"
    assert parse_speak("<think>hmm {not json}</think>" + json.dumps(good))["private_note"] == "n"
    assert parse_speak("reasoning...</think>\n" + json.dumps(good))["private_note"] == "n"
    assert parse_speak("Sure! " + json.dumps(good) + " Hope that helps")["private_note"] == "n"
    for bad in ("nope", "[1,2]", json.dumps({"private_note": "n"}), json.dumps({"private_note": 3, "public_message": "x"}),
                json.dumps({"private_note": "n", "public_message": "   "})):
        with pytest.raises(ReplyError):
            parse_speak(bad)
    with pytest.raises(ReplyError) as e:
        parse_speak(json.dumps({"private_note": "n", "public_message": "w " * 61}))
    assert e.value.too_long_only


def test_parse_guess_variants():
    assert parse_guess('{"private_note": "", "guess": 42}')["guess"] == 42
    assert parse_guess('{"private_note": "", "guess": 42.0}')["guess"] == 42
    assert parse_guess('{"private_note": "", "guess": " 7 "}')["guess"] == 7
    for bad in ('{"private_note": "", "guess": 0}', '{"private_note": "", "guess": 101}',
                '{"private_note": "", "guess": 4.5}', '{"private_note": "", "guess": true}',
                '{"private_note": "", "guess": "forty"}', '{"guess": 5}'):
        with pytest.raises(ReplyError):
            parse_guess(bad)


async def test_call_retries_then_succeeds_and_logs_every_raw_reply():
    game = game_mid_round(posted=1)
    skin = SKINS["seance"]
    client = Canned("garbage", '{"private_note": "x", "guess": 500}', '{"private_note": "ok", "guess": 55}')
    out, log = await call_medium(client, skin, "SYS", game, 1, GUESS, [], None, max_attempts=3)
    assert out == {"private_note": "ok", "guess": 55}
    assert [a["raw"] for a in log.attempts] == ["garbage", '{"private_note": "x", "guess": 500}', '{"private_note": "ok", "guess": 55}']
    assert log.attempts[0]["error"] and log.attempts[1]["error"] and log.attempts[2]["error"] is None
    assert log.prompt_tokens == 30 and log.completion_tokens == 15
    # The retry shows the model its bad reply and the error.
    last = client.sent[-1]
    assert last[-2] == {"role": "assistant", "content": '{"private_note": "x", "guess": 500}'}
    assert "500" in last[-1]["content"]


async def test_call_fails_after_max_attempts():
    game = game_mid_round(posted=1)
    client = Canned("a", "b")
    with pytest.raises(AgentCallFailed) as e:
        await call_medium(client, SKINS["seance"], "SYS", game, 1, GUESS, [], None, max_attempts=2)
    assert [a["raw"] for a in e.value.log.attempts] == ["a", "b"]


async def test_overlong_message_truncated_only_on_last_attempt():
    game = game_mid_round(posted=0)
    long = json.dumps({"private_note": "n", "public_message": "word " * 80})
    client = Canned(long, long)
    out, log = await call_medium(client, SKINS["seance"], "SYS", game, 0, SPEAK, [], None, max_attempts=2)
    assert log.truncated and len(log.attempts) == 2
    assert G.word_count(out["public_message"]) == G.MAX_MESSAGE_WORDS


@pytest.mark.parametrize("skin", list(SKINS.values()), ids=lambda s: s.name)
def test_solo_prompts_never_hint_at_an_alliance(skin):
    sys_text = system_prompt(skin, 1, ROLE_SOLO, None).lower()
    texts = [sys_text]
    for phase, posted in ((SPEAK, 0), (GUESS, 1)):
        texts.append(user_prompt(skin, game_mid_round(posted=posted), 1, phase, notes_for(1, 2), None).lower())
    for t in texts:
        for word in ALLIANCE_WORDS:
            assert not re.search(rf"\b{word}", t), word


def test_ally_prompts():
    skin = SKINS["seance"]
    improvise = system_prompt(skin, 0, ROLE_ALLY, 2)
    given = system_prompt(skin, 0, ROLE_ALLY_GIVEN_CODE, 2)
    assert "you and Cora are secret allies" in improvise and "you and Cora are secret allies" in given
    assert "Improvise one" in improvise and "agreed on this code" not in improvise
    assert "agreed on this code" in given and "candles" in given and "Improvise" not in given
    up = user_prompt(skin, game_mid_round(posted=1), 0, GUESS, notes_for(0, 2), 2)
    assert "Pooled points so far for you and Cora" in up
    with pytest.raises(ValueError):
        system_prompt(skin, 0, ROLE_ALLY, None)
    with pytest.raises(ValueError):
        system_prompt(skin, 0, ROLE_SOLO, 2)


@pytest.mark.parametrize("phase,posted", [(SPEAK, 0), (GUESS, 1)])
@pytest.mark.parametrize("seat", range(4))
def test_user_prompt_shows_only_own_visions_and_notes(seat, phase, posted):
    skin = SKINS["seance"]
    game = game_mid_round(seed=21, rounds_done=3, posted=posted)
    notes = notes_for(seat, 3) + [{"speak": [f"seat {seat} speak note r4 p1"]} if posted else {"speak": []}]
    text = user_prompt(skin, game, seat, phase, notes, None)
    rounds = game.setup.rounds[:4]
    own = {(r.visions[seat].lo, r.visions[seat].hi) for r in rounds}

    def shows(lo, hi):
        return re.search(rf"(?<![\d-]){lo} to {hi}(?!\d)", text) is not None

    for lo, hi in own:
        assert shows(lo, hi)
    for rnd in game.setup.rounds:
        for other, v in enumerate(rnd.visions):
            if other != seat and (v.lo, v.hi) not in own:
                assert not shows(v.lo, v.hi)
    for other in range(4):
        if other != seat:
            assert f"seat {other} speak note" not in text and f"seat {other} guess note" not in text
    assert f"seat {seat} speak note r3 p1" in text
    for future in game.setup.rounds[4:]:
        assert f"Round {future.index + 1} of" not in text
        v = future.visions[seat]
        if (v.lo, v.hi) not in own:
            assert not shows(v.lo, v.hi)
    if phase == SPEAK:
        # Simultaneous posting: no current-round message is visible while writing.
        assert "current post" not in text and "everyone writes at the same time" in text
        assert f"seat {seat} speak note r4" not in text
    else:
        assert all(f"current post 1 words from seat {s}" in text for s in range(4))
        assert f"seat {seat} speak note r4 p1" in text


def test_two_phase_prompts():
    skin = SKINS["seance"]
    sys_text = system_prompt(skin, 1, ROLE_SOLO, None, posting_phases=2)
    assert "This happens twice each round" in sys_text
    assert "twice" not in system_prompt(skin, 1, ROLE_SOLO, None)
    game = game_mid_round(posted=1, posting_phases=2)
    notes = notes_for(1, 2, 2) + [{"speak": ["seat 1 speak note r3 p1"]}]
    second = user_prompt(skin, game, 1, SPEAK, notes, None)
    assert "This is posting 2 of 2 this round." in second
    assert all(f"current post 1 words from seat {s}" in second for s in range(4))
    assert "seat 1 speak note r3 p1" in second
    assert '"seat 1 speak note r1 p1" then "seat 1 speak note r1 p2"' in second
    with pytest.raises(RuntimeError):
        user_prompt(skin, game, 1, GUESS, notes, None)  # second posting still open
