"""The Anthropic backend: player client, prompt caching, and the batch monitor.
No network: the SDK client is replaced by fakes."""

from types import SimpleNamespace

import pytest

from seance import claude as C
from seance import claude_monitor as CM
from seance.agents import GUESS, GUESS_SCHEMA, SPEAK, SPEAK_SCHEMA, call_medium, system_prompt, user_prompt
from seance.conditions import LABELS, build_specs
from seance.detectors import llm_monitor as M
from seance.exploratory import random_names_view_fn
from seance.fake import ScriptedModel
from seance.inference import Completion, load_settings
from seance.run import play_game
from seance.skins import get_skin

from .helpers import game_mid_round, notes_for


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for k in list(os.environ):
        if k.startswith("SEANCE_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


# --- the Qwen path is untouched ------------------------------------------------------


def test_openai_settings_and_records_are_unchanged(tmp_path):
    env = tmp_path / ".env"
    env.write_text("SEANCE_BASE_URL=http://x/v1\nSEANCE_API_KEY=k\nSEANCE_MODEL=Qwen/Qwen3-8B\n")
    assert C.provider(env) == "openai"
    assert set(load_settings(env).describe()) == {
        "base_url", "model", "temperature", "max_tokens", "json_mode", "enable_thinking", "extra_body"
    }
    assert Completion(text="", model="m").usage is None


async def test_fake_game_record_has_no_usage_fields():
    spec = build_specs(1, 0, 0, seed=1)[0]
    rec = await play_game(spec, ScriptedModel(seed=1))
    assert "usage" not in rec["tokens"]
    assert all("usage" not in a for c in rec["calls"] for a in c["attempts"])


def test_provider_must_be_known(tmp_path, monkeypatch):
    monkeypatch.setenv("SEANCE_PROVIDER", "Anthropic")
    assert C.provider(tmp_path / "none") == "anthropic"
    monkeypatch.setenv("SEANCE_PROVIDER", "bedrock")
    with pytest.raises(RuntimeError):
        C.provider(tmp_path / "none")


# --- request parameters --------------------------------------------------------------


def test_thinking_per_model():
    assert C.thinking_params("claude-haiku-5-5") == {"thinking": {"type": "disabled"}}
    assert C.thinking_params("claude-sonnet-5-5") == {"thinking": {"type": "between_tools"}}
    assert C.thinking_params("claude-opus-5-5") == {"output_config": {"effort": "low"}}
    with pytest.raises(ValueError):
        C.thinking_params("claude-haiku-4-5")


def test_request_params_map_system_schema_and_send_no_temperature():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    p = C.request_params("claude-opus-5-5", 2048, msgs, GUESS_SCHEMA)
    assert p["system"] == "sys" and p["messages"] == [{"role": "user", "content": "hi"}]
    assert p["output_config"]["effort"] == "low"
    schema = p["output_config"]["format"]["schema"]
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert schema["properties"]["guess"] == {"type": "integer"}  # bounds stripped, checked by the parser
    assert GUESS_SCHEMA["properties"]["guess"]["minimum"] == 1  # original untouched
    assert "temperature" not in p and "top_p" not in p
    assert "output_config" not in C.request_params("claude-haiku-5-5", 10, msgs, None)


def test_settings_describe_names_the_differences(tmp_path, monkeypatch):
    monkeypatch.setenv("SEANCE_PROVIDER", "anthropic")
    monkeypatch.setenv("SEANCE_MODEL", "claude-haiku-5-5")
    s = C.load_anthropic_settings(tmp_path / "none")
    d = s.describe()
    assert d["provider"] == "anthropic" and d["model"] == "claude-haiku-5-5" and d["temperature"] is None
    assert d["thinking"] == {"type": "disabled"} and d["enable_thinking"] is False and d["max_tokens"] == 2048
    assert "test-key" not in str(d)
    monkeypatch.setenv("SEANCE_MODEL", "Qwen/Qwen3-8B")
    with pytest.raises(RuntimeError):
        C.load_anthropic_settings(tmp_path / "none")


# --- prompt caching ------------------------------------------------------------------


def _prompt(rounds_done, phase=SPEAK, posted=0):
    game = game_mid_round(rounds_done=rounds_done, posted=posted)
    return user_prompt(get_skin("seance"), game, 1, phase, notes_for(1, rounds_done + 1), None)


def test_first_round_prompt_is_sent_as_is():
    assert C.cached_user_content(_prompt(0)) == _prompt(0)


def test_cache_blocks_rebuild_the_text_and_mark_the_last_finished_round():
    for rounds_done in (1, 3, 9):
        text = _prompt(rounds_done)
        blocks = C.cached_user_content(text)
        assert "".join(b["text"] for b in blocks) == text
        marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
        assert len(marked) == 1
        assert blocks[marked[0]]["text"].startswith(f"Round {rounds_done} of 10:")
        assert blocks[marked[0] + 1]["text"].startswith("Points so far: ")
        assert len(blocks) == rounds_done + 1


def test_cached_prefix_is_stable_within_and_across_rounds():
    speak, guess, nxt = _prompt(3), _prompt(3, GUESS, posted=1), _prompt(4)
    bs, bg, bn = (C.cached_user_content(t) for t in (speak, guess, nxt))
    # the speak and guess calls of a round share every block up to the breakpoint
    assert bs[:3] == bg[:3]
    # the next round repeats those blocks unchanged, so the look-back finds them
    assert [b["text"] for b in bn[:3]] == [b["text"] for b in bs[:3]]


async def test_client_sends_cached_blocks_and_records_usage():
    client = C.AnthropicClient(C.AnthropicSettings(model="claude-haiku-5-5"))
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"private_note": "n", "public_message": "m"}')],
            model="claude-haiku-5-5",
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, cache_read_input_tokens=900, cache_creation_input_tokens=100, output_tokens=7),
        )

    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": _prompt(3)}]
    comp = await client.complete(msgs, "seance_speak", SPEAK_SCHEMA)
    assert sent["system"] == "sys" and sent["thinking"] == {"type": "disabled"} and "temperature" not in sent
    assert isinstance(sent["messages"][0]["content"], list)
    assert comp.prompt_tokens == 1010 and comp.completion_tokens == 7 and comp.finish_reason == "end_turn"
    assert comp.usage["cache_read_input_tokens"] == 900
    assert msgs[1]["content"] == _prompt(3)  # caller's messages not modified


async def test_game_record_sums_usage(monkeypatch):
    class WithUsage(ScriptedModel):
        async def complete(self, messages, schema_name, schema):
            c = await super().complete(messages, schema_name, schema)
            u = {"input_tokens": 1, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 3, "output_tokens": 4}
            return Completion(text=c.text, model=c.model, usage=u)

    rec = await play_game(build_specs(1, 0, 0, seed=1)[0], WithUsage(seed=1))
    n = rec["tokens"]["attempts"]
    assert rec["tokens"]["usage"] == {"input_tokens": n, "cache_read_input_tokens": 2 * n,
                                      "cache_creation_input_tokens": 3 * n, "output_tokens": 4 * n}


def test_cost():
    u = {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 0, "output_tokens": 1_000_000}
    assert C.cost("claude-haiku-5-5", u) == pytest.approx(0.10 + 0.01 + 0.50)
    assert C.cost("claude-opus-5-5", u, batch=True) == pytest.approx((4 + 0.4 + 20) / 2)


# --- the batch monitor ---------------------------------------------------------------


async def _records(n=2):
    specs = build_specs(1, 1, 0, seed=3)[:n]
    return [await play_game(s, ScriptedModel(seed=3)) for s in specs]


def test_frozen_prompt_hashes():
    assert CM.Config("claude-haiku-5-5", "plain").describe(3)["prompt_sha256"] == "212c62e53f7d4d02"
    d = CM.Config("claude-opus-5-5", "plus_factors").describe(3)
    assert d["prompt_sha256"] == "bd978314f3e56eb4" and d["prompt_variant"] == "plus_factors"
    assert d["effort"] == "low" and d["temperature"] is None and d["name_seed"] == 20261027


async def test_prompt_is_the_qwen_monitors_prompt():
    recs = await _records()
    view_fn = random_names_view_fn()
    for variant, tag in ((None, "plain"), ("plus_factors", "plus_factors")):
        seen = []

        class Capture(ScriptedModel):
            async def complete(self, messages, schema_name, schema):
                seen.append((messages, schema))
                keys = list(schema["properties"]["probabilities"]["properties"])
                return Completion(text=_answer(keys, "none"), model="m")

        mon = M.LLMMonitor(Capture(), settings={"temperature": 0.7}, n_readings=1, prompt_variant=variant)
        await mon.read_once(view_fn(recs[0]))
        readings = CM.make_readings("fresh", recs[:1], [CM.Config("claude-sonnet-5-5", tag)], 3, view_fn)
        assert len(readings) == 3
        for r in readings:
            assert r.messages == [seen[0][0][0]]
            assert r.params()["messages"] == [seen[0][0][0]]
            assert r.params()["output_config"]["format"]["schema"] == C.output_schema(seen[0][1])
            assert r.keys == M.label_order(view_fn(recs[0])["names"])
            assert r.aliases == view_fn(recs[0]).get("aliases")


def test_custom_ids_are_valid_and_unique():
    ids = {CM.custom_id("fresh", f"p{g:03d}", CM.Config(m, p, s), i)
           for g in range(3) for m in C.THINKING for p in CM.PROMPTS for s in (True, False) for i in range(3)}
    assert len(ids) == 3 * 3 * 2 * 2 * 3
    with pytest.raises(ValueError):
        CM.custom_id("runs/final", "g000", CM.Config("claude-opus-5-5", "plain"), 0)


def _answer(keys, top):
    probs = {k: (0.7 if k == top else 0.05) for k in keys}
    import json

    return json.dumps({"reasoning": "r", "probabilities": probs})


def _ok(text, stop="end_turn"):
    usage = SimpleNamespace(input_tokens=100, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=20)
    content = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(type="succeeded", message=SimpleNamespace(content=content, usage=usage, stop_reason=stop, stop_details=SimpleNamespace(category="reasoning_extraction")))


async def test_parse_error_retry_and_refusal():
    recs = await _records(1)
    r1, r2, r3 = CM.make_readings("fresh", recs, [CM.Config("claude-haiku-5-5", "plain")], 3, random_names_view_fn())
    CM.apply_result(r1, _ok("not json"), "b1")
    assert r1.result is None and r1.attempts[0]["error"]
    assert r1.messages[1] == {"role": "assistant", "content": "not json"}
    assert r1.messages[2]["content"] == M.retry_message(r1.attempts[0]["error"])
    CM.apply_result(r2, _ok("", stop="refusal"), "b1")
    assert r2.result is None and r2.attempts[0]["error"].startswith("refusal") and len(r2.messages) == 1
    CM.apply_result(r3, SimpleNamespace(type="errored", error=SimpleNamespace(error=SimpleNamespace(message="overloaded"))), "b1")
    assert r3.result is None and "overloaded" in r3.attempts[0]["error"] and len(r3.messages) == 1


class FakeBatches:
    """Answers each batch out of order; the first answer for `bad` is unparsable."""

    def __init__(self, bad: str):
        self.bad, self.n, self.store = bad, 0, {}

    def create(self, requests):
        self.n += 1
        bid = f"b{self.n}"
        self.store[bid] = list(reversed(requests))
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended", request_counts=SimpleNamespace(processing=0, succeeded=1, errored=0))

    def results(self, bid):
        for req in self.store[bid]:
            cid, params = req["custom_id"], req["params"]
            prompt = params["messages"][0]["content"]
            keys = list(params["output_config"]["format"]["schema"]["properties"]["probabilities"]["properties"])
            text = "oops" if (cid == self.bad and len(params["messages"]) == 1) else _answer(keys, keys[0] if "-r0" in cid else "none")
            assert prompt
            yield SimpleNamespace(custom_id=cid, result=_ok(text))


async def test_batches_map_results_by_custom_id_and_retry():
    recs = await _records(2)
    view_fn = random_names_view_fn()
    configs = [CM.Config("claude-haiku-5-5", "plain"), CM.Config("claude-opus-5-5", "plus_factors")]
    readings = CM.make_readings("fresh", recs, configs, 3, view_fn)
    bad = readings[4].custom_id
    fake = FakeBatches(bad)
    ids = CM.run_batches(SimpleNamespace(messages=SimpleNamespace(batches=fake)), readings, 3, poll_s=0, log=lambda *_: None)
    assert ids == ["b1", "b2"]
    assert all(r.result is not None for r in readings)
    retried = next(r for r in readings if r.custom_id == bad)
    assert len(retried.attempts) == 2 and retried.attempts[0]["error"]
    rows = CM.game_rows(readings, 3)
    assert len(rows) == 4
    for (config, gid), row in rows.items():
        assert set(row["probs"]) == set(LABELS) and abs(sum(row["probs"].values()) - 1) < 1e-9
        assert row["detector"]["model"] == config.model and row["detector"]["usage"]["output_tokens"] > 0
        first = view_fn(next(r for r in recs if r["game_id"] == gid))
        top = M._canonical(M.label_order(first["names"])[0], first.get("aliases"))
        # reading 0 named the first pair, readings 1-2 "none": the mean keeps both
        assert row["probs"][top] == pytest.approx((0.7 / 1.0 + 0.05 * 2) / 3)
        assert len(row["detail"]["readings"]) == 3


async def test_game_with_a_failed_reading_is_left_out():
    recs = await _records(1)
    readings = CM.make_readings("fresh", recs, [CM.Config("claude-haiku-5-5", "plain")], 3, random_names_view_fn())
    for r in readings[:2]:
        CM.apply_result(r, _ok(_answer(r.keys, "none")), "b1")
    CM.apply_result(readings[2], _ok("bad"), "b1")
    assert CM.game_rows(readings, 3) == {}
