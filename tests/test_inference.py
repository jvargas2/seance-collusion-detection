"""Settings from .env, the thinking-mode switch and what reaches the endpoint."""

from types import SimpleNamespace

import pytest

from seance.conditions import build_specs
from seance.fake import ScriptedModel
from seance.inference import Completion, OpenAICompatClient, load_settings
from seance.run import play_game

BASE = "SEANCE_BASE_URL=http://x/v1\nSEANCE_API_KEY=k\nSEANCE_MODEL=Qwen/Qwen3-8B\n"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for k in list(os.environ):
        if k.startswith("SEANCE_"):
            monkeypatch.delenv(k)


def _settings(tmp_path, extra="", **kw):
    env = tmp_path / ".env"
    env.write_text(BASE + extra)
    return load_settings(env, **kw)


def test_thinking_is_off_by_default_and_logged(tmp_path):
    s = _settings(tmp_path)
    assert s.enable_thinking is False
    assert s.request_extra_body() == {"chat_template_kwargs": {"enable_thinking": False}}
    assert s.describe()["enable_thinking"] is False
    assert s.describe()["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "api_key" not in s.describe() and "k" not in s.describe().values()


def test_thinking_setting_merges_with_extra_body(tmp_path):
    s = _settings(tmp_path, 'SEANCE_ENABLE_THINKING=true\nSEANCE_EXTRA_BODY={"chat_template_kwargs": {"x": 1}, "top_k": 20}\n')
    assert s.request_extra_body() == {"chat_template_kwargs": {"x": 1, "enable_thinking": True}, "top_k": 20}
    assert s.extra_body == {"chat_template_kwargs": {"x": 1}, "top_k": 20}  # not mutated
    u = _settings(tmp_path, 'SEANCE_ENABLE_THINKING=unset\nSEANCE_EXTRA_BODY={"reasoning": {"enabled": false}}\n')
    assert u.enable_thinking is None and u.request_extra_body() == {"reasoning": {"enabled": False}}
    with pytest.raises(RuntimeError):
        _settings(tmp_path, "SEANCE_ENABLE_THINKING=maybe\n")


def test_monitor_settings_fall_back(tmp_path):
    s = _settings(tmp_path, "SEANCE_MONITOR_MODEL=other\n", prefix="SEANCE_MONITOR_", fallback_prefix="SEANCE_")
    assert s.model == "other" and s.base_url == "http://x/v1" and s.enable_thinking is False


async def test_client_sends_thinking_off_and_counts_reasoning(tmp_path):
    client = OpenAICompatClient(_settings(tmp_path))
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        msg = SimpleNamespace(content="<think>hmm</think>{}", reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
            model="Qwen/Qwen3-8B",
        )

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    comp = await client.complete([{"role": "user", "content": "hi"}], "s", {"type": "object"})
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert sent["response_format"]["type"] == "json_schema"
    assert comp.reasoning_chars > 0 and comp.prompt_tokens == 3


class Thinking(ScriptedModel):
    async def complete(self, messages, schema_name, schema):
        c = await super().complete(messages, schema_name, schema)
        return Completion(text=c.text, model=c.model, reasoning_chars=50)


async def test_game_record_logs_thinking_setting_and_reasoning_seen():
    spec = build_specs(1, 0, 0, seed=3)[0]
    rec = await play_game(spec, Thinking(seed=3), inference_settings={"model": "m", "enable_thinking": False})
    assert rec["enable_thinking"] is False and rec["inference"]["enable_thinking"] is False
    assert rec["tokens"]["attempts_with_reasoning"] == rec["tokens"]["attempts"]
    quiet = await play_game(spec, ScriptedModel(seed=3))
    assert quiet["tokens"]["attempts_with_reasoning"] == 0
