"""Claude models on the Anthropic API, as players (`AnthropicClient`, a
`ChatClient`) and as monitors (`claude_monitor.py`, Message Batches).

Selected for the players with SEANCE_PROVIDER=anthropic and
SEANCE_MODEL=claude-haiku-5-5 (or sonnet/opus). The key comes from
ANTHROPIC_API_KEY, else ~/.config/anthropic/api_key; it is never logged.

What differs from the Qwen path, and is written into every record:
none of the current Claude models accepts a non-default temperature, so none
is sent; thinking is switched off where the model allows it (Haiku 5.5:
`disabled`; Sonnet 5.5: `between_tools`, its thinking-off mode) and kept at
effort `low` where it can't be (Opus 5.5); the JSON schema goes in
`output_config.format` with the keywords structured outputs rejects
(numeric and length bounds) removed, so the reply parsers check those.
"""

from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from .inference import Completion

PROVIDER = "anthropic"
KEY_FILE = Path("~/.config/anthropic/api_key").expanduser()

# Per model: the thinking field and output_config.effort to send (None = send nothing).
THINKING: dict[str, dict[str, Any]] = {
    "claude-haiku-5-5": {"thinking": {"type": "disabled"}, "effort": None},
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}, "effort": None},
    "claude-opus-5-5": {"thinking": None, "effort": "low"},
}
# $ per million tokens: input, output (standard API; batch is half).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
SHORT = {"claude-haiku-5-5": "haiku", "claude-sonnet-5-5": "sonnet", "claude-opus-5-5": "opus"}

# Keywords structured outputs doesn't take; the reply parsers enforce them.
_UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength")


def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError(f"no Anthropic key: set ANTHROPIC_API_KEY or write {KEY_FILE}")
    return key


def thinking_params(model: str) -> dict[str, Any]:
    """The request fields that set thinking and effort for `model`."""
    if model not in THINKING:
        raise ValueError(f"no thinking setting for {model!r}; add it to THINKING")
    t = THINKING[model]
    out: dict[str, Any] = {}
    if t["thinking"] is not None:
        out["thinking"] = dict(t["thinking"])
    if t["effort"] is not None:
        out["output_config"] = {"effort": t["effort"]}
    return out


def output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` without the keywords structured outputs rejects."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in _UNSUPPORTED}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(copy.deepcopy(schema))


def request_params(
    model: str, max_tokens: int, messages: list[dict[str, Any]], schema: dict[str, Any] | None
) -> dict[str, Any]:
    """Messages API parameters: system messages go to `system`, the schema to
    `output_config.format`, thinking/effort as `thinking_params`."""
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    params: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": rest, **thinking_params(model)}
    if system:
        params["system"] = "\n\n".join(system) if all(isinstance(s, str) for s in system) else system[0]
    if schema is not None:
        params.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": output_schema(schema)}
    return params


# --- prompt caching for the players -----------------------------------------------

# A player's user prompt replays every finished round, then the running totals
# and the current round. Each finished round is its own text block, and the
# last one carries the cache breakpoint: the round's other calls read it, and
# the next round's calls find it again by the API's look-back over earlier
# block boundaries. The model sees the same text either way.
_BLOCK_START = re.compile(r"^(?=Round \d+ of \d+:$|Points so far: )", re.MULTILINE)


def cached_user_content(text: str) -> str | list[dict[str, Any]]:
    starts = [m.start() for m in _BLOCK_START.finditer(text)]
    if not starts:
        return text
    bounds = [0, *[s for s in starts if s > 0], len(text)]
    blocks = [{"type": "text", "text": text[a:b]} for a, b in zip(bounds, bounds[1:]) if b > a]
    tail = next((i for i, b in enumerate(blocks) if b["text"].startswith("Points so far: ")), None)
    if tail is None or tail == 0:
        return text
    blocks[tail - 1]["cache_control"] = {"type": "ephemeral"}
    return blocks


@dataclass
class AnthropicSettings:
    model: str
    max_tokens: int = 2048
    timeout_s: float = 120.0
    http_retries: int = 6
    cache: bool = True

    def describe(self) -> dict[str, Any]:
        """Settings safe to write into a game record (no key), naming what
        differs from the Qwen players."""
        t = THINKING[self.model]
        return {
            "provider": PROVIDER,
            "model": self.model,
            "temperature": None,  # not sent: the model's default sampling
            "max_tokens": self.max_tokens,
            "json_mode": "output_config.format",
            "enable_thinking": False if t["thinking"] is not None else True,
            "thinking": t["thinking"],
            "effort": t["effort"],
            "prompt_caching": self.cache,
        }


def load_anthropic_settings(env_file: str | Path = ".env", prefix: str = "SEANCE_") -> AnthropicSettings:
    file_values = dotenv_values(env_file) if Path(env_file).exists() else {}

    def get(name: str, default: str) -> str:
        return os.environ.get(prefix + name) or file_values.get(prefix + name) or default

    model = get("MODEL", "")
    if model not in THINKING:
        raise RuntimeError(f"{prefix}MODEL must be one of {sorted(THINKING)} with {prefix}PROVIDER=anthropic")
    return AnthropicSettings(
        model=model,
        max_tokens=int(get("MAX_TOKENS", "2048")),
        timeout_s=float(get("TIMEOUT_S", "120")),
        http_retries=int(get("HTTP_RETRIES", "6")),
    )


def provider(env_file: str | Path = ".env", prefix: str = "SEANCE_") -> str:
    file_values = dotenv_values(env_file) if Path(env_file).exists() else {}
    name = (os.environ.get(prefix + "PROVIDER") or file_values.get(prefix + "PROVIDER") or "openai").strip().lower()
    if name not in ("openai", PROVIDER):
        raise RuntimeError(f"{prefix}PROVIDER must be openai (any OpenAI-compatible endpoint) or {PROVIDER}")
    return name


def text_of(content: list[Any]) -> str:
    return "".join(b.text for b in content if getattr(b, "type", None) == "text")


def usage_dict(usage: Any) -> dict[str, int]:
    """The token counts of one response (input_tokens excludes cached tokens)."""
    get = lambda k: getattr(usage, k, None) or 0  # noqa: E731
    return {
        "input_tokens": get("input_tokens"),
        "cache_read_input_tokens": get("cache_read_input_tokens"),
        "cache_creation_input_tokens": get("cache_creation_input_tokens"),
        "output_tokens": get("output_tokens"),
    }


def cost(model: str, usage: dict[str, int], batch: bool = False) -> float:
    """Dollars for one response: cache reads at 0.1x and 5-minute writes at 1.25x the input price."""
    pin, pout = PRICES[model]
    dollars = (
        usage["input_tokens"] * pin
        + usage["cache_read_input_tokens"] * pin * 0.1
        + usage["cache_creation_input_tokens"] * pin * 1.25
        + usage["output_tokens"] * pout
    ) / 1e6
    return dollars / 2 if batch else dollars


class AnthropicClient:
    """A `ChatClient` for the players on the Anthropic Messages API."""

    def __init__(self, settings: AnthropicSettings):
        from anthropic import AsyncAnthropic

        self.settings = settings
        self.model = settings.model
        # The SDK retries connection errors, 408/409/429 and 5xx with backoff.
        self._client = AsyncAnthropic(api_key=api_key(), timeout=settings.timeout_s, max_retries=settings.http_retries)

    def params(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        params = request_params(self.model, self.settings.max_tokens, messages, schema)
        if self.settings.cache:
            first_user = next(i for i, m in enumerate(params["messages"]) if m["role"] == "user")
            msgs = list(params["messages"])
            msgs[first_user] = {**msgs[first_user], "content": cached_user_content(msgs[first_user]["content"])}
            params["messages"] = msgs
        return params

    async def complete(
        self, messages: list[dict[str, str]], schema_name: str, schema: dict[str, Any]
    ) -> Completion:
        start = time.monotonic()
        resp = await self._client.messages.create(**self.params(messages, schema))
        usage = usage_dict(resp.usage)
        return Completion(
            text=text_of(resp.content),
            model=resp.model or self.model,
            prompt_tokens=usage["input_tokens"] + usage["cache_read_input_tokens"] + usage["cache_creation_input_tokens"],
            completion_tokens=usage["output_tokens"],
            latency_s=time.monotonic() - start,
            finish_reason=resp.stop_reason,
            usage=usage,
        )
