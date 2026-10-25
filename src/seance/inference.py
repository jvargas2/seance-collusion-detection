"""One OpenAI-compatible chat client, configured from `.env`.

Modal (our vLLM app), OpenRouter and Nebius all speak the OpenAI chat API, so
switching provider means changing SEANCE_BASE_URL / SEANCE_API_KEY /
SEANCE_MODEL in `.env` and nothing else. See `.env.example`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values

JSON_MODES = ("json_schema", "json_object", "none")


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    reasoning_chars: int = 0  # thinking text returned (separately or as <think> in the content)
    reasoning: str = ""  # that thinking text itself
    usage: dict[str, int] | None = None  # provider token counts (Anthropic: with cache reads/writes)


class ChatClient(Protocol):
    model: str

    async def complete(
        self, messages: list[dict[str, str]], schema_name: str, schema: dict[str, Any]
    ) -> Completion: ...


@dataclass
class InferenceSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    json_mode: str = "json_schema"
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 120.0
    http_retries: int = 4
    # Qwen3-family thinking mode, sent as vLLM's chat_template_kwargs.enable_thinking.
    # False by default: thinking multiplies output tokens 5-10x. None sends nothing
    # (for providers that reject the field; set their equivalent in extra_body).
    enable_thinking: bool | None = False

    def request_extra_body(self) -> dict[str, Any]:
        body = json.loads(json.dumps(self.extra_body))  # deep copy
        if self.enable_thinking is not None:
            kwargs = body.setdefault("chat_template_kwargs", {})
            kwargs["enable_thinking"] = self.enable_thinking
        return body

    def describe(self) -> dict[str, Any]:
        """Settings safe to write into a game record (no key)."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "enable_thinking": self.enable_thinking,
            "extra_body": self.request_extra_body(),
        }


def load_settings(
    env_file: str | Path = ".env", prefix: str = "SEANCE_", fallback_prefix: str | None = None
) -> InferenceSettings:
    """Read `<prefix>*` settings from the environment, then `env_file`.

    With `fallback_prefix`, any setting missing under `prefix` is read under
    the fallback instead (the monitor uses SEANCE_MONITOR_*, falling back to
    SEANCE_*)."""
    file_values = dotenv_values(env_file) if Path(env_file).exists() else {}

    def lookup(key: str) -> str | None:
        return os.environ.get(key) or file_values.get(key) or None

    def get(name: str, default: str | None = None) -> str | None:
        value = lookup(prefix + name)
        if value is None and fallback_prefix is not None:
            value = lookup(fallback_prefix + name)
        return value if value is not None else default

    missing = [prefix + k for k in ("BASE_URL", "API_KEY", "MODEL") if not get(k)]
    if missing:
        raise RuntimeError(f"missing {', '.join(missing)}: copy .env.example to .env and fill it in")
    thinking = (get("ENABLE_THINKING", "false") or "").strip().lower()
    if thinking not in ("true", "false", "unset"):
        raise RuntimeError(f"{prefix}ENABLE_THINKING must be true, false or unset")
    json_mode = get("JSON_MODE", "json_schema")
    if json_mode not in JSON_MODES:
        raise RuntimeError(f"{prefix}JSON_MODE must be one of {JSON_MODES}")
    return InferenceSettings(
        base_url=get("BASE_URL"),
        api_key=get("API_KEY"),
        model=get("MODEL"),
        temperature=float(get("TEMPERATURE", "0.7")),
        max_tokens=int(get("MAX_TOKENS", "2048")),
        json_mode=json_mode,
        extra_body=json.loads(get("EXTRA_BODY", "{}")),
        timeout_s=float(get("TIMEOUT_S", "120")),
        http_retries=int(get("HTTP_RETRIES", "4")),
        enable_thinking={"true": True, "false": False, "unset": None}[thinking],
    )


class OpenAICompatClient:
    """Async client for any OpenAI-compatible chat endpoint."""

    def __init__(self, settings: InferenceSettings):
        from openai import AsyncOpenAI

        self.settings = settings
        self.model = settings.model
        # The SDK retries connection errors, 429s and 5xx with backoff.
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=settings.http_retries,
        )

    def _response_format(self, schema_name: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        mode = self.settings.json_mode
        if mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    async def complete(
        self, messages: list[dict[str, str]], schema_name: str, schema: dict[str, Any]
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        fmt = self._response_format(schema_name, schema)
        if fmt is not None:
            kwargs["response_format"] = fmt
        extra_body = self.settings.request_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        start = time.monotonic()
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        text = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
        thinking = reasoning if isinstance(reasoning, str) else ""
        if "<think>" in text or "</think>" in text:
            thinking += text.split("</think>", 1)[0]
        return Completion(
            text=text,
            reasoning_chars=len(thinking),
            reasoning=thinking,
            model=resp.model or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.monotonic() - start,
            finish_reason=choice.finish_reason,
        )


class LimitedClient:
    """Caps the number of in-flight calls across every game sharing it."""

    def __init__(self, inner: ChatClient, max_concurrent: int):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.inner = inner
        self.model = inner.model
        self.max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self.in_flight = 0
        self.peak_in_flight = 0

    async def complete(
        self, messages: list[dict[str, str]], schema_name: str, schema: dict[str, Any]
    ) -> Completion:
        async with self._sem:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            try:
                return await self.inner.complete(messages, schema_name, schema)
            finally:
                self.in_flight -= 1
