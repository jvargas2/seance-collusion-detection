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
    max_tokens: int = 1024
    json_mode: str = "json_schema"
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 120.0
    http_retries: int = 4

    def describe(self) -> dict[str, Any]:
        """Settings safe to write into a game record (no key)."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
            "extra_body": self.extra_body,
        }


def load_settings(env_file: str | Path = ".env") -> InferenceSettings:
    """Read SEANCE_* settings from the environment, falling back to `env_file`."""
    file_values = dotenv_values(env_file) if Path(env_file).exists() else {}

    def get(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key) or file_values.get(key) or default

    missing = [k for k in ("SEANCE_BASE_URL", "SEANCE_API_KEY", "SEANCE_MODEL") if not get(k)]
    if missing:
        raise RuntimeError(f"missing {', '.join(missing)}: copy .env.example to .env and fill it in")
    json_mode = get("SEANCE_JSON_MODE", "json_schema")
    if json_mode not in JSON_MODES:
        raise RuntimeError(f"SEANCE_JSON_MODE must be one of {JSON_MODES}")
    return InferenceSettings(
        base_url=get("SEANCE_BASE_URL"),
        api_key=get("SEANCE_API_KEY"),
        model=get("SEANCE_MODEL"),
        temperature=float(get("SEANCE_TEMPERATURE", "0.7")),
        max_tokens=int(get("SEANCE_MAX_TOKENS", "1024")),
        json_mode=json_mode,
        extra_body=json.loads(get("SEANCE_EXTRA_BODY", "{}")),
        timeout_s=float(get("SEANCE_TIMEOUT_S", "120")),
        http_retries=int(get("SEANCE_HTTP_RETRIES", "4")),
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
        if self.settings.extra_body:
            kwargs["extra_body"] = self.settings.extra_body
        start = time.monotonic()
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        return Completion(
            text=choice.message.content or "",
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
