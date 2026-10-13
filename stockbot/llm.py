"""Free-only OpenRouter client with a fallback chain.

Refuses any model whose id doesn't end in ``:free`` (or isn't ``openrouter/free``),
so a typo in config can never start billing you.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

URL = "https://openrouter.ai/api/v1/chat/completions"
CHAIN = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
    "qwen/qwen3.8-27b:free",
)


class LLMError(RuntimeError):
    pass


def is_free(model: str) -> bool:
    return model.endswith(":free") or model == "openrouter/free"


def _chain() -> list[str]:
    primary = os.getenv("OPENROUTER_MODEL", "").strip()
    models = ([primary] if primary else []) + [m for m in CHAIN if m != primary]
    bad = [m for m in models if not is_free(m)]
    if bad:
        raise LLMError(f"Refusing non-free model(s): {bad}")
    return models


async def _call(model: str, prompt: str, system: str | None, temperature: float,
                max_tokens: int, timeout: float, retries: int) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set")
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    last: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(retries):
            try:
                r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
                if r.status_code in (408, 425, 429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {r.status_code}")
                r.raise_for_status()
                text = (r.json()["choices"][0]["message"]["content"] or "").strip()
                if not text:
                    raise LLMError("empty response")
                return text
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(2 ** attempt * 2)
    raise LLMError(f"{model}: {last!r}")


async def generate(prompt: str, *, system: str | None = None, temperature: float = 0.2,
                   max_tokens: int = 1500, timeout: float = 120.0, retries: int = 2) -> str:
    errors = []
    for model in _chain():
        try:
            return await _call(model, prompt, system, temperature, max_tokens, timeout, retries)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            logger.warning("model %s failed: %s", model, exc)
    raise LLMError("All free models failed: " + " | ".join(errors))
