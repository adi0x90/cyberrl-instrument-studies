"""OpenRouter client with an explicit reasoning-token cap.

The published harness (`propverifier.harness.models.chat`) exposes only
`max_tokens`, which reasoning-capable models spend *before* emitting any content.
On the native-pack repair task that produced empty completions in 87/96 calls at
2,400 tokens and still ~90 % at 8,000.

OpenRouter accepts a `reasoning: {"max_tokens": N}` field that reserves the rest
of the completion budget for content.  This module adds that field and records
`finish_reason` so exhausted-budget calls are visible instead of silently empty.

The published harness is left untouched: the expression-arm pools must remain
comparable with the earlier studies, which used it unchanged.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def _key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    return key


def chat_capped(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 1.0,
    seed: int | None = None,
    max_tokens: int = 8000,
    reasoning_max_tokens: int = 1024,
    timeout: float = 300.0,
    retries: int = 3,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"max_tokens": reasoning_max_tokens},
    }
    if seed is not None:
        body["seed"] = seed

    last_error = None
    for attempt in range(retries):
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            choice = payload["choices"][0]
            usage = payload.get("usage") or {}
            return {
                "text": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read()[:300]!r}"
            if exc.code in (401, 400):
                raise RuntimeError(last_error) from exc
        except Exception as exc:  # noqa: BLE001 - transient network/JSON
            last_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"chat_capped failed after {retries} attempts: {last_error}")


__all__ = ["chat_capped"]
