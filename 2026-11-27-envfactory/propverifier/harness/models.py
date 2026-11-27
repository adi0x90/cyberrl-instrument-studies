"""OpenRouter chat client with a deterministic, offline mock mode.

Real mode
---------
POST ``https://openrouter.ai/api/v1/chat/completions`` with the key read from the
``OPENROUTER_API_KEY`` environment variable *at call time*.  The key is never
stored, cached, logged, or read from a file.  429/5xx responses (and transport
errors) are retried up to ``MAX_RETRIES`` times with exponential backoff plus
jitter; a per-call timeout is always passed to httpx.

Mock mode
---------
``mock=True`` never touches the network and never needs a key.  The reply is
chosen by ``sha256(mock_key(model, messages))``:

1. If ``<experiment>/mocks/responses.json`` has a ``canned`` entry under that
   hash, that text/token counts are returned verbatim.  Canned replies are
   therefore stable across machines and runs.
2. Otherwise a small deterministic generator picks one template (and one
   expression) from the ``fallback`` section of that file using disjoint bytes of
   the same digest, and substitutes ``{expression}``.  Unknown prompts thus still
   return a syntactically valid reply.

The digest is over a canonical serialisation of ``{"model": ..., "messages": ...}``
with sorted keys and no insignificant whitespace, so structurally identical
requests always hash identically.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://openrouter.ai/api/v1"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"

#: The only three model ids this experiment may ever call.  Never substitute.
MODEL_IDS: tuple[str, ...] = (
    "tencent/hy4-flash",
    "z-ai/glm-5.3-flash",
    "deepseek/deepseek-v4.1-flash-1017",
)

API_KEY_ENV = "OPENROUTER_API_KEY"

MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0
BACKOFF_JITTER_S = 0.5

#: Path of the deterministic mock corpus.  Tests may monkeypatch this.
MOCK_RESPONSES_PATH = Path(__file__).resolve().parent.parent / "mocks" / "responses.json"

_MODEL_ID_SET = frozenset(MODEL_IDS)

#: Indirection so tests can replace the sleeper without patching global ``time``.
_sleep = time.sleep


@dataclass(frozen=True)
class ModelReply:
    """One completion.  ``raw`` is the decoded OpenRouter JSON (or mock metadata)."""

    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    http_status: int
    raw: dict


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_model(model: str) -> None:
    if model not in _MODEL_ID_SET:
        raise ValueError(
            f"unknown model id {model!r}; this experiment only permits {list(MODEL_IDS)}"
        )


def _validate_messages(messages: list[dict]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {role, content} dicts")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be a dict")
        if not isinstance(message.get("role"), str) or not message["role"]:
            raise ValueError(f"messages[{index}].role must be a non-empty string")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"messages[{index}].content must be a string")


def _api_key() -> str:
    """Read the key from the environment at call time (never from a file)."""

    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set in the environment. This harness never reads a "
            "key from disk; export the variable for real calls, or pass mock=True for "
            "the offline deterministic mode."
        )
    return key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_models(mock: bool = False) -> list[str]:
    """Return the three permitted model ids.

    Both modes expose exactly the same three ids: mock mode is a deterministic
    stand-in for each real model, not a different catalogue.
    """

    return list(MODEL_IDS)


def chat(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    seed: int | None = None,
    # 900 is too small for reasoning-capable models: tencent/hy4-flash and
    # deepseek/deepseek-v4.1-flash-1017 emit a `reasoning` field first and return
    # content=null with finish_reason="length" when the budget is exhausted.
    max_tokens: int = 1600,
    timeout: float = 120.0,
    mock: bool = False,
) -> ModelReply:
    """Request one completion from OpenRouter (or the deterministic mock).

    Raises
    ------
    ValueError
        Unknown model id or malformed ``messages``.
    RuntimeError
        Missing ``OPENROUTER_API_KEY`` in real mode, a non-retryable HTTP error, a
        malformed response body, or transient failures that outlive the retries.
    """

    _validate_model(model)
    _validate_messages(messages)

    if mock:
        return _mock_reply(model, messages, temperature=temperature, seed=seed)

    if timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")

    api_key = _api_key()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = "no attempt was made"
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.post(
                CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except httpx.RequestError as exc:  # transport error: transient
            last_error = f"transport error: {exc.__class__.__name__}: {exc}"
            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt)
                continue
            break

        status = response.status_code
        if status == 429 or 500 <= status < 600:
            last_error = f"HTTP {status}: {_body_excerpt(response.text)}"
            if attempt < MAX_RETRIES:
                _sleep_backoff(attempt)
                continue
            break

        if status >= 400:
            # Non-transient client error: fail fast, no retry.
            raise RuntimeError(
                f"OpenRouter returned HTTP {status}: {_body_excerpt(response.text)}"
            )

        return _parse_response(model, response, status)

    raise RuntimeError(
        f"OpenRouter chat failed after {MAX_RETRIES} retries for model {model!r}: {last_error}"
    )


# ---------------------------------------------------------------------------
# Real-response parsing
# ---------------------------------------------------------------------------


def _body_excerpt(text: str, limit: int = 300) -> str:
    flat = " ".join(str(text).split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def _content_text(content: Any) -> str:
    """Normalise message content (string, or a list of typed parts) to text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    raise RuntimeError(f"unsupported message content type {type(content).__name__!r}")


def _parse_response(model: str, response: httpx.Response, status: int) -> ModelReply:
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"OpenRouter returned non-JSON body (HTTP {status}): {_body_excerpt(response.text)}"
        )

    if not isinstance(data, dict):
        raise RuntimeError("OpenRouter returned JSON that is not an object")

    try:
        message = data["choices"][0]["message"]
        text = _content_text(message.get("content"))
    except (KeyError, IndexError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "OpenRouter response is missing choices[0].message.content: "
            f"{_body_excerpt(str(data))}"
        ) from exc

    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens"))

    return ModelReply(
        model=model,
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        http_status=status,
        raw=data,
    )


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _sleep_backoff(attempt: int) -> None:
    """Exponential backoff with full jitter on the additive component."""

    delay = BACKOFF_BASE_S * (2**attempt) + random.uniform(0.0, BACKOFF_JITTER_S)
    _sleep(delay)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------


def canonical_prompt(model: str, messages: list[dict]) -> str:
    """Canonical JSON used for hashing mock requests."""

    payload = {"model": model, "messages": messages}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def mock_key(model: str, messages: list[dict]) -> str:
    """Deterministic sha256 key for a (model, messages) pair."""

    return hashlib.sha256(canonical_prompt(model, messages).encode("utf-8")).hexdigest()


def _load_mock_config() -> dict:
    path = Path(MOCK_RESPONSES_PATH)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"mock corpus is missing or unreadable at {path}: {exc}") from exc
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"mock corpus at {path} is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"mock corpus at {path} must be a JSON object")
    return config


def _estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (mock metadata only)."""

    return max(1, (len(text) + 3) // 4)


def _mock_reply(
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    seed: int | None,
) -> ModelReply:
    key = mock_key(model, messages)
    config = _load_mock_config()
    prompt_estimate = _estimate_tokens(canonical_prompt(model, messages))

    canned = config.get("canned")
    entry = canned.get(key) if isinstance(canned, dict) else None
    if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"].strip():
        text = entry["text"]
        prompt_tokens = _as_int(entry.get("prompt_tokens")) or prompt_estimate
        completion_tokens = _as_int(entry.get("completion_tokens")) or _estimate_tokens(text)
        source = "canned"
    else:
        text = _generate_mock_text(config, key)
        prompt_tokens = prompt_estimate
        completion_tokens = _estimate_tokens(text)
        source = "generated"

    raw = {
        "mock": True,
        "source": source,
        "mock_key": key,
        "model": model,
        "temperature": temperature,
        "seed": seed,
    }
    return ModelReply(
        model=model,
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        http_status=200,
        raw=raw,
    )


def _generate_mock_text(config: dict, key: str) -> str:
    """Deterministic generator fallback for prompts with no canned entry."""

    fallback = config.get("fallback")
    if not isinstance(fallback, dict):
        fallback = {}

    templates = [t for t in fallback.get("templates", []) if isinstance(t, str) and t.strip()]
    expressions = [e for e in fallback.get("expressions", []) if isinstance(e, str) and e.strip()]
    if not templates:
        templates = ['{"expression": "{expression}"}']
    if not expressions:
        expressions = ["True"]

    # Disjoint digest bytes: two independent deterministic choices.
    template_index = int(key[0:8], 16) % len(templates)
    expression_index = int(key[8:16], 16) % len(expressions)
    return templates[template_index].replace("{expression}", expressions[expression_index])


__all__ = [
    "API_KEY_ENV",
    "BASE_URL",
    "CHAT_COMPLETIONS_URL",
    "ModelReply",
    "MODEL_IDS",
    "MOCK_RESPONSES_PATH",
    "available_models",
    "canonical_prompt",
    "chat",
    "mock_key",
]
