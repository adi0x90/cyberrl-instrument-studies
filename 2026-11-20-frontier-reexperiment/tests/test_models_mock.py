"""Tests for the OpenRouter client's offline mock mode and retry policy.

No test contacts the network: real-mode tests monkeypatch ``harness.models.httpx.post``
and the key accessor, and no real credential is read or written anywhere.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness.models as models  # noqa: E402
from harness.models import (  # noqa: E402
    MODEL_IDS,
    ModelReply,
    MOCK_RESPONSES_PATH,
    available_models,
    chat,
    mock_key,
)

# Must stay byte-identical to the prompts used to build mocks/responses.json.
PROMPT_A = [
    {
        "role": "system",
        "content": "You are a careful Python security engineer. Reply with a replacement Python expression only.",
    },
    {
        "role": "user",
        "content": "Replace REPLACE_ME in `def check(req, resource): return REPLACE_ME` so that only a same-tenant writer may write. Return the replacement expression.",
    },
]
PROMPT_B = [
    {
        "role": "user",
        "content": "Return an expression that accepts only strings starting with the prefix p.",
    },
]


def load_corpus() -> dict:
    return json.loads(MOCK_RESPONSES_PATH.read_text(encoding="utf-8"))


def test_available_models_returns_exactly_the_three_ids() -> None:
    expected = ["tencent/hy4-flash", "z-ai/glm-5.3-flash", "deepseek/deepseek-v4.1-flash-1017"]
    assert available_models(mock=True) == expected
    assert available_models() == expected
    assert len(expected) == 3
    assert tuple(expected) == MODEL_IDS


def test_mock_key_is_the_sha256_of_the_canonical_prompt() -> None:
    payload = {"model": MODEL_IDS[0], "messages": PROMPT_A}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert mock_key(MODEL_IDS[0], PROMPT_A) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("model", MODEL_IDS)
@pytest.mark.parametrize("messages", [PROMPT_A, PROMPT_B], ids=["prompt_a", "prompt_b"])
def test_canned_replies_match_the_corpus(model: str, messages: list[dict]) -> None:
    corpus = load_corpus()
    entry = corpus["canned"][mock_key(model, messages)]

    reply = chat(model, messages, mock=True)

    assert isinstance(reply, ModelReply)
    assert reply.model == model
    assert reply.text == entry["text"]
    assert reply.prompt_tokens == entry["prompt_tokens"]
    assert reply.completion_tokens == entry["completion_tokens"]
    assert reply.http_status == 200
    assert reply.raw["mock"] is True
    assert reply.raw["source"] == "canned"


def test_mock_mode_is_deterministic() -> None:
    first = chat(MODEL_IDS[0], PROMPT_A, mock=True, temperature=0.7, seed=11)
    second = chat(MODEL_IDS[0], PROMPT_A, mock=True, temperature=0.7, seed=11)
    assert first == second
    assert first.raw["seed"] == 11


def test_unknown_prompt_uses_the_deterministic_generator_fallback() -> None:
    messages = [{"role": "user", "content": "a prompt that is not in the canned corpus"}]
    key = mock_key(MODEL_IDS[0], messages)
    corpus = load_corpus()
    templates = corpus["fallback"]["templates"]
    expressions = corpus["fallback"]["expressions"]
    expected = templates[int(key[0:8], 16) % len(templates)].replace(
        "{expression}", expressions[int(key[8:16], 16) % len(expressions)]
    )

    reply = chat(MODEL_IDS[0], messages, mock=True)

    assert reply.raw["source"] == "generated"
    assert reply.text == expected
    assert reply.text.strip()
    assert reply.prompt_tokens > 0
    assert reply.completion_tokens > 0


def test_fallback_replies_are_still_valid_envelope_inputs() -> None:
    from harness.envelopes import admit_fence, admit_strict, admit_toolcall

    messages = [{"role": "user", "content": "another unseen prompt for envelope sanity"}]
    reply = chat(MODEL_IDS[1], messages, mock=True)
    assert any(
        admit(reply.text).ok for admit in (admit_strict, admit_fence, admit_toolcall)
    ), f"fallback reply was not admissible by any envelope: {reply.text!r}"


def test_mock_mode_never_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reply = chat(MODEL_IDS[2], PROMPT_A, mock=True)
    assert reply.text


def test_mock_mode_never_calls_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_post(*args, **kwargs):
        raise AssertionError("mock mode must not make HTTP calls")

    monkeypatch.setattr(models.httpx, "post", forbidden_post)
    assert chat(MODEL_IDS[1], PROMPT_A, mock=True).text


def test_real_mode_without_a_key_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        chat(MODEL_IDS[0], PROMPT_A)


def test_unknown_model_id_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="unknown model id"):
        chat("openai/gpt-4o", PROMPT_A, mock=True)
    with pytest.raises(ValueError, match="unknown model id"):
        chat("openai/gpt-4o", PROMPT_A)


@pytest.mark.parametrize(
    "messages",
    [[], [{"role": "user"}], [{"role": "user", "content": 3}], "not a list"],
)
def test_malformed_messages_are_rejected(messages: object) -> None:
    with pytest.raises(ValueError):
        chat(MODEL_IDS[0], messages, mock=True)  # type: ignore[arg-type]


def test_model_reply_is_frozen() -> None:
    reply = chat(MODEL_IDS[0], PROMPT_A, mock=True)
    assert dataclasses.is_dataclass(reply)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Real-mode retry policy (httpx is monkeypatched; nothing leaves the machine)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, payload: object = None):
        self.status_code = status
        self._payload = {} if payload is None else payload
        if isinstance(self._payload, Exception):
            self.text = ""
        else:
            self.text = json.dumps(self._payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


SUCCESS = {
    "choices": [{"message": {"content": "req.tenant_id == resource.tenant_id"}}],
    "usage": {"prompt_tokens": 17, "completion_tokens": 5},
}


def install_scripted_post(monkeypatch: pytest.MonkeyPatch, scripted: list) -> list[dict]:
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = scripted.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(models.httpx, "post", fake_post)
    monkeypatch.setattr(models, "_api_key", lambda: "unit-test-placeholder")
    return calls


def test_retries_429_and_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = [FakeResponse(429, {"error": "rate limited"}), FakeResponse(503), FakeResponse(200, SUCCESS)]
    calls = install_scripted_post(monkeypatch, scripted)
    delays: list[float] = []
    monkeypatch.setattr(models, "_sleep", delays.append)

    reply = chat(MODEL_IDS[0], PROMPT_A, timeout=7.5)

    assert len(calls) == 3
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0]["timeout"] == 7.5
    assert calls[0]["headers"]["Content-Type"] == "application/json"
    assert calls[0]["headers"]["Authorization"].startswith("Bearer ")
    assert calls[0]["json"]["model"] == MODEL_IDS[0]
    assert len(delays) == 2
    assert delays[1] > delays[0] > 0
    assert reply.text == "req.tenant_id == resource.tenant_id"
    assert reply.prompt_tokens == 17
    assert reply.completion_tokens == 5
    assert reply.http_status == 200
    assert reply.raw == SUCCESS


def test_retry_budget_is_three_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = [FakeResponse(500) for _ in range(4)]
    calls = install_scripted_post(monkeypatch, scripted)
    monkeypatch.setattr(models, "_sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="failed after 3 retries"):
        chat(MODEL_IDS[0], PROMPT_A)

    assert len(calls) == 4  # initial attempt + 3 retries


def test_transport_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = [models.httpx.ConnectError("boom"), FakeResponse(200, SUCCESS)]
    calls = install_scripted_post(monkeypatch, scripted)
    monkeypatch.setattr(models, "_sleep", lambda _seconds: None)

    reply = chat(MODEL_IDS[0], PROMPT_A)

    assert len(calls) == 2
    assert reply.http_status == 200


def test_client_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = [FakeResponse(400, {"error": "bad request"})]
    calls = install_scripted_post(monkeypatch, scripted)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        chat(MODEL_IDS[0], PROMPT_A)

    assert len(calls) == 1


def test_malformed_success_bodies_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    install_scripted_post(monkeypatch, [FakeResponse(200, {"choices": []})])
    with pytest.raises(RuntimeError, match="missing choices"):
        chat(MODEL_IDS[0], PROMPT_A)

    install_scripted_post(monkeypatch, [FakeResponse(200, ValueError("not json"))])
    with pytest.raises(RuntimeError, match="non-JSON body"):
        chat(MODEL_IDS[0], PROMPT_A)


def test_missing_usage_records_zero_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    install_scripted_post(
        monkeypatch,
        [FakeResponse(200, {"choices": [{"message": {"content": "True"}}]})],
    )
    reply = chat(MODEL_IDS[0], PROMPT_A)
    assert (reply.prompt_tokens, reply.completion_tokens) == (0, 0)
