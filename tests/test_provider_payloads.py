"""Provider payload parsing, pinned to real documented response shapes.

The payload shape is the single most likely thing to break on first contact
with a live provider. These tests encode the shapes the docs actually show, so
a regression is caught offline rather than 2,000 calls into a run.

Baseten shape verified against https://www.baseten.co/library/inkling/
(retrieved 2026-07-24).
"""

from __future__ import annotations

import json

import pytest

from idq.providers import OpenAICompatProvider, TerminalError
from idq.providers.base import extract_usage

# Abridged from Baseten's published example response for Inkling. Note that
# content and reasoning_content are BOTH populated - the chain of thought is a
# sibling field, not a replacement.
BASETEN_RESPONSE = {
    "id": "chatcmpl-xxxx",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "The vehicle is stopped at the line.\n\nAnswer: D",
                "role": "assistant",
                "reasoning_content": (
                    "Looking at the scene, option D seems right. But wait - "
                    "option A is also plausible. Let me reconsider. B is out."
                ),
            },
        }
    ],
    "created": 1784140413,
    "model": "inferact/inkling-nvfp4",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 295,
        "prompt_tokens": 9,
        "total_tokens": 304,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 182,
            "rejected_prediction_tokens": 0,
        },
        "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
    },
}


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload

    @property
    def text(self):
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("IDQ_TEST_KEY", "not-a-real-key")
    return OpenAICompatProvider(
        base_url="https://inference.example.co/v1", api_key_env="IDQ_TEST_KEY"
    )


def call(provider, monkeypatch, payload, status=200):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse(payload, status=status)

    monkeypatch.setattr("idq.providers.openai_compat.requests.post", fake_post)
    return provider, captured


def test_reasoning_tokens_are_extracted():
    """The field RQ2 depends on."""
    usage = extract_usage(BASETEN_RESPONSE)
    assert usage["reasoning_tokens"] == 182
    assert usage["reasoning_tokens_reported"] is True
    assert usage["completion_tokens"] == 295
    assert usage["cached_prompt_tokens"] == 0


def test_missing_reasoning_tokens_is_none_not_zero():
    """A missing measurement and a measured zero are different claims."""
    payload = {"usage": {"completion_tokens": 100, "prompt_tokens": 10}}
    usage = extract_usage(payload)
    assert usage["reasoning_tokens"] is None
    assert usage["reasoning_tokens_reported"] is False


def test_reasoning_content_is_not_folded_into_the_answer(provider, monkeypatch):
    """The chain of thought must stay out of the parser's input.

    The reasoning here mentions A, B and D. If it were concatenated onto the
    content, the answer parser would see multiple candidate letters and either
    flag ambiguity or pick the wrong one.
    """
    p, _ = call(provider, monkeypatch, BASETEN_RESPONSE)
    resp = p.complete(system="s", user="u", model_string="thinkingmachines/inkling")

    assert resp.text == "The vehicle is stopped at the line.\n\nAnswer: D"
    assert "Let me reconsider" not in resp.text
    assert "Let me reconsider" in resp.reasoning_text

    from idq.parse import parse_answer_with_options

    parsed = parse_answer_with_options(resp.text, ["A", "B", "C", "D"], ["", "", "", ""])
    assert parsed.letter == "D"


def test_served_model_is_captured_for_quantization_disclosure(provider, monkeypatch):
    """'inferact/inkling-nvfp4' tells us the weights were served at NVFP4."""
    p, _ = call(provider, monkeypatch, BASETEN_RESPONSE)
    resp = p.complete(system="s", user="u", model_string="thinkingmachines/inkling")
    assert resp.served_model == "inferact/inkling-nvfp4"
    assert "nvfp4" in resp.served_model


def test_empty_content_falls_back_to_reasoning_text(provider, monkeypatch):
    """Some models emit only reasoning when they hit the token ceiling."""
    payload = json.loads(json.dumps(BASETEN_RESPONSE))
    payload["choices"][0]["message"]["content"] = ""
    p, _ = call(provider, monkeypatch, payload)
    resp = p.complete(system="s", user="u", model_string="thinkingmachines/inkling")
    assert resp.text.startswith("Looking at the scene")


def test_thinking_effort_is_sent_as_reasoning_effort(provider, monkeypatch):
    p, captured = call(provider, monkeypatch, BASETEN_RESPONSE)
    p.complete(system="s", user="u", model_string="m", thinking_effort="low")
    assert captured["json"]["reasoning_effort"] == "low"
    assert captured["url"].endswith("/chat/completions")


def test_images_become_content_parts(provider, monkeypatch):
    p, captured = call(provider, monkeypatch, BASETEN_RESPONSE)
    p.complete(system="s", user="u", images=["data:image/jpeg;base64,AAAA"], model_string="m")
    content = captured["json"]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_api_key_comes_from_env_and_is_not_logged(provider, monkeypatch):
    p, captured = call(provider, monkeypatch, BASETEN_RESPONSE)
    p.complete(system="s", user="u", model_string="m")
    assert captured["headers"]["Authorization"] == "Bearer not-a-real-key"
    assert "not-a-real-key" not in json.dumps(captured["json"])


def test_missing_credential_is_terminal_not_retryable(monkeypatch):
    monkeypatch.delenv("IDQ_ABSENT_KEY", raising=False)
    p = OpenAICompatProvider(base_url="https://x/v1", api_key_env="IDQ_ABSENT_KEY")
    with pytest.raises(TerminalError) as exc:
        p.complete(system="s", user="u", model_string="m")
    assert exc.value.reason == "missing_credential"


@pytest.mark.parametrize(
    "status,reason",
    [(401, "auth"), (403, "auth"), (404, "unknown_model"), (400, "http_400")],
)
def test_client_errors_are_terminal(provider, monkeypatch, status, reason):
    p, _ = call(provider, monkeypatch, {"error": "nope"}, status=status)
    with pytest.raises(TerminalError) as exc:
        p.complete(system="s", user="u", model_string="m")
    assert exc.value.reason == reason


def test_context_overflow_is_classified(provider, monkeypatch):
    p, _ = call(
        provider, monkeypatch,
        "maximum context length exceeded for this request", status=400,
    )
    with pytest.raises(TerminalError) as exc:
        p.complete(system="s", user="u", model_string="m")
    assert exc.value.reason == "context_overflow"


def test_empty_choices_is_terminal(provider, monkeypatch):
    p, _ = call(provider, monkeypatch, {"choices": [], "model": "m"})
    with pytest.raises(TerminalError) as exc:
        p.complete(system="s", user="u", model_string="m")
    assert exc.value.reason == "empty_choices"


def test_rate_limit_retries_then_succeeds(provider, monkeypatch):
    """429 is retryable: back off and try again rather than losing the call."""
    monkeypatch.setattr("idq.providers.openai_compat.time.sleep", lambda *a, **k: None)
    calls = {"n": 0}

    def flaky_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResponse({"error": "slow down"}, status=429)
        return FakeResponse(BASETEN_RESPONSE, status=200)

    monkeypatch.setattr("idq.providers.openai_compat.requests.post", flaky_post)
    resp = provider.complete(system="s", user="u", model_string="m")
    assert resp.attempts == 3
    assert resp.usage["reasoning_tokens"] == 182
