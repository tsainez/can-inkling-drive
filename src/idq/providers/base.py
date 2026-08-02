"""Provider interface and error taxonomy.

The error split matters operationally. "Do not cache errors so they retry" is
right for rate limits and 5xx, and wrong for a malformed request or a refusal:
those retry forever across every resume and burn budget on calls that can never
succeed. So terminal failures are cached as terminal, with a reason, and show
up in the invalid-output metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class RetryableError(Exception):
    """429, 5xx, timeout, connection reset. Not cached; retried."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class TerminalError(Exception):
    """4xx other than 429, content refusal, context overflow.

    Cached as a terminal outcome so we stop paying to rediscover it.
    """

    def __init__(self, message: str, reason: str, status: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.status = status


@dataclass
class ProviderResponse:
    text: str
    raw: dict
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    status: int | None = None
    attempts: int = 1
    # Providers serving Inkling return the chain of thought in a sibling field
    # rather than inline in content. Captured separately so thinking length
    # stays measurable even if a provider stops reporting reasoning_tokens.
    reasoning_text: str = ""
    # What the provider says it actually served, e.g. "inferact/inkling-nvfp4".
    # This is where quantization gets disclosed, and it belongs in the paper.
    served_model: str = ""
    # A gateway may route one model slug to several upstream providers. Capture
    # the actual provider returned by the gateway rather than trusting intent.
    served_provider: str = ""


class Provider(Protocol):
    name: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        images: list[str] | None,
        model_string: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        thinking_effort: str | None,
    ) -> ProviderResponse: ...


def extract_usage(payload: dict) -> dict:
    """Normalize usage across OpenAI-compatible providers.

    reasoning_tokens is the field RQ2 depends on. Providers that omit it are
    recorded as None rather than zero - a missing measurement and a measured
    zero are different claims, and conflating them would quietly fabricate data.
    """
    usage = (payload or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens")

    cached = None
    prompt_details = usage.get("prompt_tokens_details") or {}
    if "cached_tokens" in prompt_details:
        cached = prompt_details.get("cached_tokens")

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": reasoning,
        "cached_prompt_tokens": cached,
        "reasoning_tokens_reported": reasoning is not None,
    }
