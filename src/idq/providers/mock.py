"""Offline mock provider.

Exists so the entire pipeline can be verified end to end for free before any
money is spent. The uniform style must score at chance - that is the smoke
test's proof that scoring is not lying.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from .base import ProviderResponse, TerminalError, RetryableError

_LETTERS = re.compile(r"(?:^|\n)([A-H])\.\s")


@dataclass
class MockProvider:
    """Deterministic given (seed, model_string, prompt).

    styles:
      uniform  - picks a letter uniformly at random. Expected accuracy is
                 mean(1/n_options) across the question set.
      messy    - wraps the answer in varied formats to exercise every parser
                 branch, including unparseable and ambiguous output.
      flaky    - raises retryable errors on a fixed fraction of calls, to test
                 that the cache resumes rather than losing work.
    """

    name: str = "mock"
    style: str = "uniform"
    seed: int = 0
    fail_rate: float = 0.0
    calls: int = 0
    _failed_once: set = field(default_factory=set)

    def complete(
        self,
        *,
        system: str,
        user: str,
        images: list[str] | None = None,
        model_string: str = "mock/uniform",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        seed: int = 0,
        thinking_effort: str | None = None,
    ) -> ProviderResponse:
        self.calls += 1

        letters = _LETTERS.findall("\n" + user)
        if not letters:
            letters = ["A", "B", "C", "D"]

        rng = random.Random(f"{self.seed}|{seed}|{model_string}|{user}")

        if self.fail_rate and rng.random() < self.fail_rate:
            fingerprint = hash(user)
            if fingerprint not in self._failed_once:
                self._failed_once.add(fingerprint)
                raise RetryableError("mock transient failure", status=503)

        if model_string.endswith("/refuser"):
            raise TerminalError("mock refusal", reason="content_refusal", status=400)

        choice = rng.choice(letters)

        if self.style == "messy":
            text = _messy(choice, rng, letters)
        else:
            thinking = "<think>" + ("considering the scene. " * rng.randint(4, 40)) + "</think>"
            text = f"{thinking}\nAnswer: {choice}"

        completion_tokens = max(1, len(text) // 4)
        reasoning_tokens = max(0, text.count("considering") * 3)

        raw = {
            "id": f"mock-{self.calls}",
            "object": "chat.completion",
            "model": model_string,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": max(1, len(user) // 4),
                "completion_tokens": completion_tokens,
                "total_tokens": max(1, len(user) // 4) + completion_tokens,
                "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
            },
        }

        from .base import extract_usage

        return ProviderResponse(
            text=text,
            raw=raw,
            usage=extract_usage(raw),
            latency_ms=float(rng.randint(200, 4000)),
            status=200,
            attempts=1,
        )


def _messy(choice: str, rng: random.Random, letters: list[str]) -> str:
    """Formats chosen to hit each parser branch, including the failure branches."""
    roll = rng.random()
    if roll < 0.18:
        return choice
    if roll < 0.34:
        return f"\\boxed{{{choice}}}"
    if roll < 0.50:
        return f"Looking at the scene carefully.\n\n{choice}"
    if roll < 0.66:
        return f"The final answer is {choice}."
    if roll < 0.80:
        return "I cannot determine this from the information provided."
    if roll < 0.90:
        others = [x for x in letters if x != choice]
        other = rng.choice(others) if others else choice
        return f"It could be {choice} or {other}, both are defensible."
    return "..."
