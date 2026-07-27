"""OpenAI-compatible /chat/completions provider.

Targets Together / Fireworks / Baseten / Modal style endpoints. Payload shape
is the single most likely thing to break on first contact with a real provider,
which is why the 20-call pilot exists.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import requests

from .base import ProviderResponse, RetryableError, TerminalError, extract_usage

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class OpenAICompatProvider:
    base_url: str
    api_key_env: str
    name: str = "openai_compat"
    timeout_s: float = 180.0
    max_attempts: int = 6
    backoff_base_s: float = 1.5
    backoff_cap_s: float = 60.0
    extra_body: dict | None = None

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise TerminalError(
                f"environment variable {self.api_key_env} is not set",
                reason="missing_credential",
            )
        return key

    def complete(
        self,
        *,
        system: str,
        user: str,
        images: list[str] | None = None,
        model_string: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        seed: int = 0,
        thinking_effort: str | None = None,
    ) -> ProviderResponse:
        content: list | str
        if images:
            content = [{"type": "text", "text": user}]
            for data_url in images:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            content = user

        payload = {
            "model": model_string,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if thinking_effort:
            # Field name varies by provider. Confirm against provider docs on
            # the 20-call pilot before trusting it; an ignored field looks
            # identical to a respected one in the response.
            payload["reasoning_effort"] = thinking_effort
        if self.extra_body:
            payload.update(self.extra_body)

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last_exc = RetryableError(f"transport error: {exc}")
                self._sleep(attempt)
                continue

            latency_ms = (time.monotonic() - started) * 1000.0

            if resp.status_code in RETRYABLE_STATUS:
                last_exc = RetryableError(
                    f"http {resp.status_code}: {resp.text[:400]}", status=resp.status_code
                )
                self._sleep(attempt, resp.headers.get("Retry-After"))
                continue

            if resp.status_code >= 400:
                raise TerminalError(
                    f"http {resp.status_code}: {resp.text[:800]}",
                    reason=_terminal_reason(resp.status_code, resp.text),
                    status=resp.status_code,
                )

            try:
                data = resp.json()
            except ValueError:
                raise TerminalError(
                    f"non-JSON response: {resp.text[:400]}", reason="malformed_response",
                    status=resp.status_code,
                )

            choices = data.get("choices") or []
            if not choices:
                raise TerminalError(
                    f"no choices in response: {str(data)[:400]}", reason="empty_choices",
                    status=resp.status_code,
                )

            message = choices[0].get("message") or {}
            text = message.get("content") or ""
            # Baseten and Fireworks both return the chain of thought in
            # message.reasoning_content, ALONGSIDE a populated content field.
            # Keep them separate: folding reasoning into text would feed the
            # model's own deliberation to the answer parser, where a letter
            # mentioned mid-thought could be mistaken for the final answer.
            reasoning_text = message.get("reasoning_content") or ""
            if not text and reasoning_text:
                text = reasoning_text

            return ProviderResponse(
                text=text,
                raw=data,
                usage=extract_usage(data),
                latency_ms=latency_ms,
                status=resp.status_code,
                attempts=attempt,
                reasoning_text=reasoning_text,
                served_model=str(data.get("model") or ""),
            )

        raise last_exc or RetryableError("exhausted retries")

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), self.backoff_cap_s))
                return
            except (TypeError, ValueError):
                pass
        delay = min(self.backoff_base_s ** attempt, self.backoff_cap_s)
        time.sleep(delay * (0.5 + random.random()))


def _terminal_reason(status: int, body: str) -> str:
    low = (body or "").lower()
    if "context" in low and ("length" in low or "window" in low):
        return "context_overflow"
    if "content" in low and ("filter" in low or "policy" in low):
        return "content_refusal"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "unknown_model"
    return f"http_{status}"
