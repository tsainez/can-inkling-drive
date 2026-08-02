"""Run configuration.

Credentials are read from os.environ and nowhere else. No key is ever written
to the cache, to a config file, or to a log line.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

from .prompts import CONDITIONS, default_prompt_key


@dataclass(frozen=True)
class ModelSpec:
    """A model as a provider actually serves it.

    quantization and served_by are recorded because an open-weights MoE served
    at NVFP4 is not the same system as the same weights at BF16, and providers
    do not always volunteer which one you are getting. If it is unknown, say
    unknown in the paper rather than implying BF16.
    """

    label: str
    model_string: str
    served_by: str
    base_url: str = ""
    api_key_env: str = ""
    quantization: str = "unknown"
    usd_per_1m_input: float | None = None
    usd_per_1m_output: float | None = None
    price_quoted_on: str = ""
    price_note: str = ""

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


@dataclass(frozen=True)
class DecodeParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    # Inkling exposes a thinking-effort control. Pinned at one documented
    # value for the whole study; sweeping it is future work. It is part of the
    # cache key so a change forces re-collection rather than mixing settings.
    thinking_effort: str | None = None
    # Gateways disagree on whether reasoning is a top-level string or an
    # OpenRouter-style object. It changes the request and therefore belongs in
    # the cache key alongside the other decode parameters.
    reasoning_format: str = "reasoning_effort"
    include_sampling_params: bool = True
    max_tokens_field: str = "max_tokens"


@dataclass(frozen=True)
class RunConfig:
    model: ModelSpec
    condition: str
    seed: int = 0
    decode: DecodeParams = field(default_factory=DecodeParams)
    prompt_key: str = ""
    corruption: str = ""
    corruption_severity: int = 0

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; known: {CONDITIONS}")
        if not self.prompt_key:
            object.__setattr__(self, "prompt_key", default_prompt_key(self.condition))
        if self.condition == "corrupt" and not self.corruption:
            raise ValueError("condition 'corrupt' requires a corruption name")

    def key_fields(self) -> dict:
        """Everything that can change an answer goes in the cache key.

        Omitting decode params or corruption severity here would let a
        parameter change silently reuse stale records.
        """
        return {
            "model_string": self.model.model_string,
            "served_by": self.model.served_by,
            "quantization": self.model.quantization,
            "condition": self.condition,
            "seed": self.seed,
            "prompt_key": self.prompt_key,
            "decode": asdict(self.decode),
            "corruption": self.corruption,
            "corruption_severity": self.corruption_severity,
        }


MOCK_MODEL = ModelSpec(
    label="mock-uniform",
    model_string="mock/uniform",
    served_by="mock",
    quantization="n/a",
    usd_per_1m_input=0.0,
    usd_per_1m_output=0.0,
)
