"""Versioned, content-hashed prompt templates.

Changing a prompt mid-study without tracking it silently invalidates every
comparison already collected. So: templates are immutable, versioned, and their
hash is written into every cache record and into the cache key. Edit a template
in place and the harness will treat it as a new prompt and re-collect, rather
than mixing two prompts under one label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .hashing import sha256_str

# Conditions.
#
# blind_tags vs blind_notags exist because DriveLM question text embeds object
# references like <c1,CAM_FRONT,1088.3,497.5>. Those tags name a camera and a
# pixel location, so a "text-only" run that leaves them in is not actually
# blind. Running both lets us decompose the grounding gap into three parts:
# what the image contributes, what the annotation schema leaks, and what
# language priors alone recover.
CONDITIONS = ("clean", "blind_tags", "blind_notags", "corrupt")

BLIND_CONDITIONS = ("blind_tags", "blind_notags")
IMAGE_CONDITIONS = ("clean", "corrupt")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str

    @property
    def hash(self) -> str:
        return sha256_str(f"{self.name}\x00{self.version}\x00{self.system}\x00{self.user}")

    def render(self, *, question: str, options_block: str) -> tuple[str, str]:
        user = self.user.format(question=question, options_block=options_block)
        return self.system, user


_ANSWER_INSTRUCTION = (
    "Answer with exactly one option letter. "
    "End your reply with a final line of the form 'Answer: X' where X is the letter."
)

MCQ_V1 = PromptTemplate(
    name="mcq",
    version="v1",
    system=(
        "You are analyzing a road scene for an autonomous driving research benchmark. "
        "Answer the multiple-choice question. " + _ANSWER_INSTRUCTION
    ),
    user="{question}\n\n{options_block}\n\nAnswer:",
)

MCQ_BLIND_V1 = PromptTemplate(
    name="mcq_blind",
    version="v1",
    system=(
        "You are answering a multiple-choice question about a road scene. "
        "No image is provided. Choose the option you judge most likely to be correct. "
        + _ANSWER_INSTRUCTION
    ),
    user="{question}\n\n{options_block}\n\nAnswer:",
)

REGISTRY: dict[str, PromptTemplate] = {
    f"{t.name}@{t.version}": t for t in (MCQ_V1, MCQ_BLIND_V1)
}


def get_prompt(key: str) -> PromptTemplate:
    if key not in REGISTRY:
        raise KeyError(f"unknown prompt {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def default_prompt_key(condition: str) -> str:
    if condition in BLIND_CONDITIONS:
        return "mcq_blind@v1"
    return "mcq@v1"


def format_options_block(letters: Sequence[str], texts: Sequence[str]) -> str:
    return "\n".join(f"{ltr}. {txt}" for ltr, txt in zip(letters, texts))
