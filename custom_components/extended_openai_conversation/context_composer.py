"""Helpers to compose contextual system prompts within token budgets."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


AVERAGE_CHARS_PER_TOKEN = 4


@dataclass
class ContextSlice:
    """A composed slice of context."""

    label: str
    text: str
    tokens: int
    trimmed: bool


@dataclass
class ContextComposition:
    """Return value from compose_system_sections."""

    content: str
    slices: dict[str, ContextSlice]


def estimate_tokens(text: str | None) -> int:
    """Roughly estimate the token footprint for a piece of text."""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / AVERAGE_CHARS_PER_TOKEN))


def _truncate_to_budget(text: str, budget: int) -> tuple[str, int, bool]:
    """Clamp text to the provided token budget."""

    if budget <= 0 or not text:
        return "", 0, bool(text)

    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= budget:
        return text, estimated_tokens, False

    tokens_remaining = budget
    words: Iterable[str] = re.split(r"(\s+)", text)
    builder: list[str] = []
    for chunk in words:
        token_cost = estimate_tokens(chunk)
        if token_cost > tokens_remaining:
            break
        builder.append(chunk)
        tokens_remaining -= token_cost

    truncated = "".join(builder).strip()
    return truncated, estimate_tokens(truncated), True


def compose_system_sections(
    profile: str | None,
    scratchpad: str | None,
    retrieved: str | None,
    *,
    budget_profile: int,
    budget_scratchpad: int,
    budget_retrieved: int,
) -> ContextComposition:
    """Compose the structured system sections respecting configured budgets."""

    composition: dict[str, ContextSlice] = {}
    sections: list[str] = []

    def _register(label: str, text: str | None, budget: int, key: str) -> None:
        trimmed_text, tokens, trimmed = _truncate_to_budget(text or "", budget)
        if trimmed_text:
            sections.append(f"[{label}]\n{trimmed_text}")
        composition[key] = ContextSlice(
            label=label, text=trimmed_text, tokens=tokens, trimmed=trimmed
        )

    _register("Profile Digest", profile, budget_profile, "profile")
    _register("Session Scratchpad", scratchpad, budget_scratchpad, "scratchpad")
    _register("Retrieved Memory", retrieved, budget_retrieved, "retrieved")

    content = "\n\n".join(filter(None, sections))
    return ContextComposition(content=content, slices=composition)
