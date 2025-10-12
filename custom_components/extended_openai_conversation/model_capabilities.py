"""Model capability heuristics for routing requests."""

from __future__ import annotations

import re
from dataclasses import dataclass


_REASONING_PATTERN = re.compile(r"(gpt-5.*thinking|o[0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class ModelCapabilities:
    """Heuristic capabilities detected for a given model identifier."""

    supports_responses: bool
    supports_chat: bool
    supports_reasoning: bool
    supports_streaming: bool
    prefers_responses: bool


def detect_model_capabilities(model: str | None) -> ModelCapabilities:
    """Return inferred capabilities for ``model`` using lightweight heuristics."""

    if not model:
        return ModelCapabilities(
            supports_responses=True,
            supports_chat=True,
            supports_reasoning=False,
            supports_streaming=True,
            prefers_responses=False,
        )

    identifier = model.lower()
    supports_reasoning = bool(_REASONING_PATTERN.search(identifier))

    # Assume modern models support both APIs and streaming by default; callers may
    # still override strategy or feature flags explicitly via options.
    supports_responses = True
    supports_chat = True
    supports_streaming = True

    prefers_responses = supports_reasoning

    return ModelCapabilities(
        supports_responses=supports_responses,
        supports_chat=supports_chat,
        supports_reasoning=supports_reasoning,
        supports_streaming=supports_streaming,
        prefers_responses=prefers_responses,
    )
