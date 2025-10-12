"""Helpers for detecting OpenAI client capabilities."""

from __future__ import annotations

import inspect
from typing import Callable


def detect_responses_max_tokens_param(
    create_method: Callable | None,
) -> str | None:
    """Return the supported max tokens parameter name for Responses."""

    if create_method is None:
        return None

    try:
        parameters = inspect.signature(create_method).parameters
    except (TypeError, ValueError):
        return None

    if "max_output_tokens" in parameters:
        return "max_output_tokens"

    if "max_completion_tokens" in parameters:
        return "max_completion_tokens"

    return None


def detect_chat_tools_support(create_method: Callable | None) -> bool:
    """Return True if the chat completions endpoint supports the tools parameter."""

    if create_method is None:
        return False

    try:
        parameters = inspect.signature(create_method).parameters
    except (TypeError, ValueError):
        return False

    return "tools" in parameters
