"""Lightweight router to pre-classify memory intents."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RouterDecision:
    """Router output passed to the OpenAI request builder."""

    chat_tool_choice: dict | str
    responses_tool_choice: dict | str
    forced_tool: str | None
    normalized_text: str | None
    detected_tool: str | None = None


def _normalize_text(text: str, match: re.Match[str]) -> str:
    suffix = text[match.end() :].strip()
    return suffix or text.strip()


def classify_intent(
    text: str,
    *,
    write_pattern: str,
    search_pattern: str,
    force_tools: bool,
) -> RouterDecision:
    """Return routing metadata for the current user input."""

    if not text:
        return RouterDecision(
            chat_tool_choice="auto",
            responses_tool_choice="auto",
            forced_tool=None,
            normalized_text=None,
            detected_tool=None,
        )

    write_regex = re.compile(write_pattern, re.IGNORECASE)
    search_regex = re.compile(search_pattern, re.IGNORECASE)

    write_match = write_regex.search(text)
    if write_match:
        normalized = _normalize_text(text, write_match)
        if force_tools:
            return RouterDecision(
                chat_tool_choice={
                    "type": "function",
                    "function": {"name": "memory.write"},
                },
                responses_tool_choice={"type": "tool", "name": "memory.write"},
                forced_tool="memory.write",
                normalized_text=normalized,
                detected_tool="memory.write",
            )
        return RouterDecision(
            chat_tool_choice="auto",
            responses_tool_choice="auto",
            forced_tool=None,
            normalized_text=None,
            detected_tool="memory.write",
        )

    search_match = search_regex.search(text)
    if search_match:
        normalized = _normalize_text(text, search_match)
        if force_tools:
            return RouterDecision(
                chat_tool_choice={
                    "type": "function",
                    "function": {"name": "memory.search"},
                },
                responses_tool_choice={"type": "tool", "name": "memory.search"},
                forced_tool="memory.search",
                normalized_text=normalized,
                detected_tool="memory.search",
            )
        return RouterDecision(
            chat_tool_choice="auto",
            responses_tool_choice="auto",
            forced_tool=None,
            normalized_text=None,
            detected_tool="memory.search",
        )

    return RouterDecision(
        chat_tool_choice="auto",
        responses_tool_choice="auto",
        forced_tool=None,
        normalized_text=None,
        detected_tool=None,
    )
