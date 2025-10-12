"""Helpers to adapt Responses API payloads to chat-completion-like objects."""

from __future__ import annotations

import json
from typing import Any, Iterable

REASONING_MODELS = {"gpt-4.1"}
REASONING_PREFIXES = ("gpt-5", "o3", "o4")


def model_is_reasoning(model: str | None) -> bool:
    """Return True if the model should be treated as a reasoning model."""
    if not model:
        return False
    normalized = model.lower()
    if any(normalized.startswith(prefix) for prefix in REASONING_PREFIXES):
        return True
    if "-thinking" in normalized:
        return True
    return normalized in REASONING_MODELS


def responses_to_chat_like(response: Any) -> dict[str, Any]:
    """Normalize a Responses API response to a Chat Completions style payload."""
    data = _to_dict(response)

    assistant_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    output_items = data.get("output") or []
    if isinstance(output_items, Iterable) and not isinstance(output_items, (dict, str)):
        for item in output_items:
            item_dict = _to_dict(item)
            content_items = item_dict.get("content") or []
            if isinstance(content_items, Iterable) and not isinstance(content_items, (dict, str)):
                for content in content_items:
                    content_dict = _to_dict(content)
                    _extract_content_piece(content_dict, assistant_text_parts, tool_calls)

    if not assistant_text_parts:
        output_text = data.get("output_text") or data.get("text")
        if isinstance(output_text, str) and output_text:
            assistant_text_parts.append(output_text)

    assistant_text = "".join(assistant_text_parts)

    usage = _normalize_usage(data.get("usage") or {})

    finish_reason = "tool_calls" if tool_calls else "stop"

    message = {
        "role": "assistant",
        "content": assistant_text,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    chat_payload = {
        "id": data.get("id"),
        "object": "chat.completion",
        "created": data.get("created"),
        "model": data.get("model"),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
    }

    if usage:
        chat_payload["usage"] = usage

    return chat_payload


def _extract_content_piece(
    content: dict[str, Any],
    text_parts: list[str],
    tool_calls: list[dict[str, Any]],
) -> None:
    content_type = content.get("type")

    if content_type in {"output_text", "text"}:
        text_value = content.get("text")
        if isinstance(text_value, dict):
            text_value = text_value.get("value") or text_value.get("text")
        if isinstance(text_value, str):
            text_parts.append(text_value)
        return

    if content_type == "tool_call":
        tool = content.get("tool_call") or content
        name = tool.get("name")
        arguments = tool.get("arguments")
        if isinstance(arguments, (dict, list)):
            arguments_str = json.dumps(arguments)
        elif isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments or {})
        call_id = (
            tool.get("id")
            or tool.get("call_id")
            or content.get("id")
            or f"call_{len(tool_calls)}"
        )
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name or "",
                    "arguments": arguments_str,
                },
            }
        )
        return

    # Some SDK versions may emit tool call deltas as top-level list
    if content_type == "tool_calls":
        for tool in content.get("tool_calls") or []:
            _extract_content_piece(tool, text_parts, tool_calls)


def _normalize_usage(raw_usage: dict[str, Any]) -> dict[str, Any]:
    if not raw_usage:
        return {}

    prompt_tokens = raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens")
    completion_tokens = raw_usage.get("completion_tokens") or raw_usage.get("output_tokens")
    total_tokens = raw_usage.get("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    usage: dict[str, Any] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens

    return usage


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    return {}
