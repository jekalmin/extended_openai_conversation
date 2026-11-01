"""Stateless Responses API transport for supported GPT-6 and GPT-6.1 models.

Provider output (including encrypted reasoning) lives in AssistantContent.native,
not a process-global cache or persisted conversation. HA owns its lifetime.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from copy import deepcopy
from dataclasses import replace
import json
from typing import Any, cast

from openai.types.responses import ResponseInputParam

from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from .const import (
    CONF_EXTRA_BODY,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_SERVICE_TIER,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DOMAIN,
)
from .exceptions import ParseArgumentsFailed, TokenLengthExceededError
from .helpers import get_reasoning_effort_options

NATIVE_KEY = f"{DOMAIN}.responses"


def clear_native_history(hass: Any, agent_ids: set[str]) -> None:
    """Discard provider state on integration unload without clearing HA text."""
    from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS

    for chat_log in hass.data.get(DATA_CHAT_LOGS, {}).values():
        for index, content in enumerate(chat_log.content):
            if (
                content.role == "assistant"
                and content.agent_id in agent_ids
                and isinstance(content.native, dict)
                and content.native.get("provider") == NATIVE_KEY
            ):
                chat_log.content[index] = replace(content, native=None)


def response_input(chat_log: conversation.ChatLog, model: str) -> ResponseInputParam:
    """Replay retained provider items, or convert older HA conversation content.

    Missing results after an interruption are answered as errors, never replayed
    against the house. A completed result already in HA is always preserved.
    """
    items: list[dict[str, Any]] = []
    pending: set[str] = set()

    def finish_interrupted() -> None:
        for call_id in sorted(pending):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"error": "interrupted_tool_call"}),
                }
            )
        pending.clear()

    for content in chat_log.content:
        if content.role in {"system", "user"}:
            finish_interrupted()
            items.append({"role": content.role, "content": content.content})
        elif content.role == "assistant":
            finish_interrupted()
            native = getattr(content, "native", None)
            if (
                isinstance(native, dict)
                and native.get("provider") == NATIVE_KEY
                and native.get("model") == model
            ):
                output = deepcopy(native["output"])
                items.extend(output)
                pending.update(
                    item["call_id"]
                    for item in output
                    if item["type"] == "function_call"
                )
            else:
                if content.content:
                    items.append({"role": "assistant", "content": content.content})
                for call in content.tool_calls or []:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.tool_name,
                            "arguments": json.dumps(call.tool_args),
                        }
                    )
                    pending.add(call.id)
        elif content.role == "tool_result" and content.tool_call_id in pending:
            result = getattr(content, "result", None)
            data = result.data if result is not None else content.tool_result
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": content.tool_call_id,
                    "output": json.dumps(data, ensure_ascii=False),
                }
            )
            pending.remove(content.tool_call_id)
    finish_interrupted()
    return cast(ResponseInputParam, items)


def request_parameters(
    model: str,
    options: Mapping[str, Any],
    function_tools: list[dict[str, Any]],
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Responses parameters without changing existing YAML tool schemas."""
    effort = options.get(CONF_REASONING_EFFORT, DEFAULT_REASONING_EFFORT)
    if effort not in get_reasoning_effort_options(model):
        raise HomeAssistantError(f"Unsupported reasoning effort for {model}: {effort}")
    # extra_body is a Chat Completions/backend escape hatch. It must not silently
    # override store, reasoning, model or input on this new transport.
    if (options.get(CONF_EXTRA_BODY) or "").strip():
        raise HomeAssistantError("extra_body is not supported with the Responses API")
    params: dict[str, Any] = {
        "model": model,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": effort},
        "max_output_tokens": options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
        "service_tier": options.get(CONF_SERVICE_TIER, DEFAULT_SERVICE_TIER),
    }
    if function_tools:
        params["tools"] = [
            {"type": "function", "strict": False, **deepcopy(tool["spec"])}
            for tool in function_tools
        ]
        # Some existing no-argument YAML functions have parameters: null.
        for tool in params["tools"]:
            if tool.get("parameters") is None:
                tool["parameters"] = {"type": "object", "properties": {}}
        params["tool_choice"] = "auto"
    if response_format is not None:
        params["text"] = {"format": {"type": "json_schema", **response_format}}
    return params


async def transform_stream(
    stream: Any,
    model: str,
    max_tokens: int,
    stats: dict[str, int],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Stream visible text; advertise actions only after a complete response.

    Completed output contains the fully assembled function arguments, avoiding
    execution of truncated or partially received JSON during a broken stream.
    """
    yield {"role": "assistant"}
    async for event in stream:
        if event.type == "response.output_text.delta":
            yield {"content": event.delta}
        elif event.type in {"response.failed", "error"}:
            error = getattr(event, "error", None)
            if error is None and hasattr(event, "response"):
                error = event.response.error
            raise HomeAssistantError(f"Responses API failed: {error}")
        elif event.type == "response.incomplete":
            details = event.response.incomplete_details
            if details and details.reason == "max_output_tokens":
                raise TokenLengthExceededError(max_tokens)
            raise HomeAssistantError(f"Incomplete Responses API response: {details}")
        elif event.type == "response.completed":
            response = event.response
            if response.usage:
                stats.update(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            output = [item.model_dump(exclude_none=True) for item in response.output]
            calls = []
            seen: set[str] = set()
            for item in output:
                if item["type"] == "function_call":
                    call_id = item["call_id"]
                    if call_id in seen or item.get("status") != "completed":
                        raise HomeAssistantError("Invalid or incomplete function call")
                    seen.add(call_id)
                    try:
                        arguments = json.loads(item["arguments"])
                        if not isinstance(arguments, dict):
                            raise ValueError("Function arguments must be an object")
                    except (ValueError, TypeError) as err:
                        raise ParseArgumentsFailed(item["arguments"]) from err
                    calls.append(
                        llm.ToolInput(
                            id=call_id,
                            tool_name=item["name"],
                            tool_args=arguments,
                            external=True,
                        )
                    )
            yield {"native": {"provider": NATIVE_KEY, "model": model, "output": output}}
            if calls:
                yield {"tool_calls": calls}
            return
    raise HomeAssistantError("Responses API stream ended before response.completed")
