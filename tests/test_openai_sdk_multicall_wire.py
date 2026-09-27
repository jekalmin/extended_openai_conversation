"""Multiple provider calls stay paired through real OpenAI SDK SSE parsing."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
)
from tests.test_openai_sdk_wire import (
    _chat_chunk,
    _chat_log,
    _chat_text_stream,
    _client,
    _close_client,
    _entity,
    _json_body,
    _response_object,
    _responses_text_stream,
    _sse,
    _stream_response,
    _tool,
    _tool_result,
    _Wire,
)

_CALLS = (
    ("call-multi-a", "light.kitchen"),
    ("call-multi-b", "light.hall"),
)


def _responses_calls() -> bytes:
    events: list[dict[str, Any]] = []
    output = []
    for index, (call_id, entity_id) in enumerate(_CALLS):
        arguments = json.dumps({"entity_id": entity_id})
        item = {
            "id": f"fc-multi-{index}",
            "type": "function_call",
            "call_id": call_id,
            "name": "get_state",
            "arguments": arguments,
            "status": "completed",
        }
        output.append(item)
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "arguments": "", "status": "in_progress"},
                    "sequence_number": len(events),
                },
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": index,
                    "item_id": item["id"],
                    "name": item["name"],
                    "arguments": arguments,
                    "sequence_number": len(events) + 1,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item,
                    "sequence_number": len(events) + 2,
                },
            ]
        )
    events.append(
        {
            "type": "response.completed",
            "response": _response_object("resp-multi", output),
            "sequence_number": len(events),
        }
    )
    return _sse(events)


def _chat_calls() -> bytes:
    return _sse(
        [
            _chat_chunk(
                delta={
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "get_state",
                                "arguments": json.dumps({"entity_id": entity_id}),
                            },
                        }
                        for index, (call_id, entity_id) in enumerate(_CALLS)
                    ],
                },
                finish_reason="tool_calls",
            )
        ]
    )


@pytest.mark.parametrize("api_mode", [API_MODE_RESPONSES, API_MODE_CHAT_COMPLETIONS])
async def test_two_sdk_wire_calls_keep_ids_arguments_and_outputs_paired(
    hass, api_mode: str
) -> None:
    first = _responses_calls() if api_mode == API_MODE_RESPONSES else _chat_calls()
    second = (
        _responses_text_stream("Both done")
        if api_mode == API_MODE_RESPONSES
        else _chat_text_stream("Both done")
    )
    wire = _Wire([_stream_response(first), _stream_response(second)])
    client = _client(wire)
    entity = _entity(hass, client, api_mode)
    executed: list[tuple[str, str]] = []

    async def execute(_tool_definition, tool_input, _context, _exposed):
        executed.append((tool_input.id, tool_input.tool_args["entity_id"]))
        return _tool_result(entity, tool_input, tool_input.tool_args["entity_id"])

    entity._execute_function_tool = AsyncMock(side_effect=execute)
    chat_log = _chat_log(hass)
    try:
        await entity._async_handle_chat_log(chat_log, [_tool()], [])
    finally:
        await _close_client(client)
    assert executed == list(_CALLS)
    assert len(wire.requests) == 2
    continuation = _json_body(wire.requests[1])
    if api_mode == API_MODE_RESPONSES:
        outputs = [
            item
            for item in continuation["input"]
            if item.get("type") == "function_call_output"
        ]
        observed = [
            (item["call_id"], json.loads(item["output"])["result"]) for item in outputs
        ]
    else:
        outputs = [
            item for item in continuation["messages"] if item.get("role") == "tool"
        ]
        observed = [
            (item["tool_call_id"], json.loads(item["content"])["result"])
            for item in outputs
        ]
    assert observed == list(_CALLS)
    assert chat_log.content[-1].content == "Both done"
