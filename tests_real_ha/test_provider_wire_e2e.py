"""End-to-end provider-wire acceptance through Home Assistant's public API."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import httpx
import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    DEFAULT_CONF_FUNCTION_TOOLS,
)
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry

_ENTITY_ID = "light.provider_wire"
_TOOL_CALL_ID = "call-provider-wire"
_TOOL_ARGUMENTS = {
    "list": [
        {
            "domain": "light",
            "service": "turn_off",
            "service_data": {"entity_id": [_ENTITY_ID]},
        }
    ]
}


def _chat_sse_tool_call(
    call_id: str = _TOOL_CALL_ID,
    name: str = "execute_services",
    arguments: dict[str, Any] | None = None,
) -> bytes:
    arguments = _TOOL_ARGUMENTS if arguments is None else arguments
    chunk = {
        "id": "chatcmpl-provider-wire-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5.6",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    arguments, separators=(",", ":")
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def _chat_sse_text(text: str) -> bytes:
    chunk = {
        "id": "chatcmpl-provider-wire-2",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5.6",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def _response_object(response_id: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 500,
        "model": "gpt-5.6",
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": None,
    }


def _responses_sse_tool_call(
    call_id: str = _TOOL_CALL_ID,
    name: str = "execute_services",
    tool_arguments: dict[str, Any] | None = None,
) -> bytes:
    arguments = json.dumps(
        _TOOL_ARGUMENTS if tool_arguments is None else tool_arguments,
        separators=(",", ":"),
    )
    item = {
        "id": "fc-provider-wire",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "arguments": "", "status": "in_progress"},
            "sequence_number": 0,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
            "sequence_number": 1,
        },
        {
            "type": "response.completed",
            "response": _response_object("resp-provider-wire-1", [item]),
            "sequence_number": 2,
        },
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def _responses_sse_text(text: str) -> bytes:
    item = {
        "id": "msg-provider-wire",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []},
            "sequence_number": 0,
        },
        {
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": text,
            "item_id": item["id"],
            "logprobs": [],
            "output_index": 0,
            "sequence_number": 1,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
            "sequence_number": 2,
        },
        {
            "type": "response.completed",
            "response": _response_object("resp-provider-wire-2", [item]),
            "sequence_number": 3,
        },
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


class _ScriptedWire:
    """Intercept only the SDK's outbound HTTP call and return wire-format replies."""

    def __init__(self, replies: list[bytes | tuple[int, dict[str, Any]]]) -> None:
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def send(
        self, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        del args, kwargs
        body = json.loads(request.content.decode())
        self.requests.append({"path": request.url.path, "body": body})
        index = len(self.requests) - 1
        assert index < len(self._replies), "Unexpected extra OpenAI SDK request"
        reply = self._replies[index]
        if isinstance(reply, tuple):
            status, payload = reply
            return httpx.Response(
                status,
                headers={"content-type": "application/json"},
                json=payload,
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=reply,
            request=request,
        )


async def _agent(hass: HomeAssistant, api_mode: str):
    tool = deepcopy(DEFAULT_CONF_FUNCTION_TOOLS[0])
    entry = _make_entry(
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: api_mode,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [tool],
        },
    )
    await _setup_entry(hass, entry)
    return conversation.async_get_agent(hass, entry.entry_id)


def _raw_client(agent: Any) -> Any:
    """Unwrap integration instrumentation while leaving the real SDK intact."""
    client = agent._client
    while hasattr(client, "_delegate"):
        client = client._delegate
    return client


def _install_wire(monkeypatch: Any, agent: Any, replies: list[Any]) -> _ScriptedWire:
    wire = _ScriptedWire(replies)
    # This is deliberately below responses.create/chat.completions.create. The SDK
    # must still serialize the request, choose the endpoint and parse the SSE stream.
    monkeypatch.setattr(_raw_client(agent)._client, "send", wire.send)
    return wire


@pytest.mark.parametrize("api_mode", [API_MODE_RESPONSES, API_MODE_CHAT_COMPLETIONS])
async def test_complex_function_schema_survives_real_sdk_request_wire(
    hass: HomeAssistant, monkeypatch: Any, api_mode: str
) -> None:
    """Only HTTP transport is mocked; the SDK serializes the complete schema."""
    schema = {
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string", "description": "Lumière 東京"},
                        "brightness": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                            "maximum": 255,
                        },
                        "mode": {
                            "type": "string",
                            "enum": [f"mode_{index}" for index in range(24)],
                        },
                    },
                    "required": ["entity_id"],
                    "additionalProperties": False,
                },
            },
            "metadata": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        "required": ["targets"],
        "additionalProperties": False,
    }
    tool = {
        "spec": {
            "name": "complex_schema_wire",
            "description": "Nested actions for Éireann 東京",
            "parameters": schema,
        },
        "function": {"type": "template", "value_template": "ok"},
        "enabled": True,
    }
    entry = _make_entry(
        "Complex schema wire",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: api_mode,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [tool],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            _responses_sse_text("Schema sent")
            if api_mode == API_MODE_RESPONSES
            else _chat_sse_text("Schema sent")
        ],
    )
    result = await _say(hass, agent)
    assert _speech(result) == "Schema sent"
    body = wire.requests[0]["body"]
    emitted = next(
        item
        for item in body["tools"]
        if (
            item.get("name")
            if api_mode == API_MODE_RESPONSES
            else item.get("function", {}).get("name")
        )
        == "complex_schema_wire"
    )
    assert (
        emitted["parameters"]
        if api_mode == API_MODE_RESPONSES
        else emitted["function"]["parameters"]
    ) == schema


@pytest.mark.parametrize("api_mode", [API_MODE_RESPONSES, API_MODE_CHAT_COMPLETIONS])
@pytest.mark.parametrize(
    ("service_data", "expected_brightness"),
    [
        ({"entity_id": [_ENTITY_ID], "brightness": 100}, 100),
        ({"entity_id": [_ENTITY_ID], "brightness": "100"}, 100),
        ({"entity_id": [_ENTITY_ID], "brightness": None}, None),
        ({"entity_id": [_ENTITY_ID], "brightness": [100]}, None),
        ({"entity_id": [_ENTITY_ID], "brightness": 256}, None),
        ({"entity_id": [_ENTITY_ID], "brightness": 100, "unknown": True}, None),
        (None, None),
        ([], None),
    ],
)
async def test_provider_argument_shapes_gate_real_ha_service(
    hass: HomeAssistant,
    monkeypatch: Any,
    api_mode: str,
    service_data: Any,
    expected_brightness: int | None,
) -> None:
    """The SDK JSON, configured schema, and HA service see one typed boundary."""
    tool = deepcopy(DEFAULT_CONF_FUNCTION_TOOLS[0])
    service_schema = tool["spec"]["parameters"]["properties"]["list"]["items"][
        "properties"
    ]["service_data"]
    service_schema["properties"]["brightness"] = {
        "type": "integer",
        "minimum": 0,
        "maximum": 255,
    }
    service_schema["additionalProperties"] = False
    entry = _make_entry(
        "Typed service wire",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: api_mode,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [tool],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    calls = []

    async def turn_on(call: Any) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_on", turn_on)
    hass.states.async_set(_ENTITY_ID, "off")
    async_expose_entity(hass, conversation.DOMAIN, _ENTITY_ID, True)
    arguments = {
        "list": [
            {
                "domain": "light",
                "service": "turn_on",
                "service_data": service_data,
            }
        ]
    }
    first = (
        _responses_sse_tool_call(tool_arguments=arguments)
        if api_mode == API_MODE_RESPONSES
        else _chat_sse_tool_call(arguments=arguments)
    )
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            first,
            (
                _responses_sse_text("Done")
                if api_mode == API_MODE_RESPONSES
                else _chat_sse_text("Done")
            ),
        ],
    )
    await _say(hass, agent)
    if expected_brightness is None:
        assert calls == []
    else:
        assert len(calls) == 1
        assert calls[0].data["brightness"] == expected_brightness
        assert type(calls[0].data["brightness"]) is int
    assert wire.requests[0]["path"] == (
        "/v1/responses" if api_mode == API_MODE_RESPONSES else "/v1/chat/completions"
    )


async def _prepare_service(hass: HomeAssistant) -> list[Any]:
    calls: list[Any] = []

    async def turn_off(call: Any) -> None:
        calls.append(call)

    hass.services.async_register("light", "turn_off", turn_off)
    hass.states.async_set(_ENTITY_ID, "on", {"friendly_name": "Provider Wire"})
    async_expose_entity(hass, conversation.DOMAIN, _ENTITY_ID, True)
    return calls


async def _say(hass: HomeAssistant, agent: Any) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass=hass,
        text="Turn off the provider wire test light",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=agent.entry.entry_id,
    )


def _speech(result: conversation.ConversationResult) -> str:
    assert result.response.error_code is None
    return result.response.as_dict()["speech"]["plain"]["speech"]


def _tool_result_from_chat_request(request: dict[str, Any]) -> dict[str, Any]:
    tool_message = next(
        item for item in request["messages"] if item.get("role") == "tool"
    )
    assert tool_message["tool_call_id"] == _TOOL_CALL_ID
    return json.loads(tool_message["content"])


def _tool_result_from_responses_request(request: dict[str, Any]) -> dict[str, Any]:
    tool_item = next(
        item for item in request["input"] if item.get("type") == "function_call_output"
    )
    assert tool_item["call_id"] == _TOOL_CALL_ID
    return json.loads(tool_item["output"])


async def test_chat_completions_real_sdk_wire_executes_ha_service_once(
    hass: HomeAssistant, monkeypatch: Any
) -> None:
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    calls = await _prepare_service(hass)
    wire = _install_wire(
        monkeypatch,
        agent,
        [_chat_sse_tool_call(), _chat_sse_text("The test light is off.")],
    )

    result = await _say(hass, agent)

    assert _speech(result) == "The test light is off."
    assert len(calls) == 1
    assert calls[0].data["entity_id"] == [_ENTITY_ID]
    assert [request["path"] for request in wire.requests] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert wire.requests[0]["body"]["stream"] is True
    assert any(
        tool["function"]["name"] == "execute_services"
        for tool in wire.requests[0]["body"]["tools"]
    )
    tool_result = _tool_result_from_chat_request(wire.requests[1]["body"])
    assert tool_result["result"][0]["success"] is True


async def test_responses_real_sdk_wire_executes_ha_service_once(
    hass: HomeAssistant, monkeypatch: Any
) -> None:
    agent = await _agent(hass, API_MODE_RESPONSES)
    calls = await _prepare_service(hass)
    wire = _install_wire(
        monkeypatch,
        agent,
        [_responses_sse_tool_call(), _responses_sse_text("The test light is off.")],
    )

    result = await _say(hass, agent)

    assert _speech(result) == "The test light is off."
    assert len(calls) == 1
    assert calls[0].data["entity_id"] == [_ENTITY_ID]
    assert [request["path"] for request in wire.requests] == [
        "/v1/responses",
        "/v1/responses",
    ]
    assert wire.requests[0]["body"]["stream"] is True
    assert any(
        tool["type"] == "function" and tool["name"] == "execute_services"
        for tool in wire.requests[0]["body"]["tools"]
    )
    tool_result = _tool_result_from_responses_request(wire.requests[1]["body"])
    assert tool_result["result"][0]["success"] is True


async def test_second_provider_request_failure_does_not_repeat_side_effect(
    hass: HomeAssistant, monkeypatch: Any
) -> None:
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    calls = await _prepare_service(hass)
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            _chat_sse_tool_call(),
            (
                400,
                {
                    "error": {
                        "message": "provider-wire failure after tool execution",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "provider_wire_test",
                    }
                },
            ),
        ],
    )

    result = await _say(hass, agent)

    assert result.response.error_code is not None
    assert (
        "problem talking to OpenAI"
        in result.response.as_dict()["speech"]["plain"]["speech"]
    )
    assert len(calls) == 1
    assert calls[0].data["entity_id"] == [_ENTITY_ID]
    assert len(wire.requests) == 2
    # The failed second request reached the SDK only after the integration had
    # serialized the real tool result; the side effect is never replayed/retried.
    tool_result = _tool_result_from_chat_request(wire.requests[1]["body"])
    assert tool_result["result"][0]["success"] is True
