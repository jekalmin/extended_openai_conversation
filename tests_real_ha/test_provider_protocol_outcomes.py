"""Malformed SDK streams yield handled errors on both public processing paths."""

import json

import httpx
import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    DOMAIN,
)
from custom_components.extended_openai_conversation_responses.debug import (
    get_debug_manager,
)
from tests_real_ha.test_provider_wire_e2e import (
    _agent,
    _chat_sse_text,
    _install_wire,
    _responses_sse_text,
    _say,
)


def _duplicated_native_content():
    events = _responses_sse_text("Invalid repeated content").decode().split("\n\n")
    result = []
    for event in events:
        if not event:
            continue
        result.append(event)
        if (
            json.loads(event.removeprefix("data: "))["type"]
            == "response.output_item.done"
        ):
            result.append(event)
    return ("\n\n".join(result) + "\n\n").encode()


@pytest.mark.parametrize(
    "api_mode,body",
    [
        (API_MODE_CHAT_COMPLETIONS, b"data: {invalid\n\n"),
        (API_MODE_RESPONSES, b"data: {invalid\n\n"),
        (API_MODE_RESPONSES, _duplicated_native_content()),
        (
            API_MODE_RESPONSES,
            _responses_sse_text("Duplicate terminal")
            + _responses_sse_text("Duplicate terminal"),
        ),
    ],
    ids=["chat-json", "responses-json", "responses-duplicate", "responses-terminal"],
)
@pytest.mark.parametrize("route", ["assist", "process"])
async def test_protocol_error_is_handled_and_next_request_recovers(
    hass, monkeypatch, api_mode, body, route
):
    agent = await _agent(hass, api_mode)
    debug = get_debug_manager(hass, agent.entry.entry_id, agent.subentry.subentry_id)
    debug.configure(enabled=True)
    good = (
        _responses_sse_text("Healthy next request")
        if api_mode == API_MODE_RESPONSES
        else _chat_sse_text("Healthy next request")
    )
    wire = _install_wire(monkeypatch, agent, [body, good])

    async def turn():
        if route == "assist":
            result = await _say(hass, agent)
            return result.response.error_code is not None, result.response.as_dict()[
                "speech"
            ]["plain"]["speech"]
        result = await hass.services.async_call(
            DOMAIN,
            "process",
            {"agent_id": agent.entity_id, "text": "hello"},
            blocking=True,
            return_response=True,
        )
        return result.get("error") is not None, result["response"]

    failed, speech = await turn()
    assert "malformed data" in speech
    if route == "assist":
        assert failed
    assert not agent._usage.runs[-1].successful
    assert not debug.summaries()[0]["successful"]
    assert debug.summaries()[0]["error_type"] == "ProviderStreamError"
    failed, speech = await turn()
    assert speech == "Healthy next request"
    assert not failed
    assert agent._usage.runs[-1].successful
    assert debug.summaries()[0]["successful"]
    assert len(wire.requests) == 2


@pytest.mark.parametrize("api_mode", [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES])
async def test_invalid_history_argument_has_one_failed_terminal_outcome(
    hass, monkeypatch, api_mode
):
    from custom_components.extended_openai_conversation_responses.const import (
        CONF_API_MODE,
        CONF_CHAT_MODEL,
        CONF_FUNCTION_TOOL_ERROR_RECOVERY,
        CONF_FUNCTION_TOOLS,
    )
    from homeassistant.components import conversation
    from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
    from tests_real_ha.test_provider_wire_e2e import (
        _chat_sse_tool_call,
        _responses_sse_tool_call,
    )

    tool = {
        "spec": {
            "name": "read_history",
            "description": "Read history",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string"},
                    "entity_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "function": {"type": "native", "name": "get_history"},
        "enabled": True,
    }
    entry = _make_entry(
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: api_mode,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [tool],
            CONF_FUNCTION_TOOL_ERROR_RECOVERY: True,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    debug = get_debug_manager(hass, entry.entry_id, agent.subentry.subentry_id)
    debug.configure(enabled=True)
    arguments = {"start_time": "not-a-date", "entity_ids": ["sensor.history_probe"]}
    body = (
        _chat_sse_tool_call(name="read_history", arguments=arguments)
        if api_mode == API_MODE_CHAT_COMPLETIONS
        else _responses_sse_tool_call(name="read_history", tool_arguments=arguments)
    )
    _install_wire(monkeypatch, agent, [body])
    result = await _say(hass, agent)
    assert result.response.error_code is not None
    assert (
        "start_time must be an ISO 8601 datetime"
        in result.response.as_dict()["speech"]["plain"]["speech"]
    )
    assert not agent._usage.runs[-1].successful
    assert not debug.summaries()[0]["successful"]
    assert debug.summaries()[0]["error_type"] == agent._usage.runs[-1].error_type


@pytest.mark.parametrize("api_mode", [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES])
async def test_partial_stream_disconnect_is_failed_in_debug_and_usage(
    hass, monkeypatch, api_mode
):
    from tests.test_openai_sdk_malformed_wire import (
        _chat_partial_text_stream,
        _responses_partial_text_stream,
    )
    from tests_real_ha.test_provider_wire_e2e import _raw_client

    agent = await _agent(hass, api_mode)
    debug = get_debug_manager(hass, agent.entry.entry_id, agent.subentry.subentry_id)
    debug.configure(enabled=True)

    class DisconnectedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                _chat_partial_text_stream()
                if api_mode == API_MODE_CHAT_COMPLETIONS
                else _responses_partial_text_stream()
            )
            raise httpx.ReadError("provider disconnected after partial content")

    async def send(request, *args, **kwargs):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=DisconnectedStream(),
            request=request,
        )

    monkeypatch.setattr(_raw_client(agent)._client, "send", send)
    result = await _say(hass, agent)
    assert result.response.error_code is not None
    assert not agent._usage.runs[-1].successful
    assert not debug.summaries()[0]["successful"]
    assert debug.summaries()[0]["error_type"] == agent._usage.runs[-1].error_type
