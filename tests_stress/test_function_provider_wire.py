"""On-demand Function Group loads and executes a real local tool on SDK wire."""

from __future__ import annotations

import asyncio
import json

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_knowledge_provider_wire_e2e import (
    _chat_sse_tool_call,
    _chat_tool_result,
    _tool_names,
)
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _install_wire,
    _raw_client,
    _speech,
)
from tests_stress.conftest import record


@pytest.mark.parametrize("recreate", [False, True])
async def test_edited_tool_is_not_rebound_after_provider_reply(
    hass: HomeAssistant, monkeypatch, stress_trace: list[dict], recreate: bool
) -> None:
    """Provider arguments advertised against A never dispatch edited B."""
    name = "stale_provider_tool"
    original = {
        "spec": {
            "name": name,
            "description": "Original",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "template", "value_template": "ORIGINAL"},
        "enabled": True,
    }
    edited = {
        **original,
        "spec": {**original["spec"], "description": "Edited"},
        "function": {"type": "template", "value_template": "MUST_NOT_EXECUTE"},
    }
    entry = _make_entry(
        "Stale provider tool",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_FUNCTION_TOOLS: [original],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            _chat_sse_tool_call("stale-call", name, {}),
            _chat_sse_text("unexpected continuation"),
        ],
    )
    send = wire.send
    entered, release = asyncio.Event(), asyncio.Event()

    async def suspended_send(request, *args, **kwargs):
        if not wire.requests:
            entered.set()
            await release.wait()
        return await send(request, *args, **kwargs)

    monkeypatch.setattr(_raw_client(agent)._client, "send", suspended_send)
    turn = asyncio.create_task(
        conversation.async_converse(
            hass=hass,
            text="Call the tool",
            conversation_id=None,
            context=Context(),
            language="en",
            agent_id=entry.entry_id,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=10)
    subentry = next(iter(entry.subentries.values()))
    if recreate:
        hass.config_entries.async_update_subentry(
            entry, subentry, data={**subentry.data, CONF_FUNCTION_TOOLS: []}
        )
        subentry = next(iter(entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**subentry.data, CONF_FUNCTION_TOOLS: [edited]}
    )
    release.set()
    result = await asyncio.wait_for(turn, timeout=10)
    assert result.response.error_code is not None
    assert len(wire.requests) == 1
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    record(stress_trace, "summary", layer="real SDK wire", stale_tool_rejections=1)


async def test_group_load_tool_execution_result_and_session_isolation(
    hass: HomeAssistant,
    monkeypatch,
    stress_trace: list[dict],
) -> None:
    tool_name = "enhanced_local_status"
    group_id = "enhanced-status-group"
    entry = _make_entry(
        "Enhanced function execution",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_FUNCTION_TOOLS: [
                {
                    "spec": {
                        "name": tool_name,
                        "description": "Return the local status marker",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "function": {
                        "type": "template",
                        "value_template": "LOCAL-TOOL-RESULT-東京",
                    },
                    "enabled": True,
                }
            ],
            CONF_FUNCTION_GROUPS: [
                {
                    "id": group_id,
                    "name": "Enhanced status",
                    "description": "Nightly local result",
                    "loading_mode": "on_demand",
                    "functions": [tool_name],
                    "enabled": True,
                }
            ],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            _chat_sse_tool_call(
                "call-load", "load_function_groups", {"groups": [group_id]}
            ),
            _chat_sse_tool_call("call-status", tool_name, {}),
            _chat_sse_text("Status delivered"),
            _chat_sse_text("Fresh session"),
        ],
    )

    async def say(text: str, conversation_id=None):
        return await conversation.async_converse(
            hass=hass,
            text=text,
            conversation_id=conversation_id,
            context=Context(),
            language="en",
            agent_id=entry.entry_id,
        )

    first = await say("Load and execute the local status tool")
    assert _speech(first) == "Status delivered"
    assert len(wire.requests) == 3
    names = [
        _tool_names(request["body"], API_MODE_CHAT_COMPLETIONS)
        for request in wire.requests
    ]
    assert "load_function_groups" in names[0] and tool_name not in names[0]
    assert tool_name in names[1] and tool_name in names[2]
    loaded = _chat_tool_result(wire.requests[1]["body"], "call-load")
    assert loaded["status"] == "success"
    tool_message = next(
        item
        for item in wire.requests[2]["body"]["messages"]
        if item.get("role") == "tool" and item.get("tool_call_id") == "call-status"
    )
    executed = json.loads(tool_message["content"])
    assert executed["result"] == "LOCAL-TOOL-RESULT-東京"

    other = await say("Start a separate conversation")
    assert _speech(other) == "Fresh session"
    assert other.conversation_id != first.conversation_id
    assert tool_name not in _tool_names(
        wire.requests[3]["body"], API_MODE_CHAT_COMPLETIONS
    )
    record(
        stress_trace,
        "summary",
        layer="provider-wire",
        public_turns=2,
        provider_requests=4,
        actual_tool_executions=2,
        local_function_executions=1,
        template_function_executions=1,
    )


async def test_script_function_executes_local_ha_service_on_provider_wire(
    hass: HomeAssistant,
    monkeypatch,
    stress_trace: list[dict],
) -> None:
    tool_name = "enhanced_script_wire"
    entity_id = "light.enhanced_script_wire"
    calls = []

    async def turn_off(call):
        calls.append(call)

    hass.services.async_register("light", "turn_off", turn_off)
    hass.states.async_set(entity_id, "on")
    entry = _make_entry(
        "Enhanced script provider execution",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_FUNCTION_TOOLS: [
                {
                    "spec": {
                        "name": tool_name,
                        "description": "Turn off one local fixture light",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "function": {
                        "type": "script",
                        "sequence": [
                            {
                                "action": "light.turn_off",
                                "data": {"entity_id": entity_id},
                            }
                        ],
                    },
                    "enabled": True,
                }
            ],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            _chat_sse_tool_call("call-enhanced-script", tool_name, {}),
            _chat_sse_text("Script executed"),
        ],
    )
    result = await conversation.async_converse(
        hass=hass,
        text="Run the local script",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
    )
    assert _speech(result) == "Script executed"
    assert len(wire.requests) == 2
    assert tool_name in _tool_names(wire.requests[0]["body"], API_MODE_CHAT_COMPLETIONS)
    assert len(calls) == 1
    assert calls[0].data["entity_id"] == entity_id
    message = next(
        item
        for item in wire.requests[1]["body"]["messages"]
        if item.get("role") == "tool"
        and item.get("tool_call_id") == "call-enhanced-script"
    )
    assert "result" in json.loads(message["content"])
    record(
        stress_trace,
        "summary",
        layer="provider-wire",
        public_turns=1,
        provider_requests=2,
        actual_function_executions=1,
        script_function_executions=1,
        ha_service_calls=1,
    )
