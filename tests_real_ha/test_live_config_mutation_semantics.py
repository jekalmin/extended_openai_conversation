"""Acceptance for deliberate live configuration mutation semantics."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_REASONING_EFFORT,
)
from custom_components.extended_openai_conversation_responses.function_groups import (
    assemble_function_tools,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    async_management_command,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_provider_wire_e2e import (
    _ScriptedWire,
    _chat_sse_text,
    _install_wire,
    _raw_client,
    _speech,
)
from tests_real_ha.test_knowledge_provider_wire_e2e import _chat_sse_tool_call

_GROUP_ID = "live-group"
_TOOL_NAME = "live_status"
_REPLACEMENT_TOOL_NAME = "live_status_v2"


def _tool(name: str, description: str, *, enabled: bool = True) -> dict[str, Any]:
    return {
        "spec": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "template", "value_template": description},
        "enabled": enabled,
    }


def _group(functions: list[str], *, enabled: bool = True) -> dict[str, Any]:
    return {
        "id": _GROUP_ID,
        "name": "Live Group",
        "description": "Functions whose definitions are resolved live.",
        "loading_mode": "on_demand",
        "functions": functions,
        "enabled": enabled,
    }


def _names(assembly) -> set[str]:
    return {tool["spec"]["name"] for tool in assembly.tools}


def _rule(*, model: str = "gpt-6-astra", enabled: bool = True) -> dict[str, Any]:
    return {
        "name": "Persistent conversation route",
        "enabled": enabled,
        "phrases": ["use the deep route"],
        "match_type": "equals",
        "action_type": "model_routing",
        "action": {
            "model": model,
            "reasoning_effort": "xhigh",
            "scope": "conversation",
            "reset": False,
            "continue_to_ai": True,
            "success_response": "Route selected",
        },
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": 0,
    }


async def _say(
    hass: HomeAssistant,
    agent: Any,
    text: str,
    conversation_id: str | None = None,
) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass=hass,
        text=text,
        conversation_id=conversation_id,
        context=Context(),
        language="en",
        agent_id=agent.entry.entry_id,
    )



class _GatedFirstReplyWire(_ScriptedWire):
    """Expose the request before releasing the provider's first tool-call reply."""

    def __init__(self, replies: list[Any]) -> None:
        super().__init__(replies)
        self.request_sent = asyncio.Event()
        self.release_reply = asyncio.Event()
        self.first_body: dict[str, Any] | None = None

    async def send(self, request, *args: Any, **kwargs: Any):
        if not self.requests and self.first_body is None:
            import json

            self.first_body = json.loads(request.content.decode())
            self.request_sent.set()
            await self.release_reply.wait()
        return await super().send(request, *args, **kwargs)


async def _tool_revision(hass: HomeAssistant, agent: Any) -> str:
    result = await async_management_command(
        hass,
        "admin-user",
        True,
        {
            "section": "configuration",
            "action": "get",
            "entry_id": agent.entry.entry_id,
            "subentry_id": agent.subentry.subentry_id,
        },
    )
    revision = result["revision"]
    assert isinstance(revision, str)
    return revision


async def _mutate_tool(
    hass: HomeAssistant,
    agent: Any,
    *,
    action: str,
    revision: str,
    **extra: Any,
) -> dict[str, Any]:
    return await async_management_command(
        hass,
        "admin-user",
        True,
        {
            "section": "tools",
            "action": action,
            "entry_id": agent.entry.entry_id,
            "subentry_id": agent.subentry.subentry_id,
            "revision": revision,
            **extra,
        },
    )


async def _run_gated_tool_turn(
    hass: HomeAssistant,
    monkeypatch: Any,
    agent: Any,
    replies: list[bytes],
) -> tuple[Any, _GatedFirstReplyWire, asyncio.Task[Any]]:
    wire = _GatedFirstReplyWire(replies)
    monkeypatch.setattr(_raw_client(agent)._client, "send", wire.send)
    task = asyncio.create_task(_say(hass, agent, "Use the live status tool."))
    await asyncio.wait_for(wire.request_sent.wait(), timeout=10)
    assert wire.first_body is not None
    assert any(
        tool.get("function", {}).get("name") == _TOOL_NAME
        for tool in wire.first_body.get("tools", [])
    )
    return agent, wire, task

def test_loaded_group_selection_resolves_current_configuration_each_turn() -> None:
    """Loaded group IDs select live definitions rather than frozen tool snapshots."""
    loaded = {_GROUP_ID}
    initial_tool = _tool(_TOOL_NAME, "Initial live definition")
    initial = assemble_function_tools([initial_tool], [_group([_TOOL_NAME])], loaded)
    assert _names(initial) == {_TOOL_NAME}
    assert loaded == {_GROUP_ID}
    assert initial.tools[0]["spec"]["description"] == "Initial live definition"

    edited_tool = _tool(_TOOL_NAME, "Edited live definition")
    edited = assemble_function_tools([edited_tool], [_group([_TOOL_NAME])], loaded)
    assert _names(edited) == {_TOOL_NAME}
    assert loaded == {_GROUP_ID}
    assert edited.tools[0]["spec"]["description"] == "Edited live definition"

    replacement = _tool(_REPLACEMENT_TOOL_NAME, "Replacement group member")
    remapped = assemble_function_tools(
        [replacement], [_group([_REPLACEMENT_TOOL_NAME])], loaded
    )
    assert _names(remapped) == {_REPLACEMENT_TOOL_NAME}
    assert loaded == {_GROUP_ID}

    disabled_tool = _tool(_REPLACEMENT_TOOL_NAME, "Disabled member", enabled=False)
    unavailable = assemble_function_tools(
        [disabled_tool], [_group([_REPLACEMENT_TOOL_NAME])], loaded
    )
    assert _REPLACEMENT_TOOL_NAME not in _names(unavailable)
    assert loaded == set()

    loaded.add(_GROUP_ID)
    disabled_group = assemble_function_tools(
        [replacement], [_group([_REPLACEMENT_TOOL_NAME], enabled=False)], loaded
    )
    assert _REPLACEMENT_TOOL_NAME not in _names(disabled_group)
    assert loaded == set()


async def test_conversation_route_survives_rule_edit_disable_and_delete(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """A matched conversation route is copied state, not a live rule reference."""
    entry = _make_entry(
        "Live routing semantics",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "medium",
            CONF_FUNCTION_TOOLS: [],
            CONF_FUNCTION_GROUPS: [],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None

    created = await agent._request_rules.async_create(_rule())
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            _chat_sse_text("Routed first turn."),
            _chat_sse_text("Edited rule did not rewrite state."),
            _chat_sse_text("Disabled rule did not rewrite state."),
            _chat_sse_text("Deleted rule did not rewrite state."),
        ],
    )

    first = await _say(hass, agent, "use the deep route")
    assert _speech(first) == "Routed first turn."
    assert wire.requests[0]["body"]["model"] == "gpt-6-astra"
    assert wire.requests[0]["body"]["reasoning_effort"] == "xhigh"

    edited_rule = deepcopy(created)
    edited_rule["action"] = {
        **edited_rule["action"],
        "model": "gpt-5.6",
        "reasoning_effort": "medium",
    }
    await agent._request_rules.async_update(created["id"], edited_rule)
    second = await _say(hass, agent, "continue after editing", first.conversation_id)
    assert _speech(second) == "Edited rule did not rewrite state."
    assert wire.requests[1]["body"]["model"] == "gpt-6-astra"
    assert wire.requests[1]["body"]["reasoning_effort"] == "xhigh"

    disabled_rule = deepcopy(edited_rule)
    disabled_rule["enabled"] = False
    await agent._request_rules.async_update(created["id"], disabled_rule)
    third = await _say(hass, agent, "continue after disabling", first.conversation_id)
    assert _speech(third) == "Disabled rule did not rewrite state."
    assert wire.requests[2]["body"]["model"] == "gpt-6-astra"
    assert wire.requests[2]["body"]["reasoning_effort"] == "xhigh"

    await agent._request_rules.async_delete(created["id"])
    fourth = await _say(hass, agent, "continue after deleting", first.conversation_id)
    assert _speech(fourth) == "Deleted rule did not rewrite state."
    assert wire.requests[3]["body"]["model"] == "gpt-6-astra"
    assert wire.requests[3]["body"]["reasoning_effort"] == "xhigh"


async def test_config_entry_reload_resets_transient_conversation_route(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """A full agent reload is an explicit boundary for transient routing state."""
    entry = _make_entry(
        "Routing reload boundary",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "medium",
            CONF_FUNCTION_TOOLS: [],
            CONF_FUNCTION_GROUPS: [],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    await agent._request_rules.async_create(_rule())

    first_wire = _install_wire(
        monkeypatch, agent, [_chat_sse_text("Conversation route established.")]
    )
    first = await _say(hass, agent, "use the deep route")
    assert _speech(first) == "Conversation route established."
    assert first_wire.requests[0]["body"]["model"] == "gpt-6-astra"
    assert first_wire.requests[0]["body"]["reasoning_effort"] == "xhigh"

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    reloaded = conversation.async_get_agent(hass, entry.entry_id)
    assert reloaded is not None
    assert reloaded is not agent

    second_wire = _install_wire(
        monkeypatch, reloaded, [_chat_sse_text("Configured defaults restored.")]
    )
    second = await _say(hass, reloaded, "continue after reload", first.conversation_id)
    assert _speech(second) == "Configured defaults restored."
    assert second_wire.requests[0]["body"]["model"] == "gpt-5.6"
    assert second_wire.requests[0]["body"]["reasoning_effort"] == "medium"

async def test_provider_exposed_tool_disabled_before_call_fails_closed(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """A tool disabled after request serialization must not execute from that request."""
    entry = _make_entry(
        "Live tool disable race",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [_tool(_TOOL_NAME, "Initially exposed implementation")],
            CONF_FUNCTION_GROUPS: [],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None

    executed: list[dict[str, Any]] = []
    original_execute = agent._execute_function_tool

    async def record_execute(function_tool, *args):
        executed.append(deepcopy(function_tool))
        return await original_execute(function_tool, *args)

    monkeypatch.setattr(agent, "_execute_function_tool", record_execute)
    revision = await _tool_revision(hass, agent)
    _agent, wire, task = await _run_gated_tool_turn(
        hass,
        monkeypatch,
        agent,
        [_chat_sse_tool_call("call-live-disable", _TOOL_NAME, {})],
    )

    await _mutate_tool(
        hass,
        agent,
        action="set_enabled",
        revision=revision,
        name=_TOOL_NAME,
        enabled=False,
    )
    wire.release_reply.set()
    result = await task
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert result.response.error_code is not None
    assert executed == []


async def test_provider_exposed_tool_deleted_before_call_fails_closed(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """A tool deleted after request serialization must not execute from that request."""
    entry = _make_entry(
        "Live tool delete race",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [_tool(_TOOL_NAME, "Initially exposed implementation")],
            CONF_FUNCTION_GROUPS: [],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None

    executed: list[dict[str, Any]] = []
    original_execute = agent._execute_function_tool

    async def record_execute(function_tool, *args):
        executed.append(deepcopy(function_tool))
        return await original_execute(function_tool, *args)

    monkeypatch.setattr(agent, "_execute_function_tool", record_execute)
    revision = await _tool_revision(hass, agent)
    _agent, wire, task = await _run_gated_tool_turn(
        hass,
        monkeypatch,
        agent,
        [_chat_sse_tool_call("call-live-delete", _TOOL_NAME, {})],
    )

    await _mutate_tool(
        hass,
        agent,
        action="delete",
        revision=revision,
        name=_TOOL_NAME,
        confirm=True,
    )
    wire.release_reply.set()
    result = await task
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert result.response.error_code is not None
    assert executed == []


async def test_provider_tool_call_rejects_disabled_then_restored_config(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """A provider call from A cannot execute after Function Tools go A-B-A."""
    entry = _make_entry(
        "Function Tool ABA",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [_tool(_TOOL_NAME, "Original implementation")],
            CONF_FUNCTION_GROUPS: [],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    original = deepcopy(dict(agent.subentry.data))
    executed: list[dict[str, Any]] = []
    original_execute = agent._execute_function_tool

    async def record_execute(function_tool, *args):
        executed.append(deepcopy(function_tool))
        return await original_execute(function_tool, *args)

    monkeypatch.setattr(agent, "_execute_function_tool", record_execute)
    _agent, wire, task = await _run_gated_tool_turn(
        hass,
        monkeypatch,
        agent,
        [_chat_sse_tool_call("call-aba", _TOOL_NAME, {})],
    )
    disabled = deepcopy(original)
    disabled_tools = deepcopy(disabled[CONF_FUNCTION_TOOLS])
    disabled_tools[0]["enabled"] = False
    disabled[CONF_FUNCTION_TOOLS] = disabled_tools
    hass.config_entries.async_update_subentry(entry, agent.subentry, data=disabled)
    hass.config_entries.async_update_subentry(entry, agent.subentry, data=original)
    assert dict(agent.subentry.data) == original
    wire.release_reply.set()
    result = await task
    assert result.response.error_code is not None
    assert executed == []

    await hass.async_block_till_done()
    fresh_agent = conversation.async_get_agent(hass, entry.entry_id)
    assert fresh_agent is not None
    fresh_wire = _install_wire(
        monkeypatch, fresh_agent, [_chat_sse_text("Fresh request succeeded.")]
    )
    fresh = await _say(hass, fresh_agent, "try again")
    assert _speech(fresh) == "Fresh request succeeded."
    assert len(fresh_wire.requests) == 1
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def test_provider_exposed_tool_edit_uses_latest_definition_before_execution(
    hass: HomeAssistant,
    monkeypatch: Any,
) -> None:
    """An old exposed schema resolves to the current implementation at dispatch."""
    entry = _make_entry(
        "Live tool edit race",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [_tool(_TOOL_NAME, "Old implementation")],
            CONF_FUNCTION_GROUPS: [],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None

    executed: list[dict[str, Any]] = []
    original_execute = agent._execute_function_tool

    async def record_execute(function_tool, *args):
        executed.append(deepcopy(function_tool))
        return await original_execute(function_tool, *args)

    monkeypatch.setattr(agent, "_execute_function_tool", record_execute)
    revision = await _tool_revision(hass, agent)
    _agent, wire, task = await _run_gated_tool_turn(
        hass,
        monkeypatch,
        agent,
        [
            _chat_sse_tool_call("call-live-edit", _TOOL_NAME, {}),
            _chat_sse_text("Latest implementation used."),
        ],
    )

    edited = _tool(_TOOL_NAME, "New implementation")
    await _mutate_tool(
        hass,
        agent,
        action="save",
        revision=revision,
        tool=edited,
        original_name=_TOOL_NAME,
    )
    wire.release_reply.set()
    result = await task
    await hass.async_block_till_done()
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert _speech(result) == "Latest implementation used."
    assert len(executed) == 1
    assert executed[0]["function"]["value_template"] == "New implementation"
