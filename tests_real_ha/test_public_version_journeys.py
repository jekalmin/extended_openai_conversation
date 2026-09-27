"""Compact public EOAI journeys shared by supported Home Assistant versions."""

from copy import deepcopy
from unittest.mock import AsyncMock, patch

from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses.const import (
    CONF_API_MODE,
    CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_MEMORY_MODE,
    CONF_REASONING_EFFORT,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_CONF_FUNCTION_TOOLS,
    DOMAIN,
    MEMORY_MODE_MANUAL,
)
from custom_components.extended_openai_conversation_responses.memory import (
    async_get_memory,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
)
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_backup_transfer_protocol import (
    _download_archive,
    _transfer_call,
    _upload_archive,
    _user_token,
)
from tests_real_ha.test_management_backend_acceptance import (
    _admin_client,
    _fresh_reload,
    _management_call,
)
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _chat_sse_tool_call,
    _install_wire,
    _speech,
)


async def _say(hass: HomeAssistant, entry, text: str, *, user_id=None):
    return await conversation.async_converse(
        hass=hass,
        text=text,
        conversation_id=None,
        context=Context(user_id=user_id),
        language="en",
        agent_id=entry.entry_id,
    )


def _agent(hass: HomeAssistant, entry):
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    return agent


async def test_config_flow_basic_assist_and_healthy_reload(hass, monkeypatch):
    """A public config flow creates an agent that serves Assist across reload."""
    with patch(
        "custom_components.extended_openai_conversation_responses.config_flow.get_authenticated_client",
        new_callable=AsyncMock,
    ):
        started = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        assert started["type"] is FlowResultType.FORM
        created = await hass.config_entries.flow.async_configure(
            started["flow_id"],
            {CONF_API_KEY: "sk-public-version-journey", CONF_SKIP_AUTHENTICATION: True},
        )
    assert created["type"] is FlowResultType.CREATE_ENTRY
    entry = created["result"]
    subentry = next(
        item
        for item in entry.subentries.values()
        if item.subentry_type == "conversation"
    )
    hass.config_entries.async_update_subentry(
        entry,
        subentry,
        data={
            **subentry.data,
            CONF_API_MODE: "chat_completions",
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "none",
        },
    )
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED

    wire = _install_wire(monkeypatch, _agent(hass, entry), [_chat_sse_text("Ready")])
    assert _speech(await _say(hass, entry, "Hello")) == "Ready"
    assert len(wire.requests) == 1
    assert wire.requests[0]["path"] == "/v1/chat/completions"

    assert await hass.config_entries.async_unload(entry.entry_id)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    again = _install_wire(
        monkeypatch, _agent(hass, entry), [_chat_sse_text("Ready again")]
    )
    assert _speech(await _say(hass, entry, "Hello again")) == "Ready again"
    assert len(again.requests) == 1


async def test_function_tool_side_effect_and_provider_continuation(hass, monkeypatch):
    """The SDK tool call reaches one HA service and gets a continuation."""
    tool = deepcopy(DEFAULT_CONF_FUNCTION_TOOLS[0])
    entry = _make_entry(
        "Public Function Journey",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: "chat_completions",
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "none",
            CONF_FUNCTION_TOOLS: [tool],
        },
    )
    await _setup_entry(hass, entry)
    calls = []

    async def turn_off(call):
        calls.append(call)

    entity_id = "light.public_version_journey"
    hass.services.async_register("light", "turn_off", turn_off)
    hass.states.async_set(entity_id, "on")
    async_expose_entity(hass, conversation.DOMAIN, entity_id, True)
    wire = _install_wire(
        monkeypatch,
        _agent(hass, entry),
        [
            _chat_sse_tool_call(
                arguments={
                    "list": [
                        {
                            "domain": "light",
                            "service": "turn_off",
                            "service_data": {"entity_id": [entity_id]},
                        }
                    ]
                }
            ),
            _chat_sse_text("Light handled"),
        ],
    )
    assert _speech(await _say(hass, entry, "Turn off the light")) == "Light handled"
    assert len(calls) == 1
    assert calls[0].data["entity_id"] == [entity_id]
    assert len(wire.requests) == 2
    assert all(item["path"] == "/v1/chat/completions" for item in wire.requests)


async def test_management_save_durable_rule_and_conversation_after_reload(
    hass, hass_ws_client, monkeypatch
):
    """Management updates and a durable local rule remain usable after reload."""
    entry = _make_entry(
        "Public Management Journey",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: "chat_completions",
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "none",
        },
    )
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)
    before = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    saved = await _management_call(
        client,
        entry=entry,
        section="configuration",
        action="update",
        revision=before["revision"],
        config={CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES: 47},
    )
    assert saved["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] == 47
    calls = []

    async def record(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "record", record)
    await _management_call(
        client,
        entry=entry,
        section="request_rules",
        action="create",
        rule={
            "name": "Public durable rule",
            "enabled": True,
            "phrases": ["public local action"],
            "match_type": "equals",
            "action_type": "local_action",
            "action": {
                "actions": [
                    {"action": "rule_probe.record", "data": {"message": "persisted"}}
                ],
                "success_response": "Rule retained",
                "failure_response": "Rule failed",
            },
            "matching_behavior": "defaults",
            "matching": dict(DEFAULT_MATCHING),
            "order": 0,
        },
    )
    await _fresh_reload(hass, entry)
    effective = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    assert effective["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] == 47
    guard = _install_wire(monkeypatch, _agent(hass, entry), [])
    assert _speech(await _say(hass, entry, "public local action")) == "Rule retained"
    assert calls == ["persisted"]
    assert guard.requests == []
    wire = _install_wire(
        monkeypatch, _agent(hass, entry), [_chat_sse_text("Still conversing")]
    )
    assert _speech(await _say(hass, entry, "Normal Assist")) == "Still conversing"
    assert len(wire.requests) == 1


async def test_backup_restore_privacy_boundary_and_assist(
    hass, hass_ws_client, monkeypatch
):
    """An admin restores durable state; a regular user cannot export it."""
    entry = _make_entry(
        "Public Backup Journey",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: "chat_completions",
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "none",
            CONF_MEMORY_MODE: MEMORY_MODE_MANUAL,
        },
    )
    await _setup_entry(hass, entry)
    subentry = next(
        item
        for item in entry.subentries.values()
        if item.subentry_type == "conversation"
    )
    admin = await _admin_client(hass, hass_ws_client, user_id="version-admin")
    normal = MockUser(id="version-regular", name="Version Regular")
    normal_client = await hass_ws_client(hass, await _user_token(hass, normal))
    denied = await _transfer_call(
        normal_client, entry=entry, action="export_start", data={"mode": "full"}
    )
    assert denied["success"] is False
    assert denied["error"]["code"] == "unauthorized"

    memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    owner = "user:version-admin"
    created = await memory.async_add(
        owner, "Original durable marker", "test", "explicit"
    )
    original_id = created["memory"]["memory_id"]
    archive, metadata = await _download_archive(admin, entry=entry, mode="full")
    assert await memory.async_delete(owner, [original_id]) == 1
    await memory.async_add(owner, "Mutated marker", "test", "explicit")
    session = await _upload_archive(
        admin, entry=entry, archive=archive, filename=metadata["filename"]
    )
    inspected = await _transfer_call(
        admin, entry=entry, action="import_inspect", data={"session_id": session}
    )
    assert inspected["success"] and inspected["result"]["valid"]
    restored = await _transfer_call(
        admin,
        entry=entry,
        action="import_restore",
        data={
            "session_id": session,
            "preview_token": inspected["result"]["preview_token"],
        },
    )
    assert restored["success"] and restored["result"]["status"] == "restored"
    current = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    assert [
        (item.memory_id, item.content) for item in await current.async_list(owner)
    ] == [(original_id, "Original durable marker")]
    wire = _install_wire(
        monkeypatch, _agent(hass, entry), [_chat_sse_text("Restored and ready")]
    )
    assert _speech(await _say(hass, entry, "Are you ready?")) == "Restored and ready"
    assert len(wire.requests) == 1
