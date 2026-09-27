"""Real Home Assistant acceptance tests for the management WebSocket backend."""

from __future__ import annotations

from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import (
    CLIENT_ID,
    MockConfigEntry,
    MockUser,
)

from custom_components.extended_openai_conversation_responses import (
    memory as memory_module,
    request_rules as request_rules_module,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    CONF_MEMORY_MODE,
    CONF_SKIP_AUTHENTICATION,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
    FUNCTION_GROUP_LOADING_ON_DEMAND,
    MEMORY_MODE_MANUAL,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    WS_COMMAND,
)
from homeassistant.components.frontend import DATA_PANELS
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

ADMIN_ID = "management-acceptance-admin"


@pytest.mark.asyncio
async def test_only_unified_memory_and_knowledge_commands_are_registered(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """Real startup leaves legacy APIs unreachable and preserves Knowledge CRUD."""
    entry = _entry("Unified Data Acceptance")
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)

    assert "extended-openai-memory" not in hass.data[DATA_PANELS]
    assert "extended-openai" in hass.data[DATA_PANELS]
    registered_routes = "\n".join(
        str(resource) for resource in hass.http.app.router.resources()
    )
    assert "memory-panel.js" not in registered_routes
    assert "memory-management-panel.js" not in registered_routes

    for legacy_command in ("manage", "knowledge"):
        await client.send_json_auto_id(
            {"type": f"{DOMAIN}/{legacy_command}", "action": "agents"}
        )
        response = await client.receive_json()
        assert response["success"] is False
        assert response["error"]["code"] == "unknown_command"

    memories = await _management_call(
        client, entry=entry, section="memories", action="list"
    )
    assert memories["memories"] == []
    await _management_call(
        client,
        entry=entry,
        section="memories",
        action="add",
        content="Bins go out Friday",
        category="home",
        importance="high",
        subject="Bins",
        key="bins.day",
        valid_from="2026-09-01T00:00:00Z",
    )
    listed_memory = (
        await _management_call(client, entry=entry, section="memories", action="list")
    )["memories"][0]
    updated_memory = await _management_call(
        client,
        entry=entry,
        section="memories",
        action="update",
        memory_id=listed_memory["memory_id"],
        content="Bins go out Thursday",
        expected_revision=listed_memory["revision"],
        refresh_confirmation=False,
        clear_fields=["subject"],
    )
    assert updated_memory["memory"]["subject"] is None
    assert (
        updated_memory["memory"]["last_confirmed_at"]
        == listed_memory["last_confirmed_at"]
    )
    conflict = await _management_response(
        client,
        entry=entry,
        section="memories",
        action="update",
        memory_id=listed_memory["memory_id"],
        content="Stale edit",
        expected_revision=listed_memory["revision"],
    )
    assert conflict["success"] is False
    assert "changed since it was loaded" in conflict["error"]["message"]
    created = await _management_call(
        client,
        entry=entry,
        section="knowledge",
        action="create",
        title="Manual",
        content="Original content",
        enabled=False,
    )
    source_id = created["source"]["source_id"]
    listed = await _management_call(
        client, entry=entry, section="knowledge", action="list"
    )
    assert listed["stats"]["knowledge_source_count"] == 1
    assert listed["stats"]["knowledge_enabled_source_count"] == 0
    assert listed["sources"][0]["enabled"] is False
    assert "content" not in listed["sources"][0]
    await _management_call(
        client,
        entry=entry,
        section="knowledge",
        action="update",
        source_id=source_id,
        content="Updated content",
        enabled=True,
    )
    fetched = await _management_call(
        client,
        entry=entry,
        section="knowledge",
        action="get",
        source_id=source_id,
    )
    assert fetched["source"]["content"] == "Updated content"
    assert fetched["source"]["enabled"] is True

    normal_user = MockUser(id="knowledge-non-admin", is_owner=False)
    assert normal_user.is_admin is False
    normal_user.add_to_hass(hass)
    token = await hass.auth.async_create_refresh_token(normal_user, CLIENT_ID)
    normal_client = await hass_ws_client(
        hass, hass.auth.async_create_access_token(token)
    )
    for action in ("list", "get", "create", "update", "delete"):
        response = await _management_response(
            normal_client,
            entry=entry,
            section="knowledge",
            action=action,
            source_id=source_id,
            confirm=True,
        )
        assert response["success"] is False
        assert "Administrator permission" in response["error"]["message"]

    unconfirmed = await _management_response(
        client,
        entry=entry,
        section="knowledge",
        action="delete",
        source_id=source_id,
    )
    assert unconfirmed["success"] is False
    assert "confirmation" in unconfirmed["error"]["message"]
    deleted = await _management_call(
        client,
        entry=entry,
        section="knowledge",
        action="delete",
        source_id=source_id,
        confirm=True,
    )
    assert deleted["deleted"] == 1
    assert deleted["stats"]["knowledge_source_count"] == 0


def _entry(title: str = "Management Backend Acceptance") -> MockConfigEntry:
    """Build one local-only conversation entry for management acceptance tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data={
            CONF_API_KEY: "sk-management-backend-acceptance",
            CONF_SKIP_AUTHENTICATION: True,
        },
        version=CONFIG_ENTRY_VERSION,
        subentries_data=[
            {
                "data": {CONF_MEMORY_MODE: MEMORY_MODE_MANUAL},
                "subentry_type": "conversation",
                "title": f"{title} Conversation",
                "unique_id": None,
            }
        ],
    )


async def _setup_entry(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Load the integration through Home Assistant's real config-entry manager."""
    if hass.config_entries.async_get_entry(entry.entry_id) is None:
        entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED


def _conversation_subentry(entry: MockConfigEntry):
    """Return the entry's conversation subentry."""
    return next(
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == "conversation"
    )


async def _admin_client(
    hass: HomeAssistant,
    hass_ws_client: Any,
    *,
    user_id: str = ADMIN_ID,
    name: str = "Management Acceptance Admin",
) -> Any:
    """Create a genuine authenticated Home Assistant admin WebSocket client."""
    admin = MockUser(id=user_id, name=name, is_owner=True)
    admin.add_to_hass(hass)
    refresh_token = await hass.auth.async_create_refresh_token(admin, CLIENT_ID)
    access_token = hass.auth.async_create_access_token(refresh_token)
    return await hass_ws_client(hass, access_token)


async def _management_response(
    client: Any,
    *,
    entry: MockConfigEntry,
    section: str,
    action: str,
    **payload: Any,
) -> dict[str, Any]:
    """Call the registered management command and return the raw HA WS response."""
    subentry = _conversation_subentry(entry)
    await client.send_json_auto_id(
        {
            "type": WS_COMMAND,
            "section": section,
            "action": action,
            "entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
            **payload,
        }
    )
    return await client.receive_json()


async def _management_call(
    client: Any,
    *,
    entry: MockConfigEntry,
    section: str,
    action: str,
    **payload: Any,
) -> dict[str, Any]:
    """Call the registered management command and require a successful response."""
    response = await _management_response(
        client,
        entry=entry,
        section=section,
        action=action,
        **payload,
    )
    assert response["success"], response
    return response["result"]


def _evict_durable_management_managers(hass: HomeAssistant) -> None:
    """Discard process-local manager objects while retaining HA's durable stores."""
    hass.data.pop(memory_module._MEMORY_MANAGERS, None)
    hass.data.pop(request_rules_module._MANAGERS, None)
    hass.data.pop(request_rules_module._RUNTIMES, None)


async def _fresh_reload(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Reload the entry with fresh durable managers, approximating a restart boundary."""
    # Config-subentry writes can schedule the integration's update listener. Let any
    # such reload finish before deliberately crossing our own unload/load boundary.
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.NOT_LOADED

    _evict_durable_management_managers(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED


@pytest.mark.asyncio
async def test_prompt_preview_actions_reach_real_websocket_without_mutating_config(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    """Both preview aliases return assembled data without saving a draft."""
    entry = _entry("Management prompt preview")
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)
    before = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    for action in ("prompt_preview", "request_preview"):
        preview = await _management_call(
            client, entry=entry, section="configuration", action=action, config={}
        )
        assert isinstance(preview["prompt"], str) and preview["prompt"]
        assert isinstance(preview["notes"], list)
    after = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    assert after["revision"] == before["revision"]
    assert after["config"] == before["config"]


@pytest.mark.asyncio
async def test_configuration_round_trip_through_management_websocket(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """Configuration written over WS survives a genuine integration reload."""
    entry = _entry("Configuration Acceptance")
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
        title="Configuration Acceptance Saved",
        config={CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES: 47},
    )
    assert saved["title"] == "Configuration Acceptance Saved"
    assert saved["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] == 47

    await _fresh_reload(hass, entry)

    reloaded = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    assert reloaded["title"] == "Configuration Acceptance Saved"
    assert reloaded["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] == 47


@pytest.mark.asyncio
async def test_configuration_live_metadata_uses_non_reserved_websocket_field(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """Live configuration metadata must cross the genuine HA WS schema boundary."""
    entry = _entry("Live Metadata Acceptance")
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)

    result = await _management_call(
        client,
        entry=entry,
        section="configuration",
        action="live_metadata",
        metadata_keys=["local_handling", "exposed_attribute_catalog"],
    )

    assert "local_handling" in result
    assert "exposed_attribute_catalog" in result


@pytest.mark.asyncio
async def test_stale_configuration_revision_cannot_overwrite_newer_save(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """Two genuine HA clients must not lose a newer configuration update."""
    entry = _entry("Configuration Concurrency Acceptance")
    await _setup_entry(hass, entry)
    client_a = await _admin_client(
        hass,
        hass_ws_client,
        user_id="management-concurrency-a",
        name="Management Concurrency A",
    )
    client_b = await _admin_client(
        hass,
        hass_ws_client,
        user_id="management-concurrency-b",
        name="Management Concurrency B",
    )

    snapshot_a = await _management_call(
        client_a, entry=entry, section="configuration", action="get"
    )
    snapshot_b = await _management_call(
        client_b, entry=entry, section="configuration", action="get"
    )
    assert snapshot_b["revision"] == snapshot_a["revision"]
    assert snapshot_b["title"] == snapshot_a["title"]
    baseline_timeout = snapshot_a["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES]

    winner = await _management_call(
        client_a,
        entry=entry,
        section="configuration",
        action="update",
        revision=snapshot_a["revision"],
        title="Configuration Concurrency Winner",
        config={CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES: 41},
    )
    assert winner["title"] == "Configuration Concurrency Winner"
    assert winner["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] == 41
    assert winner["revision"] != snapshot_a["revision"]

    stale_response = await _management_response(
        client_b,
        entry=entry,
        section="configuration",
        action="update",
        revision=snapshot_b["revision"],
        title="Configuration Concurrency Stale",
        config={CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES: 52},
    )
    assert stale_response["success"] is False
    assert "changed in another tab" in stale_response["error"]["message"].lower()

    authoritative = await _management_call(
        client_b, entry=entry, section="configuration", action="get"
    )
    assert authoritative["title"] == "Configuration Concurrency Winner"
    assert authoritative["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] == 41
    assert authoritative["revision"] == winner["revision"]

    # After re-reading the authoritative revision, the stale client is healthy and
    # can make a normal subsequent write. Restore the original values as cleanup.
    recovered = await _management_call(
        client_b,
        entry=entry,
        section="configuration",
        action="update",
        revision=authoritative["revision"],
        title=snapshot_a["title"],
        config={CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES: baseline_timeout},
    )
    assert recovered["title"] == snapshot_a["title"]
    assert recovered["config"][CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] == baseline_timeout
    assert recovered["revision"] != winner["revision"]


@pytest.mark.asyncio
async def test_function_tools_and_groups_round_trip_through_management_websocket(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """Function Tool and Group management writes survive a real reload."""
    entry = _entry("Function Acceptance")
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)

    tool = {
        "spec": {
            "name": "acceptance_user_lookup",
            "description": "Return the current Home Assistant user's display name.",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "native", "name": "get_user_from_user_id"},
    }
    tool_saved = await _management_call(
        client,
        entry=entry,
        section="tools",
        action="save",
        tool=tool,
    )
    assert any(
        item["spec"]["name"] == "acceptance_user_lookup"
        for item in tool_saved["functions"]
    )

    # The Function Tool write may trigger the config-entry update listener. Wait for
    # that normal lifecycle work before issuing the dependent group mutation.
    await hass.async_block_till_done()

    group = {
        "id": "acceptance_group",
        "name": "Acceptance Group",
        "description": "Functions persisted by the real-HA management acceptance test.",
        "loading_mode": FUNCTION_GROUP_LOADING_ON_DEMAND,
        "functions": ["acceptance_user_lookup"],
        "enabled": True,
    }
    group_saved = await _management_call(
        client,
        entry=entry,
        section="tools",
        action="save_group",
        group=group,
    )
    assert any(
        item["id"] == "acceptance_group"
        and item["functions"] == ["acceptance_user_lookup"]
        for item in group_saved["function_groups"]
    )

    await _fresh_reload(hass, entry)

    reloaded = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    reloaded_tool = next(
        item
        for item in reloaded["config"]["functions"]
        if item["spec"]["name"] == "acceptance_user_lookup"
    )
    assert reloaded_tool["function"] == {
        "type": "native",
        "name": "get_user_from_user_id",
    }
    reloaded_group = next(
        item
        for item in reloaded["config"]["function_groups"]
        if item["id"] == "acceptance_group"
    )
    assert reloaded_group["functions"] == ["acceptance_user_lookup"]
    assert reloaded_group["loading_mode"] == FUNCTION_GROUP_LOADING_ON_DEMAND


@pytest.mark.asyncio
async def test_request_rules_round_trip_through_management_websocket(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """A Request Rule created over WS is rehydrated from storage after reload."""
    entry = _entry("Request Rules Acceptance")
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)

    created = await _management_call(
        client,
        entry=entry,
        section="request_rules",
        action="create",
        rule={
            "id": "acceptance-good-night",
            "name": "Acceptance good night",
            "enabled": True,
            "phrases": ["acceptance good night"],
            "match_type": "equals",
            "action_type": "local_action",
            "action": {
                "actions": [
                    {
                        "domain": "script",
                        "service": "turn_on",
                        "target": {"entity_id": ["script.acceptance_goodnight"]},
                        "data": {},
                    }
                ],
                "success_response": "Acceptance complete",
                "failure_response": "Acceptance failed safely",
            },
            "matching_behavior": "defaults",
            "matching": dict(request_rules_module.DEFAULT_MATCHING),
            "order": 0,
        },
    )
    assert created["rule"]["id"] == "acceptance-good-night"

    await _fresh_reload(hass, entry)

    reloaded = await _management_call(
        client, entry=entry, section="request_rules", action="list"
    )
    rule = next(
        item for item in reloaded["rules"] if item["id"] == "acceptance-good-night"
    )
    assert rule["name"] == "Acceptance good night"
    assert rule["phrases"] == ["acceptance good night"]
    assert rule["action"]["success_response"] == "Acceptance complete"


@pytest.mark.asyncio
async def test_memories_round_trip_through_management_websocket(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """A Memory created over WS is rehydrated from storage after reload."""
    entry = _entry("Memory Acceptance")
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)

    created = await _management_call(
        client,
        entry=entry,
        section="memories",
        action="add",
        content="Management WebSocket persistence acceptance marker.",
        category="acceptance",
    )
    assert created["memory"]["content"] == (
        "Management WebSocket persistence acceptance marker."
    )

    await _fresh_reload(hass, entry)

    reloaded = await _management_call(
        client, entry=entry, section="memories", action="list"
    )
    assert reloaded["scope_id"] == f"user:{ADMIN_ID}"
    assert [(item["content"], item["category"]) for item in reloaded["memories"]] == [
        ("Management WebSocket persistence acceptance marker.", "acceptance")
    ]
