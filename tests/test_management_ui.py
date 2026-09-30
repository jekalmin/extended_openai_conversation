"""Tests for the backend management UI command boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from custom_components.extended_openai_conversation_responses import (
    management_projections,
    management_request_preview,
    management_ui,
)
from custom_components.extended_openai_conversation_responses.agent_config import (
    AgentConfigError,
    agent_config_defaults,
)
from custom_components.extended_openai_conversation_responses.const import (
    AGENT_CONFIG_EXPORT_VERSION,
    DOMAIN,
)
from custom_components.extended_openai_conversation_responses.guest_mode import (
    GuestCapabilityPolicy,
)
from custom_components.extended_openai_conversation_responses.ha_llm_tools import (
    ToolSnapshot,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    KnowledgeSource,
)
from custom_components.extended_openai_conversation_responses.memory import MemoryRecord
from custom_components.extended_openai_conversation_responses.temporary_memory import (
    TemporaryMemoryRecord,
)
from homeassistant.exceptions import HomeAssistantError


def _entry_pair():
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Jarvis",
        data=agent_config_defaults(),
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain=DOMAIN,
        title="Provider",
        data={},
        subentries={subentry.subentry_id: subentry},
    )
    return entry, subentry


def _hass(entry=None):
    config_entries = MagicMock()
    config_entries.async_get_entry.return_value = entry
    return SimpleNamespace(
        config_entries=config_entries,
        auth=SimpleNamespace(),
        config=SimpleNamespace(language="en"),
        data={},
        services=SimpleNamespace(async_call=AsyncMock()),
    )


def _message(section: str, action: str, **values):
    return {
        "section": section,
        "action": action,
        "entry_id": "entry-1",
        "subentry_id": "agent-1",
        **values,
    }


def _tool(name: str) -> dict:
    return {
        "spec": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "script", "sequence": []},
        "enabled": True,
    }


@pytest.mark.asyncio
async def test_effective_request_preview_covers_discovery_guest_filters_and_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    hass.config.language = "en"
    manager = SimpleNamespace(status=lambda: {"active": False})
    monkeypatch.setattr(
        management_request_preview, "async_get_guest_mode", AsyncMock(return_value=manager)
    )
    monkeypatch.setattr(management_request_preview, "get_loaded_temporary_memory", lambda *_: None)
    monkeypatch.setattr(
        management_request_preview,
        "async_read_temporary_memory_snapshot",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(management_request_preview.SkillManager, "get_loaded_instance", lambda: None)
    monkeypatch.setattr(management_request_preview, "get_loaded_knowledge", lambda *_: None)
    monkeypatch.setattr(
        management_request_preview,
        "render_effective_prompt",
        lambda *_args, **_kwargs: SimpleNamespace(text="prompt", sections=[]),
    )
    monkeypatch.setattr(
        management_request_preview,
        "build_provider_request_snapshot",
        lambda *_args: SimpleNamespace(
            api_mode="responses", api_kwargs={}, provider_tools=[]
        ),
    )
    monkeypatch.setattr(
        management_request_preview, "format_function_tools", lambda tools, _mode: tools
    )
    monkeypatch.setattr(
        management_request_preview,
        "assemble_integration_function_tools",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(management_request_preview, "memory_enabled", lambda _options: False)

    ha_tool = {
        "spec": {"name": "ha_light"},
        "function": {"type": "ha_llm", "source_id": "assist", "tool_name": "Light"},
        "enabled": True,
    }
    custom = _tool("allowed")
    options = {
        **agent_config_defaults(),
        management_request_preview.CONF_TEMPORARY_MEMORY: "balanced",
        management_request_preview.CONF_MEMORY_AUTO_RETRIEVE_LIMIT: 0,
    }
    configured = [ha_tool, custom]
    monkeypatch.setattr(
        management_request_preview, "configured_function_tools_from_data", lambda _data: configured
    )
    discover = AsyncMock(return_value=ToolSnapshot())
    monkeypatch.setattr(management_request_preview, "async_discover", discover)
    monkeypatch.setattr(
        management_request_preview,
        "resolve_guest_policy",
        lambda *_: GuestCapabilityPolicy.unrestricted(),
    )
    monkeypatch.setattr(management_request_preview, "validate_function_groups", lambda *_: [])
    monkeypatch.setattr(
        management_request_preview,
        "assemble_function_tools",
        lambda tools, *_args: SimpleNamespace(tools=tools),
    )
    monkeypatch.setattr(management_request_preview, "get_exposed_entities", lambda _hass: [])

    trusted = await management_request_preview.async_preview_effective_request(
        hass, entry, subentry, options, "user"
    )
    assert trusted["prompt"] == "prompt"
    assert not any(note.startswith("Query-derived") for note in trusted["notes"])
    discover.assert_awaited_once()

    guest_policy = GuestCapabilityPolicy(
        True,
        readable_entity_ids=frozenset({"sensor.allowed"}),
        configured_tool_names=frozenset({"allowed"}),
    )
    groups = [
        {"id": "public", "guest_allowed": True, "functions": ["allowed", "blocked"]},
        {"id": "private", "guest_allowed": False, "functions": ["private"]},
    ]
    guest_tools = [custom, _tool("blocked"), _tool("private"), ha_tool]
    monkeypatch.setattr(
        management_request_preview, "configured_function_tools_from_data", lambda _data: guest_tools
    )
    monkeypatch.setattr(management_request_preview, "resolve_guest_policy", lambda *_: guest_policy)
    monkeypatch.setattr(
        management_request_preview, "validate_function_groups", lambda *_: deepcopy(groups)
    )
    monkeypatch.setattr(
        management_request_preview,
        "get_exposed_entities",
        lambda _hass: [
            {"entity_id": "sensor.allowed"},
            {"entity_id": "sensor.private"},
        ],
    )
    monkeypatch.setattr(
        management_request_preview.ConversationContinuity,
        "identity_key",
        lambda *_args: (None, "none"),
    )

    guest = await management_request_preview.async_preview_effective_request(
        hass, entry, subentry, options, "user"
    )
    assert any("temporary memories are excluded" in note for note in guest["notes"])
    assert discover.await_count == 1

    monkeypatch.setattr(
        management_request_preview, "configured_function_tools_from_data", lambda _data: []
    )
    monkeypatch.setattr(management_request_preview, "validate_function_groups", lambda *_: [])
    empty = await management_request_preview.async_preview_effective_request(
        hass, entry, subentry, options, "user"
    )
    request_settings = next(
        section for section in empty["sections"] if section["key"] == "request_settings"
    )
    assert "tool_choice" not in request_settings["content"]


@pytest.mark.asyncio
async def test_request_rule_test_defaults_duplicate_move_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    rules = SimpleNamespace(
        async_set_defaults=AsyncMock(return_value={"case_sensitive": True}),
        async_duplicate=AsyncMock(return_value={"id": "copy"}),
        async_move=AsyncMock(return_value={"id": "rule", "priority": 2}),
        revision=lambda: "revision-2",
    )
    monkeypatch.setattr(
        management_ui, "async_get_request_rules", AsyncMock(return_value=rules)
    )
    rules.async_match = AsyncMock(return_value=None)
    with pytest.raises(HomeAssistantError, match="Test request text is required"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("request_rules", "test_match", text=" ")
        )
    result = await management_ui.async_management_command(
        hass, "admin", True, _message("request_rules", "test_match", text=" hello ")
    )
    assert result["matched"] is False
    rules.async_match.assert_awaited_once()
    hass.services.async_call.assert_not_awaited()
    assert "defaults" in await management_ui.async_management_command(
        hass, "admin", True, _message("request_rules", "defaults", defaults={})
    )

    with pytest.raises(HomeAssistantError, match="rule_id is required"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("request_rules", "duplicate")
        )
    duplicated = await management_ui.async_management_command(
        hass, "admin", True, _message("request_rules", "duplicate", rule_id="rule")
    )
    moved = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("request_rules", "move", rule_id="rule", direction="up"),
    )
    assert duplicated["rule"]["id"] == "copy"
    assert moved["rule"]["priority"] == 2
    with pytest.raises(HomeAssistantError, match="direction is required"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("request_rules", "move", rule_id="rule")
        )
    with pytest.raises(HomeAssistantError, match="Unknown Request Rules action"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("request_rules", "other", rule_id="rule")
        )


@pytest.mark.asyncio
async def test_backup_preview_and_diagnostics_success_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    prepared = SimpleNamespace(title="Backup", summary=lambda: {"agents": 1})
    inspect = Mock(return_value=prepared)
    restore = AsyncMock(return_value={"restored": True})
    preview = AsyncMock(return_value={"preview": True})
    monkeypatch.setattr(management_ui, "inspect_backup", inspect)
    monkeypatch.setattr(management_ui, "async_restore_backup", restore)
    monkeypatch.setattr(management_ui, "_async_preview_effective_request", preview)
    monkeypatch.setattr(
        management_ui, "merge_agent_config", lambda data, updates: {**data, **updates}
    )
    monkeypatch.setattr(
        management_ui,
        "async_test_agent",
        AsyncMock(return_value=SimpleNamespace(as_dict=lambda: {"healthy": True})),
    )

    assert (
        await management_ui.async_management_command(
            hass, "admin", True, _message("backup", "inspect", document={})
        )
    )["title"] == "Backup"
    assert await management_ui.async_management_command(
        hass, "admin", True, _message("backup", "restore", document={}, confirm=True)
    ) == {"restored": True}
    assert await management_ui.async_management_command(
        hass, "admin", True, _message("configuration", "request_preview", config={})
    ) == {"preview": True}
    assert await management_ui.async_management_command(
        hass, "user", True, _message("diagnostics", "test_agent")
    ) == {"healthy": True}


@pytest.mark.asyncio
async def test_ha_tool_add_validation_refresh_and_group_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))

    with pytest.raises(HomeAssistantError, match="Select up to 1000"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("tools", "ha_add", tools="bad")
        )
    monkeypatch.setattr(
        management_ui,
        "validate_reference",
        Mock(side_effect=ValueError("bad reference")),
    )
    with pytest.raises(HomeAssistantError, match="bad reference"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("tools", "ha_add", tools=[{}])
        )

    reference = {"source_id": "assist", "tool_name": "Light"}
    monkeypatch.setattr(management_ui, "validate_reference", lambda _value: reference)
    monkeypatch.setattr(management_ui, "reference_key", lambda _value: "assist:Light")
    snapshot = SimpleNamespace(tools={
        "assist:Light": SimpleNamespace(
            source_label="Assist",
            tool=SimpleNamespace(description="Control lights"),
        )
    })
    monkeypatch.setattr(
        management_ui, "async_discover", AsyncMock(return_value=snapshot)
    )
    hass.config_entries.async_get_entry.return_value = None
    with pytest.raises(HomeAssistantError, match="Agent no longer exists"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("tools", "ha_add", tools=[reference])
        )

    hass.config_entries.async_get_entry.return_value = entry
    monkeypatch.setattr(
        management_ui, "configured_function_tools_from_data", lambda _data: []
    )
    group = {"id": "group", "functions": []}
    monkeypatch.setattr(management_ui, "validate_function_groups", lambda *_: [group])
    with pytest.raises(HomeAssistantError, match="Function Group no longer exists"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _message("tools", "ha_add", tools=[reference], group_id="missing"),
        )

    added = _tool("ha_light")
    added["function"] = {
        "type": "ha_llm",
        "source_id": "assist",
        "api_id": "assist",
        "tool_name": "Light",
    }
    monkeypatch.setattr(management_ui, "new_reference_tool", lambda *_args: added)
    persist = Mock(return_value={"status": "saved"})
    monkeypatch.setattr(management_ui, "_persist_function_configuration", persist)
    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("tools", "ha_add", tools=[reference], group_id="group"),
    )
    assert result == {
        "status": "saved",
        "ha_saved": {"ha_light": {
            "available": True,
            "name": "Light",
            "source": "Assist",
            "description": "Control lights",
        }},
    }
    assert group["functions"] == ["ha_light"]


@pytest.mark.asyncio
async def test_function_tool_validation_and_save_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    tools = [_tool("one"), _tool("two")]
    monkeypatch.setattr(
        management_ui,
        "configured_function_tools_from_data",
        lambda _data: deepcopy(tools),
    )
    monkeypatch.setattr(management_ui, "validate_function_groups", lambda *_: [])
    monkeypatch.setattr(
        management_ui,
        "validate_function_tools",
        lambda value: deepcopy(value) if isinstance(value, list) else [],
    )
    persist = Mock(return_value={"status": "saved", "revision": "next"})
    monkeypatch.setattr(management_ui, "_persist_function_configuration", persist)

    assert await management_ui.async_management_command(
        hass, "admin", True, _message("tools", "validate", tools=[])
    ) == {"valid": True, "errors": {}, "config": []}
    assert await management_ui.async_management_command(
        hass, "admin", True, _message("tools", "validate_current")
    ) == {"valid": True, "errors": {}}

    cases = [
        (_message("tools", "save"), "tool must be an object"),
        (
            _message("tools", "save", tool=_tool("one"), original_name=3),
            "original_name",
        ),
        (
            _message("tools", "save", tool=_tool("new"), original_name="missing"),
            "no longer exists",
        ),
        (
            _message("tools", "save", tool=_tool("two"), original_name="one"),
            "already exists",
        ),
    ]
    for message, error in cases:
        with pytest.raises(HomeAssistantError, match=error):
            await management_ui.async_management_command(hass, "admin", True, message)

    assert await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("tools", "save", tool=_tool("one"), original_name="one"),
    ) == {"status": "saved", "revision": "next"}


@pytest.mark.asyncio
async def test_function_enable_delete_and_group_validation_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    tools = [_tool("one")]
    groups = [{"id": "group", "functions": ["one"]}]
    monkeypatch.setattr(
        management_ui,
        "configured_function_tools_from_data",
        lambda _data: deepcopy(tools),
    )
    monkeypatch.setattr(
        management_ui, "validate_function_groups", lambda *_: deepcopy(groups)
    )
    monkeypatch.setattr(
        management_ui,
        "_persist_function_configuration",
        Mock(side_effect=lambda *_args, **_kwargs: {"status": "saved"}),
    )
    monkeypatch.setattr(
        management_ui,
        "_function_reference_state",
        AsyncMock(return_value=(object(), {"request_rules": [], "guest_mode": False})),
    )

    cases = [
        (
            _message("tools", "set_enabled", name="one", enabled="no"),
            "name and enabled",
        ),
        (
            _message("tools", "set_enabled", name="missing", enabled=True),
            "no longer exists",
        ),
        (_message("tools", "delete", name="one"), "Explicit confirmation"),
        (_message("tools", "delete", confirm=True), "name is required"),
        (
            _message("tools", "delete", name="missing", confirm=True),
            "no longer exists",
        ),
        (_message("tools", "save_group"), "group must be an object"),
        (
            _message("tools", "save_group", group={"id": "new", "functions": [3]}),
            "group functions",
        ),
        (
            _message(
                "tools",
                "save_group",
                group={"id": "new", "functions": []},
                original_id=3,
            ),
            "original_id",
        ),
        (
            _message(
                "tools",
                "save_group",
                group={"id": "new", "functions": []},
                original_id="missing",
            ),
            "Group no longer exists",
        ),
        (_message("tools", "delete_group", group_id="group"), "Explicit confirmation"),
        (_message("tools", "delete_group", confirm=True), "group_id is required"),
        (
            _message("tools", "delete_group", group_id="missing", confirm=True),
            "Group no longer exists",
        ),
    ]
    for message, error in cases:
        with pytest.raises(HomeAssistantError, match=error):
            await management_ui.async_management_command(hass, "admin", True, message)

    disabled = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("tools", "set_enabled", name="one", enabled=False),
    )
    assert disabled["references"] == {"request_rules": [], "guest_mode": False}
    assert await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("tools", "delete_group", group_id="group", confirm=True),
    ) == {"status": "saved"}
    enabled = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("tools", "set_enabled", name="one", enabled=True),
    )
    assert "references" not in enabled


@pytest.mark.asyncio
async def test_function_rename_propagates_failure_after_successful_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    tools = [_tool("old")]
    monkeypatch.setattr(
        management_ui,
        "configured_function_tools_from_data",
        lambda _data: deepcopy(tools),
    )
    monkeypatch.setattr(management_ui, "validate_function_groups", lambda *_: [])
    monkeypatch.setattr(
        management_ui, "validate_function_tools", lambda value: deepcopy(value)
    )
    monkeypatch.setattr(
        management_ui, "_agent_config_revision", lambda *_args: "revision"
    )
    rules = SimpleNamespace(
        revision=lambda: "rules-revision",
        async_rename_function_reference=AsyncMock(
            side_effect=RuntimeError("rename failed")
        ),
    )
    monkeypatch.setattr(
        management_ui,
        "_function_reference_state",
        AsyncMock(return_value=(rules, {"request_rules": [], "guest_mode": False})),
    )
    persist = Mock(
        side_effect=[
            {"status": "saved", "revision": "new-revision"},
            {"status": "rolled-back"},
        ]
    )
    monkeypatch.setattr(management_ui, "_persist_function_configuration", persist)

    with pytest.raises(RuntimeError, match="rename failed"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _message("tools", "save", tool=_tool("new"), original_name="old"),
        )
    assert persist.call_count == 2


@pytest.mark.asyncio
async def test_conversation_cleanup_and_admin_temporary_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    continuity = SimpleNamespace(async_end=AsyncMock(return_value=True))
    runtime = SimpleNamespace(end=Mock())
    reset = Mock()
    monkeypatch.setattr(management_ui, "async_get_continuity", lambda *_: continuity)
    monkeypatch.setattr(management_ui, "get_function_group_runtime", lambda *_: runtime)
    monkeypatch.setattr(management_ui, "_reset_request_rule_runtime", reset)

    assert await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("conversations", "end_active", continuity_key="key"),
    ) == {"ended": 1}
    runtime.end.assert_called_once_with("continuity:key")
    reset.assert_called_once_with(hass, "entry-1", "agent-1", "key")

    temporary = SimpleNamespace(
        async_delete=AsyncMock(return_value=1),
        async_delete_owned=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(
        management_ui, "async_get_temporary_memory", AsyncMock(return_value=temporary)
    )
    assert await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message(
            "memories",
            "temporary_delete",
            memory_id="memory",
            temporary_scope_id="user:someone-else",
        ),
    ) == {"deleted": 1}


@pytest.mark.asyncio
async def test_fallthrough_branches_and_scope_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        management_ui,
        "async_get_temporary_memory",
        AsyncMock(return_value=SimpleNamespace(owner_counts=lambda: {})),
    )
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    monkeypatch.setattr(
        management_ui,
        "async_get_guest_mode",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        management_ui, "async_get_usage", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        management_ui,
        "configured_function_tools_from_data",
        lambda _data: [],
    )
    monkeypatch.setattr(management_ui, "validate_function_groups", lambda *_: [])
    monkeypatch.setattr(
        management_ui,
        "async_get_continuity",
        lambda *_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        management_ui,
        "async_get_archive",
        AsyncMock(return_value=SimpleNamespace(scope_counts=lambda: {})),
    )
    monkeypatch.setattr(
        management_ui,
        "async_get_memory",
        AsyncMock(return_value=SimpleNamespace(scope_counts=lambda: {})),
    )
    monkeypatch.setattr(
        management_ui,
        "async_get_temporary_memory",
        AsyncMock(return_value=SimpleNamespace(owner_counts=lambda: {})),
    )
    monkeypatch.setattr(
        management_ui,
        "async_get_knowledge",
        AsyncMock(return_value=SimpleNamespace()),
    )
    from custom_components.extended_openai_conversation_responses import (
        management_loading_performance as loading,
    )

    monkeypatch.setattr(
        loading,
        "async_scope_catalog_projection",
        AsyncMock(return_value=[{"scope_id": "user:user"}]),
    )

    monkeypatch.setattr(loading, "async_get_archive", management_ui.async_get_archive)
    monkeypatch.setattr(loading, "async_get_memory", management_ui.async_get_memory)
    monkeypatch.setattr(
        loading,
        "async_get_temporary_memory",
        management_ui.async_get_temporary_memory,
    )
    catalog = await management_ui.async_management_command(
        hass, "admin", True, _message("scopes", "catalog")
    )
    assert catalog["scopes"][0]["scope_id"] == "user:user"
    assert catalog["scopes"][0].get("temporary_memory_count", 0) == 0

    with pytest.raises(HomeAssistantError, match="Unknown memories management action"):
        await management_ui.async_management_command(
            hass, "admin", True, _message("memories", "temporary_unknown")
        )

    for section in (
        "guest_mode",
        "backup",
        "configuration",
        "tools",
        "usage",
        "conversations",
        "memories",
        "knowledge",
    ):
        with pytest.raises(
            HomeAssistantError, match=f"Unknown {section} management action"
        ):
            await management_ui.async_management_command(
                hass, "admin", True, _message(section, "unknown")
            )


def test_reference_error_handles_each_reference_kind_independently() -> None:
    rules_only = management_ui._function_reference_error(
        "demo", {"request_rules": [{"name": "Rule"}], "guest_mode": False}
    )
    guest_only = management_ui._function_reference_error(
        "demo", {"request_rules": [], "guest_mode": True}
    )
    assert "Request Rules: Rule" in rules_only
    assert "Guest Mode" not in rules_only
    assert "Guest Mode custom function access" in guest_only


@pytest.mark.asyncio
async def test_conversation_cleanup_without_function_group_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, subentry = _entry_pair()
    hass = _hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    monkeypatch.setattr(
        management_ui,
        "async_get_continuity",
        lambda *_: SimpleNamespace(async_end=AsyncMock(return_value=True)),
    )
    monkeypatch.setattr(management_ui, "get_function_group_runtime", lambda *_: None)
    reset = Mock()
    monkeypatch.setattr(management_ui, "_reset_request_rule_runtime", reset)

    assert await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _message("conversations", "end_active", continuity_key="key"),
    ) == {"ended": 1}
    reset.assert_called_once()


def _residual_entry_pair():
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Jarvis",
        data=agent_config_defaults(),
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain=DOMAIN,
        title="Provider",
        data={},
        subentries={subentry.subentry_id: subentry},
    )
    return entry, subentry


def _residual_hass(entry=None):
    config_entries = MagicMock()
    config_entries.async_get_entry.return_value = entry
    return SimpleNamespace(
        config_entries=config_entries,
        auth=SimpleNamespace(),
        config=SimpleNamespace(language="en"),
        data={},
        http=SimpleNamespace(async_register_static_paths=AsyncMock()),
        services=SimpleNamespace(async_call=AsyncMock()),
    )


def _residual_message(section: str, action: str, **values):
    return {
        "section": section,
        "action": action,
        "entry_id": "entry-1",
        "subentry_id": "agent-1",
        **values,
    }


def test_entry_resolution_and_scope_validation_boundaries() -> None:
    entry, subentry = _residual_entry_pair()
    hass = _residual_hass(None)
    with pytest.raises(HomeAssistantError, match="Integration entry not found"):
        management_ui.entry_and_agent(hass, "missing", "agent-1")

    hass.config_entries.async_get_entry.return_value = SimpleNamespace(
        domain="other", subentries={}
    )
    with pytest.raises(HomeAssistantError, match="Integration entry not found"):
        management_ui.entry_and_agent(hass, "entry-1", "agent-1")

    hass.config_entries.async_get_entry.return_value = entry
    assert management_ui.entry_and_agent(hass, "entry-1", "agent-1") == (
        entry,
        subentry,
    )
    subentry.subentry_type = "not-conversation"
    with pytest.raises(HomeAssistantError, match="Conversation agent not found"):
        management_ui.entry_and_agent(hass, "entry-1", "agent-1")

    with pytest.raises(HomeAssistantError, match="scope_id must be a string"):
        management_ui._selected_scope("user", True, 7)
    with pytest.raises(HomeAssistantError, match="Unknown data scope"):
        management_ui._selected_scope("user", True, "organization:example")
    assert management_ui._memory_scope("user:alice") == "alice"
    assert management_ui._memory_scope("shared:household") == "shared:household"


def test_revision_and_reference_helpers_cover_validation_edges() -> None:
    _entry, subentry = _residual_entry_pair()
    management_ui._require_agent_config_revision(subentry, None)
    with pytest.raises(HomeAssistantError, match="revision must be a string"):
        management_ui._require_agent_config_revision(subentry, 1)

    assert management_ui._function_reference_error(
        "weather", {"request_rules": [{"id": "rule-1"}], "guest_mode": True}
    ) == (
        "Function Tool `weather` is still referenced by Request Rules: rule-1; "
        "Guest Mode custom function access. Update those references before deleting it."
    )

    valid = management_ui._validation_result(lambda: {"ok": True})
    assert valid == {"valid": True, "errors": {}, "config": {"ok": True}}

    def invalid():
        raise AgentConfigError("field", "bad value")

    assert management_ui._validation_result(invalid) == {
        "valid": False,
        "errors": {"field": "bad value"},
    }


@pytest.mark.parametrize(
    ("document", "error"),
    [
        ("config: [", "invalid JSON/YAML"),
        ([], "must be an object"),
        ({"schema": "wrong", "version": 1, "config": {}}, "export schema"),
        (
            {
                "schema": "extended_openai_conversation.agent",
                "version": AGENT_CONFIG_EXPORT_VERSION + 1,
                "config": {},
            },
            "export version",
        ),
        (
            {
                "schema": "extended_openai_conversation.agent",
                "version": AGENT_CONFIG_EXPORT_VERSION,
                "config": {},
                "unexpected": True,
            },
            "unknown fields",
        ),
        (
            {
                "schema": "extended_openai_conversation.agent",
                "version": AGENT_CONFIG_EXPORT_VERSION,
                "config": [],
            },
            "must be an object",
        ),
    ],
)
def test_import_document_rejects_malformed_boundaries(document, error) -> None:
    with pytest.raises(AgentConfigError, match=error):
        management_ui._parse_import_document(document)


def test_request_rule_function_reference_validation_edges() -> None:
    with pytest.raises(HomeAssistantError, match="rule must be an object"):
        management_ui._prepare_request_rule([])

    action = f"{DOMAIN}.call_function"
    with pytest.raises(HomeAssistantError, match="action is invalid"):
        management_ui._validate_request_rule_functions(
            {"action": {"actions": ["ignored", {"action": action, "data": {}}]}}, []
        )
    with pytest.raises(HomeAssistantError, match="arguments must be an object"):
        management_ui._validate_request_rule_functions(
            {
                "action": {
                    "actions": [
                        {
                            "action": action,
                            "data": {"function": "demo", "arguments": []},
                        }
                    ]
                }
            },
            [],
        )

    tool = {
        "spec": {
            "name": "demo",
            "parameters": {"type": "object", "required": ["city"]},
        },
        "function": {"type": "native", "name": "execute_service"},
    }
    rule = {
        "action": {
            "actions": [
                {"action": action, "data": {"function": "missing", "arguments": {}}}
            ]
        }
    }
    with pytest.raises(HomeAssistantError, match="unavailable or disabled: missing"):
        management_ui._validate_request_rule_functions(rule, [tool])

    rule["action"]["actions"][0]["data"]["function"] = "demo"
    with pytest.raises(HomeAssistantError, match="needs input: city"):
        management_ui._validate_request_rule_functions(rule, [tool])


async def test_non_admin_scope_catalog_handles_known_and_missing_users() -> None:
    hass = _residual_hass()
    hass.auth.async_get_user = AsyncMock(
        side_effect=[SimpleNamespace(name="Alice"), None]
    )
    first = await management_projections.async_scope_catalog_projection(
        hass,
        "alice",
        False,
        {"alice": 2},
        temporary_memory_counts={"user:alice": 3},
    )
    second = await management_projections.async_scope_catalog_projection(hass, "missing", False)
    assert first[0]["display_name"] == "Alice"
    assert first[0]["memory_count"] == 2
    assert first[0]["temporary_memory_count"] == 3
    assert second[0]["display_name"] == "missing"


async def test_agents_overview_skips_non_conversation_subentries(
    monkeypatch,
) -> None:
    entry, _subentry = _residual_entry_pair()
    skipped = SimpleNamespace(subentry_type="sensor")
    entry.subentries["skip"] = skipped
    hass = _residual_hass(entry)
    hass.config_entries.async_entries.return_value = [entry]
    hass.auth.async_get_users = AsyncMock(return_value=[])

    usage = SimpleNamespace(today_summary=lambda: {"total_tokens": 123})
    memory = SimpleNamespace(stats=lambda: {"memory_count": 4})
    knowledge = SimpleNamespace(source_count=2)
    guest = SimpleNamespace(status=lambda: {"state": "inactive"})
    monkeypatch.setattr(management_ui, "async_get_usage", AsyncMock(return_value=usage))
    monkeypatch.setattr(
        management_ui, "async_get_memory", AsyncMock(return_value=memory)
    )
    monkeypatch.setattr(
        management_ui, "async_get_knowledge", AsyncMock(return_value=knowledge)
    )
    monkeypatch.setattr(
        management_ui, "async_get_guest_mode", AsyncMock(return_value=guest)
    )

    result = await management_ui.async_management_command(
        hass, "admin", True, {"action": "agents"}
    )
    assert len(result["agents"]) == 1
    assert result["agents"][0]["tokens_today"] == 0
    assert result["agents"][0]["memory_count"] == 0
    assert result["is_admin"] is True


async def test_command_requires_agent_identifiers() -> None:
    with pytest.raises(
        HomeAssistantError, match="entry_id and subentry_id are required"
    ):
        await management_ui.async_management_command(
            _residual_hass(), "user", False, {"action": "get"}
        )


async def test_guest_policy_save_rejects_stale_revision_and_returns_new_baseline(
    monkeypatch,
) -> None:
    entry, subentry = _residual_entry_pair()
    # Revisions describe content: saving the default True would be a no-op.
    subentry.data["guest_mode_enabled"] = False
    hass = _residual_hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    monkeypatch.setattr(
        management_ui, "async_get_guest_mode", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(
        management_ui,
        "guest_policy_editor_snapshot",
        lambda _hass, config, _tools: config,
    )
    revision = management_ui._agent_config_revision(subentry.data, subentry.title)
    with pytest.raises(HomeAssistantError, match="changed in another tab"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message(
                "guest_mode",
                "save_policy",
                config={"guest_mode_enabled": True},
                revision="stale",
            ),
        )
    hass.config_entries.async_update_subentry.assert_not_called()
    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message(
            "guest_mode",
            "save_policy",
            config={"guest_mode_enabled": True},
            revision=revision,
        ),
    )
    saved = hass.config_entries.async_update_subentry.call_args.kwargs["data"]
    assert result["config"]["guest_mode_enabled"] is True
    assert result["revision"] == management_ui._agent_config_revision(
        saved, subentry.title
    )
    assert result["revision"] != revision


async def test_guest_backup_and_service_dispatch_edges(monkeypatch) -> None:
    entry, subentry = _residual_entry_pair()
    hass = _residual_hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    policy = SimpleNamespace(as_diagnostics=lambda: {"guest_active": False})
    guest = SimpleNamespace(
        status=lambda: {"state": "inactive"},
        async_update_trusted=AsyncMock(return_value={"state": "active"}),
        async_disable_trusted=AsyncMock(return_value={"state": "inactive"}),
    )
    monkeypatch.setattr(
        management_ui, "async_get_guest_mode", AsyncMock(return_value=guest)
    )
    monkeypatch.setattr(management_ui, "resolve_guest_policy", lambda *_: policy)

    primary = await management_ui.async_management_command(
        hass, "user", False, _residual_message("guest_mode", "get")
    )
    assert primary["status"] == {"state": "inactive"}
    assert primary["config"] == {}
    assert "policy" not in primary

    details = await management_ui.async_management_command(
        hass, "user", False, _residual_message("guest_mode", "details")
    )
    assert details["policy"] == {"guest_active": False}

    with pytest.raises(HomeAssistantError, match="config must be an object"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("guest_mode", "save_policy", config=[])
        )
    with pytest.raises(HomeAssistantError, match="unknown fields"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message("guest_mode", "save_policy", config={"unknown": True}),
        )

    updated = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message("guest_mode", "update", active_from="08:00", indefinite=True),
    )
    disabled = await management_ui.async_management_command(
        hass, "admin", True, _residual_message("guest_mode", "disable")
    )
    assert updated["status"]["state"] == "active"
    assert disabled["status"]["state"] == "inactive"

    monkeypatch.setattr(
        management_ui, "async_create_backup", AsyncMock(return_value={"backup": True})
    )
    assert await management_ui.async_management_command(
        hass, "admin", True, _residual_message("backup", "create")
    ) == {"backup": True}
    with pytest.raises(HomeAssistantError, match="Explicit confirmation"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("backup", "restore")
        )

    service_descriptions = AsyncMock(return_value={"light": {}})
    monkeypatch.setattr(
        management_ui.service_helper,
        "async_get_all_descriptions",
        service_descriptions,
    )
    assert await management_ui.async_management_command(
        hass, "admin", True, _residual_message("service_catalog", "get")
    ) == {"services": {"light": {}}}


async def test_configuration_transfer_and_validation_routes(hass, monkeypatch) -> None:
    entry, subentry = _residual_entry_pair()
    hass = _residual_hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))

    with pytest.raises(HomeAssistantError, match="config must be an object"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("configuration", "validate", config=[])
        )
    validated = await management_ui.async_management_command(
        hass, "admin", True, _residual_message("configuration", "validate", config={})
    )
    assert validated["valid"] is True, validated
    assert "model_capabilities" in validated
    with pytest.raises(HomeAssistantError, match="config must be an object"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("configuration", "update", config=[])
        )

    duplicate = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message("configuration", "duplicate", title="A deliberate copy"),
    )
    assert duplicate["title"] == "A deliberate copy"

    exported = await management_ui.async_management_command(
        hass, "admin", True, _residual_message("configuration", "export")
    )
    assert exported["document"]["schema"] == "extended_openai_conversation.agent"
    assert '"title": "Jarvis"' in exported["json"]
    document = {
        "schema": "extended_openai_conversation.agent",
        "version": AGENT_CONFIG_EXPORT_VERSION,
        "title": "Imported",
        "config": agent_config_defaults(),
    }
    preview = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message("configuration", "import_preview", document=document),
    )
    assert preview["valid"] is True
    assert preview["summary"]["model"] == document["config"]["chat_model"]

    with pytest.raises(HomeAssistantError, match="Explicit confirmation"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message("configuration", "import", document=document),
        )
    current = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message("configuration", "import", document=document, confirm=True),
    )
    assert current["status"] == "updated"
    assert current["subentry_id"] == "agent-1"
    with pytest.raises(HomeAssistantError, match="mode must be current or new"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message("configuration", "import", document=document, mode="replacement"),
        )
    created = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message("configuration", "import", document=document, mode="new"),
    )
    assert created["status"] == "created"

    with pytest.raises(HomeAssistantError, match="sample_text and config are invalid"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message("configuration", "speech_preview", sample_text=1),
        )
    with pytest.raises(HomeAssistantError, match="config must be an object"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message("configuration", "request_preview", config=[]),
        )


@dataclass
class _LatestRun:
    run_id: str


async def test_usage_dispatch_covers_all_read_routes(
    hass, management_message, monkeypatch
):
    usage = SimpleNamespace(
        latest_run=_LatestRun("run-1"),
        request_retention_days=7,
        run_retention_days=30,
        as_dict=lambda: {"requests": 3},
        today_summary=lambda: {"requests": 1},
        month_summary=lambda: {"requests": 2},
        daily={"2026-01-01": {"requests": 1}},
        runs=[],
        requests=[],
        async_clear_details=AsyncMock(return_value={"cleared": 2}),
    )
    monkeypatch.setattr(management_ui, "async_get_usage", AsyncMock(return_value=usage))
    summary = await management_ui.async_management_command(
        hass, "user", False, _residual_message("usage", "summary")
    )
    assert summary["latest"] is None
    daily = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message(
            "usage", "daily", start_date="2026-01-01", end_date="2026-01-02"
        ),
    )
    assert daily["days"][0]["requests"] == 1
    for action in ("runs", "requests"):
        page = await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message("usage", action, run_id="run-1", limit=2, offset=1),
        )
        assert page[action] == []
        assert page["limit"] == 2
        assert page["offset"] == 1
    with pytest.raises(HomeAssistantError, match="run_id is required"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("usage", "requests")
        )
    assert await management_ui.async_management_command(
        hass, "admin", True, _residual_message("usage", "retention")
    ) == {"request_days": 7, "run_days": 30}
    with pytest.raises(HomeAssistantError, match="Administrator"):
        await management_ui.async_management_command(
            hass,
            "user",
            False,
            _residual_message("usage", "clear_details", confirm=True),
        )
    result = await management_ui.async_management_command(
        hass, "admin", True, _residual_message("usage", "clear_details", confirm=True)
    )
    assert result == {"cleared": 2}
    usage.async_clear_details.assert_awaited_once_with(confirm=True)


async def test_conversation_memory_and_knowledge_dispatch(monkeypatch) -> None:
    entry, subentry = _residual_entry_pair()
    hass = _residual_hass(entry)
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))
    monkeypatch.setattr(
        management_ui, "archive_list_page", AsyncMock(return_value={"sessions": []})
    )
    monkeypatch.setattr(
        management_ui, "archive_search_page", AsyncMock(return_value={"matches": []})
    )
    monkeypatch.setattr(
        management_ui, "archive_get_page", AsyncMock(return_value={"session": "one"})
    )
    continuity = SimpleNamespace(
        async_list=AsyncMock(return_value=[{"key": "active"}]),
        async_end=AsyncMock(return_value=False),
    )
    archive = SimpleNamespace(
        async_list_sessions=AsyncMock(return_value={"sessions": []}),
        async_search=AsyncMock(return_value={"matches": []}),
        async_get=AsyncMock(return_value={"session": "one"}),
        async_delete_session=AsyncMock(return_value={"deleted": 1}),
        async_clear_scope=AsyncMock(return_value={"deleted": 2}),
        async_delete_date_range=AsyncMock(return_value={"deleted": 3}),
    )
    monkeypatch.setattr(management_ui, "async_get_continuity", lambda *_: continuity)
    monkeypatch.setattr(
        management_ui, "async_get_archive", AsyncMock(return_value=archive)
    )

    with pytest.raises(HomeAssistantError, match="Administrator"):
        await management_ui.async_management_command(
            hass, "user", False, _residual_message("conversations", "active")
        )
    assert (
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("conversations", "active")
        )
    )["active"][0]["key"] == "active"
    with pytest.raises(HomeAssistantError, match="continuity_key is required"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("conversations", "end_active")
        )
    assert await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message("conversations", "end_active", continuity_key="gone"),
    ) == {"ended": 0}

    for action, expected_key in (
        ("list", "sessions"),
        ("search", "matches"),
        ("get", "session"),
        ("delete", "deleted"),
        ("clear", "deleted"),
        ("delete_range", "deleted"),
    ):
        result = await management_ui.async_management_command(
            hass, "user", False, _residual_message("conversations", action, session_id="one")
        )
        assert expected_key in result
    assert "archive_enabled" in await management_ui.async_management_command(
        hass, "user", False, _residual_message("conversations", "settings")
    )

    temporary_record = TemporaryMemoryRecord(
        memory_id="temp-1",
        scope_id="user:user",
        content="Temporary",
        category="general",
        source="explicit",
        expires_at="2026-01-02",
        created_at="2026-01-01",
        updated_at="2026-01-01",
    )
    temporary = SimpleNamespace(
        async_list_owned=AsyncMock(return_value=[temporary_record]),
        stats=lambda: {},
        async_list_all=AsyncMock(return_value=[temporary_record]),
        async_delete_owned=AsyncMock(return_value=1),
    )
    monkeypatch.setattr(
        management_ui, "async_get_temporary_memory", AsyncMock(return_value=temporary)
    )
    listed = await management_ui.async_management_command(
        hass, "user", False, _residual_message("memories", "temporary_list")
    )
    assert listed["memories"][0]["memory_id"] == "temp-1"
    with pytest.raises(HomeAssistantError, match="scope_id must be a string"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message("memories", "temporary_delete", scope_id=2),
        )
    with pytest.raises(HomeAssistantError, match="not available"):
        await management_ui.async_management_command(
            hass,
            "user",
            False,
            _residual_message("memories", "temporary_delete", scope_id="user:other"),
        )
    assert await management_ui.async_management_command(
        hass,
        "user",
        False,
        _residual_message("memories", "temporary_delete", memory_id="temp-1"),
    ) == {"deleted": 1}

    record = MemoryRecord(
        memory_id="memory-1",
        user_id="user",
        content="Remember this",
        category="general",
        source="explicit",
        created_at="2026-01-01",
        updated_at="2026-01-01",
    )
    memory = SimpleNamespace(
        _memories={record.memory_id: record},
        async_list_page=AsyncMock(return_value=([record], False)),
        async_add=AsyncMock(return_value={"status": "created", "memory": {"memory_id": "memory-1"}}),
        async_get_many=AsyncMock(return_value=[record]),
        async_update=AsyncMock(return_value=record),
        async_delete=AsyncMock(return_value=1),
        async_clear=AsyncMock(return_value=2),
        async_reassign=AsyncMock(return_value={"reassigned": 1}),
    )
    monkeypatch.setattr(
        management_ui, "async_get_memory", AsyncMock(return_value=memory)
    )
    listed = await management_ui.async_management_command(
        hass, "user", False, _residual_message("memories", "list")
    )
    assert listed["memories"][0]["content"] == "Remember this"
    added = await management_ui.async_management_command(
        hass, "user", False, _residual_message("memories", "add", content="New")
    )
    assert added["status"] == "created"
    assert added["memory"]["memory_id"] == "memory-1"
    assert added["memory"]["revision"]
    assert (
        await management_ui.async_management_command(
            hass, "user", False, _residual_message("memories", "update", memory_id="memory-1")
        )
    )["status"] == "updated"
    assert await management_ui.async_management_command(
        hass, "user", False, _residual_message("memories", "delete", memory_id="memory-1")
    ) == {"deleted": 1}
    with pytest.raises(HomeAssistantError, match="Explicit confirmation"):
        await management_ui.async_management_command(
            hass, "user", False, _residual_message("memories", "clear")
        )
    assert await management_ui.async_management_command(
        hass, "user", False, _residual_message("memories", "clear", confirm=True)
    ) == {"deleted": 2}
    with pytest.raises(HomeAssistantError, match="Administrator"):
        await management_ui.async_management_command(
            hass,
            "user",
            False,
            _residual_message("memories", "reassign_legacy", memory_ids=[]),
        )
    with pytest.raises(HomeAssistantError, match="list of strings"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            _residual_message(
                "memories",
                "reassign_legacy",
                target_scope_id="user:user",
                memory_ids=[1],
            ),
        )
    assert await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message(
            "memories",
            "reassign_legacy",
            target_scope_id="user:user",
            memory_ids=["memory-1"],
        ),
    ) == {"reassigned": 1}

    source = KnowledgeSource(
        source_id="source-1",
        title="Notes",
        description="",
        content="Text",
        enabled=True,
        created_at="2026-01-01",
        updated_at="2026-01-01",
    )
    library = SimpleNamespace(
        total_source_count=1,
        async_list=AsyncMock(return_value=[{"source_id": "source-1"}]),
        stats=lambda: {"source_count": 1},
        async_get=AsyncMock(return_value=source),
        async_create=AsyncMock(return_value=source),
        async_update=AsyncMock(return_value=source),
        async_delete=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        management_ui, "async_get_knowledge", AsyncMock(return_value=library)
    )
    assert (
        await management_ui.async_management_command(
            hass, "user", True, _residual_message("knowledge", "list")
        )
    )["stats"] == {"source_count": 1}
    for action in ("get", "create", "update"):
        assert "source" in await management_ui.async_management_command(
            hass, "user", True, _residual_message("knowledge", action)
        )
    with pytest.raises(HomeAssistantError, match="Explicit confirmation"):
        await management_ui.async_management_command(
            hass, "user", True, _residual_message("knowledge", "delete")
        )
    deleted = await management_ui.async_management_command(
        hass, "user", True, _residual_message("knowledge", "delete", confirm=True)
    )
    assert deleted["deleted"] == 1
    assert deleted["feature_status"]["source_count"] == 1

    with pytest.raises(HomeAssistantError, match="settings must be an object"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("settings", "update", settings=[])
        )
    settings = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        _residual_message("settings", "update", settings={"archive_enabled": True}),
    )
    assert settings["settings"]["archive_enabled"] is True
    with pytest.raises(HomeAssistantError, match="Unknown unknown management action"):
        await management_ui.async_management_command(
            hass, "admin", True, _residual_message("unknown", "unknown")
        )


async def test_websocket_and_setup_wiring(monkeypatch) -> None:
    connection = SimpleNamespace(
        user=SimpleNamespace(id="user", is_admin=False),
        send_error=MagicMock(),
        send_result=MagicMock(),
    )
    monkeypatch.setattr(
        management_ui,
        "async_management_command",
        AsyncMock(side_effect=HomeAssistantError("bad request")),
    )
    websocket_handler = management_ui.websocket_management.__wrapped__
    await websocket_handler(_residual_hass(), connection, {"id": 7, "action": "x"})
    connection.send_error.assert_called_once_with(7, "invalid_request", "bad request")

    management_ui.async_management_command.side_effect = None
    management_ui.async_management_command.return_value = {"ok": True}
    await websocket_handler(_residual_hass(), connection, {"id": 8, "action": "x"})
    connection.send_result.assert_called_once_with(8, {"ok": True})

    hass = _residual_hass()
    register_command = MagicMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda callback: callback())
    register_panel = AsyncMock()
    monkeypatch.setattr(
        management_ui.websocket_api, "async_register_command", register_command
    )
    monkeypatch.setattr(
        management_ui.panel_custom, "async_register_panel", register_panel
    )
    await management_ui.async_setup_management_ui(hass)
    await management_ui.async_setup_management_ui(hass)
    hass.http.async_register_static_paths.assert_awaited_once()
    register_command.assert_called_once_with(hass, management_ui.websocket_management)
    register_panel.assert_awaited_once()
    assert register_panel.await_args.kwargs["frontend_url_path"] == "extended-openai"
    registered_assets = {
        asset.url_path
        for asset in hass.http.async_register_static_paths.await_args.args[0]
    }
    assert registered_assets == {f"/{DOMAIN}/frontend"}
    module_url = register_panel.await_args.kwargs["module_url"]
    assert module_url.startswith(f"/{DOMAIN}/frontend/assets/management-")
    assert module_url.endswith(".js")
