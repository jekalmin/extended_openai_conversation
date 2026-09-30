"""Regression tests for Knowledge source availability controls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.extended_openai_conversation_responses import management_ui
from custom_components.extended_openai_conversation_responses.const import (
    CONF_KNOWLEDGE_ENABLED,
)
from custom_components.extended_openai_conversation_responses.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    KnowledgeLibrary,
    KnowledgeStore,
)
from custom_components.extended_openai_conversation_responses.live_subentry_updates import (
    is_live_subentry_update,
)


class FakeStorage:
    """Minimal durable storage double."""

    def __init__(self, data=None) -> None:
        self.data = data

    async def async_load(self):
        return self.data

    async def async_save(self, data):
        self.data = data


async def _library(data=None) -> KnowledgeLibrary:
    library = KnowledgeLibrary(FakeStorage(data))
    await library.async_initialize()
    return library


def _stored_source(source_id: str = "source-1", *, enabled_marker=True):
    source = {
        "source_id": source_id,
        "title": "Kitchen",
        "description": "Layout",
        "content": "Tea towels beside oven",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    if enabled_marker is not None:
        source["enabled"] = enabled_marker
    return source


async def test_current_storage_migration_enables_existing_sources() -> None:
    store = KnowledgeStore.__new__(KnowledgeStore)
    old = {"sources": [_stored_source(enabled_marker=None)]}

    migrated = await store._async_migrate_func(1, 0, old)

    assert migrated["sources"][0]["enabled"] is True


async def test_legacy_sources_default_enabled_without_migration() -> None:
    library = await _library({"sources": [_stored_source(enabled_marker=None)]})

    source = await library.async_get("source-1")

    assert source.enabled is True
    assert library.source_count == 1


async def test_disable_reenable_updates_model_index_but_preserves_source() -> None:
    library = await _library()
    source = await library.async_create(
        "Kitchen", "Layout", "Tea towels beside oven", enabled=True
    )
    assert await library.async_search("tea towels")
    assert (await library.async_catalog())["total"] == 1

    disabled = await library.async_update(source.source_id, enabled=False)

    assert disabled.enabled is False
    assert library.source_count == 0
    assert library.total_source_count == 1
    assert (await library.async_get(source.source_id)).content == source.content
    assert await library.async_search("tea towels") == []
    assert (await library.async_catalog())["total"] == 0
    with pytest.raises(ValueError, match="not found"):
        await library.async_get_section(source.source_id)

    enabled = await library.async_update(source.source_id, enabled=True)

    assert enabled.enabled is True
    assert library.source_count == 1
    assert [
        result.source_id for result in await library.async_search("tea towels")
    ] == [source.source_id]


async def test_disabled_requested_source_does_not_expand_search_filter() -> None:
    library = await _library()
    enabled = await library.async_create("Enabled", "", "Shared needle phrase")
    disabled = await library.async_create(
        "Disabled", "", "Shared needle phrase", enabled=False
    )

    allowed, ignored = library.resolve_source_filter([disabled.source_id])

    assert allowed is None
    assert ignored == [disabled.source_id]
    results = await library.async_search("shared needle", [disabled.source_id])
    assert [result.source_id for result in results] == [enabled.source_id]


async def test_guest_allowlist_cannot_reenable_disabled_source() -> None:
    library = await _library()
    await library.async_create("Enabled", "", "General reference")
    disabled = await library.async_create(
        "Disabled", "", "Private reference", enabled=False
    )
    entity = ExtendedOpenAIAgentEntity.__new__(ExtendedOpenAIAgentEntity)
    entity.subentry = SimpleNamespace(data={CONF_KNOWLEDGE_ENABLED: True})
    entity._knowledge = library
    policy = SimpleNamespace(
        knowledge_access=True,
        knowledge_source_ids=frozenset({disabled.source_id}),
    )

    with patch.object(
        ExtendedOpenAIAgentEntity, "_effective_guest_policy", return_value=policy
    ):
        listed = await entity._async_execute_knowledge_tool("list", {})
        assert listed["sources"] == []
        assert listed["total"] == 0
        with pytest.raises(ValueError, match="not found"):
            await entity._async_execute_knowledge_tool(
                "get", {"source_id": disabled.source_id}
            )


async def test_backup_preserves_state_and_legacy_backup_defaults_enabled() -> None:
    library = await _library()
    disabled = await library.async_create("Manual", "", "Reference", enabled=False)

    backup = await library.async_backup_data()
    assert backup["sources"][0]["enabled"] is False

    restored_sources = KnowledgeLibrary.validate_backup_data(backup)
    restored = await _library()
    await restored.async_replace_backup(restored_sources)
    assert (await restored.async_get(disabled.source_id)).enabled is False
    assert restored.source_count == 0

    legacy = _stored_source("legacy", enabled_marker=None)
    legacy_sources = KnowledgeLibrary.validate_backup_data({"sources": [legacy]})
    assert legacy_sources[0].enabled is True


async def test_unified_management_preserves_knowledge_availability() -> None:
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Assistant",
        data={CONF_KNOWLEDGE_ENABLED: True},
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain="extended_openai_conversation_responses",
        title="OpenAI",
        subentries={"agent-1": subentry},
    )
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = entry
    library = await _library()
    base = {
        "section": "knowledge",
        "entry_id": "entry-1",
        "subentry_id": "agent-1",
    }

    with patch.object(
        management_ui,
        "async_get_knowledge",
        AsyncMock(return_value=library),
    ):
        created = await management_ui.async_management_command(
            hass,
            "admin-user",
            True,
            {
                **base,
                "action": "create",
                "title": "Manual",
                "description": "Reference",
                "content": "Service procedure",
                "enabled": False,
            },
        )
        source_id = created["source"]["source_id"]
        assert created["source"]["enabled"] is False
        assert library.source_count == 0

        listed = await management_ui.async_management_command(
            hass, "admin-user", True, {**base, "action": "list"}
        )
        assert listed["sources"][0]["enabled"] is False
        assert "content" not in listed["sources"][0]

        updated = await management_ui.async_management_command(
            hass,
            "admin-user",
            True,
            {
                **base,
                "action": "update",
                "source_id": source_id,
                "title": "Renamed manual",
            },
        )
        assert updated["source"]["title"] == "Renamed manual"
        assert updated["source"]["enabled"] is False
        assert library.source_count == 0

        reenabled = await management_ui.async_management_command(
            hass,
            "admin-user",
            True,
            {
                **base,
                "action": "update",
                "source_id": source_id,
                "enabled": True,
            },
        )
        assert reenabled["source"]["enabled"] is True
        assert library.source_count == 1

        defaulted = await management_ui.async_management_command(
            hass,
            "admin-user",
            True,
            {
                **base,
                "action": "create",
                "title": "Default",
                "description": "",
                "content": "Enabled by default",
            },
        )
        assert defaulted["source"]["enabled"] is True


async def test_management_set_enabled_updates_only_knowledge_config() -> None:
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Assistant",
        data={CONF_KNOWLEDGE_ENABLED: True},
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain="extended_openai_conversation_responses",
        title="OpenAI",
        data={},
        subentries={"agent-1": subentry},
    )
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = entry
    live_update_contexts: list[bool] = []
    hass.config_entries.async_update_subentry.side_effect = lambda *_args, **_kwargs: (
        live_update_contexts.append(is_live_subentry_update())
    )
    library = await _library()
    await library.async_create("Reference", "", "Content", enabled=True)
    base = {
        "section": "knowledge",
        "entry_id": "entry-1",
        "subentry_id": "agent-1",
    }

    with patch.object(
        management_ui,
        "async_get_knowledge",
        AsyncMock(return_value=library),
    ):
        result = await management_ui.async_management_command(
            hass,
            "admin-user",
            True,
            {**base, "action": "set_enabled", "enabled": False},
        )

    assert result["knowledge_enabled"] is False
    assert result["feature_status"]["enabled"] is False
    assert isinstance(result["revision"], str)
    hass.config_entries.async_update_subentry.assert_called_once()
    args = hass.config_entries.async_update_subentry.call_args
    assert args.args[:2] == (entry, subentry)
    assert args.kwargs["data"][CONF_KNOWLEDGE_ENABLED] is False
    assert live_update_contexts == [True]


async def test_enabled_field_requires_boolean() -> None:
    library = await _library()
    with pytest.raises(ValueError, match="enabled"):
        await library.async_create("Bad", "", "Content", enabled="yes")  # type: ignore[arg-type]


async def test_enabling_knowledge_after_adding_available_source_returns_current_status() -> (
    None
):
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Assistant",
        data={CONF_KNOWLEDGE_ENABLED: False},
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain="extended_openai_conversation_responses",
        title="OpenAI",
        data={},
        subentries={"agent-1": subentry},
    )
    hass = MagicMock()
    hass.config_entries.async_get_entry.return_value = entry
    library = await _library()
    base = {
        "section": "knowledge",
        "entry_id": entry.entry_id,
        "subentry_id": subentry.subentry_id,
    }
    with patch.object(
        management_ui, "async_get_knowledge", AsyncMock(return_value=library)
    ):
        created = await management_ui.async_management_command(
            hass,
            "admin-user",
            True,
            {
                **base,
                "action": "create",
                "title": "Reference",
                "content": "Available reference",
                "enabled": True,
            },
        )
        assert created["summary"]["enabled"] is True
        assert created["feature_status"]["state"] == "disabled"
        enabled = await management_ui.async_management_command(
            hass, "admin-user", True, {**base, "action": "set_enabled", "enabled": True}
        )
    assert enabled["feature_status"]["source_count"] == 1
    assert enabled["feature_status"]["state"] == "available"
    assert enabled["feature_status"]["label"] == "Available"
