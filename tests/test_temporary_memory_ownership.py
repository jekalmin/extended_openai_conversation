"""Canonical Temporary Memory ownership and integration-boundary tests."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation_responses import (
    management_request_preview,
    management_ui,
    temporary_memory as ownership,
    temporary_memory as temporary_module,
)
from custom_components.extended_openai_conversation_responses.scope import (
    SHARED_HOUSEHOLD_SCOPE_ID,
)
from custom_components.extended_openai_conversation_responses.temporary_memory import (
    MAX_ACTIVE_RECORDS,
    MAX_DELETE_RECORDS,
    TemporaryMemory,
    TemporaryMemoryRecord,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util


def _record(
    memory_id: str,
    *,
    owner: str | None,
    scope_id: str = "conversation:test",
    expires_delta: timedelta = timedelta(hours=1),
    updated_delta: timedelta = timedelta(),
) -> TemporaryMemoryRecord:
    now = datetime.now(UTC)
    return TemporaryMemoryRecord(
        memory_id=memory_id,
        scope_id=scope_id,
        content=f"content-{memory_id}",
        category="general",
        source="automatic",
        expires_at=(now + expires_delta).isoformat(),
        created_at=(now - timedelta(minutes=1)).isoformat(),
        updated_at=(now + updated_delta).isoformat(),
        owner_scope_id=owner,
    )


@pytest.mark.asyncio
async def test_conversation_contract_binds_and_resets_owner_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversation reads/tools inherit only the resolved retained-data owner."""
    from custom_components.extended_openai_conversation_responses import conversation

    seen: list[tuple[str, str | None]] = []

    async def active(_scope):
        seen.append(("retrieve", ownership._ACTIVE_OWNER_SCOPE_ID.get()))
        return [_record("one", owner=ownership._ACTIVE_OWNER_SCOPE_ID.get())]

    async def add(*_args):
        seen.append(("add", ownership._ACTIVE_OWNER_SCOPE_ID.get()))
        return {"ok": True}

    entity_cls = conversation.ExtendedOpenAIAgentEntity
    entity = object.__new__(entity_cls)
    entity.subentry = SimpleNamespace(data={"temporary_memory": "enabled"})
    entity._effective_guest_policy = lambda: SimpleNamespace(temporary_memory=True)
    entity._temporary_memory = SimpleNamespace(async_active=active, async_add=add)
    assert await entity._async_retrieve_temporary_memories() == []
    with pytest.raises(RuntimeError, match="temporary memory is unavailable"):
        await entity._async_execute_temporary_memory_tool("add", {})

    scope_token = conversation._ACTIVE_SCOPE.set(
        SimpleNamespace(scope_type="user", user_id="alice")
    )
    temporary_token = conversation._ACTIVE_TEMPORARY_SCOPE.set("conversation:one")
    try:
        records = await entity_cls._async_retrieve_temporary_memories(entity)
        result = await entity_cls._async_execute_temporary_memory_tool(
            entity, "add", {"content": "fact", "expires_at": "later"}
        )
    finally:
        conversation._ACTIVE_TEMPORARY_SCOPE.reset(temporary_token)
        conversation._ACTIVE_SCOPE.reset(scope_token)

    assert [record.memory_id for record in records] == ["one"]
    assert result == {"ok": True}
    assert seen == [("retrieve", "user:alice"), ("add", "user:alice")]
    assert ownership._ACTIVE_OWNER_SCOPE_ID.get() is None


@pytest.mark.asyncio
async def test_management_contract_validates_and_enriches_owner_operations(
    hass,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Management commands cannot escape Personal/Shared ownership boundaries."""
    from custom_components.extended_openai_conversation_responses import (
        management_loading_performance as loading,
        management_ui,
    )

    fallback_results = []

    async def fallback(*_args):
        return fallback_results.pop(0)

    monkeypatch.setattr(loading, "async_scope_catalog", fallback)

    class Manager:
        def stats(self) -> dict[str, int]:
            return {"active": 1}

        async def async_list_owned(self, owner: str) -> list[TemporaryMemoryRecord]:
            return [_record("owned", owner=owner, scope_id=owner)]

        async def async_delete_owned(self, owner: str, ids: list[str]) -> int:
            assert owner == "user:alice"
            assert ids == ["owned"]
            return 1

        async def async_update_owned(
            self,
            owner: str,
            memory_id: str,
            content: str | None,
            expires_at: str | None,
            category: str | None,
        ) -> TemporaryMemoryRecord:
            if content == "bad":
                raise ValueError("invalid update")
            return _record(memory_id, owner=owner, scope_id=owner)

        def owner_counts(self) -> dict[str, int]:
            return {"user:alice": 2, SHARED_HOUSEHOLD_SCOPE_ID: 4}

    manager = Manager()
    entry = SimpleNamespace(entry_id="entry")
    subentry = SimpleNamespace(subentry_id="sub")

    monkeypatch.setattr(
        management_ui,
        "entry_and_agent",
        lambda _hass, _entry_id, _subentry_id: (entry, subentry),
    )
    monkeypatch.setattr(
        management_ui,
        "_selected_scope",
        lambda _user_id, _is_admin, requested: requested or "user:alice",
    )

    async def get_manager(*_args: Any) -> Manager:
        return manager

    monkeypatch.setattr(management_ui, "async_get_temporary_memory", get_manager)

    base = {
        "section": "memories",
        "entry_id": "entry",
        "subentry_id": "sub",
        "scope_id": "user:alice",
    }
    listed = await management_ui.async_management_command(
        hass, "alice", False, base | {"action": "temporary_list"}
    )
    assert listed["scope_id"] == "user:alice"
    assert listed["memories"][0]["owner_scope_id"] == "user:alice"

    with pytest.raises(HomeAssistantError, match="memory_id is required"):
        await management_ui.async_management_command(
            hass, "alice", False, base | {"action": "temporary_delete"}
        )

    deleted = await management_ui.async_management_command(
        hass,
        "alice",
        False,
        base | {"action": "temporary_delete", "memory_id": "owned"},
    )
    assert deleted == {"deleted": 1}

    with pytest.raises(HomeAssistantError, match="must be strings when supplied"):
        await management_ui.async_management_command(
            hass,
            "alice",
            False,
            base
            | {
                "action": "temporary_update",
                "memory_id": "owned",
                "content": 123,
            },
        )

    with pytest.raises(HomeAssistantError, match="at least one Temporary Memory field"):
        await management_ui.async_management_command(
            hass,
            "alice",
            False,
            base | {"action": "temporary_update", "memory_id": "owned"},
        )

    with pytest.raises(HomeAssistantError, match="invalid update"):
        await management_ui.async_management_command(
            hass,
            "alice",
            False,
            base
            | {
                "action": "temporary_update",
                "memory_id": "owned",
                "content": "bad",
            },
        )

    updated = await management_ui.async_management_command(
        hass,
        "alice",
        False,
        base
        | {
            "action": "temporary_update",
            "memory_id": "owned",
            "content": "good",
        },
    )
    assert updated["memory"]["owner_scope_id"] == "user:alice"

    monkeypatch.setattr(
        management_ui,
        "_selected_scope",
        lambda *_args: "device:kitchen",
    )
    with pytest.raises(HomeAssistantError, match="Personal or Shared scopes"):
        await management_ui.async_management_command(
            hass, "alice", False, base | {"action": "temporary_list"}
        )

    fallback_results.extend(
        [
            {
                "scopes": [
                    {"scope_id": "user:alice", "temporary_memory_count": 2},
                    {
                        "scope_id": SHARED_HOUSEHOLD_SCOPE_ID,
                        "temporary_memory_count": 4,
                    },
                    "ignore-me",
                ]
            },
            {"scopes": "not-a-list"},
        ]
    )
    catalog = await management_ui.async_management_command(
        hass,
        "alice",
        True,
        {
            "section": "scopes",
            "action": "catalog",
            "entry_id": "entry",
            "subentry_id": "sub",
        },
    )
    assert catalog["scopes"][0]["temporary_memory_count"] == 2
    assert catalog["scopes"][1]["temporary_memory_count"] == 4

    unchanged = await management_ui.async_management_command(
        hass,
        "alice",
        True,
        {
            "section": "scopes",
            "action": "catalog",
            "entry_id": "entry",
            "subentry_id": "sub",
        },
    )
    assert unchanged == {"scopes": "not-a-list"}


async def test_direct_manager_backup_and_owned_helpers() -> None:
    store = SimpleNamespace(
        async_load=AsyncMock(return_value=None), async_save=AsyncMock()
    )
    memory = TemporaryMemory(store)
    await memory.async_initialize()
    records = [
        _record("valid", owner="user:alice"),
        _record("legacy", owner=None, scope_id=SHARED_HOUSEHOLD_SCOPE_ID),
        _record("invalid", owner="device:kitchen"),
    ]
    validated = TemporaryMemory.validate_backup_data(
        {"records": [asdict(r) for r in records]}
    )
    assert [r.memory_id for r in validated] == ["valid", "legacy"]
    assert validated[1].owner_scope_id == SHARED_HOUSEHOLD_SCOPE_ID
    await memory.async_replace_backup(records)
    assert [
        item.memory_id
        for item in await memory.async_list(owner_scope_id="user:alice")
    ] == ["valid"]
    assert [
        item.memory_id
        for item in await memory.async_list_all(
            owner_scope_id=SHARED_HOUSEHOLD_SCOPE_ID
        )
    ] == ["legacy"]
    with pytest.raises(ValueError, match="resolved Personal or Shared owner"):
        await memory.async_list()
    with pytest.raises(ValueError, match="resolved Personal or Shared owner"):
        await memory.async_list_all(owner_scope_id="device:kitchen")
    updated = await memory.async_update_owned(
        "user:alice", "valid", "new", None, "note"
    )
    assert updated.owner_scope_id == "user:alice"
    assert updated.scope_id == "conversation:test"
    assert memory.owner_counts() == {"user:alice": 1, SHARED_HOUSEHOLD_SCOPE_ID: 1}
    with pytest.raises(ValueError, match="memory_ids must contain"):
        await memory.async_delete_owned("user:alice", [])
    with pytest.raises(ValueError, match="memory_ids must contain"):
        await memory.async_delete_owned(
            "user:alice", [str(i) for i in range(MAX_DELETE_RECORDS + 1)]
        )
    assert await memory.async_delete_owned("user:alice", ["legacy"]) == 0
    assert await memory.async_delete_owned("user:alice", ["valid"]) == 1


async def test_snapshot_contract_fails_closed_before_io_and_uses_bound_owner(
    monkeypatch,
) -> None:
    from custom_components.extended_openai_conversation_responses import management_ui

    store = SimpleNamespace(
        async_load=AsyncMock(
            return_value={
                "records": [
                    asdict(_record("one", owner="user:alice")),
                    asdict(_record("foreign", owner="user:bob")),
                ]
            }
        )
    )

    def factory(*_):
        return store

    monkeypatch.setattr(temporary_module, "_temporary_memory_store", factory)
    assert (
        await temporary_module.async_read_temporary_memory_snapshot(
            object(), "entry", "sub", "scope"
        )
        == []
    )
    store.async_load.assert_not_awaited()
    token = temporary_module._ACTIVE_OWNER_SCOPE_ID.set("user:alice")
    try:
        records = await management_request_preview.async_read_temporary_memory_snapshot(
            object(), "entry", "sub", "different-continuity"
        )
    finally:
        temporary_module._ACTIVE_OWNER_SCOPE_ID.reset(token)
    assert [r.memory_id for r in records] == ["one"]
    store.async_load.assert_awaited_once()
    assert (
        management_request_preview.async_read_temporary_memory_snapshot
        is temporary_module.async_read_temporary_memory_snapshot
    )


# Low-level owner normalization and selection invariants.

def _ownership_record(
    memory_id: str,
    *,
    scope_id: str = "user:one",
    owner_scope_id: str | None = "user:one",
    expires_delta: int = 3600,
    updated_delta: int = 0,
) -> TemporaryMemoryRecord:
    now = dt_util.utcnow()
    created = (now - timedelta(seconds=30)).isoformat()
    updated = (now + timedelta(seconds=updated_delta)).isoformat()
    expires = (now + timedelta(seconds=expires_delta)).isoformat()
    return TemporaryMemoryRecord(
        memory_id=memory_id,
        scope_id=scope_id,
        content=f"content-{memory_id}",
        category="general",
        source="automatic",
        expires_at=expires,
        created_at=created,
        updated_at=updated,
        owner_scope_id=owner_scope_id,
    )


def test_owner_validation_and_resolved_scope_translation() -> None:
    assert ownership._valid_owner_scope_id(" user:abc ") == "user:abc"
    assert (
        ownership._valid_owner_scope_id(SHARED_HOUSEHOLD_SCOPE_ID)
        == SHARED_HOUSEHOLD_SCOPE_ID
    )
    assert ownership._valid_owner_scope_id("user:") is None
    assert ownership._valid_owner_scope_id("device:kitchen") is None
    assert ownership._valid_owner_scope_id(42) is None
    assert ownership._valid_owner_scope_id("user:" + "x" * 124) is None

    assert (
        ownership._owner_from_resolved_scope(
            SimpleNamespace(scope_type="user", user_id="abc")
        )
        == "user:abc"
    )
    assert (
        ownership._owner_from_resolved_scope(
            SimpleNamespace(scope_type="shared", user_id=None)
        )
        == SHARED_HOUSEHOLD_SCOPE_ID
    )
    assert (
        ownership._owner_from_resolved_scope(
            SimpleNamespace(scope_type="device", user_id="abc")
        )
        is None
    )
    assert ownership._owner_from_resolved_scope(None) is None


def test_require_owner_uses_context_and_rejects_missing_owner() -> None:
    token = ownership._ACTIVE_OWNER_SCOPE_ID.set("user:context")
    try:
        assert ownership._require_owner_scope_id() == "user:context"
    finally:
        ownership._ACTIVE_OWNER_SCOPE_ID.reset(token)

    with pytest.raises(ValueError, match="resolved Personal or Shared owner"):
        ownership._require_owner_scope_id("device:kitchen")


def test_record_owner_normalization_preserves_valid_and_migrates_only_safe_legacy() -> (
    None
):
    valid = _ownership_record("valid", owner_scope_id="user:one")
    spaced = replace(valid, memory_id="spaced", owner_scope_id=" user:one ")
    legacy_safe = _ownership_record("legacy", owner_scope_id=None, scope_id="user:legacy")
    legacy_unsafe = _ownership_record(
        "unsafe", owner_scope_id=None, scope_id="conversation:123"
    )
    invalid = _ownership_record("invalid", owner_scope_id="device:kitchen")

    assert ownership._normalize_record_owner(valid) is valid
    assert ownership._normalize_record_owner(spaced).owner_scope_id == "user:one"
    assert (
        ownership._normalize_record_owner(legacy_safe).owner_scope_id == "user:legacy"
    )
    assert ownership._normalize_record_owner(legacy_unsafe) is None
    assert ownership._normalize_record_owner(invalid) is None


@pytest.mark.asyncio
async def test_normalize_loaded_records_prunes_invalid_and_overflow_and_persists() -> (
    None
):
    records = {
        f"r{i}": _ownership_record(f"r{i}", updated_delta=i)
        for i in range(MAX_ACTIVE_RECORDS + 2)
    }
    records["invalid"] = _ownership_record("invalid", owner_scope_id="device:kitchen")
    manager = TemporaryMemory(None)
    manager._records = records
    manager._async_save_locked = AsyncMock()
    async with manager._lock:
        await manager._async_normalize_loaded_records_locked()

    assert len(manager._records) == MAX_ACTIVE_RECORDS
    assert "invalid" not in manager._records
    assert "r0" not in manager._records
    assert "r1" not in manager._records
    assert manager.invalid_owners_pruned == 1
    assert manager.overflow_pruned == 2
    manager._async_save_locked.assert_awaited_once()


@pytest.mark.asyncio
async def test_normalize_loaded_records_reconciles_after_save_failure() -> None:
    legacy = _ownership_record("legacy", owner_scope_id=None)
    manager = TemporaryMemory(None)
    manager._records = {"legacy": legacy}
    manager._store = SimpleNamespace(
        async_load=AsyncMock(return_value={"records": [asdict(legacy)]}),
        async_save=AsyncMock(side_effect=OSError("disk failed")),
    )
    with pytest.raises(OSError, match="disk failed"):
        async with manager._lock:
            await manager._async_normalize_loaded_records_locked()

    assert manager._records["legacy"].owner_scope_id == "user:one"
    assert manager._initialized


def test_records_for_owner_filters_expired_and_orders_deterministically() -> None:
    manager = SimpleNamespace(
        _records={
            "later": _ownership_record("later", expires_delta=7200),
            "earlier": _ownership_record("earlier", expires_delta=3600),
            "other": _ownership_record("other", owner_scope_id="user:other"),
            "expired": _ownership_record("expired", expires_delta=-10),
        }
    )
    memory = TemporaryMemory(None)
    memory._records = manager._records
    result = memory._records_for_owner("user:one")
    assert [item.memory_id for item in result] == ["earlier", "later"]


@pytest.mark.parametrize("entry_id", [None, "", 123])
async def test_temporary_memory_rejects_invalid_selection(hass, entry_id):
    with pytest.raises(HomeAssistantError, match="entry_id is required"):
        await management_ui.async_management_command(
            hass,
            "user",
            False,
            {
                "section": "memories",
                "action": "temporary_list",
                "entry_id": entry_id,
                "subentry_id": "agent",
            },
        )
    hass.config_entries.async_get_entry.assert_not_called()


async def test_backup_restore_enforces_global_record_ceiling_and_keeps_newest() -> None:
    raw = [
        asdict(_ownership_record(f"r{index}", updated_delta=index))
        for index in range(MAX_ACTIVE_RECORDS + 4)
    ]
    with pytest.raises(ValueError, match="temporary memory count is invalid"):
        TemporaryMemory.validate_backup_data({"records": raw})
    validated = [
        TemporaryMemory.validate_backup_data({"records": [record]})[0]
        for record in raw
    ]

    store = SimpleNamespace(
        async_load=AsyncMock(return_value=None),
        async_save=AsyncMock(),
    )
    memory = TemporaryMemory(store)
    await memory.async_initialize()
    await memory.async_replace_backup(validated)

    restored = await memory.async_list_owned("user:one")
    assert len(restored) == MAX_ACTIVE_RECORDS
    ids = {record.memory_id for record in restored}
    assert f"r{MAX_ACTIVE_RECORDS + 3}" in ids
    assert "r0" not in ids
