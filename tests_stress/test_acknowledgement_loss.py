"""Nightly real-Store commits whose caller loses the acknowledgement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.extended_openai_conversation_responses.delayed_tools import (
    DelayedToolManager,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    HomeAssistantKnowledgeStorage,
    KnowledgeLibrary,
)
from custom_components.extended_openai_conversation_responses.memory import (
    HomeAssistantMemoryStorage,
    PersistentMemory,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    STORAGE_VERSION as RULES_VERSION,
    RequestRules,
    RequestRuleStore,
)
from homeassistant.core import HomeAssistant
from tests_stress.conftest import record
from tests_stress.test_os_storage_faults import real_store_io as real_store_io


@pytest.mark.parametrize(
    "owner", ("persistent_memory", "request_rules", "knowledge", "delayed_tools")
)
@pytest.mark.usefixtures("real_store_io")
async def test_atomic_commit_survives_lost_acknowledgement(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    owner: str,
) -> None:
    """Cancellation after HA's atomic write must retain one authoritative object."""
    entry_id, subentry_id = "ack-entry", "ack-agent"
    if owner == "persistent_memory":
        storage = HomeAssistantMemoryStorage(hass, entry_id, subentry_id)
        manager = PersistentMemory(storage)
        await manager.async_initialize()
        store = storage._store
        mutate = manager.async_add(
            "ack-user", "Committed despite lost reply", "ack", "explicit"
        )
    elif owner == "request_rules":
        store = RequestRuleStore(
            hass, RULES_VERSION, "extended_openai_conversation.ack_rules"
        )
        manager = RequestRules(store)
        await manager.async_initialize()
        before_revision = manager.revision()
        mutate = manager.async_set_groups(
            [{"id": "ack-group", "name": "Acknowledgement group"}],
            expected_revision=before_revision,
        )
    elif owner == "knowledge":
        storage = HomeAssistantKnowledgeStorage(hass, entry_id, subentry_id)
        manager = KnowledgeLibrary(storage)
        await manager.async_initialize()
        store = storage._store
        mutate = manager.async_create(
            "Acknowledgement source", "Committed source", "Persisted content"
        )
    else:
        manager = DelayedToolManager(hass)
        await manager.async_setup()
        store = manager._store
        entity = SimpleNamespace(
            entry=SimpleNamespace(entry_id=entry_id),
            subentry=SimpleNamespace(subentry_id=subentry_id),
        )
        mutate = manager.async_schedule(
            entity, "scheduled_ack_tool", {"delay": {"hours": 1}}, None
        )

    committed = asyncio.Event()
    release_ack = asyncio.Event()
    original_write = store._async_write_data

    async def write_then_lose_ack(data: dict[str, Any]) -> None:
        await original_write(data)
        committed.set()
        await release_ack.wait()

    with monkeypatch.context() as patch:
        patch.setattr(store, "_async_write_data", write_then_lose_ack)
        task = asyncio.create_task(mutate)
        await asyncio.wait_for(committed.wait(), timeout=10)
        assert Path(store.path).is_file()
        task.cancel()
        release_ack.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    if owner == "persistent_memory":
        runtime = await manager.async_list("ack-user")
        reloaded = PersistentMemory(
            HomeAssistantMemoryStorage(hass, entry_id, subentry_id)
        )
        await reloaded.async_initialize()
        disk = await reloaded.async_list("ack-user")
        assert len(runtime) == len(disk) == 1
        assert runtime[0].memory_id == disk[0].memory_id
        retry = await manager.async_add(
            "ack-user", "Committed despite lost reply", "ack", "explicit"
        )
        assert retry["status"] == "duplicate"
        assert retry["memory"]["memory_id"] == disk[0].memory_id
    elif owner == "request_rules":
        reloaded = RequestRules(
            RequestRuleStore(
                hass, RULES_VERSION, "extended_openai_conversation.ack_rules"
            )
        )
        await reloaded.async_initialize()
        assert (
            manager.snapshot()["groups"]
            == reloaded.snapshot()["groups"]
            == [{"id": "ack-group", "name": "Acknowledgement group"}]
        )
        assert manager.revision() != before_revision
        assert reloaded.revision() != before_revision
        with pytest.raises(ValueError, match="changed in another tab"):
            await manager.async_set_groups([], expected_revision=before_revision)
        with pytest.raises(ValueError, match="changed in another tab"):
            await reloaded.async_set_groups([], expected_revision=before_revision)
    elif owner == "knowledge":
        reloaded = KnowledgeLibrary(
            HomeAssistantKnowledgeStorage(hass, entry_id, subentry_id)
        )
        await reloaded.async_initialize()
        runtime = await manager.async_list()
        disk = await reloaded.async_list()
        assert len(runtime) == len(disk) == 1
        assert runtime[0]["source_id"] == disk[0]["source_id"]
        assert (
            await reloaded.async_get(disk[0]["source_id"])
        ).content == "Persisted content"
    else:
        reloaded = DelayedToolManager(hass)
        await reloaded.async_setup()
        assert len(manager._records) == len(reloaded._records) == 1
        assert set(manager._records) == set(reloaded._records)
        assert next(iter(reloaded._records.values())).tool_name == "scheduled_ack_tool"

    record(
        stress_trace,
        "acknowledgement_loss",
        owner=owner,
        seam="real_ha_atomic_write_after_commit",
        runtime_disk_converged=True,
        duplicate_objects=False,
    )
