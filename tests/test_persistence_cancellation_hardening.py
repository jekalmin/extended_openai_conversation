"""Cancellation regressions for shared durable-manager persistence guards."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import timedelta
from typing import Any

import pytest

from custom_components.extended_openai_conversation_responses.knowledge import (
    KnowledgeLibrary,
)
from custom_components.extended_openai_conversation_responses.memory import (
    PersistentMemory,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    RequestRules,
)
from custom_components.extended_openai_conversation_responses.temporary_memory import (
    TemporaryMemory,
)
from homeassistant.util import dt as dt_util

_MANAGER_TYPES = ("memory", "temporary_memory", "knowledge", "request_rules")


class BlockingStorage:
    """Storage double whose next save can be paused and failed deterministically."""

    def __init__(self) -> None:
        self.data: dict[str, Any] | None = None
        self.block_saves = False
        self.fail_saves = False
        self.fail_after_write = False
        self.save_started = asyncio.Event()
        self.release_save = asyncio.Event()

    async def async_load(self) -> dict[str, Any] | None:
        return deepcopy(self.data)

    async def async_save(self, data: dict[str, Any]) -> None:
        candidate = deepcopy(data)
        if self.block_saves:
            self.save_started.set()
            await self.release_save.wait()
        if self.fail_saves:
            raise RuntimeError("simulated Store failure")
        self.data = candidate
        if self.fail_after_write:
            self.fail_after_write = False
            raise RuntimeError("simulated lost Store acknowledgement")

    def arm(self, *, fail: bool = False) -> None:
        """Pause subsequent saves until explicitly released."""
        self.block_saves = True
        self.fail_saves = fail
        self.save_started = asyncio.Event()
        self.release_save = asyncio.Event()

    def fail_immediately(self) -> None:
        """Make subsequent saves fail without blocking."""
        self.block_saves = False
        self.fail_saves = True


async def _create_manager(kind: str, storage: BlockingStorage) -> Any:
    if kind == "memory":
        manager = PersistentMemory(storage)  # type: ignore[arg-type]
    elif kind == "temporary_memory":
        manager = TemporaryMemory(storage)  # type: ignore[arg-type]
    elif kind == "knowledge":
        manager = KnowledgeLibrary(storage)  # type: ignore[arg-type]
    elif kind == "request_rules":
        manager = RequestRules(storage)  # type: ignore[arg-type]
    else:  # pragma: no cover - protected by parametrization
        raise AssertionError(kind)
    await manager.async_initialize()
    return manager


def _snapshot(kind: str, manager: Any) -> Any:
    if kind == "memory":
        value = {"memories": dict(manager._memories)}
    elif kind == "temporary_memory":
        value = {
            "records": dict(manager._records),
            "expired_pruned": manager.expired_pruned,
        }
    elif kind == "knowledge":
        value = {"sources": dict(manager._sources)}
    elif kind == "request_rules":
        value = {
            "defaults": manager._defaults,
            "wording_groups": manager._wording_groups,
            "groups": manager._groups,
            "rules": manager._rules,
        }
    else:  # pragma: no cover - protected by parametrization
        raise AssertionError(kind)
    return deepcopy(value)


def _committed_snapshot(kind, manager):
    if kind == "temporary_memory":
        records, expired_pruned = manager._committed_state
        return deepcopy({"records": records, "expired_pruned": expired_pruned})
    if kind == "memory":
        return deepcopy({"memories": manager._committed_state.memories})
    return deepcopy(manager._committed_state)


async def _mutate(kind: str, manager: Any, marker: str) -> None:
    if kind == "memory":
        await manager.async_add(
            "user-1",
            f"Persistent memory {marker}",
            "test",
            "explicit",
        )
        return
    if kind == "temporary_memory":
        await manager.async_add(
            "user-1",
            f"Temporary memory {marker}",
            (dt_util.utcnow() + timedelta(days=30)).isoformat(),
            "test",
            owner_scope_id="user:test-owner",
        )
        return
    if kind == "knowledge":
        await manager.async_create(
            f"Knowledge {marker}",
            "Reference",
            f"Knowledge body {marker}",
        )
        return
    if kind == "request_rules":
        if marker == "first":
            settings = {
                "word_forms": False,
                "wording_alternatives": True,
                "fuzzy": True,
                "fuzzy_threshold": 85,
            }
        else:
            settings = {
                "word_forms": True,
                "wording_alternatives": False,
                "fuzzy": True,
                "fuzzy_threshold": 80,
            }
        await manager.async_set_defaults(settings)
        return
    raise AssertionError(kind)  # pragma: no cover


@pytest.mark.parametrize("kind", _MANAGER_TYPES)
async def test_cancellation_waits_for_successful_commit_and_advances_snapshot(
    kind: str,
) -> None:
    """Caller cancellation is deferred until each manager's save commits."""
    storage = BlockingStorage()
    manager = await _create_manager(kind, storage)
    baseline = _snapshot(kind, manager)
    baseline_storage = deepcopy(storage.data)

    storage.arm()
    mutation = asyncio.create_task(_mutate(kind, manager, "first"))
    await asyncio.wait_for(storage.save_started.wait(), timeout=1)

    mutation.cancel()
    await asyncio.sleep(0)
    assert not mutation.done()

    # A second cancellation request must still not tear down the in-flight save.
    mutation.cancel()
    await asyncio.sleep(0)
    assert not mutation.done()

    storage.release_save.set()
    with pytest.raises(asyncio.CancelledError):
        await mutation

    committed = _snapshot(kind, manager)
    assert committed != baseline
    assert storage.data != baseline_storage
    assert _committed_snapshot(kind, manager) == committed

    # Prove the cancellation-success snapshot became the rollback baseline rather
    # than merely leaving the newer live state in RAM accidentally.
    storage.fail_immediately()
    with pytest.raises(RuntimeError, match="simulated Store failure"):
        await _mutate(kind, manager, "second")
    assert _snapshot(kind, manager) == committed


@pytest.mark.parametrize("kind", _MANAGER_TYPES)
async def test_cancellation_waits_for_failed_commit_then_rolls_back(kind: str) -> None:
    """A cancelled caller sees cancellation only after failed persistence rolls back."""
    storage = BlockingStorage()
    manager = await _create_manager(kind, storage)
    baseline = _snapshot(kind, manager)
    baseline_storage = deepcopy(storage.data)

    storage.arm(fail=True)
    mutation = asyncio.create_task(_mutate(kind, manager, "first"))
    await asyncio.wait_for(storage.save_started.wait(), timeout=1)

    mutation.cancel()
    await asyncio.sleep(0)
    assert not mutation.done()

    storage.release_save.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await mutation

    assert isinstance(cancelled.value.__cause__, RuntimeError)
    assert _snapshot(kind, manager) == baseline
    assert storage.data == baseline_storage
    assert _committed_snapshot(kind, manager) == baseline


@pytest.mark.parametrize("kind", _MANAGER_TYPES)
async def test_cancelled_ambiguous_failure_reconciles_before_propagating_cancellation(
    kind: str,
) -> None:
    """Pending cancellation keeps precedence after a post-write failure is settled."""
    storage = BlockingStorage()
    manager = await _create_manager(kind, storage)
    baseline = _snapshot(kind, manager)
    storage.arm()
    storage.fail_after_write = True
    mutation = asyncio.create_task(_mutate(kind, manager, "first"))
    await asyncio.wait_for(storage.save_started.wait(), timeout=1)

    mutation.cancel()
    await asyncio.sleep(0)
    assert not mutation.done()
    storage.release_save.set()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await mutation
    assert isinstance(cancelled.value.__cause__, RuntimeError)
    assert _snapshot(kind, manager) != baseline
    assert _committed_snapshot(kind, manager) == _snapshot(kind, manager)


@pytest.mark.parametrize("kind", ["memory", "knowledge", "request_rules"])
async def test_cancelled_store_restores_committed_state(kind):
    storage = BlockingStorage()
    manager = await _create_manager(kind, storage)
    await _mutate(kind, manager, "first")
    committed = _snapshot(kind, manager)

    async def cancelled_save(data):
        raise asyncio.CancelledError

    storage.async_save = cancelled_save
    with pytest.raises(asyncio.CancelledError):
        await _mutate(kind, manager, "second")
    assert _snapshot(kind, manager) == committed


@pytest.mark.parametrize("kind", _MANAGER_TYPES)
async def test_failed_acknowledgement_reconciles_to_candidate_and_rebuilds_indexes(
    kind: str,
) -> None:
    """A Store exception after candidate persistence must adopt that candidate."""
    storage = BlockingStorage()
    manager = await _create_manager(kind, storage)
    storage.fail_after_write = True
    with pytest.raises(RuntimeError, match="lost Store acknowledgement"):
        await _mutate(kind, manager, "first")

    if kind == "memory":
        assert storage.data is not None
        assert len(manager._memories) == len(storage.data["memories"]) == 1
        memory = next(iter(manager._memories.values()))
        assert set(manager._key_index.values()) <= set(manager._memories)
        assert (await manager.async_search(memory.user_id, "Persistent memory first"))
    elif kind == "knowledge":
        assert storage.data is not None
        assert len(manager._sources) == len(storage.data["sources"]) == 1
        assert await manager.async_search("Knowledge body first")
        assert all(
            key[0] in manager._sources
            for keys in manager._token_index.values()
            for key in keys
        )
    elif kind == "temporary_memory":
        assert storage.data is not None
        assert len(manager._records) == len(storage.data["records"]) == 1
        active = await manager.async_active("user-1", "user:test-owner")
        assert [record.content for record in active] == ["Temporary memory first"]
    else:
        assert storage.data is not None
        assert manager._defaults == storage.data["defaults"]
        assert manager._matching_snapshot.phrases == ()


@pytest.mark.parametrize("kind", _MANAGER_TYPES)
async def test_unreadable_store_invalidates_manager_without_masking_write_error(
    kind: str,
) -> None:
    """When disk cannot be read, managers fail closed and later reload safely."""
    storage = BlockingStorage()
    manager = await _create_manager(kind, storage)
    storage.fail_after_write = True
    original_load = storage.async_load

    async def unreadable_load() -> dict[str, Any] | None:
        raise OSError("simulated Store read failure")

    storage.async_load = unreadable_load
    with pytest.raises(RuntimeError, match="lost Store acknowledgement"):
        await _mutate(kind, manager, "first")

    assert not manager._initialized
    storage.async_load = original_load
    await manager.async_initialize()
    assert manager._initialized
    assert storage.data is not None
