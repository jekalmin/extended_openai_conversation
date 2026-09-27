"""Narrow OS failure probes at HA's actual atomic fsync/replace seam."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
import errno
from pathlib import Path

import atomicwrites
import pytest

from custom_components.extended_openai_conversation_responses.conversation_archive import (
    HomeAssistantArchiveStorage,
)
from custom_components.extended_openai_conversation_responses.delayed_tools import (
    DelayedToolManager,
)
from custom_components.extended_openai_conversation_responses.guest_mode import (
    GuestModeManager,
)
from custom_components.extended_openai_conversation_responses.intercom import (
    IntercomManager,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    HomeAssistantKnowledgeStorage,
    KnowledgeLibrary,
)
from custom_components.extended_openai_conversation_responses.memory import (
    HomeAssistantMemoryStorage,
)
from custom_components.extended_openai_conversation_responses.model_catalog_manager import (
    ModelCatalogManager,
)
from custom_components.extended_openai_conversation_responses.quiet_hours_runtime import (
    QuietHoursManager,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    STORAGE_VERSION as RULES_VERSION,
    RequestRules,
    RequestRuleStore,
)
from custom_components.extended_openai_conversation_responses.restore_recovery import (
    _async_write_journal_verified,
    _journal_store,
)
from custom_components.extended_openai_conversation_responses.temporary_memory import (
    TemporaryMemory,
    _temporary_memory_store,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util import file as ha_file
from tests_stress.conftest import record


def _raise_os_error(number: int):
    def fail(*args, **kwargs):
        del args, kwargs
        raise OSError(number, "seeded private storage failure")

    return fail


def _files(path: str) -> set[str]:
    return {child.name for child in Path(path).parent.iterdir()}


@pytest.fixture
def real_store_io(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Undo pytest-HA's in-memory Store shim for these OS-boundary probes."""
    hass.config.config_dir = str(tmp_path)

    async def write_to_disk(store: Store, data: dict) -> None:
        # All selected EOAI stores serialize in the executor; retain HA's real
        # _write_data -> write_utf8_file_atomic -> fsync/replace path.
        await store.hass.async_add_executor_job(store._write_data, data)

    async def load_from_disk(store: Store):
        return await store._async_load_data()

    # Restore the plugin's Store shim before its own fixture tears down. A
    # function-scoped monkeypatch teardown runs too late for its autospec.
    with monkeypatch.context() as scoped:
        scoped.setattr(Store, "_async_write_data", write_to_disk)
        scoped.setattr(Store, "_async_load", load_from_disk)
        yield


async def test_knowledge_fsync_enospc_rolls_back_and_recovers(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    real_store_io: None,
) -> None:
    del real_store_io
    storage = HomeAssistantKnowledgeStorage(hass, "disk-entry", "disk-agent")
    library = KnowledgeLibrary(storage)
    await library.async_initialize()
    first = await library.async_create("Saved title", "Description", "Saved content")
    path = storage._store.path
    before_files = _files(path)
    before_bytes = Path(path).read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(atomicwrites, "_proper_fsync", _raise_os_error(errno.ENOSPC))
        with pytest.raises(OSError) as error:
            await library.async_update(first.source_id, content="Undurable content")
    assert error.value.errno == errno.ENOSPC
    assert (await library.async_get(first.source_id)).content == "Saved content"
    assert Path(path).read_bytes() == before_bytes
    assert _files(path) == before_files
    restarted = KnowledgeLibrary(
        HomeAssistantKnowledgeStorage(hass, "disk-entry", "disk-agent")
    )
    await restarted.async_initialize()
    assert (await restarted.async_get(first.source_id)).content == "Saved content"
    await library.async_update(first.source_id, content="Recovered content")
    recovered = KnowledgeLibrary(
        HomeAssistantKnowledgeStorage(hass, "disk-entry", "disk-agent")
    )
    await recovered.async_initialize()
    assert (await recovered.async_get(first.source_id)).content == "Recovered content"
    record(
        stress_trace,
        "os_storage_fault",
        store="knowledge",
        seam="fsync",
        errno="ENOSPC",
    )


async def test_request_rules_atomic_replace_erofs_rolls_back_and_recovers(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    real_store_io: None,
) -> None:
    del real_store_io
    key = "extended_openai_conversation.disk_fault_rules"
    store = RequestRuleStore(hass, RULES_VERSION, key)
    rules = RequestRules(store)
    await rules.async_initialize()
    await rules.async_set_groups(
        [{"id": "saved", "name": "Saved"}], expected_revision=rules.revision()
    )
    path = store.path
    before_files = _files(path)
    before_bytes = Path(path).read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(atomicwrites, "replace_atomic", _raise_os_error(errno.EROFS))
        with pytest.raises(OSError) as error:
            await rules.async_set_groups(
                [{"id": "lost", "name": "Lost"}], expected_revision=rules.revision()
            )
    assert error.value.errno == errno.EROFS
    assert rules.snapshot()["groups"] == [{"id": "saved", "name": "Saved"}]
    assert Path(path).read_bytes() == before_bytes
    assert _files(path) == before_files
    restarted = RequestRules(RequestRuleStore(hass, RULES_VERSION, key))
    await restarted.async_initialize()
    assert restarted.snapshot()["groups"] == [{"id": "saved", "name": "Saved"}]
    await rules.async_set_groups(
        [{"id": "recovered", "name": "Recovered"}], expected_revision=rules.revision()
    )
    recovered = RequestRules(RequestRuleStore(hass, RULES_VERSION, key))
    await recovered.async_initialize()
    assert recovered.snapshot()["groups"] == [{"id": "recovered", "name": "Recovered"}]
    record(
        stress_trace,
        "os_storage_fault",
        store="request_rules",
        seam="replace",
        errno="EROFS",
    )


async def test_restore_journal_replace_eacces_never_claims_commit(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    real_store_io: None,
) -> None:
    del real_store_io
    store = _journal_store(hass, "disk-entry", "disk-agent")
    saved = {"phase": "saved", "private_marker": "do-not-log-this"}
    assert await _async_write_journal_verified(store, saved)
    path = store.path
    before_files = _files(path)
    before_bytes = Path(path).read_bytes()
    with monkeypatch.context() as fault:
        fault.setattr(atomicwrites, "replace_atomic", _raise_os_error(errno.EACCES))
        assert not await _async_write_journal_verified(
            store, {"phase": "not-committed"}
        )
    assert Path(path).read_bytes() == before_bytes
    assert _files(path) == before_files
    assert await _journal_store(hass, "disk-entry", "disk-agent").async_load() == saved
    assert await _async_write_journal_verified(store, {"phase": "recovered"})
    assert await _journal_store(hass, "disk-entry", "disk-agent").async_load() == {
        "phase": "recovered"
    }
    record(
        stress_trace,
        "os_storage_fault",
        store="restore_journal",
        seam="replace",
        errno="EACCES",
    )


@pytest.mark.parametrize(
    "owner",
    (
        "persistent_memory",
        "temporary_memory",
        "archive_metadata",
        "archive_partition",
        "guest_mode",
        "delayed_tools",
        "model_catalogue",
        "quiet_hours",
        "broadcast_settings",
    ),
)
async def test_transactional_store_writer_failure_is_observable_and_retryable(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    real_store_io: None,
    owner: str,
) -> None:
    """Every transaction owner must see HA's real atomic writer failure."""
    del real_store_io
    archive = HomeAssistantArchiveStorage(hass, "disk-entry", "disk-agent")
    stores = {
        "persistent_memory": HomeAssistantMemoryStorage(
            hass, "disk-entry", "disk-agent"
        )._store,
        "temporary_memory": _temporary_memory_store(hass, "disk-entry", "disk-agent"),
        "archive_metadata": archive._metadata,
        "archive_partition": archive._partition_store("2026-09"),
        "guest_mode": GuestModeManager(hass, "disk-entry", "disk-agent")._store,
        "delayed_tools": DelayedToolManager(hass)._store,
        "model_catalogue": ModelCatalogManager(hass).store,
        "quiet_hours": QuietHoursManager(hass)._store,
        "broadcast_settings": IntercomManager(hass)._store,
    }
    store = stores[owner]
    await store.async_save({"generation": "A"})
    before = Path(store.path).read_bytes()
    with monkeypatch.context() as fault:
        if store._atomic_writes:
            fault.setattr(atomicwrites, "replace_atomic", _raise_os_error(errno.EACCES))
        else:
            fault.setattr(ha_file.os, "replace", _raise_os_error(errno.EACCES))
        with pytest.raises(OSError) as error:
            await store.async_save({"generation": "B"})
    assert error.value.errno == errno.EACCES
    assert Path(store.path).read_bytes() == before
    assert await store.async_load() == {"generation": "A"}
    await store.async_save({"generation": "C"})
    assert await store.async_load() == {"generation": "C"}
    record(
        stress_trace,
        "durable_mutation",
        owner=owner,
        classification="transactional_durable",
        failure_phase="atomic_replace",
        runtime_rolled_back=True,
        reload_preserved_prior_generation=True,
        recovery_write_succeeded=True,
    )


async def test_temporary_memory_failure_does_not_publish_or_reload_new_fact(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    real_store_io: None,
) -> None:
    del real_store_io
    manager = TemporaryMemory(_temporary_memory_store(hass, "temp-entry", "temp-agent"))
    await manager.async_initialize()
    expiry = (dt_util.utcnow() + timedelta(days=1)).isoformat()
    first = await manager.async_add(
        "scope", "Known fact", expiry, owner_scope_id="user:alice"
    )
    with monkeypatch.context() as fault:
        fault.setattr(atomicwrites, "replace_atomic", _raise_os_error(errno.ENOSPC))
        with pytest.raises(OSError, match="Private storage write failed"):
            await manager.async_add(
                "scope", "Undurable fact", expiry, owner_scope_id="user:alice"
            )
    assert len(manager._records) == 1
    assert first["memory"]["memory_id"] in manager._records
    restarted = TemporaryMemory(
        _temporary_memory_store(hass, "temp-entry", "temp-agent")
    )
    await restarted.async_initialize()
    assert set(restarted._records) == set(manager._records)
    await manager.async_add(
        "scope", "Recovered fact", expiry, owner_scope_id="user:alice"
    )
    assert len(manager._records) == 2


async def test_guest_schedule_failure_does_not_publish_and_retries(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    real_store_io: None,
) -> None:
    del real_store_io
    manager = GuestModeManager(hass, "guest-entry", "guest-agent")
    await manager.async_initialize()
    assert manager._schedule is None
    with monkeypatch.context() as fault:
        fault.setattr(atomicwrites, "replace_atomic", _raise_os_error(errno.EROFS))
        with pytest.raises(OSError, match="Private storage write failed"):
            await manager.async_update_trusted(indefinite=True)
    assert manager._schedule is None
    restarted = GuestModeManager(hass, "guest-entry", "guest-agent")
    await restarted.async_initialize()
    assert restarted._schedule is None
    await manager.async_update_trusted(indefinite=True)
    assert manager._schedule is not None


async def test_broadcast_switch_failure_does_not_report_enabled(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    real_store_io: None,
) -> None:
    del real_store_io
    manager = IntercomManager(hass)
    await manager.async_initialize()
    assert manager.enabled is False
    with monkeypatch.context() as fault:
        fault.setattr(ha_file.os, "replace", _raise_os_error(errno.EACCES))
        with pytest.raises(OSError, match="Private storage write failed"):
            await manager.async_set_enabled(True)
    assert manager.enabled is False
    restarted = IntercomManager(hass)
    await restarted.async_initialize()
    assert restarted.enabled is False
    await manager.async_set_enabled(True)
    assert manager.enabled is True
