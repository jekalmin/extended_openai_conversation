"""Narrow OS failure probes at HA's actual atomic fsync/replace seam."""

from __future__ import annotations

import errno
from pathlib import Path

import atomicwrites
import pytest

from custom_components.extended_openai_conversation_responses.knowledge import (
    HomeAssistantKnowledgeStorage,
    KnowledgeLibrary,
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
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
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
) -> None:
    """Undo pytest-HA's in-memory Store shim for these OS-boundary probes."""
    hass.config.config_dir = str(tmp_path)

    async def write_to_disk(store: Store, data: dict) -> None:
        # All selected EOAI stores serialize in the executor; retain HA's real
        # _write_data -> write_utf8_file_atomic -> fsync/replace path.
        await store.hass.async_add_executor_job(store._write_data, data)

    async def load_from_disk(store: Store):
        return await store._async_load_data()

    monkeypatch.setattr(Store, "_async_write_data", write_to_disk)
    monkeypatch.setattr(Store, "_async_load", load_from_disk)


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
