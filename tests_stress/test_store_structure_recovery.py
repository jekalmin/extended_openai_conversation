"""Nightly genuine Store recovery from structurally invalid data."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_KNOWLEDGE_ENABLED,
    CONF_TEMPORARY_MEMORY,
    SUBSYSTEM_STATUS_KEY,
    TEMPORARY_MEMORY_BALANCED,
)
from custom_components.extended_openai_conversation_responses.usage import RequestUsage
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_corrupt_subsystem_store_startup_isolation import (
    _purge_cached_managers,
    _real_store_io,  # noqa: F401 - imported fixture is registered for this module
)
from tests_stress.conftest import record


@pytest.mark.no_fail_on_log_exception
@pytest.mark.asyncio
async def test_two_corrupt_stores_leave_usage_intact_and_recover_independently(
    hass: HomeAssistant,
    _real_store_io: None,  # noqa: F811 - genuine Store fixture
    stress_trace: list[dict],
) -> None:
    """Wrong-shape and future-version stores degrade separately on one cold start."""
    entry = _make_entry(
        "Simultaneous store corruption",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_KNOWLEDGE_ENABLED: True,
            CONF_TEMPORARY_MEMORY: TEMPORARY_MEMORY_BALANCED,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert agent._knowledge is not None
    assert agent._temporary_memory is not None
    assert agent._usage is not None
    await agent._knowledge.async_create(
        "Healthy source", "Reference", "Preserved source"
    )
    await agent._temporary_memory.async_add(
        "user:corruption-owner",
        "Preserved temporary note",
        (dt_util.utcnow() + timedelta(hours=1)).isoformat(),
        "acceptance",
        owner_scope_id="user:corruption-owner",
    )
    await agent._usage.async_record_request(
        successful=True, usage=RequestUsage(total_tokens=7)
    )
    await hass.async_block_till_done()
    knowledge_path = Path(agent._knowledge._storage._store.path)
    temporary_path = Path(agent._temporary_memory._store.path)
    original_knowledge = await hass.async_add_executor_job(
        knowledge_path.read_text, "utf-8"
    )
    original_temporary = await hass.async_add_executor_job(
        temporary_path.read_text, "utf-8"
    )
    wrong_shape = json.loads(original_knowledge)
    wrong_shape["data"]["sources"] = {"wrong": "mapping"}
    wrong_shape_text = json.dumps(wrong_shape)
    future = json.loads(original_temporary)
    future["version"] = 999
    future_text = json.dumps(future)

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    _purge_cached_managers(hass.data)
    await hass.async_add_executor_job(
        knowledge_path.write_text, wrong_shape_text, "utf-8"
    )
    await hass.async_add_executor_job(temporary_path.write_text, future_text, "utf-8")
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    degraded = conversation.async_get_agent(hass, entry.entry_id)
    assert degraded is not None
    assert degraded._knowledge is None
    assert degraded._temporary_memory is None
    assert degraded._usage is not None
    assert degraded._usage.totals.api_request_count == 1
    status = hass.data[SUBSYSTEM_STATUS_KEY][
        (entry.entry_id, degraded.subentry.subentry_id)
    ]
    assert status["knowledge"]["status"] != "healthy"
    assert status["temporary_memory"]["status"] != "healthy"
    assert status["usage"]["status"] == "healthy"
    assert (
        await hass.async_add_executor_job(knowledge_path.read_text, "utf-8")
        == wrong_shape_text
    )
    assert (
        await hass.async_add_executor_job(temporary_path.read_text, "utf-8")
        == future_text
    )

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    _purge_cached_managers(hass.data)
    await hass.async_add_executor_job(
        knowledge_path.write_text, original_knowledge, "utf-8"
    )
    await hass.async_add_executor_job(
        temporary_path.write_text, original_temporary, "utf-8"
    )
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    recovered = conversation.async_get_agent(hass, entry.entry_id)
    assert recovered is not None
    assert recovered._knowledge is not None
    assert recovered._knowledge.source_count == 1
    assert recovered._temporary_memory is not None
    notes = await recovered._temporary_memory.async_active(
        "user:corruption-owner", owner_scope_id="user:corruption-owner"
    )
    assert [note.content for note in notes] == ["Preserved temporary note"]
    assert recovered._usage is not None
    assert recovered._usage.totals.api_request_count == 1
    record(stress_trace, "simultaneous_store_recovery", corrupt_stores=2)


@pytest.mark.no_fail_on_log_exception
@pytest.mark.asyncio
async def test_invalid_store_structure_stays_durable_until_repaired(
    hass: HomeAssistant,
    _real_store_io: None,  # noqa: F811 - fixture name deliberately matches import
    stress_trace: list[dict],
) -> None:
    """A valid JSON envelope with invalid data degrades only its owner."""
    entry = _make_entry(
        "Invalid knowledge structure",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_KNOWLEDGE_ENABLED: True,
            CONF_TEMPORARY_MEMORY: TEMPORARY_MEMORY_BALANCED,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert agent._knowledge is not None
    assert agent._temporary_memory is not None
    await agent._knowledge.async_create(
        "Durable source", "Reference", "Important durable knowledge"
    )
    path = Path(agent._knowledge._storage._store.path)
    original = await hass.async_add_executor_job(path.read_text, "utf-8")
    payload = json.loads(original)
    payload["data"]["sources"] = {"invalid": "mapping-instead-of-list"}
    damaged = json.dumps(payload)

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    _purge_cached_managers(hass.data)
    await hass.async_add_executor_job(path.write_text, damaged, "utf-8")

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    degraded = conversation.async_get_agent(hass, entry.entry_id)
    assert degraded is not None
    assert degraded._knowledge is None
    assert degraded._temporary_memory is not None
    assert await hass.async_add_executor_job(path.read_text, "utf-8") == damaged
    subentry_id = degraded.subentry.subentry_id
    status = hass.data[SUBSYSTEM_STATUS_KEY][(entry.entry_id, subentry_id)]
    assert status["knowledge"]["status"] != "healthy"
    assert status["temporary_memory"]["status"] == "healthy"

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    _purge_cached_managers(hass.data)
    await hass.async_add_executor_job(path.write_text, original, "utf-8")
    assert await hass.config_entries.async_setup(entry.entry_id)
    restored = conversation.async_get_agent(hass, entry.entry_id)
    assert restored is not None
    assert restored._knowledge is not None
    assert restored._knowledge.source_count == 1
    record(
        stress_trace,
        "summary",
        layer="Real HA",
        invalid_store_structure_recoveries=1,
    )


@pytest.mark.no_fail_on_log_exception
@pytest.mark.asyncio
async def test_knowledge_read_failure_preserves_store_and_sibling_state(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    _real_store_io: None,  # noqa: F811 - imported fixture is intentionally reused
    stress_trace: list[dict],
) -> None:
    """A disk-like read error must not turn durable sources into an empty save."""
    entry = _make_entry(
        "Knowledge read failure",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_KNOWLEDGE_ENABLED: True,
            CONF_TEMPORARY_MEMORY: TEMPORARY_MEMORY_BALANCED,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert agent._knowledge is not None
    await agent._knowledge.async_create(
        "Read failure source", "Reference", "Keep this durable knowledge"
    )
    path = Path(agent._knowledge._storage._store.path)
    original_bytes = await hass.async_add_executor_job(path.read_bytes)
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    _purge_cached_managers(hass.data)

    real_load = Store._async_load

    async def fail_knowledge_load(store: Store, *args, **kwargs):
        if Path(store.path) == path:
            raise OSError("Nightly knowledge Store read failure")
        return await real_load(store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Store, "_async_load", fail_knowledge_load)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        degraded = conversation.async_get_agent(hass, entry.entry_id)
        assert degraded is not None
        assert degraded._knowledge is None
        assert degraded._temporary_memory is not None
        assert await hass.async_add_executor_job(path.read_bytes) == original_bytes
        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        _purge_cached_managers(hass.data)

    assert await hass.config_entries.async_setup(entry.entry_id)
    restored = conversation.async_get_agent(hass, entry.entry_id)
    assert restored is not None
    assert restored._knowledge is not None
    assert restored._knowledge.source_count == 1
    record(
        stress_trace,
        "summary",
        layer="Real HA",
        store_read_failure_recoveries=1,
    )


@pytest.mark.no_fail_on_log_exception
@pytest.mark.asyncio
async def test_knowledge_write_failure_rolls_back_and_survives_reload(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    _real_store_io: None,  # noqa: F811 - imported fixture is intentionally reused
    stress_trace: list[dict],
) -> None:
    """A failed real Store write leaves its prior complete generation intact."""
    entry = _make_entry(
        "Knowledge write failure",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_KNOWLEDGE_ENABLED: True,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    knowledge = agent._knowledge
    assert knowledge is not None
    await knowledge.async_create("Before failure", "Reference", "Durable version")
    path = Path(knowledge._storage._store.path)
    original_bytes = await hass.async_add_executor_job(path.read_bytes)
    real_write = Store._async_write_data

    async def fail_knowledge_write(store: Store, *args, **kwargs):
        if Path(store.path) == path:
            raise OSError("Nightly knowledge Store write failure")
        return await real_write(store, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Store, "_async_write_data", fail_knowledge_write)
        with pytest.raises(OSError, match="Nightly knowledge Store write failure"):
            await knowledge.async_create(
                "Uncommitted source", "Reference", "Must not be published"
            )
    assert knowledge.source_count == 1
    assert await hass.async_add_executor_job(path.read_bytes) == original_bytes

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    _purge_cached_managers(hass.data)
    assert await hass.config_entries.async_setup(entry.entry_id)
    restarted = conversation.async_get_agent(hass, entry.entry_id)
    assert restarted is not None
    assert restarted._knowledge is not None
    assert restarted._knowledge.source_count == 1
    await restarted._knowledge.async_create(
        "After recovery", "Reference", "The Store is writable again"
    )
    assert restarted._knowledge.source_count == 2
    record(
        stress_trace,
        "summary",
        layer="Real HA",
        store_write_failure_recoveries=1,
    )
