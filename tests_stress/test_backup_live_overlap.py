"""Nightly backup restore overlap with an active public conversation."""

from __future__ import annotations

import asyncio

import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses import backup
from custom_components.extended_openai_conversation_responses.agent_maintenance import (
    get_agent_maintenance_gate,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_ARCHIVE_ENABLED,
    CONF_MEMORY_MODE,
    MEMORY_MODE_MANUAL,
)
from custom_components.extended_openai_conversation_responses.conversation_archive import (
    async_get_archive,
)
from custom_components.extended_openai_conversation_responses.memory import (
    async_get_memory,
)
from custom_components.extended_openai_conversation_responses.usage import (
    async_get_usage,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import (
    _conversation_subentry,
    _make_entry,
    _setup_entry,
)
from tests_real_ha.test_management_backend_acceptance import (
    _admin_client,
    _fresh_reload,
    _management_call,
)
from tests_stress.conftest import record


@pytest.mark.asyncio
async def test_restore_waits_for_active_turn_then_becomes_authoritative(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A restore cannot interleave with or be overwritten by an older live turn."""
    entry = _make_entry(
        "Restore overlap",
        include_ai_task=False,
        conversation_options={
            CONF_ARCHIVE_ENABLED: True,
            CONF_MEMORY_MODE: MEMORY_MODE_MANUAL,
        },
    )
    MockUser(id="restore-owner", name="Restore Owner").add_to_hass(hass)
    await _setup_entry(hass, entry)
    subentry = _conversation_subentry(entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    archive = await async_get_archive(hass, entry.entry_id, subentry.subentry_id)
    usage = await async_get_usage(hass, entry.entry_id, subentry.subentry_id)

    await memory.async_add(
        "restore-owner", "RESTORED-AUTHORITATIVE-MARKER", "acceptance", "explicit"
    )
    target = await backup.async_collect_backup_snapshot(hass, entry, subentry)
    for item in await memory.async_list("restore-owner"):
        assert await memory.async_delete("restore-owner", [item.memory_id]) == 1
    await memory.async_add(
        "restore-owner", "PRE-RESTORE-LIVE-MARKER", "acceptance", "explicit"
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_model(log: conversation.ChatLog, **kwargs) -> None:
        del kwargs
        entered.set()
        await release.wait()
        log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=agent.entity_id,
                content="Turn completed before restore.",
            )
        )

    monkeypatch.setattr(agent, "_async_handle_chat_log", blocked_model)
    turn = asyncio.create_task(
        conversation.async_converse(
            hass=hass,
            text="This turn must finish before backup restore can commit.",
            conversation_id=None,
            context=Context(user_id="restore-owner"),
            language="en",
            agent_id=entry.entry_id,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=10)

    restore = asyncio.create_task(
        backup.async_restore_backup(hass, entry, subentry, target)
    )
    await asyncio.sleep(0)
    assert not restore.done(), "restore crossed the active conversation maintenance lease"

    release.set()
    result = await asyncio.wait_for(turn, timeout=15)
    assert result.response.error_code is None
    restored = await asyncio.wait_for(restore, timeout=15)
    assert restored["status"] == "restored"
    await hass.async_block_till_done()

    memories = await memory.async_list("restore-owner")
    assert [item.content for item in memories] == ["RESTORED-AUTHORITATIVE-MARKER"]
    assert (await archive.async_list_sessions("user:restore-owner", limit=20))[
        "sessions"
    ] == []
    assert usage.totals.conversation_count == target["usage"]["totals"][
        "conversation_count"
    ]
    assert usage.totals.api_request_count == target["usage"]["totals"][
        "api_request_count"
    ]

    final = await backup.async_collect_backup_snapshot(hass, entry, subentry)
    for section in (
        "memories",
        "temporary_memories",
        "knowledge",
        "archive",
        "usage",
        "guest_mode",
        "request_rules",
    ):
        assert final[section] == target[section], section

    record(
        stress_trace,
        "summary",
        layer="Real HA Assist + backup restore",
        overlapping_live_restores=1,
        public_turns=1,
        restore_waited_for_active_turn=1,
    )


@pytest.mark.asyncio
async def test_restore_drains_paused_management_memory_commit(
    hass: HomeAssistant,
    hass_ws_client,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A pre-restore Management write must finish before B becomes authoritative."""
    entry = _make_entry(
        "Restore management overlap",
        include_ai_task=False,
        conversation_options={CONF_MEMORY_MODE: MEMORY_MODE_MANUAL},
    )
    await _setup_entry(hass, entry)
    subentry = _conversation_subentry(entry)
    client = await _admin_client(
        hass, hass_ws_client, user_id="restore-mutation-admin", name="Restore Mutation Admin"
    )
    memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    owner = "restore-mutation-admin"
    target_record = await memory.async_add(
        owner, "RESTORED-GENERATION-B", "acceptance", "explicit"
    )
    target = await backup.async_collect_backup_snapshot(hass, entry, subentry)
    assert await memory.async_delete(owner, [target_record["memory"]["memory_id"]]) == 1

    entered = asyncio.Event()
    release = asyncio.Event()
    save = memory._storage.async_save

    async def paused_save(data):
        entered.set()
        await release.wait()
        await save(data)

    monkeypatch.setattr(memory._storage, "async_save", paused_save)
    mutation = asyncio.create_task(
        _management_call(
            client,
            entry=entry,
            section="memories",
            action="add",
            content="STALE-GENERATION-A",
            category="acceptance",
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=10)
    restore = asyncio.create_task(
        backup.async_restore_backup(hass, entry, subentry, target)
    )
    await asyncio.sleep(0)
    assert not restore.done(), "restore crossed a paused durable Management commit"

    release.set()
    assert (await asyncio.wait_for(mutation, timeout=15))["status"] == "created"
    assert (await asyncio.wait_for(restore, timeout=15))["status"] == "restored"
    assert [item.content for item in await memory.async_list(owner)] == [
        "RESTORED-GENERATION-B"
    ]
    assert (await backup.async_collect_backup_snapshot(hass, entry, subentry))[
        "memories"
    ] == target["memories"]

    await _fresh_reload(hass, entry)
    reloaded = await _management_call(
        client, entry=entry, section="memories", action="list"
    )
    assert [item["content"] for item in reloaded["memories"]] == [
        "RESTORED-GENERATION-B"
    ]
    record(
        stress_trace,
        "summary",
        layer="Real HA Management + backup restore",
        paused_durable_mutations=1,
        restore_waited_for_commit=1,
    )


@pytest.mark.asyncio
async def test_restore_wins_after_committed_management_write_loses_ack(
    hass: HomeAssistant,
    hass_ws_client,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A committed write with a lost reply drains before restore becomes authority."""
    entry = _make_entry(
        "Restore committed acknowledgement loss",
        include_ai_task=False,
        conversation_options={CONF_MEMORY_MODE: MEMORY_MODE_MANUAL},
    )
    await _setup_entry(hass, entry)
    subentry = _conversation_subentry(entry)
    client = await _admin_client(
        hass,
        hass_ws_client,
        user_id="restore-ack-admin",
        name="Restore Ack Admin",
    )
    target_memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    target_record = await target_memory.async_add(
        "restore-ack-admin", "RESTORE-AUTHORITATIVE", "acceptance", "explicit"
    )
    target = await backup.async_collect_backup_snapshot(hass, entry, subentry)
    assert await target_memory.async_delete(
        "restore-ack-admin", [target_record["memory"]["memory_id"]]
    ) == 1

    commit_complete = asyncio.Event()
    release_ack = asyncio.Event()
    save = target_memory._storage.async_save

    async def commit_then_pause_ack(data):
        await save(data)
        commit_complete.set()
        await release_ack.wait()

    monkeypatch.setattr(target_memory._storage, "async_save", commit_then_pause_ack)
    caller = asyncio.create_task(
        _management_call(
            client,
            entry=entry,
            section="memories",
            action="add",
            content="COMMITTED-BUT-RESTORED-AWAY",
            category="acceptance",
        )
    )
    await asyncio.wait_for(commit_complete.wait(), timeout=10)

    gate = get_agent_maintenance_gate(hass, entry.entry_id, subentry.subentry_id)
    restore = asyncio.create_task(
        backup.async_restore_backup(hass, entry, subentry, target)
    )
    for _ in range(100):
        if gate._waiting_writers:  # noqa: SLF001 - deterministic lease boundary
            break
        await asyncio.sleep(0)
    assert gate._waiting_writers, "restore did not queue behind the committed mutation"
    assert not restore.done()

    # The command was sent and the Store commit completed. Cancelling only the
    # caller's receive coroutine models a lost WebSocket acknowledgement.
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release_ack.set()

    restored = await asyncio.wait_for(restore, timeout=15)
    assert restored["status"] == "restored"
    await hass.async_block_till_done()

    live_memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    live_contents = [
        item.content for item in await live_memory.async_list("restore-ack-admin")
    ]
    assert live_contents == ["RESTORE-AUTHORITATIVE"]
    assert (
        await backup.async_collect_backup_snapshot(hass, entry, subentry)
    )["memories"] == target["memories"]

    await _fresh_reload(hass, entry)
    reload_client = await _admin_client(
        hass,
        hass_ws_client,
        user_id="restore-ack-admin",
        name="Restore Ack Reload Admin",
    )
    reloaded = await _management_call(
        reload_client, entry=entry, section="memories", action="list"
    )
    assert [item["content"] for item in reloaded["memories"]] == [
        "RESTORE-AUTHORITATIVE"
    ]
    record(
        stress_trace,
        "summary",
        layer="Real HA Management + committed Store write + backup restore",
        committed_writes_with_lost_ack=1,
        restore_waited_for_commit=1,
        authoritative_restore_after_reload=True,
    )
