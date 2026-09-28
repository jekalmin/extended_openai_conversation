"""Nightly removal races across active public Assist work."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import MappingProxyType
from typing import Any

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
)
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_provider_wire_e2e import (
    _agent,
    _chat_sse_text,
    _install_wire,
    _raw_client,
    _speech,
)
from tests_stress.conftest import record


async def _say(hass: HomeAssistant, entry_id: str, text: str) -> Any:
    return await conversation.async_converse(
        hass=hass,
        text=text,
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry_id,
    )


def _block_provider(monkeypatch: pytest.MonkeyPatch, agent: Any):
    wire = _install_wire(
        monkeypatch, agent, [_chat_sse_text("Removed runtime must not survive.")]
    )
    original_send = wire.send
    response_ready = asyncio.Event()
    release = asyncio.Event()

    async def blocked_send(*args: Any, **kwargs: Any) -> Any:
        response = await original_send(*args, **kwargs)
        response_ready.set()
        await release.wait()
        return response

    monkeypatch.setattr(_raw_client(agent)._client, "send", blocked_send)
    return wire, response_ready, release


@pytest.mark.asyncio
async def test_config_entry_removal_during_provider_turn_leaves_no_runtime(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """Removing a loaded entry must retire an in-flight agent generation."""
    old_agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    entry_id = old_agent.entry.entry_id
    wire, response_ready, release = _block_provider(monkeypatch, old_agent)
    turn = asyncio.create_task(_say(hass, entry_id, "Begin before entry removal"))

    try:
        await asyncio.wait_for(response_ready.wait(), timeout=10)
        assert len(wire.requests) == 1
        result = await asyncio.wait_for(
            hass.config_entries.async_remove(entry_id), timeout=15
        )
        assert result["require_restart"] is False
        assert hass.config_entries.async_get_entry(entry_id) is None
        assert conversation.async_get_agent(hass, entry_id) is None
    finally:
        turn.cancel()
        release.set()
        with suppress(asyncio.CancelledError):
            await turn

    await hass.async_block_till_done()
    assert hass.config_entries.async_get_entry(entry_id) is None
    assert conversation.async_get_agent(hass, entry_id) is None
    record(
        stress_trace,
        "summary",
        layer="Real HA",
        active_entry_removals=1,
        provider_requests=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap_recreation", [False, True])
async def test_conversation_subentry_removal_retires_old_turn_and_replacement_is_fresh(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    overlap_recreation: bool,
) -> None:
    """A removed subentry cannot leak into a same-title replacement runtime."""
    entry = _make_entry(
        "Active subentry removal",
        include_ai_task=True,
        conversation_options={CONF_API_MODE: API_MODE_CHAT_COMPLETIONS},
    )
    await _setup_entry(hass, entry)
    old_agent = conversation.async_get_agent(hass, entry.entry_id)
    assert old_agent is not None
    old_subentry = old_agent.subentry
    old_subentry_id = old_subentry.subentry_id
    old_data = dict(old_subentry.data)

    wire, response_ready, release = _block_provider(monkeypatch, old_agent)
    turn = asyncio.create_task(
        _say(hass, entry.entry_id, "Begin before conversation subentry removal")
    )
    replacement = ConfigSubentry(
        data=MappingProxyType(old_data),
        subentry_type="conversation",
        title=old_subentry.title if overlap_recreation else "Replacement conversation",
        unique_id=None,
    )
    try:
        await asyncio.wait_for(response_ready.wait(), timeout=10)
        assert len(wire.requests) == 1
        assert hass.config_entries.async_remove_subentry(entry, old_subentry_id)
        await hass.async_block_till_done()
        assert old_subentry_id not in entry.subentries
        assert conversation.async_get_agent(hass, entry.entry_id) is None
        if overlap_recreation:
            assert hass.config_entries.async_add_subentry(entry, replacement)
            await hass.async_block_till_done()
            assert replacement.subentry_id != old_subentry_id
            assert conversation.async_get_agent(hass, entry.entry_id) is not old_agent
            release.set()
            await asyncio.wait_for(asyncio.gather(turn, return_exceptions=True), 10)
    finally:
        if not turn.done():
            turn.cancel()
        release.set()
        await asyncio.gather(turn, return_exceptions=True)

    if not overlap_recreation:
        assert hass.config_entries.async_add_subentry(entry, replacement)
        await hass.async_block_till_done()
    assert replacement.subentry_id != old_subentry_id
    new_agent = conversation.async_get_agent(hass, entry.entry_id)
    assert new_agent is not None
    assert new_agent is not old_agent
    assert new_agent.subentry.subentry_id == replacement.subentry_id
    assert len(wire.requests) == 1

    fresh_wire = _install_wire(
        monkeypatch, new_agent, [_chat_sse_text("Replacement runtime is healthy.")]
    )
    result = await _say(hass, entry.entry_id, "Use the replacement conversation")
    assert _speech(result) == "Replacement runtime is healthy."
    assert len(fresh_wire.requests) == 1
    record(
        stress_trace,
        "summary",
        layer="Real HA",
        active_subentry_removals=1,
        replacement_subentries=1,
        overlap_recreation=overlap_recreation,
        provider_requests=2,
    )
