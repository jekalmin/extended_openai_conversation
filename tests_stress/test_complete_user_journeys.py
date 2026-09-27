"""Distinct, stateful user journeys across real HA and mocked SDK transport."""

from __future__ import annotations

from time import monotonic
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.extended_openai_conversation_responses import (
    agent_config,
    backup,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_API_PROVIDER,
    CONF_BASE_URL,
    CONF_SKIP_AUTHENTICATION,
    DOMAIN,
    GUEST_POLICY_VERSION,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    async_get_knowledge,
)
from custom_components.extended_openai_conversation_responses.memory import (
    async_get_memory,
)
from custom_components.extended_openai_conversation_responses.request import (
    build_provider_request_snapshot,
)
from homeassistant.config_entries import SOURCE_USER
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_management_backend_acceptance import (
    _admin_client,
    _conversation_subentry,
    _fresh_reload,
    _management_call,
)
from tests_stress.conftest import record
from tests_stress.generated_valid_states import normalized_state
from tests_stress.generated_valid_transitions import JOURNEYS, _named_paths, fingerprint
from tests_stress.test_generated_state_lifecycle import _assert_wire, _converse


async def _installed_entry(hass: HomeAssistant, journey: str):
    if journey != "installation-to-mature":
        entry = _make_entry(
            journey,
            include_ai_task=False,
            conversation_options={
                "functions": [],
                "guest_policy_version": GUEST_POLICY_VERSION,
            },
        )
        await _setup_entry(hass, entry)
        return entry
    with patch(
        "custom_components.extended_openai_conversation_responses.config_flow.get_authenticated_client",
        new_callable=AsyncMock,
    ):
        started = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        created = await hass.config_entries.flow.async_configure(
            started["flow_id"],
            {
                CONF_NAME: "Maturing assistant",
                CONF_API_KEY: "sk-local-journey",
                CONF_BASE_URL: "https://example.test/v1",
                CONF_SKIP_AUTHENTICATION: True,
                CONF_API_PROVIDER: "openai",
            },
        )
    assert created["type"] is FlowResultType.CREATE_ENTRY
    entry = created["result"]
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def _save(client, entry, normalized: dict) -> None:
    before = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    await _management_call(
        client,
        entry=entry,
        section="configuration",
        action="update",
        revision=before["revision"],
        config=normalized,
    )


async def _assert_current(client, entry, intended: dict) -> None:
    current = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    assert agent_config.normalize_agent_config(current["config"]) == intended


@pytest.mark.parametrize("journey", tuple(JOURNEYS), ids=tuple(JOURNEYS))
async def test_complete_user_journey(
    hass: HomeAssistant,
    hass_ws_client,
    monkeypatch,
    journey: str,
    stress_seed: int,
    stress_trace: list[dict],
) -> None:
    """Walk independent features through edits, reload, SDK, and recovery."""
    started = monotonic()
    path = next(path for path in _named_paths() if path.family == JOURNEYS[journey])
    entry = await _installed_entry(hass, journey)
    client = await _admin_client(hass, hass_ws_client)
    subentry = _conversation_subentry(entry)
    first, first_api = normalized_state(path.states[0])
    # The first turn precedes all edits, including installation's config-flow save.
    initial = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    installed = agent_config.normalize_agent_config(initial["config"])
    installed_api = build_provider_request_snapshot(installed, {}).api_mode
    request = await _converse(
        hass, entry, monkeypatch, installed_api, installed["chat_model"]
    )
    _assert_wire(request, installed, installed_api)
    checkpoint = None
    for step, state in enumerate(path.states):
        intended, api = normalized_state(state)
        record(
            stress_trace,
            "journey_step_start",
            seed=stress_seed,
            journey=journey,
            step=step,
            fingerprint=fingerprint(state),
            model=intended["chat_model"],
            api=api,
        )
        await _save(client, entry, intended)
        await _fresh_reload(hass, entry)
        await _assert_current(client, entry, intended)
        request = await _converse(hass, entry, monkeypatch, api, intended["chat_model"])
        _assert_wire(request, intended, api)
        snapshot = await backup.async_collect_backup_snapshot(hass, entry, subentry)
        assert (
            agent_config.normalize_agent_config(snapshot["agent"]["config"]) == intended
        )
        if step == 1:
            checkpoint = snapshot
            if journey in {
                "installation-to-mature",
                "long-lived-evolution",
                "extensive-backup-restore",
                "guest-private-boundary",
            }:
                memory = await async_get_memory(
                    hass, entry.entry_id, subentry.subentry_id
                )
                knowledge = await async_get_knowledge(
                    hass, entry.entry_id, subentry.subentry_id
                )
                created = await memory.async_add(
                    "user:journey-owner",
                    f"Private {journey} marker",
                    "acceptance",
                    "explicit",
                )
                assert created["memory"]["memory_id"]
                source = await knowledge.async_create(
                    f"{journey} handbook",
                    "Journey reference",
                    f"Knowledge marker for {journey}",
                )
                assert source.source_id
                checkpoint = await backup.async_collect_backup_snapshot(
                    hass, entry, subentry
                )
    assert checkpoint is not None
    # A restore after later edits must rehydrate the earlier configuration and stores.
    if journey in {
        "installation-to-mature",
        "long-lived-evolution",
        "extensive-backup-restore",
        "function-heavy-evolution",
    }:
        restored = await backup.async_restore_backup(hass, entry, subentry, checkpoint)
        assert restored["status"] == "restored"
        await hass.async_block_till_done()
        await _fresh_reload(hass, entry)
        recovered = agent_config.normalize_agent_config(checkpoint["agent"]["config"])
        await _assert_current(client, entry, recovered)
        restored_state = path.states[1]
        _, api = normalized_state(restored_state)
        request = await _converse(
            hass, entry, monkeypatch, api, recovered["chat_model"]
        )
        _assert_wire(request, recovered, api)
    if journey == "installation-to-mature":
        assert await hass.config_entries.async_unload(entry.entry_id)
        assert await hass.config_entries.async_remove(entry.entry_id)
    record(
        stress_trace,
        "complete_journey",
        seed=stress_seed,
        journey=journey,
        start=fingerprint(path.states[0]),
        end=fingerprint(path.states[-1]),
        steps=len(path.states) - 1,
        model=first["chat_model"],
        api=first_api,
        elapsed_seconds=round(monotonic() - started, 3),
    )
