"""Generated valid state changes through Management, reload, backup, and SDK."""

from __future__ import annotations

from time import monotonic

from custom_components.extended_openai_conversation_responses import (
    agent_config,
    backup,
)
from custom_components.extended_openai_conversation_responses.const import (
    GUEST_POLICY_VERSION,
)
from homeassistant.core import HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_management_backend_acceptance import (
    _admin_client,
    _conversation_subentry,
    _fresh_reload,
    _management_call,
)
from tests_stress.conftest import record
from tests_stress.generated_valid_states import normalized_state
from tests_stress.generated_valid_transitions import fingerprint, generate_transitions
from tests_stress.test_generated_state_lifecycle import _assert_wire, _converse


async def test_generated_configuration_transitions_remain_live_and_reversible(
    hass: HomeAssistant,
    hass_ws_client,
    monkeypatch,
    stress_seed: int,
    stress_trace: list[dict],
) -> None:
    started = monotonic()
    suite = generate_transitions(stress_seed)
    entry = _make_entry(
        "Generated transition states",
        include_ai_task=False,
        conversation_options={
            "functions": [],
            "guest_policy_version": GUEST_POLICY_VERSION,
        },
    )
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)
    for path in suite.paths:
        path_started = monotonic()
        initial, _ = normalized_state(path.states[0])
        for step, state in enumerate(path.states):
            intended, api = normalized_state(state)
            record(
                stress_trace,
                "transition_step_start",
                path=path.path_id,
                step=step,
                fingerprint=fingerprint(state),
                model=intended["chat_model"],
                api=api,
            )
            before = await _management_call(
                client, entry=entry, section="configuration", action="get"
            )
            await _management_call(
                client,
                entry=entry,
                section="configuration",
                action="update",
                revision=before["revision"],
                config=intended,
            )
            await _fresh_reload(hass, entry)
            current = await _management_call(
                client, entry=entry, section="configuration", action="get"
            )
            persisted = agent_config.normalize_agent_config(current["config"])
            assert persisted == intended, (path.path_id, step)
            snapshot = await backup.async_collect_backup_snapshot(
                hass, entry, _conversation_subentry(entry)
            )
            assert (
                agent_config.normalize_agent_config(snapshot["agent"]["config"])
                == intended
            )
            request = await _converse(
                hass, entry, monkeypatch, api, intended["chat_model"]
            )
            _assert_wire(request, intended, api)
            manager_counts = {
                str(key): len(value)
                for key, value in hass.data.items()
                if isinstance(key, str)
                and key.startswith("extended_openai_conversation_responses.")
                and key.endswith("_managers")
                and isinstance(value, dict)
            }
            assert all(count <= 1 for count in manager_counts.values()), (
                path.path_id,
                step,
                manager_counts,
            )
        if path.states[0] == path.states[-1]:
            assert persisted == initial, path.path_id
        record(
            stress_trace,
            "transition_path",
            path=path.path_id,
            family=path.family,
            start=fingerprint(path.states[0]),
            end=fingerprint(path.states[-1]),
            steps=len(path.states) - 1,
            elapsed_seconds=round(monotonic() - path_started, 3),
        )
    record(
        stress_trace,
        "transition_suite",
        **suite.evidence(stress_seed),
        elapsed_seconds=round(monotonic() - started, 3),
    )
