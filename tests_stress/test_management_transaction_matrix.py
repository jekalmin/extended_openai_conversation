"""Nightly fault boundaries for linked Function and Request Rule edits."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.extended_openai_conversation_responses.agent_config import (
    configured_function_tools_from_data,
    validate_function_groups,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_SKIP_AUTHENTICATION,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
)
from custom_components.extended_openai_conversation_responses.management_function_repair import (
    persisted_config_projection,
    require_agent_config_revision,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    _persist_function_configuration,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    async_get_request_rules,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.exceptions import HomeAssistantError
from tests_stress.conftest import record


async def _entry(hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Management transaction",
        data={CONF_API_KEY: "sk-local", CONF_SKIP_AUTHENTICATION: True},
        version=CONFIG_ENTRY_VERSION,
        subentries_data=[
            {
                "data": {},
                "subentry_type": "conversation",
                "title": "Agent",
                "unique_id": None,
            }
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry, next(iter(entry.subentries.values()))


def _group(group_id: str, names: list[str]) -> dict:
    return {
        "id": group_id,
        "name": group_id,
        "description": "Nightly linked edit",
        "loading_mode": "always",
        "functions": names,
        "enabled": True,
    }


def _tool(name: str) -> dict:
    return {
        "spec": {
            "name": name,
            "description": "Nightly transaction tool",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "template", "value_template": "transaction marker"},
        "enabled": True,
    }


@pytest.mark.parametrize(
    ("field", "intermediate"),
    [
        (CONF_FUNCTION_TOOLS, [_tool("nightly_aba")]),
        (CONF_FUNCTION_GROUPS, [_group("nightly_aba", ["nightly_aba"])]),
        ("guest_mode_enabled", False),
        ("voice_device_mappings", {"kitchen": "user:one"}),
        ("exposed_entities_enabled", False),
    ],
)
async def test_agent_config_aba_rejects_suspended_management_writer(
    hass, stress_trace, field, intermediate
) -> None:
    """A restored value cannot validate a token from before two HA updates."""
    entry, subentry = await _entry(hass)
    original = dict(subentry.data)
    if field == CONF_FUNCTION_GROUPS:
        original[CONF_FUNCTION_TOOLS] = [_tool("nightly_aba")]
        hass.config_entries.async_update_subentry(entry, subentry, data=original)
    assert original.get(field) != intermediate
    expected = persisted_config_projection(subentry).revision
    entered, resume = asyncio.Event(), asyncio.Event()

    async def stale_writer() -> None:
        entered.set()
        await resume.wait()
        require_agent_config_revision(subentry, expected)

    task = asyncio.create_task(stale_writer())
    await entered.wait()
    hass.config_entries.async_update_subentry(
        entry, subentry, data={**original, field: intermediate}
    )
    hass.config_entries.async_update_subentry(entry, subentry, data=original)
    await hass.async_block_till_done()
    assert dict(subentry.data) == original
    resume.set()
    with pytest.raises(HomeAssistantError, match="changed in another tab"):
        await task
    record(stress_trace, "agent_config_aba", field=field, revisions=3)
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()


async def test_function_tools_and_groups_commit_as_one_subentry_revision(
    hass,
    monkeypatch,
    stress_trace,
) -> None:
    entry, subentry = await _entry(hass)
    original = deepcopy(dict(subentry.data))
    tools = [_tool("nightly_one"), _tool("nightly_two")]
    groups = [_group("nightly_group", ["nightly_one", "nightly_two"])]
    update = hass.config_entries.async_update_subentry

    def fail_before_write(*args, **kwargs):
        raise RuntimeError("injected subentry write failure")

    monkeypatch.setattr(hass.config_entries, "async_update_subentry", fail_before_write)
    with pytest.raises(RuntimeError, match="injected subentry"):
        _persist_function_configuration(hass, entry, subentry, tools, groups)
    assert dict(subentry.data) == original
    monkeypatch.setattr(hass.config_entries, "async_update_subentry", update)
    _persist_function_configuration(hass, entry, subentry, tools, groups)
    assert {
        tool["spec"]["name"]
        for tool in configured_function_tools_from_data(subentry.data)
    } >= {"nightly_one", "nightly_two"}
    assert validate_function_groups(
        subentry.data[CONF_FUNCTION_GROUPS],
        configured_function_tools_from_data(subentry.data),
    )[0]["functions"] == ["nightly_one", "nightly_two"]
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    subentry = next(iter(entry.subentries.values()))
    assert validate_function_groups(
        subentry.data[CONF_FUNCTION_GROUPS],
        configured_function_tools_from_data(subentry.data),
    )[0]["functions"] == ["nightly_one", "nightly_two"]
    record(stress_trace, "function_group_transaction", injected_phases=1, reloads=1)


async def test_rule_group_and_references_rollback_on_store_failure(
    hass,
    monkeypatch,
    stress_trace,
) -> None:
    entry, subentry = await _entry(hass)
    rules = await async_get_request_rules(hass, entry.entry_id, subentry.subentry_id)
    await rules.async_set_groups([{"id": "old", "name": "Old"}])
    await rules.async_create(
        {
            "name": "Nightly linked rule",
            "phrases": ["nightly transaction marker"],
            "match_type": "equals",
            "action_type": "local_action",
            "action": {"actions": [{"action": "script.turn_on"}]},
            "group_id": "old",
        }
    )
    before = rules.snapshot()
    save = rules._store.async_save

    async def fail_before_write(_value):
        raise RuntimeError("injected rule store failure")

    monkeypatch.setattr(rules._store, "async_save", fail_before_write)
    with pytest.raises(RuntimeError, match="injected rule store"):
        await rules.async_set_groups([{"id": "new", "name": "New"}])
    assert rules.snapshot() == before
    monkeypatch.setattr(rules._store, "async_save", save)
    result = await rules.async_set_groups([{"id": "new", "name": "New"}])
    assert result["groups"][0]["id"] == "new"
    assert result["rules"][0]["group_id"] is None
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    reloaded = await async_get_request_rules(hass, entry.entry_id, subentry.subentry_id)
    assert reloaded.snapshot()["groups"] == result["groups"]
    assert reloaded.snapshot()["rules"][0]["group_id"] is None
    record(stress_trace, "rule_group_transaction", injected_phases=1, reloads=1)
