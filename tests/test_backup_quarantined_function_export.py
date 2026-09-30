"""Regression coverage for exporting agents with quarantined Function Tools."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from custom_components.extended_openai_conversation_responses import (
    backup,
    function_dependency_integrity,
    transfer,
)
from custom_components.extended_openai_conversation_responses.agent_config import (
    AgentConfigError,
    agent_config_defaults,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    DOMAIN,
    SERVICE_CALL_FUNCTION,
)
from homeassistant.exceptions import HomeAssistantError


def _broken_config() -> dict:
    config = agent_config_defaults()
    tools = yaml.safe_load(config[CONF_FUNCTION_TOOLS])
    assert isinstance(tools, list) and tools
    invalid = deepcopy(tools[0])
    invalid["spec"]["name"] = "unavailable_reminder"
    invalid["function"] = {
        "type": "native",
        "name": "reminders.unavailable",
    }
    config[CONF_FUNCTION_TOOLS] = yaml.safe_dump(
        [tools[0], invalid], sort_keys=False, allow_unicode=True
    )
    config[CONF_FUNCTION_GROUPS] = [
        {
            "id": "mixed",
            "name": "Mixed",
            "description": "Contains a quarantined member",
            "loading_mode": "always",
            "functions": [tools[0]["spec"]["name"], "unavailable_reminder"],
        }
    ]
    return config


def test_export_snapshot_preserves_raw_quarantined_function_fields() -> None:
    config = _broken_config()

    snapshot = backup.export_configuration_snapshot(config)

    assert snapshot[CONF_FUNCTION_TOOLS] == config[CONF_FUNCTION_TOOLS]
    assert snapshot[CONF_FUNCTION_GROUPS] == config[CONF_FUNCTION_GROUPS]
    assert yaml.safe_load(snapshot[CONF_FUNCTION_TOOLS])[1]["function"]["name"] == (
        "reminders.unavailable"
    )


def test_export_snapshot_does_not_hide_unrelated_invalid_configuration() -> None:
    config = agent_config_defaults()
    config["max_tokens"] = 0

    with pytest.raises(AgentConfigError, match="must be at least 1"):
        backup.export_configuration_snapshot(config)


async def test_full_backup_collection_succeeds_with_quarantined_function(
    monkeypatch,
) -> None:
    config = _broken_config()
    entry = SimpleNamespace(entry_id="entry-1")
    subentry = SimpleNamespace(subentry_id="agent-1", title="Jarvis", data=config)

    payloads = (
        {"memories": []},
        {"records": []},
        {"sources": []},
        {"sessions": [], "turns": []},
        {"totals": {}, "daily": {}, "requests": [], "runs": []},
        {"schedule": None},
        {"storage_version": 1, "defaults": {}, "rules": []},
    )
    managers = tuple(
        SimpleNamespace(async_backup_data=AsyncMock(return_value=payload))
        for payload in payloads
    )
    monkeypatch.setattr(backup, "_managers", AsyncMock(return_value=managers))

    snapshot = await backup.async_collect_backup_snapshot(
        SimpleNamespace(), entry, subentry
    )

    exported = snapshot["agent"]["config"]
    assert exported[CONF_FUNCTION_TOOLS] == config[CONF_FUNCTION_TOOLS]
    assert exported[CONF_FUNCTION_GROUPS] == config[CONF_FUNCTION_GROUPS]


async def test_setup_export_collection_succeeds_with_quarantined_function(
    hass,
    monkeypatch,
) -> None:
    config = _broken_config()
    entry = SimpleNamespace(entry_id="entry-1")
    subentry = SimpleNamespace(subentry_id="agent-1", title="Jarvis", data=config)
    rules = SimpleNamespace(
        async_backup_data=AsyncMock(
            return_value={"storage_version": 1, "defaults": {}, "rules": []}
        )
    )
    monkeypatch.setattr(
        transfer, "async_get_request_rules", AsyncMock(return_value=rules)
    )

    document = await transfer.async_collect_transfer_snapshot(
        hass, entry, subentry, mode="setup"
    )

    exported = document["sections"][transfer.SECTION_CONFIGURATION]
    assert exported[CONF_FUNCTION_TOOLS] == config[CONF_FUNCTION_TOOLS]
    assert exported[CONF_FUNCTION_GROUPS] == config[CONF_FUNCTION_GROUPS]


def test_setup_import_accepts_quarantined_function_for_repair() -> None:
    config = _broken_config()
    prepared = transfer.inspect_transfer(
        {
            "schema": transfer.LEGACY_AGENT_SCHEMA,
            "version": transfer.AGENT_CONFIG_EXPORT_VERSION,
            "title": "Imported",
            "config": config,
        },
        "target-agent",
    )

    assert prepared.config is not None
    assert prepared.config[CONF_FUNCTION_TOOLS] == config[CONF_FUNCTION_TOOLS]
    assert prepared.config[CONF_FUNCTION_GROUPS] == config[CONF_FUNCTION_GROUPS]


async def test_request_rule_restore_allows_only_explicitly_quarantined_name(
    hass,
) -> None:
    rule = {
        "action": {
            "actions": [
                {
                    "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
                    "data": {"function": "unavailable_reminder", "arguments": {}},
                }
            ]
        }
    }

    await function_dependency_integrity.async_validate_request_rule_functions(
        hass,
        rule,
        [],
        quarantined_names={"unavailable_reminder"},
    )

    with pytest.raises(HomeAssistantError, match="unavailable or disabled"):
        await function_dependency_integrity.async_validate_request_rule_functions(
            hass,
            rule,
            [],
        )
