"""Focused regressions for roadmap PR2 Function Tool concurrency integrity."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from homeassistant.exceptions import HomeAssistantError

from custom_components.extended_openai_conversation_responses.agent_config import (
    agent_config_defaults,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_GUEST_ALLOWED_FUNCTION_NAMES,
    DOMAIN,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    async_management_command,
)


def _tool(name: str) -> dict:
    return {
        "spec": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "native", "name": "execute_service"},
    }


def _setup_entry(hass):
    data = agent_config_defaults()
    data[CONF_FUNCTION_TOOLS] = yaml.safe_dump([_tool("old_tool")], sort_keys=False)
    data[CONF_FUNCTION_GROUPS] = [
        {
            "id": "group",
            "name": "Group",
            "description": "Grouped functions",
            "loading_mode": "always",
            "functions": ["old_tool"],
        }
    ]
    data[CONF_GUEST_ALLOWED_FUNCTION_NAMES] = []
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Jarvis",
        data=data,
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain=DOMAIN,
        title="Provider",
        data={},
        subentries={subentry.subentry_id: subentry},
    )
    hass.config_entries.async_get_entry.return_value = entry
    return entry, subentry


class _Rules:
    def __init__(self, *, rename_error: Exception | None = None, on_rename=None) -> None:
        self.rename_error = rename_error
        self.on_rename = on_rename
        self.expected_revision = None

    def revision(self) -> str:
        return "rules-revision"

    def function_references(self, _function_name: str):
        return []

    async def async_rename_function_reference(
        self,
        _old_name: str,
        _new_name: str,
        *,
        expected_revision: str | None = None,
    ) -> int:
        self.expected_revision = expected_revision
        if self.on_rename is not None:
            self.on_rename()
        if self.rename_error is not None:
            raise self.rename_error
        return 0


def _concurrent_data(subentry, name: str) -> dict:
    newer = deepcopy(subentry.data)
    tools = yaml.safe_load(newer[CONF_FUNCTION_TOOLS])
    tools.append(_tool(name))
    newer[CONF_FUNCTION_TOOLS] = yaml.safe_dump(tools, sort_keys=False)
    return newer


@pytest.mark.parametrize("action", ["save", "delete"])
@pytest.mark.parametrize("aba", [False, True])
async def test_tool_cross_store_mutation_rejects_agent_change_during_rule_lookup(
    hass, monkeypatch, action, aba
) -> None:
    entry, subentry = _setup_entry(hass)
    rules = _Rules()
    newer = _concurrent_data(subentry, "concurrent_tool")
    original = deepcopy(subentry.data)

    async def lookup_rules(*_args, **_kwargs):
        subentry.data = newer
        if aba:
            subentry.data = deepcopy(original)
        return rules

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.management_ui.async_get_request_rules",
        lookup_rules,
    )

    message = {
        "section": "tools",
        "action": action,
        "entry_id": entry.entry_id,
        "subentry_id": subentry.subentry_id,
    }
    if action == "save":
        message.update(original_name="old_tool", tool=_tool("new_tool"))
    else:
        message.update(name="old_tool", confirm=True)

    with pytest.raises(HomeAssistantError, match="Configuration changed in another tab"):
        await async_management_command(hass, "admin", True, message)

    hass.config_entries.async_update_subentry.assert_not_called()
    assert (
        any(
            tool["spec"]["name"] == "concurrent_tool"
            for tool in yaml.safe_load(subentry.data[CONF_FUNCTION_TOOLS])
        )
        is not aba
    )


async def test_tool_rename_uses_request_rule_revision(hass, monkeypatch) -> None:
    entry, subentry = _setup_entry(hass)
    rules = _Rules()
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.management_ui.async_get_request_rules",
        AsyncMock(return_value=rules),
    )

    await async_management_command(
        hass,
        "admin",
        True,
        {
            "section": "tools",
            "action": "save",
            "entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
            "original_name": "old_tool",
            "tool": _tool("new_tool"),
        },
    )

    assert rules.expected_revision == "rules-revision"


async def test_tool_rename_rollback_does_not_overwrite_newer_agent_edit(
    hass, monkeypatch
) -> None:
    entry, subentry = _setup_entry(hass)
    writes: list[dict] = []

    def persist(_entry, _subentry, *, data, **_kwargs):
        writes.append(data)
        subentry.data = data

    hass.config_entries.async_update_subentry.side_effect = persist

    def concurrent_edit() -> None:
        subentry.data = _concurrent_data(subentry, "concurrent_tool")

    rules = _Rules(rename_error=ValueError("stale rules"), on_rename=concurrent_edit)
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.management_ui.async_get_request_rules",
        AsyncMock(return_value=rules),
    )

    with pytest.raises(
        HomeAssistantError,
        match="newer configuration was preserved",
    ):
        await async_management_command(
            hass,
            "admin",
            True,
            {
                "section": "tools",
                "action": "save",
                "entry_id": entry.entry_id,
                "subentry_id": subentry.subentry_id,
                "original_name": "old_tool",
                "tool": _tool("new_tool"),
            },
        )

    assert len(writes) == 1
    assert any(
        tool["spec"]["name"] == "concurrent_tool"
        for tool in yaml.safe_load(subentry.data[CONF_FUNCTION_TOOLS])
    )
