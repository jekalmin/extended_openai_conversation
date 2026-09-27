"""Focused edge-case coverage for persisted Function Tool recovery."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from custom_components.extended_openai_conversation_responses import management_ui
from custom_components.extended_openai_conversation_responses.agent_config import (
    agent_config_defaults,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
)
from custom_components.extended_openai_conversation_responses.management_function_repair import (
    async_function_repair,
    function_tools_issue,
    isolated_function_tools,
)
from homeassistant.exceptions import HomeAssistantError


class _FakeConfigEntries:
    def __init__(self) -> None:
        self.updates = 0

    def async_update_subentry(
        self, _entry: Any, subentry: Any, *, data: dict[str, Any], **_kwargs: Any
    ) -> None:
        subentry.data = data
        self.updates += 1


def _invalid_legacy_tool_data() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    defaults = agent_config_defaults()
    tools = yaml.safe_load(defaults[CONF_FUNCTION_TOOLS])
    assert isinstance(tools, list) and tools
    parameters = tools[0]["spec"].setdefault(
        "parameters", {"type": "object", "properties": {}}
    )
    parameters["description"] = 123
    return (
        {
            CONF_FUNCTION_TOOLS: yaml.safe_dump(
                tools, sort_keys=False, allow_unicode=True
            ),
            CONF_FUNCTION_GROUPS: deepcopy(defaults[CONF_FUNCTION_GROUPS]),
        },
        tools,
    )


def _mixed_legacy_tool_data() -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    defaults = agent_config_defaults()
    tools = yaml.safe_load(defaults[CONF_FUNCTION_TOOLS])
    assert isinstance(tools, list) and tools
    valid_tool = deepcopy(tools[0])
    broken_tool = deepcopy(tools[0])
    broken_tool["spec"]["name"] = f"{broken_tool['spec']['name']}_broken"
    broken_tool["spec"].setdefault(
        "parameters", {"type": "object", "properties": {}}
    )["description"] = 123
    mixed = [valid_tool, broken_tool]
    return (
        {
            CONF_FUNCTION_TOOLS: yaml.safe_dump(
                mixed, sort_keys=False, allow_unicode=True
            ),
            CONF_FUNCTION_GROUPS: deepcopy(defaults[CONF_FUNCTION_GROUPS]),
        },
        mixed,
        valid_tool,
    )


def _entry_and_subentry(data: dict[str, Any]) -> tuple[Any, Any]:
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Broken agent",
        data=data,
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        title="Extended OpenAI",
        data={},
        subentries={subentry.subentry_id: subentry},
    )
    return entry, subentry


def test_function_tools_issue_isolates_malformed_yaml() -> None:
    """Malformed persisted YAML must not collapse the agent catalogue."""
    configured, issue = function_tools_issue({CONF_FUNCTION_TOOLS: "[unterminated"})

    assert configured == []
    assert issue is not None


def test_isolated_function_tools_keeps_valid_siblings() -> None:
    """Per-tool repair isolates a bad tool without presenting valid siblings as broken."""
    data, _mixed, valid_tool = _mixed_legacy_tool_data()

    valid, invalid, issue = isolated_function_tools(data)

    assert issue is not None
    assert len(valid) == 1
    assert valid[0]["spec"]["name"] == valid_tool["spec"]["name"]
    assert len(invalid) == 1
    assert invalid[0]["index"] == 1
    assert invalid[0]["name"].endswith("_broken")
    assert "description" in invalid[0]["validation_error"]


def test_function_tools_issue_returns_valid_subset_when_one_tool_is_bad() -> None:
    """A persisted bad tool is excluded while independently valid tools remain usable."""
    data, _mixed, valid_tool = _mixed_legacy_tool_data()

    configured, issue = function_tools_issue(data)

    assert issue is not None
    assert [tool["spec"]["name"] for tool in configured] == [
        valid_tool["spec"]["name"]
    ]


@pytest.mark.asyncio
async def test_function_repair_get_returns_only_invalid_tool_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair metadata identifies only bad tools while retaining the raw collection."""
    data, mixed, _valid_tool = _mixed_legacy_tool_data()
    entry, subentry = _entry_and_subentry(data)
    hass = SimpleNamespace(data={}, config_entries=_FakeConfigEntries())
    monkeypatch.setattr(
        management_ui,
        "entry_and_agent",
        lambda *_args, **_kwargs: (entry, subentry),
    )

    repair = await async_function_repair(
        hass,
        "admin",
        True,
        {
            "action": "get",
            "entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
        },
    )

    assert repair["tools"] == mixed
    assert len(repair["invalid_tools"]) == 1
    assert repair["invalid_tools"][0]["index"] == 1
    assert repair["invalid_tools"][0]["tool"] == mixed[1]


@pytest.mark.asyncio
async def test_function_repair_save_one_replaces_only_selected_invalid_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-tool repair persists while the valid sibling stays intact."""
    data, _mixed, valid_tool = _mixed_legacy_tool_data()
    entry, subentry = _entry_and_subentry(data)
    config_entries = _FakeConfigEntries()
    hass = SimpleNamespace(data={}, config_entries=config_entries)
    monkeypatch.setattr(
        management_ui, "entry_and_agent", lambda *_args, **_kwargs: (entry, subentry)
    )
    before = await async_function_repair(
        hass,
        "admin",
        True,
        {"action": "get", "entry_id": entry.entry_id, "subentry_id": subentry.subentry_id},
    )
    replacement = deepcopy(valid_tool)
    replacement["spec"]["name"] = "repaired_tool"

    saved = await async_function_repair(
        hass,
        "admin",
        True,
        {
            "action": "save_one",
            "entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
            "revision": before["revision"],
            "index": 1,
            "tool": replacement,
        },
    )
    assert saved["revision"] != before["revision"]
    assert config_entries.updates == 1
    persisted_tools, issue = function_tools_issue(dict(subentry.data))
    assert issue is None
    assert persisted_tools[0] == valid_tool
    assert persisted_tools[1]["spec"]["name"] == "repaired_tool"


@pytest.mark.asyncio
async def test_function_repair_save_replaces_invalid_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole-field repair publishes a valid replacement and new revision."""
    data, _mixed, valid_tool = _mixed_legacy_tool_data()
    entry, subentry = _entry_and_subentry(data)
    config_entries = _FakeConfigEntries()
    hass = SimpleNamespace(data={}, config_entries=config_entries)
    monkeypatch.setattr(
        management_ui, "entry_and_agent", lambda *_args, **_kwargs: (entry, subentry)
    )
    before = await async_function_repair(
        hass,
        "admin",
        True,
        {"action": "get", "entry_id": entry.entry_id, "subentry_id": subentry.subentry_id},
    )
    saved = await async_function_repair(
        hass,
        "admin",
        True,
        {
            "action": "save",
            "entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
            "revision": before["revision"],
            "tools": [valid_tool],
        },
    )
    assert saved["valid"] is True
    assert saved["revision"] != before["revision"]
    assert config_entries.updates == 1
    assert function_tools_issue(dict(subentry.data))[1] is None


@pytest.mark.asyncio
async def test_function_repair_rejects_stale_raw_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated concurrent config change invalidates a repair write."""
    data, tools = _invalid_legacy_tool_data()
    entry, subentry = _entry_and_subentry(data)
    config_entries = _FakeConfigEntries()
    hass = SimpleNamespace(data={}, config_entries=config_entries)
    monkeypatch.setattr(
        management_ui,
        "entry_and_agent",
        lambda *_args, **_kwargs: (entry, subentry),
    )

    repair = await async_function_repair(
        hass,
        "admin",
        True,
        {
            "action": "get",
            "entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
        },
    )
    subentry.data = {**subentry.data, "concurrent_change": True}
    repaired_tools = deepcopy(tools)
    repaired_tools[0]["spec"]["parameters"].pop("description")

    with pytest.raises(HomeAssistantError, match="changed in another tab"):
        await async_function_repair(
            hass,
            "admin",
            True,
            {
                "action": "save",
                "entry_id": entry.entry_id,
                "subentry_id": subentry.subentry_id,
                "revision": repair["revision"],
                "tools": repaired_tools,
            },
        )

    assert config_entries.updates == 0


@pytest.mark.asyncio
async def test_function_repair_requires_admin() -> None:
    """Persisted agent repair remains an administrator-only operation."""
    data, _tools = _invalid_legacy_tool_data()
    entry, subentry = _entry_and_subentry(data)
    hass = SimpleNamespace(data={}, config_entries=_FakeConfigEntries())

    with pytest.raises(HomeAssistantError, match="Administrator permission"):
        await async_function_repair(
            hass,
            "user-1",
            False,
            {
                "action": "get",
                "entry_id": entry.entry_id,
                "subentry_id": subentry.subentry_id,
            },
        )
