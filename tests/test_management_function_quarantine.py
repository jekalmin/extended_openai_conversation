"""Regression coverage for management behavior while Function Tools need repair."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from custom_components.extended_openai_conversation_responses import (
    management_function_quarantine as quarantine,
)
from custom_components.extended_openai_conversation_responses.agent_config import (
    agent_config_defaults,
)
from custom_components.extended_openai_conversation_responses.agent_test import (
    AgentTestResult,
    TestCheck,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_API_PROVIDER,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
)


def _mixed_unknown_native_data() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    defaults = agent_config_defaults()
    tools = yaml.safe_load(defaults[CONF_FUNCTION_TOOLS])
    assert isinstance(tools, list) and tools
    valid_tool = deepcopy(tools[0])
    invalid_tool = deepcopy(valid_tool)
    invalid_tool["spec"]["name"] = "schedule_deferred_action"
    invalid_tool["function"] = {
        "type": "native",
        "name": "deferred_actions.create_safe",
    }
    mixed = [valid_tool, invalid_tool]
    return (
        {
            **defaults,
            CONF_FUNCTION_TOOLS: yaml.safe_dump(
                mixed, sort_keys=False, allow_unicode=True
            ),
            CONF_FUNCTION_GROUPS: deepcopy(defaults[CONF_FUNCTION_GROUPS]),
        },
        valid_tool,
        invalid_tool,
    )


def test_safe_function_configuration_quarantines_unknown_native() -> None:
    """Management-only normalization keeps valid siblings and removes bad natives."""
    data, valid_tool, invalid_tool = _mixed_unknown_native_data()

    safe = quarantine._safe_function_configuration(data)
    parsed = yaml.safe_load(safe[CONF_FUNCTION_TOOLS])

    assert [tool["spec"]["name"] for tool in parsed] == [valid_tool["spec"]["name"]]
    assert invalid_tool["spec"]["name"] not in {
        name
        for group in safe[CONF_FUNCTION_GROUPS]
        for name in group.get("functions", [])
    }


def test_management_merge_preserves_repair_owned_function_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving Guest Mode cannot silently rewrite quarantined Function Tools."""
    data, _valid_tool, _invalid_tool = _mixed_unknown_native_data()
    original_tools = data[CONF_FUNCTION_TOOLS]
    original_groups = deepcopy(data[CONF_FUNCTION_GROUPS])

    monkeypatch.setattr(
        quarantine,
        "_STRICT_MERGE_AGENT_CONFIG",
        lambda source, updates: {**dict(source), **updates},
    )
    token = quarantine._ALLOW_QUARANTINED_TOOLS.set(True)
    try:
        merged = quarantine._management_merge_agent_config(
            data, {"guest_mode_enabled": True}
        )
    finally:
        quarantine._ALLOW_QUARANTINED_TOOLS.reset(token)

    assert merged["guest_mode_enabled"] is True
    assert merged[CONF_FUNCTION_TOOLS] == original_tools
    assert merged[CONF_FUNCTION_GROUPS] == original_groups


@pytest.mark.parametrize("section", ["request_rules", "guest_mode", "tools"])
def test_tolerant_management_scope_is_limited_and_reset_after_errors(
    monkeypatch, section
):
    sentinel = [{"spec": {"name": "valid"}, "function": {"type": "template"}}]
    monkeypatch.setattr(quarantine, "_usable_function_tools", lambda _: sentinel)

    def strict(_):
        raise AssertionError("strict parser used")

    monkeypatch.setattr(quarantine, "_STRICT_CONFIGURED_TOOLS", strict)
    with pytest.raises(RuntimeError, match="handler failed"):
        with quarantine.management_function_tools(section):
            assert (
                quarantine._management_configured_tools({CONF_FUNCTION_TOOLS: "broken"})
                == sentinel
            )
            raise RuntimeError("handler failed")
    with pytest.raises(AssertionError, match="strict parser used"):
        quarantine._management_configured_tools({})
    with quarantine.management_function_tools("configuration"):
        with pytest.raises(AssertionError, match="strict parser used"):
            quarantine._management_configured_tools({})


@pytest.mark.asyncio
async def test_diagnostics_quarantines_invalid_tools_and_reports_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad Function Tool no longer prevents provider diagnostics from running."""
    data, valid_tool, invalid_tool = _mixed_unknown_native_data()
    entry = SimpleNamespace(entry_id="entry-1", runtime_data=object(), data={})
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        title="Assistant",
        data=data,
    )
    observed: dict[str, Any] = {}

    async def original(_hass: Any, _entry: Any, safe_subentry: Any) -> AgentTestResult:
        observed["tools"] = yaml.safe_load(safe_subentry.data[CONF_FUNCTION_TOOLS])
        return AgentTestResult(
            "Passed",
            [TestCheck("Provider request", "Passed", "Request succeeded")],
        )

    monkeypatch.setattr(quarantine, "async_test_configured_agent", original)

    result = await quarantine._tolerant_agent_test(SimpleNamespace(), entry, subentry)

    assert [tool["spec"]["name"] for tool in observed["tools"]] == [
        valid_tool["spec"]["name"]
    ]
    assert invalid_tool["spec"]["name"] not in {
        tool["spec"]["name"] for tool in observed["tools"]
    }
    assert result.status == "Warning"
    assert result.checks[-1].name == "Function Tools"
    assert result.checks[-1].status == "Warning"
    assert "schedule_deferred_action" in result.checks[-1].message


@pytest.mark.asyncio
async def test_overview_fallback_uses_persisted_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrelated tool damage must not become Unknown provider / No model selected."""

    entry = SimpleNamespace(
        entry_id="entry-1",
        runtime_data=None,
        data={CONF_API_PROVIDER: "openai"},
    )
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        data={CONF_CHAT_MODEL: "gpt-5.6-luna"},
    )

    from custom_components.extended_openai_conversation_responses import (
        management_setup_health,
    )

    def fail_health(*args, **kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(
        management_setup_health, "build_setup_health_facts", fail_health
    )
    result = management_setup_health.add_setup_health(
        object(), entry, subentry, {"agent": {}}, is_admin=True
    )
    runtime = result["setup_health"]["provider_runtime"]
    assert runtime["provider"] == "openai"
    assert runtime["model"] == "gpt-5.6-luna"
    assert runtime["client_loaded"] is False


def test_management_quarantine_clean_paths_use_strict_helpers_and_reset_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {CONF_FUNCTION_TOOLS: "clean"}
    strict_tools = Mock(return_value=[{"spec": {"name": "clean"}}])
    strict_groups = Mock(return_value=[{"id": "g", "functions": ["clean"]}])
    strict_merge = Mock(return_value={"merged": True})
    monkeypatch.setattr(quarantine, "function_tools_issue", lambda _raw: ([], None))
    monkeypatch.setattr(quarantine, "_STRICT_CONFIGURED_TOOLS", strict_tools)
    monkeypatch.setattr(quarantine, "_STRICT_VALIDATE_FUNCTION_GROUPS", strict_groups)
    monkeypatch.setattr(quarantine, "_STRICT_MERGE_AGENT_CONFIG", strict_merge)

    token = quarantine._QUARANTINED_FUNCTION_NAMES.set(frozenset({"stale"}))
    try:
        assert quarantine._usable_function_tools(data) == [{"spec": {"name": "clean"}}]
        assert quarantine._QUARANTINED_FUNCTION_NAMES.get() == frozenset()
    finally:
        quarantine._QUARANTINED_FUNCTION_NAMES.reset(token)

    groups = [{"id": "g", "functions": ["clean"]}]
    tools = [{"spec": {"name": "clean"}}]
    assert quarantine._management_validate_function_groups(groups, tools) == [
        {"id": "g", "functions": ["clean"]}
    ]
    strict_groups.assert_called_once_with(groups, tools)

    assert quarantine._management_merge_agent_config(data, {"x": 1}) == {
        "merged": True
    }
    strict_merge.assert_called_once_with(data, {"x": 1})


def test_management_group_validation_filters_only_quarantined_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def strict(groups, tools):
        captured["groups"] = groups
        captured["tools"] = tools
        return groups

    monkeypatch.setattr(quarantine, "_STRICT_VALIDATE_FUNCTION_GROUPS", strict)
    allow = quarantine._ALLOW_QUARANTINED_TOOLS.set(True)
    names = quarantine._QUARANTINED_FUNCTION_NAMES.set(
        frozenset({"broken", "missing"})
    )
    try:
        result = quarantine._management_validate_function_groups(
            [
                {"id": "g", "functions": ["keep", "broken", 123]},
                {"id": "no-functions"},
                "not-a-group",
            ],
            [{"spec": {"name": "keep"}}],
        )
    finally:
        quarantine._QUARANTINED_FUNCTION_NAMES.reset(names)
        quarantine._ALLOW_QUARANTINED_TOOLS.reset(allow)

    assert result[0]["functions"] == ["keep", 123]
    assert captured["groups"][0]["functions"] == ["keep", 123]


def test_restore_quarantined_group_members_preserves_hidden_raw_members() -> None:
    groups = [
        {"id": "g1", "functions": ["valid"]},
        {"id": "g2", "functions": ["already", "broken"]},
        {"id": "missing-original", "functions": ["valid"]},
    ]
    raw = [
        {"id": "g1", "functions": ["valid", "broken", "other"]},
        {"id": "g2", "functions": ["already", "broken"]},
        {"id": "malformed", "functions": "not-a-list"},
    ]

    restored = quarantine._restore_quarantined_group_members(
        groups, raw, frozenset({"broken"})
    )

    assert restored[0]["functions"] == ["valid", "broken"]
    assert restored[1]["functions"] == ["already", "broken"]
    assert restored[2]["functions"] == ["valid"]
    assert groups[0]["functions"] == ["valid"]

    assert quarantine._restore_quarantined_group_members(
        groups, raw, frozenset()
    ) == groups
    assert quarantine._restore_quarantined_group_members(
        groups, object(), frozenset({"broken"})
    ) == groups


def test_management_merge_in_quarantine_preserves_raw_function_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        CONF_FUNCTION_TOOLS: "raw-tools",
        CONF_FUNCTION_GROUPS: [{"id": "raw-group"}],
        "other": "old",
    }
    monkeypatch.setattr(
        quarantine, "function_tools_issue", lambda _raw: ([], "broken")
    )
    monkeypatch.setattr(
        quarantine,
        "_safe_function_configuration",
        lambda _raw: {"safe": True},
    )
    monkeypatch.setattr(
        quarantine,
        "_STRICT_MERGE_AGENT_CONFIG",
        lambda source, updates: {**source, **updates},
    )

    token = quarantine._ALLOW_QUARANTINED_TOOLS.set(True)
    try:
        merged = quarantine._management_merge_agent_config(
            raw, {"other": "new"}
        )
    finally:
        quarantine._ALLOW_QUARANTINED_TOOLS.reset(token)

    assert merged["other"] == "new"
    assert merged[CONF_FUNCTION_TOOLS] == "raw-tools"
    assert merged[CONF_FUNCTION_GROUPS] == [{"id": "raw-group"}]


def test_tolerant_persist_delegates_when_configuration_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = Mock(return_value={"revision": "delegated"})
    monkeypatch.setattr(
        quarantine,
        "isolated_function_tools",
        lambda _raw: ([], [], None),
    )
    monkeypatch.setattr(
        quarantine, "persist_valid_function_configuration", delegated
    )
    hass = SimpleNamespace()
    entry = SimpleNamespace()
    subentry = SimpleNamespace(data={})

    result = quarantine._tolerant_persist_function_configuration(
        hass,
        entry,
        subentry,
        [{"spec": {"name": "ok"}}],
        [],
        extra_updates={"x": 1},
        expected_revision="revision",
    )

    assert result == {"revision": "delegated"}
    delegated.assert_called_once()


def test_tolerant_persist_rejects_stale_duplicate_and_unisolatable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = [{"index": 0, "name": "broken"}]
    monkeypatch.setattr(
        quarantine,
        "isolated_function_tools",
        lambda _raw: ([], invalid, "broken"),
    )
    monkeypatch.setattr(quarantine, "repair_revision", lambda _subentry: "current")
    subentry = SimpleNamespace(data={CONF_FUNCTION_TOOLS: "raw"})

    with pytest.raises(quarantine.HomeAssistantError, match="changed in another tab"):
        quarantine._tolerant_persist_function_configuration(
            SimpleNamespace(),
            SimpleNamespace(),
            subentry,
            [],
            [],
            expected_revision="stale",
        )

    monkeypatch.setattr(
        quarantine,
        "editable_function_tools",
        lambda _raw: [{"spec": {"name": "broken"}}],
    )
    with pytest.raises(quarantine.HomeAssistantError, match="already exists"):
        quarantine._tolerant_persist_function_configuration(
            SimpleNamespace(),
            SimpleNamespace(),
            subentry,
            [{"spec": {"name": "broken"}}],
            [],
            expected_revision="current",
        )

    monkeypatch.setattr(quarantine, "editable_function_tools", lambda _raw: "bad")
    with pytest.raises(
        quarantine.HomeAssistantError, match="cannot be isolated safely"
    ):
        quarantine._tolerant_persist_function_configuration(
            SimpleNamespace(),
            SimpleNamespace(),
            subentry,
            [],
            [],
            expected_revision="current",
        )


def test_tolerant_persist_keeps_invalid_raw_tool_and_hidden_group_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_tool = {"spec": {"name": "broken"}, "function": {"type": "legacy"}}
    raw_groups = [{"id": "g", "functions": ["valid", "broken"]}]
    subentry = SimpleNamespace(
        data={
            CONF_FUNCTION_TOOLS: yaml.safe_dump([invalid_tool]),
            CONF_FUNCTION_GROUPS: raw_groups,
            "guest_mode_enabled": False,
        }
    )
    invalid = [{"index": 0, "name": "broken"}]
    monkeypatch.setattr(
        quarantine,
        "isolated_function_tools",
        lambda _raw: ([], invalid, "broken"),
    )
    monkeypatch.setattr(
        quarantine,
        "editable_function_tools",
        lambda _raw: [deepcopy(invalid_tool)],
    )
    monkeypatch.setattr(quarantine, "repair_revision", lambda _subentry: "new-revision")
    monkeypatch.setattr(
        quarantine,
        "preserve_legacy_guest_policy",
        lambda _old, new: new,
    )

    class ConfigEntries:
        def async_update_subentry(self, _entry, target, *, data):
            target.data = data

    hass = SimpleNamespace(config_entries=ConfigEntries())
    tools = [{"spec": {"name": "valid"}, "function": {"type": "template"}}]
    groups = [{"id": "g", "functions": ["valid"]}]

    result = quarantine._tolerant_persist_function_configuration(
        hass,
        SimpleNamespace(),
        subentry,
        tools,
        groups,
        extra_updates={"guest_mode_enabled": True},
        expected_revision="new-revision",
    )

    persisted = yaml.safe_load(subentry.data[CONF_FUNCTION_TOOLS])
    assert persisted == [*tools, invalid_tool]
    assert subentry.data[CONF_FUNCTION_GROUPS] == [
        {"id": "g", "functions": ["valid", "broken"]}
    ]
    assert subentry.data["guest_mode_enabled"] is True
    assert result == {
        "functions": tools,
        "function_groups": groups,
        "revision": "new-revision",
    }


@pytest.mark.asyncio
async def test_tolerant_agent_test_clean_and_collection_level_warning_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean_result = AgentTestResult(
        "Passed", [TestCheck("Provider request", "Passed", "ok")]
    )
    provider = AsyncMock(return_value=clean_result)
    monkeypatch.setattr(quarantine, "async_test_configured_agent", provider)
    monkeypatch.setattr(
        quarantine,
        "isolated_function_tools",
        lambda _raw: ([], [], None),
    )
    entry = SimpleNamespace()
    subentry = SimpleNamespace(data={}, subentry_id="agent", title="Assistant")

    assert (
        await quarantine._tolerant_agent_test(
            SimpleNamespace(), entry, subentry
        )
        is clean_result
    )

    monkeypatch.setattr(
        quarantine,
        "isolated_function_tools",
        lambda _raw: ([], [], "collection-level corruption"),
    )
    monkeypatch.setattr(
        quarantine,
        "_safe_function_configuration",
        lambda raw: raw,
    )
    warned = AgentTestResult(
        "Passed", [TestCheck("Provider request", "Passed", "ok")]
    )
    provider.return_value = warned

    result = await quarantine._tolerant_agent_test(
        SimpleNamespace(), entry, subentry
    )

    assert result.checks[-1].name == "Function Tools"
    assert result.checks[-1].status == "Warning"
    assert "configuration was quarantined" in result.checks[-1].message
