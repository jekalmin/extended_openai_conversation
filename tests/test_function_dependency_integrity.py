"""Regression coverage for Function Tool and Request Rule dependency integrity."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.extended_openai_conversation_responses import (
    function_dependency_integrity as integrity,
    management_ui,
)
from custom_components.extended_openai_conversation_responses.agent_config import (
    agent_config_defaults,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_GUEST_ALLOWED_GROUP_IDS,
    DOMAIN,
    SERVICE_CALL_FUNCTION,
)
from custom_components.extended_openai_conversation_responses.function_dependency_integrity import (
    async_rename_function_reference_recursive,
    async_validate_request_rule_functions,
    async_validate_static_function_arguments,
    recursive_function_references,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
    RequestRules,
)
from homeassistant.exceptions import HomeAssistantError


def _call(function: str = "nested_tool", arguments: dict | None = None) -> dict:
    return {
        "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
        "data": {"function": function, "arguments": arguments or {}},
    }


def _nested_rule() -> dict:
    return {
        "id": "rule-1",
        "name": "Nested functions",
        "enabled": True,
        "phrases": ["do nested work"],
        "match_type": "equals",
        "action_type": "local_action",
        "action": {
            "actions": [
                {"choose": [{"sequence": [_call()]}]},
                {"repeat": {"sequence": [_call()]}},
                {"parallel": [_call()]},
                {"sequence": [_call()]},
            ],
            "success_response": "Done",
            "failure_response": "Failed",
        },
        "matching_behavior": "defaults",
        "matching": DEFAULT_MATCHING,
        "order": 0,
    }


def _tool(parameters: dict | None = None) -> dict:
    return {
        "spec": {
            "name": "nested_tool",
            "description": "Nested test tool",
            "parameters": parameters
            or {"type": "object", "properties": {}, "additionalProperties": False},
        },
        "function": {"type": "native", "name": "execute_service"},
    }


def test_nested_function_references_cover_all_script_branches() -> None:
    manager = SimpleNamespace(_rules=[_nested_rule()])
    assert recursive_function_references(manager, "nested_tool") == [
        {"id": "rule-1", "name": "Nested functions"}
    ]
    assert recursive_function_references(manager, "other_tool") == []


async def test_nested_function_rename_reconciles_after_lost_save_acknowledgement(
    monkeypatch,
) -> None:
    manager = RequestRules(SimpleNamespace(async_save=AsyncMock()))
    manager._initialized = True
    manager._rules = [_nested_rule()]
    manager._sort_and_compile()
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.request_rules.validate_rule",
        lambda value, **_kwargs: value,
    )

    assert (
        await async_rename_function_reference_recursive(
            manager, "nested_tool", "nested_tool"
        )
        == 0
    )
    assert (
        await async_rename_function_reference_recursive(
            manager, "missing_tool", "renamed_tool"
        )
        == 0
    )

    revision = manager.revision()
    changed = await async_rename_function_reference_recursive(
        manager,
        "nested_tool",
        "renamed_tool",
        expected_revision=revision,
    )
    assert changed == 4
    assert recursive_function_references(manager, "nested_tool") == []
    assert recursive_function_references(manager, "renamed_tool") == [
        {"id": "rule-1", "name": "Nested functions"}
    ]

    async def save_then_lose_acknowledgement(candidate):
        manager._store.async_load = AsyncMock(return_value=deepcopy(candidate))
        raise OSError("directory fsync acknowledgement failed")

    manager._store.async_save = AsyncMock(side_effect=save_then_lose_acknowledgement)
    with pytest.raises(OSError, match="directory fsync acknowledgement failed"):
        await async_rename_function_reference_recursive(
            manager,
            "renamed_tool",
            "third_tool",
            expected_revision=manager.revision(),
        )
    assert manager._initialized
    assert recursive_function_references(manager, "renamed_tool") == []
    assert recursive_function_references(manager, "third_tool") == [
        {"id": "rule-1", "name": "Nested functions"}
    ]


async def test_static_request_rule_arguments_use_full_nested_function_schema(
    hass,
) -> None:
    parameters = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["safe", "fast"]},
            "payload": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "minimum": 1, "maximum": 3},
                    "code": {"type": "string", "pattern": "^[A-Z]{2}$"},
                },
                "required": ["count", "code"],
                "additionalProperties": False,
            },
        },
        "required": ["mode", "payload"],
        "additionalProperties": False,
    }
    spec = _tool(parameters)["spec"]

    with pytest.raises(HomeAssistantError, match="at least 1"):
        await async_validate_static_function_arguments(
            hass,
            spec,
            {"mode": "safe", "payload": {"count": 0, "code": "AB"}},
        )
    with pytest.raises(HomeAssistantError, match="Unknown function input"):
        await async_validate_static_function_arguments(
            hass,
            spec,
            {
                "mode": "safe",
                "payload": {"count": 1, "code": "AB", "extra": True},
            },
        )
    with pytest.raises(HomeAssistantError, match="required pattern"):
        await async_validate_static_function_arguments(
            hass,
            spec,
            {"mode": "safe", "payload": {"count": 1, "code": "bad"}},
        )


async def test_dynamic_request_rule_leaf_defers_only_that_leaf(hass) -> None:
    parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1},
            "mode": {"type": "string", "enum": ["safe", "fast"]},
        },
        "required": ["count", "mode"],
        "additionalProperties": False,
    }
    spec = _tool(parameters)["spec"]

    await async_validate_static_function_arguments(
        hass, spec, {"count": "{{ captured_count }}", "mode": "safe"}
    )
    with pytest.raises(HomeAssistantError, match="one of its choices"):
        await async_validate_static_function_arguments(
            hass, spec, {"count": "{{ captured_count }}", "mode": "invalid"}
        )
    with pytest.raises(HomeAssistantError, match="Missing required function input"):
        await async_validate_static_function_arguments(
            hass, spec, {"count": "{{ captured_count }}"}
        )


async def test_nested_rule_validation_rejects_missing_or_invalid_function_calls(
    hass,
) -> None:
    parameters = {
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1}},
        "required": ["count"],
        "additionalProperties": False,
    }
    tool = _tool(parameters)
    rule = _nested_rule()
    rule["action"]["actions"][0]["choose"][0]["sequence"][0] = _call(
        arguments={"count": 0}
    )
    with pytest.raises(HomeAssistantError, match="at least 1"):
        await async_validate_request_rule_functions(hass, rule, [tool])

    rule["action"]["actions"][0]["choose"][0]["sequence"][0] = _call(
        "missing_tool", {"count": 1}
    )
    with pytest.raises(HomeAssistantError, match="missing_tool"):
        await async_validate_request_rule_functions(hass, rule, [tool])

    malformed_rule = {
        "action": {
            "actions": [
                {"action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}", "data": {}}
            ]
        }
    }
    with pytest.raises(HomeAssistantError, match="action is invalid"):
        await async_validate_request_rule_functions(hass, malformed_rule, [tool])

    malformed_arguments_rule = {
        "action": {
            "actions": [
                {
                    "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
                    "data": {"function": "nested_tool", "arguments": []},
                }
            ]
        }
    }
    with pytest.raises(HomeAssistantError, match="arguments must be an object"):
        await async_validate_request_rule_functions(
            hass, malformed_arguments_rule, [tool]
        )

    disabled_tool = _tool(parameters)
    disabled_tool["enabled"] = False
    valid_rule = {
        "action": {"actions": [_call(arguments={"count": 1})]}
    }
    with pytest.raises(HomeAssistantError, match="unavailable or disabled"):
        await async_validate_request_rule_functions(hass, valid_rule, [disabled_tool])

    await async_validate_request_rule_functions(hass, valid_rule, [tool])


async def test_tool_mutations_reject_stale_loaded_revision(hass, monkeypatch) -> None:
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Jarvis",
        data=agent_config_defaults(),
    )
    entry = SimpleNamespace(
        entry_id="entry-1", domain=DOMAIN, subentries={"agent-1": subentry}
    )
    monkeypatch.setattr(
        management_ui, "entry_and_agent", lambda *_args: (entry, subentry)
    )
    command = management_ui.async_management_command

    for action in ("save", "set_enabled", "delete", "save_group", "delete_group"):
        with pytest.raises(
            HomeAssistantError, match="Configuration changed in another tab"
        ):
            await command(
                hass,
                "admin",
                True,
                {
                    "section": "tools",
                    "action": action,
                    "entry_id": "entry-1",
                    "subentry_id": "agent-1",
                    "revision": "stale",
                },
            )


@pytest.mark.parametrize(
    ("action", "message", "expected_ids"),
    [
        (
            "save_group",
            {
                "original_id": "old_group",
                "group": {
                    "id": "new_group",
                    "name": "New group",
                    "description": "New",
                    "functions": [],
                    "loading_mode": "always",
                },
            },
            ["new_group", "other_group"],
        ),
        (
            "delete_group",
            {"group_id": "old_group", "confirm": True},
            ["other_group"],
        ),
    ],
)
async def test_group_dependency_update_shares_config_revision_and_write(
    hass, monkeypatch, action, message, expected_ids
) -> None:
    data = agent_config_defaults()
    data[CONF_GUEST_ALLOWED_GROUP_IDS] = ["old_group", "other_group"]
    data["function_groups"] = [
        {
            "id": "old_group",
            "name": "Old group",
            "description": "Old",
            "functions": [],
            "loading_mode": "always",
        },
        {
            "id": "other_group",
            "name": "Other group",
            "description": "Other",
            "functions": [],
            "loading_mode": "always",
        },
    ]
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Jarvis",
        data=data,
    )
    entry = SimpleNamespace(
        entry_id="entry-1", domain=DOMAIN, subentries={"agent-1": subentry}
    )
    monkeypatch.setattr(
        management_ui, "entry_and_agent", lambda *_args: (entry, subentry)
    )
    revision = management_ui._agent_config_revision(subentry.data, subentry.title)
    seen: dict = {}

    def persist(
        _hass,
        _entry,
        _subentry,
        _tools,
        _groups,
        *,
        extra_updates=None,
        expected_revision=None,
    ):
        seen["extra_updates"] = extra_updates
        seen["expected_revision"] = expected_revision
        return {"revision": "next-revision"}

    monkeypatch.setattr(
        management_ui,
        "_persist_function_configuration",
        persist,
    )

    command = management_ui.async_management_command
    result = await command(
        hass,
        "admin",
        True,
        {
            "section": "tools",
            "action": action,
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "revision": revision,
            **message,
        },
    )

    assert seen["expected_revision"] == revision
    assert seen["extra_updates"][CONF_GUEST_ALLOWED_GROUP_IDS] == expected_ids
    assert result["revision"] == "next-revision"

def test_template_detection_and_schema_type_inference() -> None:
    assert integrity._is_template_string("{{ value }}") is True
    assert integrity._is_template_string("plain") is False
    assert integrity._contains_template({"a": [1, {"b": "{% if x %}"}]}) is True
    assert integrity._contains_template(("plain", 2)) is False

    assert integrity._schema_types({"type": "string"}) == {"string"}
    assert integrity._schema_types({"type": ["string", 3, "null"]}) == {
        "string",
        "null",
    }
    assert integrity._schema_types({"properties": {}}) == {"object"}
    assert integrity._schema_types({"required": []}) == {"object"}
    assert integrity._schema_types({"items": {}}) == {"array"}
    assert integrity._schema_types({}) == set()


def test_mask_dynamic_schema_preserves_static_structure_and_defensive_shapes() -> None:
    schema = {
        "type": "object",
        "enum": [{"mode": "fixed"}],
        "const": {"mode": "fixed"},
        "properties": {
            "name": {"type": "string", "minLength": 2},
            "count": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": {"type": "string", "minLength": 1},
    }
    masked = integrity._mask_dynamic_schema(
        {"name": "{{ dynamic }}", "count": 2, "extra": "{{ value }}"}, schema
    )

    assert "enum" not in masked
    assert "const" not in masked
    assert masked["properties"]["name"] == {}
    assert masked["properties"]["count"] == {
        "type": "integer",
        "minimum": 1,
    }
    assert masked["properties"]["extra"] == {}
    assert masked["additionalProperties"] is False

    array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "uniqueItems": True,
    }
    array_masked = integrity._mask_dynamic_schema(
        ["fixed", "{{ dynamic }}"], array_schema
    )
    assert array_masked["items"] == {}
    assert "uniqueItems" not in array_masked

    unchanged = {"type": "string", "minLength": 2}
    assert integrity._mask_dynamic_schema("fixed", unchanged) == unchanged

    assert integrity._mask_dynamic_schema(
        {"known": "{{ dynamic }}", "extra": "static"},
        {
            "type": "object",
            "properties": {"known": {"type": "string"}},
            "additionalProperties": True,
        },
    ) == {
        "type": "object",
        "properties": {"known": {}},
        "additionalProperties": True,
    }
    assert integrity._mask_dynamic_schema(
        {"extra": "{{ dynamic }}"},
        {"type": "object", "additionalProperties": True},
    ) == {"type": "object", "additionalProperties": True}
    assert integrity._mask_dynamic_schema(
        {"x": "{{ dynamic }}"}, array_schema
    ) == array_schema


async def test_static_descendant_validation_skips_templates_and_defensive_shapes(
    monkeypatch,
) -> None:
    validate = AsyncMock()
    monkeypatch.setattr(integrity, "async_validate_function_arguments", validate)

    await integrity._async_validate_static_descendants(
        SimpleNamespace(), "{{ dynamic }}", {"type": "string"}
    )
    validate.assert_not_awaited()

    await integrity._async_validate_static_descendants(
        SimpleNamespace(), "fixed", {"type": "string", "minLength": 2}
    )
    validate.assert_awaited_once()
    spec, arguments = validate.await_args.args[1:]
    assert spec["parameters"]["properties"]["value"]["minLength"] == 2
    assert arguments == {"value": "fixed"}

    validate.reset_mock()
    await integrity._async_validate_static_descendants(
        SimpleNamespace(),
        {"x": "{{ dynamic }}"},
        {"type": "array", "items": {"type": "string"}},
    )
    await integrity._async_validate_static_descendants(
        SimpleNamespace(),
        ["{{ dynamic }}", "static"],
        {"type": "array", "items": True},
    )
    validate.assert_not_awaited()


async def test_static_descendants_recurse_object_and_array(monkeypatch) -> None:
    validate = AsyncMock()
    monkeypatch.setattr(integrity, "async_validate_function_arguments", validate)
    schema = {
        "type": "object",
        "properties": {
            "known": {"type": "string"},
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "additionalProperties": {"type": "boolean"},
    }
    value = {
        "known": "fixed",
        "items": [1, "{{ later }}", 3],
        "extra": True,
        "dynamic": "{{ later }}",
    }

    await integrity._async_validate_static_descendants(SimpleNamespace(), value, schema)

    checked_values = [call.args[2]["value"] for call in validate.await_args_list]
    assert checked_values == ["fixed", 1, 3, True]


async def test_validate_static_function_arguments_rejects_bad_schema_and_masks_dynamic(
    monkeypatch,
) -> None:
    with pytest.raises(HomeAssistantError, match="input schema is invalid"):
        await integrity.async_validate_static_function_arguments(
            SimpleNamespace(), {"parameters": []}, {}
        )

    validate = AsyncMock()
    descendants = AsyncMock()
    monkeypatch.setattr(integrity, "async_validate_function_arguments", validate)
    monkeypatch.setattr(integrity, "_async_validate_static_descendants", descendants)

    spec = {
        "name": "demo",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    arguments = {"value": "{{ dynamic }}"}
    await integrity.async_validate_static_function_arguments(
        SimpleNamespace(), spec, arguments
    )

    masked_spec = validate.await_args.args[1]
    assert masked_spec["parameters"]["properties"]["value"] == {}
    descendants.assert_awaited_once()


async def test_rule_action_iteration_and_service_alias_references(monkeypatch) -> None:
    assert list(integrity._rule_script_actions({"action": {"actions": "bad"}})) == []
    assert list(integrity._rule_script_actions({})) == []

    monkeypatch.setattr(
        integrity.request_rules,
        "_iter_script_actions",
        lambda actions: iter(actions),
    )
    service = f"{integrity.DOMAIN}.{integrity.SERVICE_CALL_FUNCTION}"
    manager = SimpleNamespace(
        _rules=[
            {
                "id": "one",
                "name": "Rule one",
                "action": {
                    "actions": [
                        {"action": service, "data": {"function": "alpha"}},
                        {"action": "light.turn_on"},
                    ]
                },
            },
            {
                "id": "two",
                "name": "Rule two",
                "action": {
                    "actions": [{"service": service, "data": {"function": "beta"}}]
                },
            },
        ]
    )

    assert integrity.recursive_function_references(manager, "alpha") == [
        {"id": "one", "name": "Rule one"}
    ]
    assert integrity.recursive_function_references(manager, "beta") == [
        {"id": "two", "name": "Rule two"}
    ]


class _ReadOnlyAction(Mapping[str, object]):
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


async def test_function_rename_rejects_immutable_matching_action(monkeypatch) -> None:
    action = _ReadOnlyAction(
        {
            "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
            "data": {"function": "old_name", "arguments": {}},
        }
    )
    monkeypatch.setattr(
        integrity, "_rule_script_actions", lambda _rule: iter((action,))
    )
    manager = SimpleNamespace(
        _lock=asyncio.Lock(),
        _rules=[{"id": "rule", "name": "Rule"}],
        _require_revision_locked=Mock(),
        _sort_and_compile=Mock(),
        _async_save_locked=AsyncMock(),
    )

    with pytest.raises(ValueError, match="Request Rule action is not mutable"):
        await integrity.async_rename_function_reference_recursive(
            manager, "old_name", "new_name"
        )

    manager._async_save_locked.assert_not_awaited()
    manager._sort_and_compile.assert_not_called()


def test_guest_group_mutation_ignores_non_list_saved_ids() -> None:
    assert (
        integrity.group_reference_updates(
            {CONF_GUEST_ALLOWED_GROUP_IDS: ("old",)}, "old", "new"
        )
        == {}
    )


@pytest.mark.parametrize(
    "message",
    [
        {
            "section": "tools",
            "action": "save",
            "entry_id": None,
            "subentry_id": "agent",
        },
        {
            "section": "tools",
            "action": "delete_group",
            "entry_id": "entry",
            "subentry_id": 456,
        },
    ],
)
async def test_function_management_mutations_reject_invalid_ids_before_resolution(
    hass, monkeypatch, message
) -> None:
    select = Mock(side_effect=AssertionError("invalid selection was resolved"))
    monkeypatch.setattr(management_ui, "entry_and_agent", select)

    with pytest.raises(
        HomeAssistantError, match="entry_id and subentry_id are required"
    ):
        await management_ui.async_management_command(hass, "admin", True, message)

    select.assert_not_called()
