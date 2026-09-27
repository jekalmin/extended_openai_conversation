"""Request-lifecycle regression coverage for runtime Function Tool quarantine."""

from unittest.mock import Mock

import yaml


def _phone_tool(*, min_length):
    return {
        "spec": {
            "name": "invalid_phone_tool",
            "description": "Phone selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone": {
                        "type": "string",
                        "enum": ["home", "mobile"],
                        "minLength": min_length,
                    }
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    }


def test_runtime_quarantine_does_not_leak_into_next_request(monkeypatch) -> None:
    """A later clean request must not inherit an earlier quarantined tool name."""
    from custom_components.extended_openai_conversation_responses import agent_config

    broken = _phone_tool(min_length="legacy")
    logger = Mock()
    monkeypatch.setattr(quarantine, "_LOGGER", logger)
    first_tools = quarantine._runtime_configured_function_tools(
        {"functions": yaml.safe_dump([broken], sort_keys=False)}
    )

    assert first_tools == []
    assert quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.get() == frozenset(
        {"invalid_phone_tool"}
    )
    assert quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.get() is False
    logger.warning.assert_called_once()
    log_args = logger.warning.call_args.args
    assert "Extended OpenAI > Functions" in log_args[0]
    assert log_args[1] == "invalid_phone_tool"

    repaired = _phone_tool(min_length=1)
    second_tools = quarantine._runtime_configured_function_tools(
        {"functions": yaml.safe_dump([repaired], sort_keys=False)}
    )

    assert [tool["spec"]["name"] for tool in second_tools] == ["invalid_phone_tool"]
    assert quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.get() == frozenset()
    assert quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.get() is False

    captured = {}

    def validate(groups, function_tools):
        captured["groups"] = groups
        captured["function_tools"] = function_tools
        return groups

    monkeypatch.setattr(agent_config, "validate_function_groups", validate)
    groups = [
        {
            "id": "phone_tools",
            "name": "Phone tools",
            "description": "Phone-related tools.",
            "loading_mode": "always",
            "functions": ["invalid_phone_tool"],
        }
    ]

    result = quarantine._runtime_validate_function_groups(groups, second_tools)

    assert result[0]["functions"] == ["invalid_phone_tool"]
    assert captured["groups"][0]["functions"] == ["invalid_phone_tool"]
    assert captured["function_tools"] is second_tools


from custom_components.extended_openai_conversation_responses import (
    function_tool_quarantine as quarantine,
)


def test_runtime_quarantine_all_strips_string_group_members_and_warns(
    monkeypatch,
) -> None:
    from custom_components.extended_openai_conversation_responses import agent_config

    def strict_tools(data):
        raw = yaml.safe_load(data["functions"])
        if raw:
            raise ValueError("collection-level failure")
        return []

    captured = {}

    def validate(groups, function_tools):
        captured["groups"] = groups
        captured["function_tools"] = function_tools
        return groups

    logger = Mock()
    monkeypatch.setattr(agent_config, "configured_function_tools_from_data", strict_tools)
    monkeypatch.setattr(agent_config, "validate_function_groups", validate)
    monkeypatch.setattr(
        quarantine,
        "_isolated_function_tools",
        lambda _data: ([], [], "collection-level failure"),
    )
    monkeypatch.setattr(quarantine, "_LOGGER", logger)

    assert quarantine._runtime_configured_function_tools(
        {"functions": yaml.safe_dump([{"broken": True}])}
    ) == []
    assert quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.get() is True
    assert quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.get() == frozenset()

    groups = [
        {
            "id": "all",
            "functions": ["tool-a", 123, "tool-b"],
        },
        {"id": "malformed"},
        "not-a-group",
    ]
    result = quarantine._runtime_validate_function_groups(groups, [])

    assert result[0]["functions"] == [123]
    assert captured["groups"][0]["functions"] == [123]
    logger.warning.assert_called_once()
    assert "All configured Function Tools are invalid" in logger.warning.call_args.args[0]


def test_runtime_quarantine_named_tools_only_removes_matching_group_members(
    monkeypatch,
) -> None:
    from custom_components.extended_openai_conversation_responses import agent_config

    captured = {}

    def validate(groups, function_tools):
        captured["groups"] = groups
        return groups

    monkeypatch.setattr(agent_config, "validate_function_groups", validate)
    names_token = quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.set(
        frozenset({"broken"})
    )
    all_token = quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.set(False)
    try:
        result = quarantine._runtime_validate_function_groups(
            [
                {
                    "id": "group",
                    "functions": ["keep", "broken", 123],
                }
            ],
            [{"spec": {"name": "keep"}}],
        )
    finally:
        quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.reset(all_token)
        quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.reset(names_token)

    assert result[0]["functions"] == ["keep", 123]
    assert captured["groups"][0]["functions"] == ["keep", 123]


def test_runtime_quarantine_re_raises_when_isolation_finds_no_issue(
    monkeypatch,
) -> None:
    from custom_components.extended_openai_conversation_responses import agent_config

    def fail(_data):
        raise ValueError("strict failure")

    monkeypatch.setattr(agent_config, "configured_function_tools_from_data", fail)
    monkeypatch.setattr(
        quarantine,
        "_isolated_function_tools",
        lambda _data: ([], [], None),
    )

    import pytest

    with pytest.raises(ValueError, match="strict failure"):
        quarantine._runtime_configured_function_tools({"functions": "[]"})


def test_runtime_group_validation_without_quarantine_uses_strict_value(
    monkeypatch,
) -> None:
    from custom_components.extended_openai_conversation_responses import agent_config

    marker = object()
    validate = Mock(return_value=marker)
    monkeypatch.setattr(agent_config, "validate_function_groups", validate)
    names_token = quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.set(frozenset())
    all_token = quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.set(False)
    groups = [{"id": "clean", "functions": ["tool"]}]
    tools = [{"spec": {"name": "tool"}}]
    try:
        assert quarantine._runtime_validate_function_groups(groups, tools) is marker
    finally:
        quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.reset(all_token)
        quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.reset(names_token)

    validate.assert_called_once_with(groups, tools)
