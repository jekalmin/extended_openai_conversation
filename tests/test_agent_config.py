"""Tests for the shared conversation-agent configuration contract."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from custom_components.extended_openai_conversation_responses import agent_config
from custom_components.extended_openai_conversation_responses.agent_config import (
    MAX_AGENT_TITLE_LENGTH,
    AgentConfigError,
    agent_config_defaults,
    agent_config_options,
    agent_config_snapshot,
    configured_function_tool_metadata_from_data,
    function_tool_enabled,
    function_tool_yaml,
    merge_agent_config,
    model_capabilities,
    normalize_agent_config,
    starter_function_tool_yaml,
    validate_agent_title,
    validate_function_groups,
    validate_function_tools,
    validate_single_function_tool,
    validate_speech_regex_replacements,
)
from custom_components.extended_openai_conversation_responses.built_in_functions import (
    built_in_function_catalog,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_MEMORY_AUTO_CREATE,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MODE,
    CONF_SPEECH_REGEX_REPLACEMENTS,
)


def _native_tool(name: str = "test_tool") -> dict:
    return {
        "spec": {
            "name": name,
            "description": "Test tool",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "native", "name": "execute_service"},
    }


def test_model_change_without_effort_uses_new_models_recommended_profile() -> None:
    current = normalize_agent_config(
        {"chat_model": "gpt-6-luna", "reasoning_effort": "low", "functions": []}
    )
    changed = merge_agent_config(current, {"chat_model": "gpt-4.1", "functions": []})
    assert changed["chat_model"] == "gpt-4.1"
    assert "reasoning_effort" not in changed
    assert changed["functions"] == current["functions"]
    restored = merge_agent_config(changed, {"chat_model": "gpt-6-luna"})
    assert restored["reasoning_effort"] == "low"


def test_defaults_are_isolated_and_snapshot_parses_tools(hass) -> None:
    first = agent_config_defaults()
    first["voice_device_mappings"]["kitchen"] = "user:one"
    assert agent_config_defaults()["voice_device_mappings"] == {}
    snapshot = agent_config_snapshot(agent_config_defaults())
    assert isinstance(snapshot["functions"], list)
    assert snapshot["chat_model"]


def test_normalization_preserves_legacy_memory_fields() -> None:
    result = normalize_agent_config({CONF_MEMORY_MODE: "automatic"})
    assert result[CONF_MEMORY_ENABLED] is True
    assert result[CONF_MEMORY_AUTO_CREATE] is True


def test_normalization_accepts_legacy_numeric_selector_values() -> None:
    result = normalize_agent_config(
        {
            "archive_session_timeout_minutes": "30",
            "archive_retention_days": "90",
            "usage_request_retention_days": "7",
            "max_tokens": 750.0,
            "temperature": "0.5",
        }
    )
    assert result["archive_session_timeout_minutes"] == 30
    assert result["archive_retention_days"] == 90
    assert result["usage_request_retention_days"] == 7
    assert result["max_tokens"] == 750
    assert result["temperature"] == 0.5


def test_model_capabilities_are_model_specific() -> None:
    assert model_capabilities("gpt-5-mini")["supports_reasoning_effort"] is True
    assert model_capabilities("gpt-4o")["supports_temperature"] is True


def test_invalid_values_and_unknown_updates_are_rejected() -> None:
    with pytest.raises(AgentConfigError, match="max_tokens"):
        normalize_agent_config({"max_tokens": 0})
    with pytest.raises(AgentConfigError, match="unknown fields"):
        merge_agent_config({}, {"api_key": "must-not-be-agent-data"})


def test_function_tools_validate_yaml_schema_and_duplicates() -> None:
    assert validate_function_tools([_native_tool()])[0]["spec"]["name"] == "test_tool"
    with pytest.raises(AgentConfigError, match="invalid YAML"):
        validate_function_tools("- spec: [")
    invalid_type = _native_tool()
    invalid_type["function"] = {"type": "unknown"}
    with pytest.raises(AgentConfigError, match="unrecognized function type"):
        validate_function_tools([invalid_type])
    with pytest.raises(AgentConfigError, match="duplicate tool name"):
        validate_function_tools([_native_tool(), _native_tool()])


def test_function_tool_cache_key_avoids_yaml_for_persisted_lists(
    monkeypatch,
) -> None:
    config = {"functions": [_native_tool()]}
    monkeypatch.setattr(
        agent_config.yaml,
        "safe_dump",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("persisted list cache keys must not use PyYAML")
        ),
    )

    key = agent_config._configured_tools_yaml(config)

    assert key == agent_config._configured_tools_yaml(
        {"functions": [dict(reversed(list(_native_tool().items())))]}
    )
    assert key.startswith("[{")


def test_function_tool_metadata_uses_cached_tools_without_runtime_copy(
    monkeypatch,
) -> None:
    config = {"functions": yaml.safe_dump([_native_tool()], sort_keys=False)}
    agent_config._cached_configured_tools.cache_clear()
    monkeypatch.setattr(
        agent_config,
        "copy_runtime_function_config",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("metadata must not copy hydrated runtime configs")
        ),
    )

    metadata = configured_function_tool_metadata_from_data(config)

    assert metadata == {"usable_count": 1, "enabled_count": 1, "total_count": 1}


def test_function_tool_enabled_defaults_and_boolean_validation() -> None:
    legacy = validate_function_tools([_native_tool()])[0]
    assert "enabled" not in legacy
    assert function_tool_enabled(legacy) is True
    for value in (True, False):
        tool = _native_tool()
        tool["enabled"] = value
        assert validate_function_tools([tool])[0]["enabled"] is value
    for value in (0, 1, "false", None):
        tool = _native_tool()
        tool["enabled"] = value
        with pytest.raises(AgentConfigError, match=r"enabled.*boolean"):
            validate_function_tools([tool])


def test_built_in_catalogue_presets_are_valid_and_marks_used_native_tools() -> None:
    configured = _native_tool()
    catalog = built_in_function_catalog([configured])
    assert len(catalog) == 8
    assert all(validate_function_tools([preset["tool"]]) for preset in catalog)
    execute_service = next(
        preset for preset in catalog if preset["implementation"] == "execute_service"
    )
    assert execute_service["already_configured"] is True
    assert execute_service["tool"]["spec"]["name"] == "execute_service"
    assert (
        next(preset for preset in catalog if preset["implementation"] == "get_energy")[
            "already_configured"
        ]
        is False
    )


def test_function_tool_unknown_fields_are_preserved() -> None:
    tool = _native_tool()
    tool["x-extension"] = {"enabled": True}
    assert validate_function_tools([tool])[0]["x-extension"] == {"enabled": True}


def test_single_function_tool_yaml_round_trip_preserves_structure() -> None:
    tool = _native_tool()
    tool["x-extension"] = {"enabled": True, "labels": ["one", "two"]}
    serialized = function_tool_yaml(tool)
    assert serialized.startswith("spec:\n")
    assert not serialized.startswith("-")
    assert validate_single_function_tool(serialized) == tool


def test_single_function_tool_yaml_rejects_invalid_yaml_and_list_wrapper() -> None:
    with pytest.raises(AgentConfigError, match="invalid YAML"):
        validate_single_function_tool("spec: [")
    with pytest.raises(AgentConfigError, match="must contain an object"):
        validate_single_function_tool([_native_tool()])


def test_starter_function_tool_yaml_has_clean_single_tool_shape() -> None:
    starter = starter_function_tool_yaml()
    assert starter.startswith("spec:\n")
    assert "name: my_tool" in starter
    assert "type: native" in starter


@pytest.mark.parametrize(
    "fixture_name",
    [
        "native_execute_service_example.yaml",
        "template_example.yaml",
        "bash_example.yaml",
        "rest_example.yaml",
        "scrape_example.yaml",
        "sqlite_example.yaml",
        "script_example.yaml",
        "read_file_example.yaml",
        "write_file_example.yaml",
        "edit_file_example.yaml",
        "composite_example.yaml",
    ],
)
def test_single_tool_yaml_validates_existing_function_types(
    hass, fixture_name: str
) -> None:
    tools = yaml.safe_load(
        (Path(__file__).parent / "fixtures" / "functions" / fixture_name).read_text(
            encoding="utf-8"
        )
    )
    assert validate_single_function_tool(yaml.safe_dump(tools[0]))["spec"]["name"]


def test_regex_validation_is_field_specific_and_ordered() -> None:
    config = normalize_agent_config(
        {
            CONF_SPEECH_REGEX_REPLACEMENTS: [
                {"pattern": "HA", "replacement": "Home Assistant"}
            ]
        }
    )
    assert config[CONF_SPEECH_REGEX_REPLACEMENTS][0]["replacement"] == "Home Assistant"
    capture = normalize_agent_config(
        {
            CONF_SPEECH_REGEX_REPLACEMENTS: [
                {"pattern": "(HA)", "replacement": r"\1 assistant"}
            ]
        }
    )
    assert capture[CONF_SPEECH_REGEX_REPLACEMENTS][0]["replacement"] == r"\1 assistant"
    with pytest.raises(
        AgentConfigError, match=r"speech_regex_replacements\[0\]\.pattern"
    ):
        normalize_agent_config(
            {CONF_SPEECH_REGEX_REPLACEMENTS: [{"pattern": "[", "replacement": ""}]}
        )
    with pytest.raises(
        AgentConfigError, match=r"speech_regex_replacements\[0\]\.replacement"
    ):
        normalize_agent_config(
            {
                CONF_SPEECH_REGEX_REPLACEMENTS: [
                    {"pattern": "(HA)", "replacement": r"\2 assistant"}
                ]
            }
        )
    with pytest.raises(AgentConfigError, match="invalid replacement expression"):
        normalize_agent_config(
            {CONF_SPEECH_REGEX_REPLACEMENTS: [{"pattern": "HA", "replacement": "\\"}]}
        )


def test_option_metadata_is_authoritative_and_labeled() -> None:
    options = agent_config_options()
    assert {item["value"] for item in options["api_mode"]} == {
        "auto",
        "chat_completions",
        "responses",
    }
    assert all(
        set(item) == {"value", "label"}
        for choices in options.values()
        for item in choices
    )


def _coverage_native_tool(name: str = "demo") -> dict:
    return {
        "spec": {
            "name": name,
            "description": "Demo tool",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "native", "name": "execute_service"},
    }


def _coverage_group(
    group_id: str = "lights",
    *,
    name: str = "Lights",
    description: str = "Lighting tools",
    functions: list[str] | None = None,
) -> dict:
    return {
        "id": group_id,
        "name": name,
        "description": description,
        "loading_mode": agent_config.FUNCTION_GROUP_LOADING_MODES[0],
        "functions": ["demo"] if functions is None else functions,
    }


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda tool: tool.update(enabled="yes"), r"enabled.*boolean"),
        (lambda tool: tool.update(guest_allowed="yes"), r"guest_allowed.*boolean"),
        (lambda tool: tool.update(spec=[]), r"spec.*object"),
        (lambda tool: tool["spec"].update(unexpected=True), r"spec.*unknown fields"),
        (lambda tool: tool["spec"].update(name=""), r"spec.name.*required"),
        (lambda tool: tool["spec"].update(name="bad name"), r"spec.name.*only"),
        (lambda tool: tool["spec"].update(description=123), r"description.*string"),
        (lambda tool: tool["spec"].update(strict="yes"), r"strict.*boolean"),
        (lambda tool: tool["spec"].update(parameters=[]), r"parameters.*object"),
        (lambda tool: tool.update(function=[]), r"function.*object"),
        (
            lambda tool: tool["function"].update(type="missing"),
            r"function.type.*unrecognized",
        ),
        (
            lambda tool: tool["function"].update(name="not_a_native_implementation"),
            r"function.name.*unknown native implementation",
        ),
    ],
)
def test_function_tool_validation_rejects_malformed_metadata(mutator, match) -> None:
    tool = _coverage_native_tool()
    mutator(tool)
    with pytest.raises(AgentConfigError, match=match):
        validate_function_tools([tool])


def test_function_tool_validation_wraps_schema_and_function_errors(monkeypatch) -> None:
    tool = _coverage_native_tool()

    def reject_schema(_schema):
        raise agent_config.HomeAssistantError(
            "Function input schema is invalid: broken"
        )

    monkeypatch.setattr(agent_config, "validate_function_schema", reject_schema)
    with pytest.raises(AgentConfigError, match=r"parameters.*broken"):
        validate_function_tools([tool])

    monkeypatch.setattr(agent_config, "validate_function_schema", lambda _schema: None)

    class BrokenFunction:
        @staticmethod
        def validate_schema(_config):
            raise ValueError("bad implementation config")

    monkeypatch.setattr(agent_config, "get_function", lambda _name: BrokenFunction())
    with pytest.raises(
        AgentConfigError, match=r"configuration is invalid.*bad implementation"
    ):
        validate_function_tools([tool])


def test_function_tools_accept_none_and_reject_non_lists() -> None:
    assert validate_function_tools(None) == []
    with pytest.raises(AgentConfigError, match="top-level value must be a list"):
        validate_function_tools({"spec": {}})
    with pytest.raises(
        AgentConfigError, match=rf"{agent_config.CONF_FUNCTION_TOOLS}\[0\].*object"
    ):
        validate_function_tools(["not-a-tool"])


def test_function_groups_normalize_and_preserve_guest_allowed() -> None:
    group = _coverage_group()
    group.update(name="  Lighting  ", description="  Lighting helpers  ")
    group["guest_allowed"] = True
    group["enabled"] = False

    assert validate_function_groups([group], [_coverage_native_tool()]) == [
        {
            "id": "lights",
            "name": "Lighting",
            "description": "Lighting helpers",
            "loading_mode": agent_config.FUNCTION_GROUP_LOADING_MODES[0],
            "functions": ["demo"],
            "enabled": False,
            "guest_allowed": True,
        }
    ]


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ({}, "must be a list"),
        (["bad"], r"group.*object"),
        ([{**_coverage_group(), "unknown": True}], "unknown fields"),
        ([{**_coverage_group(), "id": "Bad ID"}], r"id.*lowercase"),
        (
            [_coverage_group("same"), _coverage_group("same", name="Other")],
            "duplicate group ID",
        ),
        (
            [
                _coverage_group("one", name="Same"),
                _coverage_group("two", name=" same "),
            ],
            "duplicate group name",
        ),
        ([{**_coverage_group(), "name": " "}], r"name.*required"),
        ([{**_coverage_group(), "name": "x" * 101}], r"name.*100"),
        ([{**_coverage_group(), "description": " "}], r"description.*required"),
        ([{**_coverage_group(), "description": "x" * 501}], r"description.*500"),
        (
            [{**_coverage_group(), "loading_mode": "not-valid"}],
            r"loading_mode.*unsupported",
        ),
        ([{**_coverage_group(), "enabled": "yes"}], r"enabled.*boolean"),
        ([{**_coverage_group(), "guest_allowed": "yes"}], r"guest_allowed.*boolean"),
        ([{**_coverage_group(), "functions": "demo"}], r"functions.*list of names"),
        ([{**_coverage_group(), "functions": ["demo", "demo"]}], "duplicate names"),
        ([{**_coverage_group(), "functions": ["missing"]}], "unknown function"),
    ],
)
def test_function_group_validation_rejects_bad_shapes(groups, match) -> None:
    with pytest.raises(AgentConfigError, match=match):
        validate_function_groups(groups, [_coverage_native_tool()])


def test_function_groups_reject_duplicate_assignment_and_excess_groups() -> None:
    with pytest.raises(AgentConfigError, match="already assigned"):
        validate_function_groups(
            [_coverage_group("one"), _coverage_group("two", name="Two")],
            [_coverage_native_tool()],
        )

    groups = [
        _coverage_group(f"g{index}", name=f"Group {index}", functions=[])
        for index in range(51)
    ]
    with pytest.raises(AgentConfigError, match="at most 50"):
        validate_function_groups(groups, [_coverage_native_tool()])


def test_speech_regex_validation_residual_matrix() -> None:
    cases = [
        ({}, "must be a list"),
        (["bad"], r"\[0\].*object"),
        ([{"pattern": "x", "replacement": "y", "extra": True}], "unknown fields"),
        ([{"pattern": "", "replacement": ""}], r"pattern.*required"),
        ([{"pattern": "x", "replacement": 1}], r"replacement.*string"),
        (
            [
                {
                    "pattern": "x" * (agent_config.MAX_SPEECH_REGEX_PATTERN_LENGTH + 1),
                    "replacement": "",
                }
            ],
            r"pattern.*too long",
        ),
        (
            [
                {
                    "pattern": "x",
                    "replacement": "y"
                    * (agent_config.MAX_SPEECH_REGEX_REPLACEMENT_LENGTH + 1),
                }
            ],
            r"replacement.*too long",
        ),
    ]
    for value, match in cases:
        with pytest.raises(AgentConfigError, match=match):
            validate_speech_regex_replacements(value)

    with pytest.raises(AgentConfigError, match="at most"):
        validate_speech_regex_replacements(
            [{"pattern": "x", "replacement": "y"}]
            * (agent_config.MAX_SPEECH_REGEX_RULES + 1)
        )


def test_legacy_number_coercion_handles_signed_decimal_and_invalid_float_strings() -> (
    None
):
    config = {
        agent_config.CONF_MAX_TOKENS: "+42.0",
        agent_config.CONF_CONTEXT_THRESHOLD: 8.0,
        agent_config.CONF_TOP_P: "not-a-number",
    }
    agent_config._coerce_legacy_numbers(config)
    assert config[agent_config.CONF_MAX_TOKENS] == 42
    assert config[agent_config.CONF_CONTEXT_THRESHOLD] == 8
    assert config[agent_config.CONF_TOP_P] == "not-a-number"


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({agent_config.CONF_MAX_TOKENS: True}, r"max_tokens.*int"),
        ({agent_config.CONF_WEB_SEARCH: 1}, r"web_search.*bool"),
        ({agent_config.CONF_PROMPT: 1}, r"prompt.*str"),
        ({agent_config.CONF_TOP_P: "bad"}, r"top_p.*int or float"),
        ({agent_config.CONF_API_MODE: "invalid"}, r"api_mode.*unsupported"),
        (
            {agent_config.CONF_ARCHIVE_RETENTION_DAYS: 123},
            r"unsupported archive retention",
        ),
        ({agent_config.CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: -1}, r"at least 0"),
        ({agent_config.CONF_CONTEXT_THRESHOLD: 0}, r"context_threshold.*at least 1"),
        ({agent_config.CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES: 0}, r"must be 1 to 1440"),
        (
            {
                agent_config.CONF_MEMORY_AUTO_RETRIEVE_LIMIT: agent_config.MAX_MEMORY_AUTO_RETRIEVE_LIMIT
                + 1
            },
            r"memory_auto_retrieve_limit.*must be 0 to",
        ),
        ({agent_config.CONF_TOP_P: 1.1}, r"top_p.*0 to 1"),
        ({agent_config.CONF_TEMPERATURE: 2.1}, r"temperature.*0 to 2"),
        (
            {agent_config.CONF_VOICE_DEVICE_MAPPINGS: []},
            r"voice_device_mappings.*must map",
        ),
        (
            {agent_config.CONF_VOICE_DEVICE_MAPPINGS: {1: "user:x"}},
            r"voice_device_mappings.*must map",
        ),
        ({agent_config.CONF_SKILLS: "skill"}, r"skills.*list of names"),
        ({agent_config.CONF_SKILLS: [1]}, r"skills.*list of names"),
        (
            {agent_config.CONF_GUEST_ALLOWED_GROUP_IDS: [""]},
            r"list of non-empty strings",
        ),
    ],
)
def test_normalize_agent_config_rejects_residual_invalid_values(config, match) -> None:
    with pytest.raises(AgentConfigError, match=match):
        normalize_agent_config(config)


def test_normalize_agent_config_template_and_list_edges() -> None:
    for key in (
        agent_config.CONF_CURRENT_DATETIME_TEMPLATE,
        agent_config.CONF_EXPOSED_ENTITIES_TEMPLATE,
    ):
        with pytest.raises(AgentConfigError, match="invalid template"):
            normalize_agent_config({key: "{{ broken"})

    result = normalize_agent_config(
        {
            agent_config.CONF_CURRENT_DATETIME_TEMPLATE: "   ",
            agent_config.CONF_EXPOSED_ENTITIES_TEMPLATE: "",
            agent_config.CONF_GUEST_ALLOWED_GROUP_IDS: [" one ", "one", "two"],
        }
    )
    assert result[agent_config.CONF_GUEST_ALLOWED_GROUP_IDS] == ["one", "two"]


def test_reasoning_default_and_explicit_validation(monkeypatch) -> None:
    monkeypatch.setattr(agent_config, "get_reasoning_effort_options", lambda _model: [])
    monkeypatch.setattr(
        agent_config,
        "get_model_config",
        lambda _model: {"recommended_profile": {}},
    )
    result = normalize_agent_config({})
    assert agent_config.CONF_REASONING_EFFORT not in result

    monkeypatch.setattr(
        agent_config, "get_reasoning_effort_options", lambda _model: ["low", "high"]
    )
    with pytest.raises(AgentConfigError, match=r"reasoning_effort.*unsupported"):
        normalize_agent_config({agent_config.CONF_REASONING_EFFORT: "medium"})


def test_skills_fail_when_loader_status_reports_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_config,
        "skill_loader_status",
        lambda *_args, **_kwargs: type(
            "Status", (), {"available": False, "reason": "loader unavailable"}
        )(),
    )
    with pytest.raises(AgentConfigError, match=r"skills.*loader unavailable"):
        normalize_agent_config({agent_config.CONF_SKILLS: ["demo"]})


def test_function_tools_supplied_in_config_are_serialized_canonically() -> None:
    result = normalize_agent_config(
        {agent_config.CONF_FUNCTION_TOOLS: [_coverage_native_tool()]}
    )
    assert isinstance(result[agent_config.CONF_FUNCTION_TOOLS], str)
    assert "name: demo" in result[agent_config.CONF_FUNCTION_TOOLS]


def test_validate_agent_title_rejects_non_string() -> None:
    """Reject persisted titles that are not strings."""
    with pytest.raises(agent_config.AgentConfigError, match="title: must be a string"):
        agent_config.validate_agent_title(123)


def test_tools_yaml_validates_then_serializes() -> None:
    """Serialize the canonical validated tool representation."""
    assert agent_config._tools_yaml([]) == "[]\n"


def test_validate_function_tools_normalizes_reference_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose invalid Home Assistant references as field-scoped config errors."""
    monkeypatch.setattr(agent_config, "is_ha_tool", lambda _tool: True)

    def invalid_reference(_value: Any) -> Any:
        raise ValueError("invalid HA tool reference")

    monkeypatch.setattr(agent_config, "validate_reference", invalid_reference)

    with pytest.raises(
        agent_config.AgentConfigError,
        match=r"functions\[0\]: invalid HA tool reference",
    ):
        agent_config.validate_function_tools(
            [{"spec": {"name": "ha_ref"}, "function": {"platform": "test"}}]
        )


def test_validate_function_groups_reserves_loader_for_on_demand_groups() -> None:
    """Prevent a configured tool from colliding with the on-demand loader tool."""
    tools = [{"spec": {"name": agent_config.FUNCTION_GROUP_LOADER_TOOL_NAME}}]
    groups = [
        {
            "id": "lights",
            "name": "Lights",
            "description": "Lighting tools",
            "loading_mode": agent_config.FUNCTION_GROUP_LOADING_ON_DEMAND,
            "functions": [],
        }
    ]

    with pytest.raises(
        agent_config.AgentConfigError,
        match="reserved when an on-demand function group is configured",
    ):
        agent_config.validate_function_groups(groups, tools)


def test_coerce_legacy_numbers_leaves_non_integral_float_unchanged() -> None:
    """Do not silently truncate non-integral legacy numeric values."""
    config = {agent_config.CONF_MAX_TOKENS: 1.5}
    agent_config._coerce_legacy_numbers(config)
    assert config[agent_config.CONF_MAX_TOKENS] == 1.5


def test_normalize_agent_config_rejects_non_object() -> None:
    """Reject non-object persisted agent configuration payloads."""
    with pytest.raises(
        agent_config.AgentConfigError, match="config: must be an object"
    ):
        agent_config.normalize_agent_config([])  # type: ignore[arg-type]


def test_agent_title_contract_is_shared_and_bounded() -> None:
    assert validate_agent_title("  Jarvis  ") == "Jarvis"
    assert validate_agent_title(None, default="Imported conversation agent") == (
        "Imported conversation agent"
    )
    assert validate_agent_title("x" * MAX_AGENT_TITLE_LENGTH) == (
        "x" * MAX_AGENT_TITLE_LENGTH
    )
    with pytest.raises(AgentConfigError, match="must not be empty"):
        validate_agent_title("   ")
    with pytest.raises(AgentConfigError, match="at most 255"):
        validate_agent_title("x" * (MAX_AGENT_TITLE_LENGTH + 1))
