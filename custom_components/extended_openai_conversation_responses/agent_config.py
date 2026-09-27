"""Authoritative conversation-agent configuration contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import json
import re
from types import MappingProxyType
from typing import Any, cast

import yaml

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import template

from .const import (
    API_MODE_OPTIONS,
    ARCHIVE_RETENTION_OPTIONS,
    CONF_ADVANCED_OPTIONS,
    CONF_API_MODE,
    CONF_ARCHIVE_ENABLED,
    CONF_ARCHIVE_MODEL_SEARCH_ENABLED,
    CONF_ARCHIVE_RETENTION_DAYS,
    CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_CONTINUE_CONVERSATION,
    CONF_CONVERSATION_CONTINUITY,
    CONF_CONVERSATION_TIMEOUT_MINUTES,
    CONF_CURRENT_DATETIME_ENABLED,
    CONF_CURRENT_DATETIME_TEMPLATE,
    CONF_EXPOSED_ENTITIES_ENABLED,
    CONF_EXPOSED_ENTITIES_TEMPLATE,
    CONF_EXPOSED_ENTITY_ATTRIBUTES,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOL_ERROR_RECOVERY,
    CONF_FUNCTION_TOOLS,
    CONF_GUEST_ALLOWED_FUNCTION_NAMES,
    CONF_GUEST_ALLOWED_GROUP_IDS,
    CONF_GUEST_CONTROL_EXCLUDED_AREAS,
    CONF_GUEST_CONTROL_EXCLUDED_DOMAINS,
    CONF_GUEST_CONTROL_EXCLUDED_ENTITIES,
    CONF_GUEST_CONTROL_EXCLUDED_LABELS,
    CONF_GUEST_CONTROLLABLE_AREAS,
    CONF_GUEST_CONTROLLABLE_DOMAINS,
    CONF_GUEST_CONTROLLABLE_ENTITIES,
    CONF_GUEST_CONTROLLABLE_LABELS,
    CONF_GUEST_EXCLUDED_AREAS,
    CONF_GUEST_EXCLUDED_DOMAINS,
    CONF_GUEST_EXCLUDED_ENTITIES,
    CONF_GUEST_EXCLUDED_LABELS,
    CONF_GUEST_FUNCTION_POLICY,
    CONF_GUEST_KNOWLEDGE_ENABLED,
    CONF_GUEST_KNOWLEDGE_POLICY,
    CONF_GUEST_KNOWLEDGE_SOURCE_IDS,
    CONF_GUEST_MODE_ENABLED,
    CONF_GUEST_POLICY_VERSION,
    CONF_GUEST_READABLE_AREAS,
    CONF_GUEST_READABLE_DOMAINS,
    CONF_GUEST_READABLE_ENTITIES,
    CONF_GUEST_READABLE_LABELS,
    CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS,
    CONF_GUEST_SHARED_MEMORY_POLICY,
    CONF_GUEST_SHARED_MEMORY_READ,
    CONF_GUEST_SHARED_MEMORY_WRITE,
    CONF_KNOWLEDGE_ENABLED,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MEMORY_AUTO_CREATE,
    CONF_MEMORY_AUTO_RETRIEVE_LIMIT,
    CONF_MEMORY_EMBEDDING_MODEL,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MODE,
    CONF_MEMORY_RETRIEVAL_MODE,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_SERVICE_TIER,
    CONF_SHARED_ARCHIVE_ENABLED,
    CONF_SHARED_MEMORY_MODE,
    CONF_SHORTEN_TOOL_CALL_ID,
    CONF_SKILLS,
    CONF_SPEECH_PROCESSING_ENABLED,
    CONF_SPEECH_REGEX_REPLACEMENTS,
    CONF_SPEECH_STRIP_MARKDOWN,
    CONF_SPEECH_STRIP_URLS,
    CONF_TEMPERATURE,
    CONF_TEMPORARY_MEMORY,
    CONF_TOP_P,
    CONF_USAGE_REQUEST_RETENTION_DAYS,
    CONF_USAGE_RUN_RETENTION_DAYS,
    CONF_VOICE_DEFAULT_USER_ID,
    CONF_VOICE_DEVICE_MAPPINGS,
    CONF_VOICE_SCOPE_POLICY,
    CONF_VOICE_UNMAPPED_POLICY,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CONTEXT,
    CONTEXT_TRUNCATE_STRATEGIES,
    CONTINUE_CONVERSATION_OPTIONS,
    CONVERSATION_CONTINUITY_OPTIONS,
    CONVERSATION_TIMEOUT_OPTIONS,
    DEFAULT_ADVANCED_OPTIONS,
    DEFAULT_API_MODE,
    DEFAULT_ARCHIVE_ENABLED,
    DEFAULT_ARCHIVE_MODEL_SEARCH_ENABLED,
    DEFAULT_ARCHIVE_RETENTION_DAYS,
    DEFAULT_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTION_TOOLS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_CONVERSATION_CONTINUITY,
    DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    DEFAULT_CURRENT_DATETIME_ENABLED,
    DEFAULT_CURRENT_DATETIME_TEMPLATE,
    DEFAULT_EXPOSED_ENTITIES_ENABLED,
    DEFAULT_EXPOSED_ENTITIES_TEMPLATE,
    DEFAULT_FUNCTION_GROUPS,
    DEFAULT_FUNCTION_TOOL_ERROR_RECOVERY,
    DEFAULT_GUEST_ENTITY_SELECTORS,
    DEFAULT_GUEST_KNOWLEDGE_ENABLED,
    DEFAULT_GUEST_MODE_ENABLED,
    DEFAULT_GUEST_SHARED_MEMORY_READ,
    DEFAULT_GUEST_SHARED_MEMORY_WRITE,
    DEFAULT_KNOWLEDGE_ENABLED,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEMORY_AUTO_RETRIEVE_LIMIT,
    DEFAULT_MEMORY_EMBEDDING_MODEL,
    DEFAULT_MEMORY_MODE,
    DEFAULT_MEMORY_RETRIEVAL_MODE,
    DEFAULT_PROMPT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    DEFAULT_SHARED_ARCHIVE_ENABLED,
    DEFAULT_SHARED_MEMORY_MODE,
    DEFAULT_SHORTEN_TOOL_CALL_ID,
    DEFAULT_SPEECH_PROCESSING_ENABLED,
    DEFAULT_SPEECH_REGEX_REPLACEMENTS,
    DEFAULT_SPEECH_STRIP_MARKDOWN,
    DEFAULT_SPEECH_STRIP_URLS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPORARY_MEMORY,
    DEFAULT_TOP_P,
    DEFAULT_USAGE_REQUEST_RETENTION_DAYS,
    DEFAULT_USAGE_RUN_RETENTION_DAYS,
    DEFAULT_VOICE_SCOPE_POLICY,
    DEFAULT_VOICE_UNMAPPED_POLICY,
    DEFAULT_WEB_SEARCH,
    DEFAULT_WEB_SEARCH_CONTEXT,
    FUNCTION_GROUP_LOADER_TOOL_NAME,
    FUNCTION_GROUP_LOADING_MODES,
    FUNCTION_GROUP_LOADING_ON_DEMAND,
    GUEST_ACCESS_POLICIES,
    GUEST_POLICY_VERSION,
    GUEST_SHARED_MEMORY_POLICIES,
    MAX_MEMORY_AUTO_RETRIEVE_LIMIT,
    MAX_SPEECH_REGEX_PATTERN_LENGTH,
    MAX_SPEECH_REGEX_REPLACEMENT_LENGTH,
    MAX_SPEECH_REGEX_RULES,
    MEMORY_MODES,
    MEMORY_RETRIEVAL_MODES,
    REASONING_EFFORT_OPTIONS,
    SERVICE_TIER_OPTIONS,
    SHARED_MEMORY_MODES,
    TEMPORARY_MEMORY_OPTIONS,
    USAGE_RETENTION_OPTIONS,
    VOICE_POLICIES,
    WEB_SEARCH_CONTEXT_OPTIONS,
)
from .function_execution import validate_function_schema
from .function_tool_policy import (
    FUNCTION_TOOL_NAME_PATTERN,
    NATIVE_FUNCTION_IMPLEMENTATIONS,
    RESERVED_FUNCTION_TOOL_NAMES,
)
from .functions import FUNCTIONS, get_function
from .functions.base import copy_runtime_function_config
from .ha_llm_tools import is_ha_tool, reference_key, validate_reference
from .helpers import get_model_config, get_reasoning_effort_options
from .local_intents import (
    CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI,
    CONF_LOCAL_INTENT_EXCLUSIONS,
    CONF_LOCAL_INTENTS_ENABLED,
    DEFAULT_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI,
    DEFAULT_LOCAL_INTENT_EXCLUSIONS,
    DEFAULT_LOCAL_INTENTS_ENABLED,
)
from .memory import get_memory_mode
from .skill_availability import skill_loader_status

CONF_GUEST_WEB_SEARCH = "guest_web_search"
DEFAULT_GUEST_WEB_SEARCH = False


class AgentConfigError(HomeAssistantError):
    """A validation error tied to one agent configuration field."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")


MAX_AGENT_TITLE_LENGTH = 255


def validate_agent_title(value: Any, *, default: str | None = None) -> str:
    """Return one canonical conversation-agent title for every persistence path."""
    if value is None and default is not None:
        value = default
    if not isinstance(value, str):
        raise AgentConfigError("title", "must be a string")
    title = value.strip()
    if not title:
        raise AgentConfigError("title", "must not be empty")
    if len(title) > MAX_AGENT_TITLE_LENGTH:
        raise AgentConfigError(
            "title", f"must be at most {MAX_AGENT_TITLE_LENGTH} characters"
        )
    return title


def _tools_yaml(value: Any) -> str:
    tools = validate_function_tools(value)
    return yaml.safe_dump(tools, sort_keys=False, allow_unicode=True)


AGENT_CONFIG_DEFAULTS = MappingProxyType(
    {
        CONF_EXPOSED_ENTITY_ATTRIBUTES: {},
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CURRENT_DATETIME_ENABLED: DEFAULT_CURRENT_DATETIME_ENABLED,
        CONF_CURRENT_DATETIME_TEMPLATE: DEFAULT_CURRENT_DATETIME_TEMPLATE,
        CONF_EXPOSED_ENTITIES_ENABLED: DEFAULT_EXPOSED_ENTITIES_ENABLED,
        CONF_EXPOSED_ENTITIES_TEMPLATE: DEFAULT_EXPOSED_ENTITIES_TEMPLATE,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_API_MODE: DEFAULT_API_MODE,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_REASONING_EFFORT: DEFAULT_REASONING_EFFORT,
        CONF_SERVICE_TIER: DEFAULT_SERVICE_TIER,
        CONF_FUNCTION_TOOLS: yaml.safe_dump(
            DEFAULT_CONF_FUNCTION_TOOLS, sort_keys=False, allow_unicode=True
        ),
        CONF_FUNCTION_GROUPS: list(DEFAULT_FUNCTION_GROUPS),
        CONF_FUNCTION_TOOL_ERROR_RECOVERY: DEFAULT_FUNCTION_TOOL_ERROR_RECOVERY,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        CONF_CONTINUE_CONVERSATION: DEFAULT_CONTINUE_CONVERSATION,
        CONF_CONVERSATION_CONTINUITY: DEFAULT_CONVERSATION_CONTINUITY,
        CONF_CONVERSATION_TIMEOUT_MINUTES: DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
        CONF_LOCAL_INTENTS_ENABLED: DEFAULT_LOCAL_INTENTS_ENABLED,
        CONF_LOCAL_INTENT_EXCLUSIONS: list(DEFAULT_LOCAL_INTENT_EXCLUSIONS),
        CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI: DEFAULT_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI,
        CONF_WEB_SEARCH: DEFAULT_WEB_SEARCH,
        CONF_WEB_SEARCH_CONTEXT: DEFAULT_WEB_SEARCH_CONTEXT,
        CONF_MEMORY_MODE: DEFAULT_MEMORY_MODE,
        CONF_TEMPORARY_MEMORY: DEFAULT_TEMPORARY_MEMORY,
        CONF_MEMORY_ENABLED: False,
        CONF_MEMORY_AUTO_CREATE: False,
        CONF_MEMORY_AUTO_RETRIEVE_LIMIT: DEFAULT_MEMORY_AUTO_RETRIEVE_LIMIT,
        CONF_MEMORY_RETRIEVAL_MODE: DEFAULT_MEMORY_RETRIEVAL_MODE,
        CONF_MEMORY_EMBEDDING_MODEL: DEFAULT_MEMORY_EMBEDDING_MODEL,
        CONF_KNOWLEDGE_ENABLED: DEFAULT_KNOWLEDGE_ENABLED,
        CONF_GUEST_MODE_ENABLED: DEFAULT_GUEST_MODE_ENABLED,
        CONF_GUEST_WEB_SEARCH: DEFAULT_GUEST_WEB_SEARCH,
        CONF_GUEST_POLICY_VERSION: GUEST_POLICY_VERSION,
        CONF_GUEST_EXCLUDED_ENTITIES: [],
        CONF_GUEST_EXCLUDED_DOMAINS: [],
        CONF_GUEST_EXCLUDED_AREAS: [],
        CONF_GUEST_EXCLUDED_LABELS: [],
        CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS: False,
        CONF_GUEST_CONTROL_EXCLUDED_ENTITIES: [],
        CONF_GUEST_CONTROL_EXCLUDED_DOMAINS: [],
        CONF_GUEST_CONTROL_EXCLUDED_AREAS: [],
        CONF_GUEST_CONTROL_EXCLUDED_LABELS: [],
        CONF_GUEST_KNOWLEDGE_POLICY: "off",
        CONF_GUEST_KNOWLEDGE_SOURCE_IDS: [],
        CONF_GUEST_FUNCTION_POLICY: "off",
        CONF_GUEST_ALLOWED_FUNCTION_NAMES: [],
        CONF_GUEST_ALLOWED_GROUP_IDS: [],
        CONF_GUEST_SHARED_MEMORY_POLICY: "off",
        CONF_GUEST_READABLE_ENTITIES: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_CONTROLLABLE_ENTITIES: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_READABLE_DOMAINS: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_CONTROLLABLE_DOMAINS: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_READABLE_AREAS: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_CONTROLLABLE_AREAS: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_READABLE_LABELS: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_CONTROLLABLE_LABELS: list(DEFAULT_GUEST_ENTITY_SELECTORS),
        CONF_GUEST_SHARED_MEMORY_READ: DEFAULT_GUEST_SHARED_MEMORY_READ,
        CONF_GUEST_SHARED_MEMORY_WRITE: DEFAULT_GUEST_SHARED_MEMORY_WRITE,
        CONF_GUEST_KNOWLEDGE_ENABLED: DEFAULT_GUEST_KNOWLEDGE_ENABLED,
        CONF_ARCHIVE_ENABLED: DEFAULT_ARCHIVE_ENABLED,
        CONF_ARCHIVE_RETENTION_DAYS: DEFAULT_ARCHIVE_RETENTION_DAYS,
        CONF_ARCHIVE_MODEL_SEARCH_ENABLED: DEFAULT_ARCHIVE_MODEL_SEARCH_ENABLED,
        CONF_SHARED_ARCHIVE_ENABLED: DEFAULT_SHARED_ARCHIVE_ENABLED,
        CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES: DEFAULT_ARCHIVE_SESSION_TIMEOUT_MINUTES,
        CONF_VOICE_SCOPE_POLICY: DEFAULT_VOICE_SCOPE_POLICY,
        CONF_VOICE_UNMAPPED_POLICY: DEFAULT_VOICE_UNMAPPED_POLICY,
        CONF_VOICE_DEFAULT_USER_ID: "",
        CONF_VOICE_DEVICE_MAPPINGS: {},
        CONF_SHARED_MEMORY_MODE: DEFAULT_SHARED_MEMORY_MODE,
        CONF_USAGE_REQUEST_RETENTION_DAYS: DEFAULT_USAGE_REQUEST_RETENTION_DAYS,
        CONF_USAGE_RUN_RETENTION_DAYS: DEFAULT_USAGE_RUN_RETENTION_DAYS,
        CONF_SHORTEN_TOOL_CALL_ID: DEFAULT_SHORTEN_TOOL_CALL_ID,
        CONF_ADVANCED_OPTIONS: DEFAULT_ADVANCED_OPTIONS,
        CONF_SKILLS: [],
        CONF_SPEECH_PROCESSING_ENABLED: DEFAULT_SPEECH_PROCESSING_ENABLED,
        CONF_SPEECH_STRIP_MARKDOWN: DEFAULT_SPEECH_STRIP_MARKDOWN,
        CONF_SPEECH_STRIP_URLS: DEFAULT_SPEECH_STRIP_URLS,
        CONF_SPEECH_REGEX_REPLACEMENTS: DEFAULT_SPEECH_REGEX_REPLACEMENTS,
    }
)

AGENT_CONFIG_FIELDS = frozenset(
    {*AGENT_CONFIG_DEFAULTS, CONF_REASONING_EFFORT, CONF_SERVICE_TIER}
)
GUEST_V2_FIELDS = frozenset(
    {
        CONF_GUEST_POLICY_VERSION,
        CONF_GUEST_WEB_SEARCH,
        CONF_GUEST_EXCLUDED_ENTITIES,
        CONF_GUEST_EXCLUDED_DOMAINS,
        CONF_GUEST_EXCLUDED_AREAS,
        CONF_GUEST_EXCLUDED_LABELS,
        CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS,
        CONF_GUEST_CONTROL_EXCLUDED_ENTITIES,
        CONF_GUEST_CONTROL_EXCLUDED_DOMAINS,
        CONF_GUEST_CONTROL_EXCLUDED_AREAS,
        CONF_GUEST_CONTROL_EXCLUDED_LABELS,
        CONF_GUEST_KNOWLEDGE_POLICY,
        CONF_GUEST_KNOWLEDGE_SOURCE_IDS,
        CONF_GUEST_FUNCTION_POLICY,
        CONF_GUEST_ALLOWED_FUNCTION_NAMES,
        CONF_GUEST_ALLOWED_GROUP_IDS,
        CONF_GUEST_SHARED_MEMORY_POLICY,
    }
)


def preserve_legacy_guest_policy(
    source: dict[str, Any], normalized: dict[str, Any]
) -> dict[str, Any]:
    """Keep an absent v2 marker absent across generic configuration workflows."""
    if CONF_GUEST_POLICY_VERSION not in source:
        for key in GUEST_V2_FIELDS:
            normalized.pop(key, None)
    return normalized


def agent_config_defaults() -> dict[str, Any]:
    """Return an isolated copy of authoritative defaults."""
    return deepcopy(dict(AGENT_CONFIG_DEFAULTS))


def _choice(value: Any, label: str | None = None) -> dict[str, Any]:
    """Return one frontend-safe configuration choice."""
    return {
        "value": value,
        "label": label or str(value).replace("_", " ").replace("-", " ").title(),
    }


def agent_config_options() -> dict[str, list[dict[str, Any]]]:
    """Return authoritative option metadata for the management frontend."""
    return {
        CONF_API_MODE: [
            _choice(item["key"], str(item["label"])) for item in API_MODE_OPTIONS
        ],
        CONF_CONTINUE_CONVERSATION: [
            _choice(value) for value in CONTINUE_CONVERSATION_OPTIONS
        ],
        CONF_CONVERSATION_CONTINUITY: [
            _choice(value) for value in CONVERSATION_CONTINUITY_OPTIONS
        ],
        CONF_CONVERSATION_TIMEOUT_MINUTES: [
            _choice(
                value,
                (f"{value // 60} hour" if value == 60 else f"{value // 60} hours")
                if value >= 60
                else f"{value} minutes",
            )
            for value in CONVERSATION_TIMEOUT_OPTIONS
        ],
        CONF_WEB_SEARCH_CONTEXT: [
            _choice(value) for value in WEB_SEARCH_CONTEXT_OPTIONS
        ],
        CONF_MEMORY_MODE: [_choice(value) for value in MEMORY_MODES],
        CONF_MEMORY_RETRIEVAL_MODE: [
            _choice(value) for value in MEMORY_RETRIEVAL_MODES
        ],
        CONF_TEMPORARY_MEMORY: [_choice(value) for value in TEMPORARY_MEMORY_OPTIONS],
        CONF_ARCHIVE_RETENTION_DAYS: [
            _choice(value, f"{value} days") for value in ARCHIVE_RETENTION_OPTIONS
        ],
        CONF_VOICE_SCOPE_POLICY: [_choice(value) for value in VOICE_POLICIES],
        CONF_VOICE_UNMAPPED_POLICY: [_choice(value) for value in VOICE_POLICIES],
        CONF_SHARED_MEMORY_MODE: [_choice(value) for value in SHARED_MEMORY_MODES],
        CONF_USAGE_REQUEST_RETENTION_DAYS: [
            _choice(value, f"{value} days" if value else "Disabled")
            for value in USAGE_RETENTION_OPTIONS
        ],
        CONF_USAGE_RUN_RETENTION_DAYS: [
            _choice(value, f"{value} days" if value else "Disabled")
            for value in USAGE_RETENTION_OPTIONS
        ],
        CONF_CONTEXT_TRUNCATE_STRATEGY: [
            _choice(item["key"], str(item["label"]))
            for item in CONTEXT_TRUNCATE_STRATEGIES
        ],
        CONF_REASONING_EFFORT: [_choice(value) for value in REASONING_EFFORT_OPTIONS],
        CONF_SERVICE_TIER: [_choice(value) for value in SERVICE_TIER_OPTIONS],
        "function_group_loading_modes": [
            _choice(value) for value in FUNCTION_GROUP_LOADING_MODES
        ],
    }


def validate_function_tools(value: Any) -> list[dict[str, Any]]:
    """Parse and validate function tools without executing them."""
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError as err:
            raise AgentConfigError(CONF_FUNCTION_TOOLS, f"invalid YAML: {err}") from err
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentConfigError(CONF_FUNCTION_TOOLS, "top-level value must be a list")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    ha_references: set[str] = set()
    for index, tool in enumerate(value):
        field = f"{CONF_FUNCTION_TOOLS}[{index}]"
        if not isinstance(tool, dict):
            raise AgentConfigError(field, "tool must be an object")
        if "enabled" in tool and not isinstance(tool["enabled"], bool):
            raise AgentConfigError(f"{field}.enabled", "must be a boolean")
        if "guest_allowed" in tool and not isinstance(tool["guest_allowed"], bool):
            raise AgentConfigError(f"{field}.guest_allowed", "must be a boolean")
        spec = tool.get("spec")
        function_config = tool.get("function")
        if not isinstance(spec, dict):
            raise AgentConfigError(f"{field}.spec", "must be an object")
        unknown_spec_fields = set(spec) - {
            "name",
            "description",
            "parameters",
            "strict",
        }
        if unknown_spec_fields:
            raise AgentConfigError(
                f"{field}.spec",
                "unknown fields: " + ", ".join(sorted(unknown_spec_fields)),
            )
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise AgentConfigError(f"{field}.spec.name", "is required")
        if not FUNCTION_TOOL_NAME_PATTERN.fullmatch(name):
            raise AgentConfigError(
                f"{field}.spec.name",
                "must contain only letters, numbers, underscores, or hyphens "
                "and be at most 64 characters",
            )
        if name in RESERVED_FUNCTION_TOOL_NAMES:
            raise AgentConfigError(
                f"{field}.spec.name", f"reserved integration tool name: {name}"
            )
        if name in names:
            raise AgentConfigError(f"{field}.spec.name", f"duplicate tool name: {name}")
        names.add(name)
        if is_ha_tool(tool):
            if set(tool) - {"spec", "function", "enabled"} or set(spec) != {"name"}:
                raise AgentConfigError(
                    field,
                    "HA tools store only a reference, local name and enabled state",
                )
            try:
                reference = validate_reference(function_config)
            except ValueError as err:
                raise AgentConfigError(field, str(err)) from err
            key = reference_key(reference)
            if key in ha_references:
                raise AgentConfigError(field, "HA tool reference already configured")
            ha_references.add(key)
            result.append(
                {
                    "spec": {"name": name},
                    "function": reference,
                    "enabled": tool.get("enabled", True),
                }
            )
            continue
        description = spec.get("description")
        if description is not None and not isinstance(description, str):
            raise AgentConfigError(f"{field}.spec.description", "must be a string")
        strict = spec.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise AgentConfigError(f"{field}.spec.strict", "must be a boolean")
        parameters = spec.get("parameters", {})
        if not isinstance(parameters, dict):
            raise AgentConfigError(f"{field}.spec.parameters", "must be an object")
        try:
            validate_function_schema(parameters)
        except HomeAssistantError as err:
            message = str(err).removeprefix("Function input schema is invalid: ")
            raise AgentConfigError(f"{field}.spec.parameters", message) from err
        if not isinstance(function_config, dict):
            raise AgentConfigError(f"{field}.function", "must be an object")
        function_type = function_config.get("type")
        if not isinstance(function_type, str) or function_type not in FUNCTIONS:
            raise AgentConfigError(
                f"{field}.function.type", f"unrecognized function type: {function_type}"
            )
        try:
            get_function(function_type).validate_schema(deepcopy(function_config))
        except Exception as err:
            raise AgentConfigError(
                f"{field}.function",
                f"configuration is invalid for {function_type}: {err}",
            ) from err
        if (
            function_type == "native"
            and function_config.get("name") not in NATIVE_FUNCTION_IMPLEMENTATIONS
        ):
            raise AgentConfigError(
                f"{field}.function.name",
                f"unknown native implementation: {function_config.get('name')}",
            )
        normalized = deepcopy(tool)
        normalized["spec"] = deepcopy(spec)
        normalized["function"] = deepcopy(function_config)
        result.append(normalized)
    return result


def function_tool_enabled(tool: dict[str, Any]) -> bool:
    """Return the single authoritative enabled state for a configured tool."""
    return tool.get("enabled", True) is True


def _configured_function_tools_from_data(data: Any) -> list[dict[str, Any]]:
    """Parse fully validated production Function Tools from agent data."""
    configured = data.get(CONF_FUNCTION_TOOLS)
    parsed = yaml.safe_load(configured) if configured else DEFAULT_CONF_FUNCTION_TOOLS
    tools = validate_function_tools(parsed)
    for tool in tools:
        if is_ha_tool(tool):
            continue
        function_config = tool["function"]
        tool["function"] = get_function(function_config["type"]).validate_schema(
            function_config
        )
    return tools


_FUNCTION_GROUP_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _validate_function_groups(
    value: Any, function_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate compact group metadata and references to configured functions."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentConfigError(CONF_FUNCTION_GROUPS, "must be a list")
    if len(value) > 50:
        raise AgentConfigError(CONF_FUNCTION_GROUPS, "supports at most 50 groups")

    tool_names = {tool["spec"]["name"] for tool in function_tools}
    group_ids: set[str] = set()
    group_names: set[str] = set()
    assigned_functions: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    allowed = {
        "id",
        "name",
        "description",
        "loading_mode",
        "functions",
        "guest_allowed",
        "enabled",
    }
    for index, group in enumerate(value):
        field = f"{CONF_FUNCTION_GROUPS}[{index}]"
        if not isinstance(group, dict):
            raise AgentConfigError(field, "group must be an object")
        unknown_fields = set(group) - allowed
        if unknown_fields:
            raise AgentConfigError(
                field, "unknown fields: " + ", ".join(sorted(unknown_fields))
            )
        group_id = group.get("id")
        if not isinstance(group_id, str) or not _FUNCTION_GROUP_ID.fullmatch(group_id):
            raise AgentConfigError(
                f"{field}.id",
                "must start with a lowercase letter and contain only lowercase "
                "letters, numbers, underscores, or hyphens (maximum 64 characters)",
            )
        if group_id in group_ids:
            raise AgentConfigError(f"{field}.id", f"duplicate group ID: {group_id}")
        group_ids.add(group_id)

        name = group.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AgentConfigError(f"{field}.name", "is required")
        name = name.strip()
        if len(name) > 100:
            raise AgentConfigError(f"{field}.name", "must be at most 100 characters")
        normalized_name = name.casefold()
        if normalized_name in group_names:
            raise AgentConfigError(f"{field}.name", f"duplicate group name: {name}")
        group_names.add(normalized_name)

        description = group.get("description")
        if not isinstance(description, str) or not description.strip():
            raise AgentConfigError(f"{field}.description", "is required")
        description = description.strip()
        if len(description) > 500:
            raise AgentConfigError(
                f"{field}.description", "must be at most 500 characters"
            )

        loading_mode = group.get("loading_mode")
        if loading_mode not in FUNCTION_GROUP_LOADING_MODES:
            raise AgentConfigError(f"{field}.loading_mode", "unsupported value")
        enabled = group.get("enabled", True)
        if not isinstance(enabled, bool):
            raise AgentConfigError(f"{field}.enabled", "must be a boolean")
        guest_allowed = group.get("guest_allowed", False)
        if not isinstance(guest_allowed, bool):
            raise AgentConfigError(f"{field}.guest_allowed", "must be a boolean")
        functions = group.get("functions", [])
        if not isinstance(functions, list) or not all(
            isinstance(name, str) for name in functions
        ):
            raise AgentConfigError(f"{field}.functions", "must be a list of names")
        if len(functions) != len(set(functions)):
            raise AgentConfigError(f"{field}.functions", "contains duplicate names")
        for function_name in functions:
            if function_name not in tool_names:
                raise AgentConfigError(
                    f"{field}.functions", f"unknown function: {function_name}"
                )
            if function_name in assigned_functions:
                raise AgentConfigError(
                    f"{field}.functions",
                    f"function {function_name} is already assigned to group "
                    f"{assigned_functions[function_name]}",
                )
            assigned_functions[function_name] = group_id
        normalized_group = {
            "id": group_id,
            "name": name,
            "description": description,
            "loading_mode": loading_mode,
            "functions": list(functions),
            "enabled": enabled,
        }
        if guest_allowed:
            normalized_group["guest_allowed"] = True
        result.append(normalized_group)

    if (
        any(
            group["loading_mode"] == FUNCTION_GROUP_LOADING_ON_DEMAND
            for group in result
        )
        and FUNCTION_GROUP_LOADER_TOOL_NAME in tool_names
    ):
        raise AgentConfigError(
            CONF_FUNCTION_TOOLS,
            f"tool name `{FUNCTION_GROUP_LOADER_TOOL_NAME}` is reserved when an "
            "on-demand function group is configured",
        )
    return result


def validate_single_function_tool(value: Any) -> dict[str, Any]:
    """Parse and validate one function tool from a clean editor document."""
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError as err:
            raise AgentConfigError(CONF_FUNCTION_TOOLS, f"invalid YAML: {err}") from err
    if isinstance(value, list):
        raise AgentConfigError(
            CONF_FUNCTION_TOOLS,
            "single-tool YAML must contain an object, not a list",
        )
    return validate_function_tools([value])[0]


def function_tool_yaml(value: Any) -> str:
    """Serialize one validated function tool as readable YAML."""
    tool = validate_single_function_tool(value)
    return yaml.safe_dump(tool, sort_keys=False, allow_unicode=True)


def starter_function_tool_yaml() -> str:
    """Return the generic YAML-first editor template for a new function tool."""
    return yaml.safe_dump(
        {
            "spec": {
                "name": "my_tool",
                "description": "Describe what this tool does.",
                "parameters": {"type": "object", "properties": {}},
            },
            "function": {"type": "native", "name": ""},
        },
        sort_keys=False,
        allow_unicode=True,
    )


def validate_speech_regex_replacements(value: Any) -> list[dict[str, str]]:
    """Validate ordered spoken-text regex replacements."""
    if not isinstance(value, list):
        raise AgentConfigError(CONF_SPEECH_REGEX_REPLACEMENTS, "must be a list")
    if len(value) > MAX_SPEECH_REGEX_RULES:
        raise AgentConfigError(
            CONF_SPEECH_REGEX_REPLACEMENTS,
            f"supports at most {MAX_SPEECH_REGEX_RULES} rules",
        )
    result = []
    for index, rule in enumerate(value):
        field = f"{CONF_SPEECH_REGEX_REPLACEMENTS}[{index}]"
        if not isinstance(rule, dict):
            raise AgentConfigError(field, "must be an object")
        unknown = set(rule) - {"pattern", "replacement"}
        if unknown:
            raise AgentConfigError(
                field, "unknown fields: " + ", ".join(sorted(unknown))
            )
        pattern = rule.get("pattern")
        replacement = rule.get("replacement")
        if not isinstance(pattern, str) or not pattern:
            raise AgentConfigError(f"{field}.pattern", "is required")
        if not isinstance(replacement, str):
            raise AgentConfigError(f"{field}.replacement", "must be a string")
        if len(pattern) > MAX_SPEECH_REGEX_PATTERN_LENGTH:
            raise AgentConfigError(f"{field}.pattern", "is too long")
        if len(replacement) > MAX_SPEECH_REGEX_REPLACEMENT_LENGTH:
            raise AgentConfigError(f"{field}.replacement", "is too long")
        try:
            compiled = re.compile(pattern)
        except re.error as err:
            raise AgentConfigError(
                f"{field}.pattern", f"invalid regular expression: {err}"
            ) from err
        try:
            compiled.sub(replacement, "")
        except re.error as err:
            raise AgentConfigError(
                f"{field}.replacement", f"invalid replacement expression: {err}"
            ) from err
        result.append({"pattern": pattern, "replacement": replacement})
    return result


def _require_type(
    config: dict[str, Any], keys: tuple[str, ...], expected: type | tuple[type, ...]
) -> None:
    label = (
        expected.__name__
        if isinstance(expected, type)
        else " or ".join(item.__name__ for item in expected)
    )
    for key in keys:
        if key in config and (
            not isinstance(config[key], expected)
            or (expected is int and isinstance(config[key], bool))
        ):
            raise AgentConfigError(key, f"must be a {label}")


def _coerce_legacy_numbers(config: dict[str, Any]) -> None:
    """Normalize numeric values stored as strings by older selector flows."""
    integer_keys = (
        CONF_CONVERSATION_TIMEOUT_MINUTES,
        CONF_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_CONTEXT_THRESHOLD,
        CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
        CONF_MEMORY_AUTO_RETRIEVE_LIMIT,
        CONF_ARCHIVE_RETENTION_DAYS,
        CONF_USAGE_REQUEST_RETENTION_DAYS,
        CONF_USAGE_RUN_RETENTION_DAYS,
    )
    for key in integer_keys:
        value = config.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"[+-]?\d+(?:\.0+)?", stripped):
                config[key] = int(stripped.split(".", maxsplit=1)[0])
        elif isinstance(value, float) and value.is_integer():
            config[key] = int(value)

    for key in (CONF_TOP_P, CONF_TEMPERATURE):
        value = config.get(key)
        if isinstance(value, str):
            try:
                config[key] = float(value.strip())
            except ValueError:
                continue


def normalize_agent_config(
    data: dict[str, Any],
    *,
    apply_defaults: bool = True,
    reject_unknown: bool = True,
    validated_functions: tuple[list[dict[str, Any]], list[dict[str, Any]] | None]
    | None = None,
    function_tools_as_list: bool = False,
) -> dict[str, Any]:
    """Validate config; reuse Function results only for identical Function inputs.

    ``validated_functions`` is an internal persisted-projection result for the
    exact Function inputs in ``data``. Quarantined inputs use the validated
    effective subset. When groups is None, groups are validated here.
    """
    if not isinstance(data, dict):
        raise AgentConfigError("config", "must be an object")
    unknown = set(data) - AGENT_CONFIG_FIELDS
    if reject_unknown and unknown:
        raise AgentConfigError(
            "config", "unknown fields: " + ", ".join(sorted(unknown))
        )
    reasoning_effort_explicit = CONF_REASONING_EFFORT in data
    result = agent_config_defaults() if apply_defaults else {}
    result.update(deepcopy(data))
    _coerce_legacy_numbers(result)

    selected_model = str(result.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL))
    reasoning_options = get_reasoning_effort_options(selected_model)
    if not reasoning_effort_explicit:
        recommended_effort = (
            get_model_config(selected_model)
            .get("recommended_profile", {})
            .get("reasoning_effort")
        )
        if recommended_effort is None:
            result.pop(CONF_REASONING_EFFORT, None)
        else:
            result[CONF_REASONING_EFFORT] = recommended_effort

    _require_type(
        result,
        (
            CONF_ARCHIVE_ENABLED,
            CONF_ARCHIVE_MODEL_SEARCH_ENABLED,
            CONF_SHARED_ARCHIVE_ENABLED,
            CONF_WEB_SEARCH,
            CONF_KNOWLEDGE_ENABLED,
            CONF_SHORTEN_TOOL_CALL_ID,
            CONF_ADVANCED_OPTIONS,
            CONF_CURRENT_DATETIME_ENABLED,
            CONF_EXPOSED_ENTITIES_ENABLED,
            CONF_FUNCTION_TOOL_ERROR_RECOVERY,
            CONF_GUEST_MODE_ENABLED,
            CONF_GUEST_WEB_SEARCH,
            CONF_GUEST_SHARED_MEMORY_READ,
            CONF_GUEST_SHARED_MEMORY_WRITE,
            CONF_GUEST_KNOWLEDGE_ENABLED,
            CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS,
            CONF_SPEECH_PROCESSING_ENABLED,
            CONF_SPEECH_STRIP_MARKDOWN,
            CONF_SPEECH_STRIP_URLS,
            CONF_LOCAL_INTENTS_ENABLED,
            CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI,
        ),
        bool,
    )
    _require_type(
        result,
        (
            CONF_PROMPT,
            CONF_CURRENT_DATETIME_TEMPLATE,
            CONF_EXPOSED_ENTITIES_TEMPLATE,
            CONF_CHAT_MODEL,
            CONF_API_MODE,
            CONF_CONTEXT_TRUNCATE_STRATEGY,
            CONF_CONTINUE_CONVERSATION,
            CONF_CONVERSATION_CONTINUITY,
            CONF_MEMORY_MODE,
            CONF_MEMORY_RETRIEVAL_MODE,
            CONF_MEMORY_EMBEDDING_MODEL,
            CONF_TEMPORARY_MEMORY,
            CONF_VOICE_SCOPE_POLICY,
            CONF_VOICE_UNMAPPED_POLICY,
            CONF_SHARED_MEMORY_MODE,
            CONF_GUEST_KNOWLEDGE_POLICY,
            CONF_GUEST_FUNCTION_POLICY,
            CONF_GUEST_SHARED_MEMORY_POLICY,
        ),
        str,
    )
    _require_type(
        result,
        (
            CONF_MAX_TOKENS,
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            CONF_CONTEXT_THRESHOLD,
            CONF_CONVERSATION_TIMEOUT_MINUTES,
            CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
            CONF_MEMORY_AUTO_RETRIEVE_LIMIT,
            CONF_GUEST_POLICY_VERSION,
        ),
        int,
    )
    _require_type(result, (CONF_TOP_P, CONF_TEMPERATURE), (int, float))

    choices: dict[str, list[Any]] = {
        CONF_API_MODE: [item["key"] for item in API_MODE_OPTIONS],
        CONF_CONTINUE_CONVERSATION: CONTINUE_CONVERSATION_OPTIONS,
        CONF_CONVERSATION_CONTINUITY: CONVERSATION_CONTINUITY_OPTIONS,
        CONF_WEB_SEARCH_CONTEXT: WEB_SEARCH_CONTEXT_OPTIONS,
        CONF_MEMORY_MODE: MEMORY_MODES,
        CONF_MEMORY_RETRIEVAL_MODE: MEMORY_RETRIEVAL_MODES,
        CONF_TEMPORARY_MEMORY: TEMPORARY_MEMORY_OPTIONS,
        CONF_ARCHIVE_RETENTION_DAYS: ARCHIVE_RETENTION_OPTIONS,
        CONF_VOICE_SCOPE_POLICY: VOICE_POLICIES,
        CONF_VOICE_UNMAPPED_POLICY: VOICE_POLICIES,
        CONF_SHARED_MEMORY_MODE: SHARED_MEMORY_MODES,
        CONF_USAGE_REQUEST_RETENTION_DAYS: USAGE_RETENTION_OPTIONS,
        CONF_USAGE_RUN_RETENTION_DAYS: USAGE_RETENTION_OPTIONS,
        CONF_CONTEXT_TRUNCATE_STRATEGY: [
            item["key"] for item in CONTEXT_TRUNCATE_STRATEGIES
        ],
        CONF_REASONING_EFFORT: reasoning_options,
        CONF_SERVICE_TIER: SERVICE_TIER_OPTIONS,
        CONF_GUEST_KNOWLEDGE_POLICY: GUEST_ACCESS_POLICIES,
        CONF_GUEST_FUNCTION_POLICY: GUEST_ACCESS_POLICIES,
        CONF_GUEST_SHARED_MEMORY_POLICY: GUEST_SHARED_MEMORY_POLICIES,
        CONF_GUEST_POLICY_VERSION: [GUEST_POLICY_VERSION],
    }
    timeout = result.get(CONF_CONVERSATION_TIMEOUT_MINUTES)
    if CONF_CONVERSATION_TIMEOUT_MINUTES in result and (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or not 1 <= timeout <= 1440
    ):
        raise AgentConfigError(CONF_CONVERSATION_TIMEOUT_MINUTES, "unsupported value")
    for key, options in choices.items():
        if key in result and result[key] not in options:
            message = (
                "unsupported archive retention"
                if key == CONF_ARCHIVE_RETENTION_DAYS
                else "unsupported value"
            )
            raise AgentConfigError(key, message)
    if result[CONF_MAX_TOKENS] < 1:
        raise AgentConfigError(CONF_MAX_TOKENS, "must be at least 1")
    if result[CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION] < 0:
        raise AgentConfigError(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, "must be at least 0"
        )
    if result[CONF_CONTEXT_THRESHOLD] < 1:
        raise AgentConfigError(CONF_CONTEXT_THRESHOLD, "must be at least 1")
    if not 1 <= result[CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES] <= 1440:
        raise AgentConfigError(
            CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES, "must be 1 to 1440"
        )
    if (
        not 0
        <= result[CONF_MEMORY_AUTO_RETRIEVE_LIMIT]
        <= MAX_MEMORY_AUTO_RETRIEVE_LIMIT
    ):
        raise AgentConfigError(
            CONF_MEMORY_AUTO_RETRIEVE_LIMIT,
            f"must be 0 to {MAX_MEMORY_AUTO_RETRIEVE_LIMIT}",
        )
    if not 0 <= float(result[CONF_TOP_P]) <= 1:
        raise AgentConfigError(CONF_TOP_P, "must be 0 to 1")
    if not 0 <= float(result[CONF_TEMPERATURE]) <= 2:
        raise AgentConfigError(CONF_TEMPERATURE, "must be 0 to 2")
    for key in (CONF_CURRENT_DATETIME_TEMPLATE, CONF_EXPOSED_ENTITIES_TEMPLATE):
        value = result[key]
        if not value.strip():
            continue
        try:
            # Normalization is used outside a running HA instance (including
            # backup import). Template.ensure_valid() dereferences hass.data
            # for a dynamic template, so compile with HA's filter environment
            # directly without binding an unavailable HomeAssistant object.
            template.TemplateEnvironment(None).compile(value)
        except Exception as err:
            concise = " ".join(str(err).split())[:300] or type(err).__name__
            raise AgentConfigError(key, f"invalid template: {concise}") from err
    mappings = result.get(CONF_VOICE_DEVICE_MAPPINGS, {})
    if not isinstance(mappings, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mappings.items()
    ):
        raise AgentConfigError(
            CONF_VOICE_DEVICE_MAPPINGS, "must map device IDs to scope owners"
        )
    skills = result.get(CONF_SKILLS, [])
    if not isinstance(skills, list) or not all(
        isinstance(skill, str) for skill in skills
    ):
        raise AgentConfigError(CONF_SKILLS, "must be a list of names")
    for key in (
        CONF_LOCAL_INTENT_EXCLUSIONS,
        CONF_GUEST_READABLE_ENTITIES,
        CONF_GUEST_CONTROLLABLE_ENTITIES,
        CONF_GUEST_READABLE_DOMAINS,
        CONF_GUEST_CONTROLLABLE_DOMAINS,
        CONF_GUEST_READABLE_AREAS,
        CONF_GUEST_CONTROLLABLE_AREAS,
        CONF_GUEST_READABLE_LABELS,
        CONF_GUEST_CONTROLLABLE_LABELS,
        CONF_GUEST_EXCLUDED_ENTITIES,
        CONF_GUEST_EXCLUDED_DOMAINS,
        CONF_GUEST_EXCLUDED_AREAS,
        CONF_GUEST_EXCLUDED_LABELS,
        CONF_GUEST_CONTROL_EXCLUDED_ENTITIES,
        CONF_GUEST_CONTROL_EXCLUDED_DOMAINS,
        CONF_GUEST_CONTROL_EXCLUDED_AREAS,
        CONF_GUEST_CONTROL_EXCLUDED_LABELS,
        CONF_GUEST_KNOWLEDGE_SOURCE_IDS,
        CONF_GUEST_ALLOWED_FUNCTION_NAMES,
        CONF_GUEST_ALLOWED_GROUP_IDS,
    ):
        values = result.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise AgentConfigError(key, "must be a list of non-empty strings")
        result[key] = list(dict.fromkeys(value.strip() for value in values))

    if validated_functions is None:
        function_tools = validate_function_tools(result.get(CONF_FUNCTION_TOOLS, []))
        function_groups = validate_function_groups(
            result.get(CONF_FUNCTION_GROUPS, []), function_tools
        )
    else:
        # The caller may only supply a result validated for these exact persisted
        # fields. Copy it so normalization never mutates the projection cache.
        function_tools, cached_groups = deepcopy(validated_functions)
        function_groups = (
            validate_function_groups(
                result.get(CONF_FUNCTION_GROUPS, []), function_tools
            )
            if cached_groups is None
            else cached_groups
        )
    result[CONF_FUNCTION_GROUPS] = function_groups
    loader_status = skill_loader_status(
        skills,
        function_tools,
        function_groups,
        max_function_calls=result[CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION],
    )
    if not loader_status.available:
        raise AgentConfigError(
            CONF_SKILLS,
            loader_status.reason or "selected Skills are not loadable",
        )
    if function_tools_as_list:
        result[CONF_FUNCTION_TOOLS] = function_tools
    elif CONF_FUNCTION_TOOLS in data and (
        validated_functions is None or not isinstance(data[CONF_FUNCTION_TOOLS], str)
    ):
        result[CONF_FUNCTION_TOOLS] = yaml.safe_dump(
            function_tools, sort_keys=False, allow_unicode=True
        )
    result[CONF_SPEECH_REGEX_REPLACEMENTS] = validate_speech_regex_replacements(
        result.get(CONF_SPEECH_REGEX_REPLACEMENTS, [])
    )
    mode = get_memory_mode(result)
    result[CONF_MEMORY_MODE] = mode
    result[CONF_MEMORY_ENABLED] = mode != "off"
    result[CONF_MEMORY_AUTO_CREATE] = mode == "automatic"
    if apply_defaults or CONF_EXPOSED_ENTITY_ATTRIBUTES in data:
        from .exposed_attributes import _validate_preferences

        result[CONF_EXPOSED_ENTITY_ATTRIBUTES] = _validate_preferences(
            data.get(CONF_EXPOSED_ENTITY_ATTRIBUTES)
        )
    return result


def merge_agent_config(
    current: dict[str, Any],
    updates: dict[str, Any],
    *,
    validated_functions: tuple[list[dict[str, Any]], list[dict[str, Any]] | None]
    | None = None,
) -> dict[str, Any]:
    """Validate updates against the final merged configuration.

    Callers may pass ``validated_functions`` only when its corresponding Function
    inputs in ``updates`` are unchanged from the authoritative source.
    """
    unknown = set(updates) - AGENT_CONFIG_FIELDS
    if unknown:
        raise AgentConfigError(
            "config", "unknown fields: " + ", ".join(sorted(unknown))
        )
    known = {key: value for key, value in current.items() if key in AGENT_CONFIG_FIELDS}
    if (
        CONF_CHAT_MODEL in updates
        and updates[CONF_CHAT_MODEL] != known.get(CONF_CHAT_MODEL)
        and CONF_REASONING_EFFORT not in updates
    ):
        # A model change with no explicit effort should use the new model's
        # recommended profile. Retaining the prior model's effort can make a
        # perfectly valid switch to a non-reasoning model unsavable.
        known.pop(CONF_REASONING_EFFORT, None)
    normalized = normalize_agent_config(
        {**known, **updates}, validated_functions=validated_functions
    )
    return {
        **{
            key: deepcopy(value)
            for key, value in current.items()
            if key not in AGENT_CONFIG_FIELDS
        },
        **normalized,
    }


def agent_config_snapshot(
    data: dict[str, Any], *, preparsed_function_tools: bool = False
) -> dict[str, Any]:
    """Return frontend-safe normalized configuration with parsed tools."""
    result = normalize_agent_config(
        {key: value for key, value in data.items() if key in AGENT_CONFIG_FIELDS},
        function_tools_as_list=preparsed_function_tools,
    )
    # normalize_agent_config already validated the tools and their groups. It
    # stores explicitly configured tools as YAML for persistence; only decode
    # that validated representation for the frontend.
    if isinstance(result[CONF_FUNCTION_TOOLS], str):
        result[CONF_FUNCTION_TOOLS] = yaml.safe_load(result[CONF_FUNCTION_TOOLS]) or []
    return result


def model_capabilities(model: str) -> dict[str, Any]:
    """Return model-specific fields and choices supported by the backend."""
    from .model_capabilities import frontend_capabilities

    capabilities: dict[str, Any] = frontend_capabilities(model)
    capabilities["reasoning_effort_options"] = get_reasoning_effort_options(model)
    return capabilities


def _configured_tools_yaml(data: Mapping[str, Any]) -> str | None:
    """Return a deterministic cache key preserving legacy empty/default semantics."""
    configured = data.get(CONF_FUNCTION_TOOLS)
    if not configured:
        return None
    if isinstance(configured, str):
        return configured
    try:
        # Persisted Function Tool collections are JSON-compatible. Using JSON here
        # keeps cache-key construction linear and avoids PyYAML's expensive emitter
        # on every Management/agent catalogue read. JSON is valid YAML, so the
        # existing cached parser can consume this representation unchanged.
        return json.dumps(
            configured,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except TypeError, ValueError:
        # Runtime-only hydrated configs may contain objects such as HA Templates.
        # Preserve the established fallback for those uncommon call sites.
        return yaml.safe_dump(
            configured,
            sort_keys=True,
            allow_unicode=True,
        )


@lru_cache(maxsize=64)
def _cached_configured_tools(raw_yaml: str | None) -> tuple[dict[str, Any], ...]:
    """Parse and validate one distinct persisted Function Tool revision once."""
    data: dict[str, Any] = {}
    if raw_yaml is not None:
        data[CONF_FUNCTION_TOOLS] = raw_yaml
    return tuple(_configured_function_tools_from_data(data))


def configured_function_tool_metadata_from_data(
    data: Mapping[str, Any],
) -> dict[str, int]:
    """Return immutable-style metadata without copying hydrated runtime configs."""
    tools = _cached_configured_tools(_configured_tools_yaml(data))
    return {
        "usable_count": len(tools),
        "enabled_count": sum(function_tool_enabled(tool) for tool in tools),
        "total_count": len(tools),
    }


def configured_function_tools_from_data(
    data: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return isolated tools from the validated configuration-revision cache."""
    # Callers historically received fresh mutable dictionaries. Keep that contract
    # while moving the much more expensive YAML/schema validation behind the cache.
    # Runtime Function configs require a container copy that keeps HA-bound Template
    # objects hydrated; their normal deepcopy is intentionally the persistence form.
    return cast(
        list[dict[str, Any]],
        copy_runtime_function_config(
            list(_cached_configured_tools(_configured_tools_yaml(data)))
        ),
    )


def _groups_cache_key(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=True, allow_unicode=True)


@lru_cache(maxsize=128)
def _cached_function_groups(
    groups_yaml: str,
    tool_names: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    groups = yaml.safe_load(groups_yaml)
    synthetic_tools = [{"spec": {"name": name}} for name in tool_names]
    return tuple(_validate_function_groups(groups, synthetic_tools))


def validate_function_groups(
    value: Any, function_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate each distinct group/tool-name configuration only once."""
    try:
        tool_names = tuple(
            str(tool["spec"]["name"])
            for tool in function_tools
            if isinstance(tool, dict)
            and isinstance(tool.get("spec"), dict)
            and isinstance(tool["spec"].get("name"), str)
        )
        groups_yaml = _groups_cache_key(value)
    except Exception:
        # Preserve the existing validation/error path for malformed unexpected data.
        return _validate_function_groups(value, function_tools)
    return deepcopy(list(_cached_function_groups(groups_yaml, tool_names)))
