"""Unified authenticated management API and single Home Assistant panel."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import logging
from time import monotonic, perf_counter
from types import MappingProxyType
from typing import Any, Final
from uuid import uuid4

import voluptuous as vol
import yaml

from homeassistant.components import panel_custom, websocket_api
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import service as service_helper

from .agent_config import (
    AGENT_CONFIG_FIELDS,
    GUEST_V2_FIELDS,
    MAX_AGENT_TITLE_LENGTH,
    AgentConfigError,
    agent_config_defaults,
    agent_config_options,
    agent_config_snapshot,
    configured_function_tools_from_data as _strict_configured_function_tools,
    function_tool_enabled,
    function_tool_yaml,
    merge_agent_config as merge_validated_agent_config,
    model_capabilities,
    normalize_agent_config,
    preserve_legacy_guest_policy,
    starter_function_tool_yaml,
    validate_agent_title,
    validate_function_tools,
    validate_single_function_tool,
)
from .agent_maintenance import management_command_lease
from .backup import async_create_backup, async_restore_backup, inspect_backup
from .built_in_functions import built_in_function_catalog
from .const import (
    AGENT_CONFIG_EXPORT_VERSION,
    CONF_CHAT_MODEL,
    CONF_CONVERSATION_TIMEOUT_MINUTES,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_GUEST_ALLOWED_FUNCTION_NAMES,
    CONF_GUEST_MODE_ENABLED,
    CONF_GUEST_POLICY_VERSION,
    CONF_KNOWLEDGE_ENABLED,
    CONF_SHARED_MEMORY_MODE,
    CONF_USAGE_REQUEST_RETENTION_DAYS,
    CONF_USAGE_RUN_RETENTION_DAYS,
    DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    DEFAULT_SHARED_MEMORY_MODE,
    DOMAIN,
    GUEST_POLICY_VERSION,
    MANAGEMENT_PANEL_TITLE,
    MANAGEMENT_PANEL_URL,
    SERVICE_CALL_FUNCTION,
    SHARED_MEMORY_DISABLED,
)
from .continuity import async_get_continuity
from .conversation_archive import async_get_archive
from .exposed_attributes import exposed_attribute_catalog
from .frontend_assets import async_register_frontend_assets, frontend_entry_url
from .function_dependency_integrity import (
    _TOOL_MUTATIONS,
    _rule_script_actions,
    async_validate_request_rule_functions,
    group_reference_updates,
)
from .function_groups import get_function_group_runtime
from .functions import FUNCTIONS
from .functions.security import FunctionSecurity, classify_tool
from .guest_mode import (
    async_get_guest_mode,
    guest_policy_editor_snapshot,
    resolve_guest_policy,
)
from .ha_llm_tools import (
    async_discover,
    is_ha_tool,
    new_reference_tool,
    reference_key,
    validate_reference,
)
from .helpers import get_exposed_entities
from .knowledge import async_get_knowledge, knowledge_source_as_dict
from .local_intents import (
    CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI,
    CONF_LOCAL_INTENT_EXCLUSIONS,
    CONF_LOCAL_INTENTS_ENABLED,
    local_handling_snapshot,
)
from .management_browser import async_browse_memories
from .management_configuration_guidance import (
    _configuration_action,
    decorate_configuration_result,
)
from .management_function_quarantine import (
    _management_configured_tools as configured_function_tools_from_data,
    _management_merge_agent_config as merge_agent_config,
    _management_validate_function_groups as validate_function_groups,
    _tolerant_agent_test as async_test_agent,
    _tolerant_persist_function_configuration,
    management_function_tools,
)
from .management_function_repair import (
    agent_config_revision as _agent_config_revision,  # noqa: F401 - test seam
    has_unavailable_native_tool,
    normalized_persisted_config_snapshot,
    peek_function_tool_health,
    persisted_config_projection,
    repair_state_for_projection,
    require_agent_config_revision as _require_agent_config_revision,
    saved_agent_config_revision,
    seed_persisted_config_projection,
)
from .management_history_queries import (
    archive_get_page,
    archive_list_page,
    archive_search_page,
    usage_breakdowns,
    usage_daily_page,
    usage_requests_page,
    usage_runs_page,
    usage_summary,
)
from .management_permissions import (
    async_quiet_hours_command,
    require_management_permission,
)
from .management_projections import settings_snapshot
from .management_request_preview import (
    async_preview_effective_request as _async_preview_effective_request,
    entry_and_agent,
)
from .memory import ANONYMOUS_USER_ID, async_get_memory
from .regex_execution import async_process_speech_text
from .request_rule_match_preview import async_request_rule_match_preview
from .request_rule_packs import (
    async_append_rule_pack,
    export_rule_pack,
    validate_rule_pack,
)
from .request_rules import (
    async_get_request_rules,
    get_request_rule_runtime,
    rule_has_sensitive_actions,
    validate_rule,
)
from .scope import SHARED_HOUSEHOLD_SCOPE_ID
from .secret_redaction import redact_secrets, restore_redacted_secrets
from .temporary_memory import async_get_temporary_memory, temporary_memory_as_dict
from .usage import async_get_usage

_PERFORMANCE_LOGGER = logging.getLogger(f"{__name__}.performance")
_MANAGEMENT_SERVER_EPOCH = uuid4().hex

WS_COMMAND = f"{DOMAIN}/management"
_UI_SETUP = f"{DOMAIN}.management_ui_setup"
_LIVE_CONFIGURATION_METADATA = frozenset(
    {"local_handling", "exposed_attribute_catalog"}
)
# These are the persisted fields that can change the local handling projection.
_LOCAL_HANDLING_CONFIG_FIELDS = frozenset(
    {
        CONF_LOCAL_INTENTS_ENABLED,
        CONF_LOCAL_INTENT_EXCLUSIONS,
        CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI,
    }
)


def _local_handling_config_changed(
    current: Mapping[str, Any], candidate: Mapping[str, Any], updates: Mapping[str, Any]
) -> bool:
    return any(
        key in updates and current.get(key) != candidate.get(key)
        for key in _LOCAL_HANDLING_CONFIG_FIELDS
    )


@lru_cache(maxsize=1)
def _cached_configuration_defaults() -> dict[str, Any]:
    """Build the immutable-source default configuration projection once."""
    return agent_config_snapshot(agent_config_defaults())


@lru_cache(maxsize=1)
def _cached_configuration_options() -> dict[str, list[dict[str, Any]]]:
    """Build static management choice metadata once."""
    return agent_config_options()


def _configuration_defaults() -> dict[str, Any]:
    return deepcopy(_cached_configuration_defaults())


def _configuration_options() -> dict[str, list[dict[str, Any]]]:
    return deepcopy(_cached_configuration_options())


@lru_cache(maxsize=64)
def _cached_model_capabilities(model: str) -> dict[str, Any]:
    """Model metadata depends only on the selected model, not the agent revision."""
    return model_capabilities(model)


def _configuration_model_capabilities(model: str) -> dict[str, Any]:
    return deepcopy(_cached_model_capabilities(model))


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def _reset_request_rule_runtime(
    hass: HomeAssistant, entry_id: str, subentry_id: str, continuity_key: str
) -> None:
    """Clear Request Rule routing state for one ended continuity session."""
    get_request_rule_runtime(hass, entry_id, subentry_id).reset(
        f"continuity:{continuity_key}"
    )


def _require_admin(is_admin: bool) -> None:
    if not is_admin:
        raise HomeAssistantError("Administrator permission is required")


def _selected_scope(user_id: str, is_admin: bool, requested: Any) -> str:
    personal = f"user:{user_id}"
    if requested is None:
        return personal
    if not isinstance(requested, str):
        raise HomeAssistantError("scope_id must be a string")
    if not is_admin and requested != personal:
        raise HomeAssistantError("This scope is not available to the current user")
    if requested in {
        SHARED_HOUSEHOLD_SCOPE_ID,
        ANONYMOUS_USER_ID,
    } or requested.startswith("user:"):
        return requested
    raise HomeAssistantError("Unknown data scope")


def _memory_scope(scope_id: str) -> str:
    return scope_id.removeprefix("user:") if scope_id.startswith("user:") else scope_id


def _validation_result(callback) -> dict[str, Any]:
    """Run configuration validation and return frontend-friendly errors."""
    try:
        value = callback()
    except AgentConfigError as err:
        return {"valid": False, "errors": {err.field: str(err).split(": ", 1)[-1]}}
    return {"valid": True, "errors": {}, "config": value}


def _validated_model_request(
    config: dict[str, Any],
    entry_data: Mapping[str, Any],
    current: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use the live resolver only at a configuration mutation boundary."""
    relevant = {
        "chat_model",
        "api_mode",
        "reasoning_effort",
        "max_tokens",
        "functions",
        "function_groups",
        "memory_enabled",
        "knowledge_enabled",
        "archive_enabled",
        "guest_mode_enabled",
        "web_search",
    }
    if current is not None:
        defaults = agent_config_defaults()
        if all(
            config.get(key) == current.get(key, defaults.get(key)) for key in relevant
        ):
            return config
    from .request import build_provider_request_snapshot

    try:
        build_provider_request_snapshot(config, entry_data)
    except HomeAssistantError as err:
        raise AgentConfigError("api_mode", str(err)) from err
    return config


def _persist_function_configuration(
    hass: HomeAssistant,
    entry: Any,
    subentry: Any,
    tools: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    extra_updates: dict[str, Any] | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Persist one revision-checked tool edit without discarding quarantined siblings."""
    return _tolerant_persist_function_configuration(
        hass,
        entry,
        subentry,
        tools,
        groups,
        extra_updates=extra_updates,
        expected_revision=expected_revision,
    )


async def _function_reference_state(
    hass: HomeAssistant,
    entry_id: str,
    subentry_id: str,
    subentry_data: MappingProxyType | dict[str, Any],
    function_name: str,
):
    """Return exact durable references to one configured Function Tool."""
    rules = await async_get_request_rules(hass, entry_id, subentry_id)
    guest_names = subentry_data.get(CONF_GUEST_ALLOWED_FUNCTION_NAMES, [])
    return rules, {
        "request_rules": rules.function_references(function_name),
        "guest_mode": isinstance(guest_names, list) and function_name in guest_names,
    }


def _function_reference_error(name: str, references: dict[str, Any]) -> str:
    """Describe semantic references that must be resolved before deletion."""
    parts: list[str] = []
    rule_names = [
        item.get("name", item.get("id", "unnamed rule"))
        for item in references.get("request_rules", [])
    ]
    if rule_names:
        parts.append("Request Rules: " + ", ".join(rule_names))
    if references.get("guest_mode"):
        parts.append("Guest Mode custom function access")
    return (
        f"Function Tool `{name}` is still referenced by "
        + "; ".join(parts)
        + ". Update those references before deleting it."
    )


def _redact_export_secrets(value: Any, *, schema: bool = False) -> Any:
    """Redact credential values while preserving exported configuration structure."""
    return redact_secrets(value, schema=schema)


def _export_agent(subentry) -> dict[str, Any]:
    """Build a versioned configuration document with best-effort redaction."""
    snapshot = preserve_legacy_guest_policy(
        dict(subentry.data), agent_config_snapshot(dict(subentry.data))
    )
    return {
        "schema": "extended_openai_conversation.agent",
        "version": AGENT_CONFIG_EXPORT_VERSION,
        "title": subentry.title,
        "config": _redact_export_secrets(snapshot),
    }


def _parse_import_document(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError as err:
            raise AgentConfigError("document", f"invalid JSON/YAML: {err}") from err
    if not isinstance(value, dict):
        raise AgentConfigError("document", "must be an object")
    if value.get("schema") != "extended_openai_conversation.agent":
        raise AgentConfigError("schema", "unsupported or missing export schema")
    if value.get("version") != AGENT_CONFIG_EXPORT_VERSION:
        raise AgentConfigError("version", "unsupported export version")
    unknown = set(value) - {"schema", "version", "title", "config"}
    if unknown:
        raise AgentConfigError(
            "document", "unknown fields: " + ", ".join(sorted(unknown))
        )
    config = restore_redacted_secrets(value.get("config"))
    if not isinstance(config, dict):
        raise AgentConfigError("config", "must be an object")
    return {
        "title": validate_agent_title(
            value.get("title"), default="Imported conversation agent"
        ),
        "config": preserve_legacy_guest_policy(config, normalize_agent_config(config)),
    }


def _prepare_request_rule(value: Any, rule_id: str | None = None) -> dict[str, Any]:
    """Assign/preserve rule identity before canonical validation."""
    if not isinstance(value, Mapping):
        raise HomeAssistantError("rule must be an object")
    raw = dict(value)
    if rule_id is not None:
        raw["id"] = rule_id
    elif not isinstance(raw.get("id"), str) or not str(raw["id"]).strip():
        raw["id"] = uuid4().hex
    return validate_rule(raw)


def _validate_request_rule_functions(
    rule: Mapping[str, Any], configured_tools: list[dict[str, Any]]
) -> None:
    """Validate configured Function references in canonical persisted actions."""
    service_action = f"{DOMAIN}.{SERVICE_CALL_FUNCTION}"
    calls: list[Mapping[str, Any]] = []
    for action in rule.get("action", {}).get("actions", []):
        if not isinstance(action, Mapping):
            continue
        if action.get("action", action.get("service")) != service_action:
            continue
        data = action.get("data")
        if not isinstance(data, Mapping) or not isinstance(data.get("function"), str):
            raise HomeAssistantError("Configured Function action is invalid")
        arguments = data.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise HomeAssistantError("Configured Function arguments must be an object")
        calls.append(data)

    available = {
        tool["spec"]["name"] for tool in configured_tools if function_tool_enabled(tool)
    }
    missing = {
        str(call["function"]) for call in calls if call["function"] not in available
    }
    if missing:
        raise HomeAssistantError(
            "Function Tool is unavailable or disabled: " + ", ".join(sorted(missing))
        )
    by_name = {tool["spec"]["name"]: tool for tool in configured_tools}
    for call in calls:
        function_name = str(call["function"])
        parameters = by_name[function_name]["spec"].get("parameters", {})
        required = (
            parameters.get("required", []) if isinstance(parameters, dict) else []
        )
        missing_inputs = set(required) - set(call.get("arguments", {}))
        if missing_inputs:
            raise HomeAssistantError(
                f"Function Tool `{function_name}` needs input: "
                + ", ".join(sorted(missing_inputs))
            )


async def _validate_request_rule_conditions(
    hass: HomeAssistant, rule: Mapping[str, Any]
) -> None:
    """Validate only the edited rule against HA's current condition platforms."""
    from homeassistant.helpers import condition as ha_condition

    for config in rule.get("conditions", []):
        try:
            await ha_condition.async_validate_condition_config(hass, deepcopy(config))
        except Exception as err:
            raise HomeAssistantError(f"Invalid Only when condition: {err}") from err


async def _review_rule_pack(
    hass: HomeAssistant,
    prepared: Mapping[str, Any],
    rules: Any,
    configured_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Inspect dependencies without executing or saving imported rules."""
    snapshot = rules.snapshot()
    if len(snapshot["rules"]) + len(prepared["rules"]) > 500:
        raise HomeAssistantError("Request Rule limit reached")
    new_group_names = {group["name"].casefold() for group in snapshot["groups"]}
    added_group_names = {
        group["name"].casefold()
        for group in prepared["groups"]
        if group["name"].casefold() not in new_group_names
    }
    if len(snapshot["groups"]) + len(added_group_names) > 100:
        raise HomeAssistantError("Group limit reached")
    group_names = {group["id"]: group["name"] for group in prepared["groups"]}
    available = {
        tool["spec"]["name"] for tool in configured_tools if function_tool_enabled(tool)
    }
    summaries = []
    for rule in prepared["rules"]:
        await _validate_request_rule_conditions(hass, rule)
        actions = list(_rule_script_actions(rule))
        function_names = sorted(
            {
                str(action["data"]["function"])
                for action in actions
                if action.get("action", action.get("service"))
                == f"{DOMAIN}.{SERVICE_CALL_FUNCTION}"
                and isinstance(action.get("data"), Mapping)
                and isinstance(action["data"].get("function"), str)
            }
        )
        missing_functions = sorted(set(function_names) - available)
        await async_validate_request_rule_functions(
            hass, rule, configured_tools, quarantined_names=missing_functions
        )
        entities = sorted(
            {
                entity
                for action in actions
                for value in (
                    [action.get("target", {}).get("entity_id")]
                    if isinstance(action.get("target"), Mapping)
                    else []
                )
                for entity in (value if isinstance(value, list) else [value])
                if isinstance(entity, str) and "{{" not in entity
            }
        )
        missing_entities = [
            entity for entity in entities if hass.states.get(entity) is None
        ]
        services = sorted(
            {
                service
                for action in actions
                if isinstance(
                    service := action.get("action", action.get("service")), str
                )
                and "." in service
                and service != f"{DOMAIN}.{SERVICE_CALL_FUNCTION}"
            }
        )
        missing_services = [
            service
            for service in services
            if not hass.services.has_service(*service.split(".", 1))
        ]
        summaries.append(
            {
                "name": rule["name"],
                "triggers": rule["phrases"][:3],
                "group": group_names.get(rule["group_id"], "Ungrouped"),
                "action_type": rule["action_type"],
                "conditions": len(rule["conditions"]),
                "function_tools": function_names,
                "entities": entities,
                "services": services,
                "missing_dependencies": [
                    *missing_functions,
                    *missing_entities,
                    *missing_services,
                ],
                "continue_to_ai": rule["action"]["continue_to_ai"],
                "continue_matching": rule["continue_matching"],
                "ai_input_mode": rule["ai_input_mode"],
                "ai_input_capture": rule["ai_input_capture"],
                "status": "needs_attention"
                if missing_functions or missing_entities or missing_services
                else "ready",
            }
        )
    return {
        "count": len(summaries),
        "ready": sum(item["status"] == "ready" for item in summaries),
        "needs_attention": sum(item["status"] != "ready" for item in summaries),
        "rules": summaries,
        "will_append": True,
        "will_disable": True,
        "new_groups": len(added_group_names),
        "revision": snapshot["revision"],
    }


def _rule_pack_digest(prepared: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _register_rule_pack_review(
    rules: Any, prepared: Mapping[str, Any], revision: str
) -> str:
    reviews = getattr(rules, "_rule_pack_reviews", {})
    now = monotonic()
    reviews = {token: record for token, record in reviews.items() if record[2] > now}
    token = uuid4().hex
    reviews[token] = (_rule_pack_digest(prepared), revision, now + 600)
    rules._rule_pack_reviews = dict(list(reviews.items())[-16:])
    return token


def _consume_rule_pack_review(
    rules: Any, token: Any, prepared: Mapping[str, Any], revision: Any
) -> None:
    reviews = getattr(rules, "_rule_pack_reviews", {})
    record = reviews.pop(token, None) if isinstance(token, str) else None
    if (
        record is None
        or record[0] != _rule_pack_digest(prepared)
        or record[1] != revision
        or record[2] <= monotonic()
    ):
        raise HomeAssistantError("Review this exact Rule Pack before importing")


@dataclass(frozen=True)
class _ManagementRequest:
    """One validated agent selection shared by explicit section handlers."""

    hass: HomeAssistant
    user_id: str
    is_admin: bool
    message: dict[str, Any]
    entry_id: str
    subentry_id: str
    entry: Any
    subentry: ConfigSubentry


def _unknown_management_action(request: _ManagementRequest) -> dict[str, Any]:
    # Preserve the existing scope boundary on legacy fall-through errors.
    _selected_scope(request.user_id, request.is_admin, request.message.get("scope_id"))
    raise HomeAssistantError(
        f"Unknown {request.message.get('section', 'overview')} management action: {request.message['action']}"
    )


async def async_request_rules_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the request rules Management section."""
    hass = request.hass
    is_admin = request.is_admin
    message = request.message
    subentry = request.subentry
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    _require_admin(is_admin)
    if action in {"test", "test_match"}:
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HomeAssistantError("Test request text is required")
        rules = await async_get_request_rules(hass, entry_id, subentry_id)
        return await async_request_rule_match_preview(hass, rules, text)
    if action in {"create", "update"}:
        # Static validation can await Home Assistant schemas. Keep the legacy
        # second read so the final rule validator sees current Function Tools.
        _entry, subentry = entry_and_agent(hass, entry_id, subentry_id)
    rules = await async_get_request_rules(hass, entry_id, subentry_id)
    if action == "rule_pack_export":
        return export_rule_pack(
            rules,
            message.get("selection"),
            message.get("group_id"),
            message.get("rule_ids"),
        )
    if action in {"rule_pack_review", "rule_pack_import"}:
        prepared = await hass.async_add_executor_job(
            validate_rule_pack, message.get("pack")
        )
        tools = configured_function_tools_from_data(subentry.data)
        review = await _review_rule_pack(hass, prepared, rules, tools)
        if action == "rule_pack_review":
            review["review_token"] = _register_rule_pack_review(
                rules, prepared, review["revision"]
            )
            return review
        if message.get("confirm") is not True:
            raise HomeAssistantError("Review and confirm the Rule Pack before import")
        if message.get("revision") != review["revision"]:
            raise HomeAssistantError("Request Rules changed after review; review again")
        _consume_rule_pack_review(
            rules, message.get("review_token"), prepared, review["revision"]
        )
        imported = await async_append_rule_pack(
            rules, prepared, expected_revision=message.get("revision")
        )
        return {**imported, "review": review}
    if action == "list":
        snapshot = rules.snapshot()
        snapshot["rules"] = [
            {**rule, "sensitive_matching_warning": rule_has_sensitive_actions(rule)}
            for rule in snapshot["rules"]
        ]
        configured_tools = (
            configured_function_tools_from_data(subentry.data)
            if hasattr(subentry, "data")
            else []
        )
        snapshot["function_catalog"] = [
            {
                "name": tool["spec"]["name"],
                "description": tool["spec"].get("description", ""),
                "parameters": tool["spec"].get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for tool in configured_tools
            if function_tool_enabled(tool) and not is_ha_tool(tool)
        ]
        return snapshot
    if action == "settings":
        return await rules.async_set_settings(
            message.get("defaults"),
            message.get("wording_groups"),
            expected_revision=message.get("revision"),
        )
    if action == "defaults":
        defaults = await rules.async_set_defaults(
            message.get("defaults"), expected_revision=message.get("revision")
        )
        return {"defaults": defaults, "revision": rules.revision()}
    if action == "wording_groups":
        wording_groups = await rules.async_set_wording_groups(
            message.get("wording_groups"),
            expected_revision=message.get("revision"),
        )
        return {"wording_groups": wording_groups, "revision": rules.revision()}
    if action == "groups":
        return await rules.async_set_groups(
            message.get("groups"), expected_revision=message.get("revision")
        )
    if action == "create":
        candidate = _prepare_request_rule(message.get("rule"))
        await _validate_request_rule_conditions(hass, candidate)
        _validate_request_rule_functions(
            candidate,
            configured_function_tools_from_data(subentry.data)
            if hasattr(subentry, "data")
            else [],
        )
        rule = await rules.async_create(
            candidate, expected_revision=message.get("revision")
        )
        return {
            "rule": rule,
            "sensitive_matching_warning": rule_has_sensitive_actions(rule),
            "revision": rules.revision(),
        }
    rule_id = message.get("rule_id")
    if not isinstance(rule_id, str):
        raise HomeAssistantError("rule_id is required")
    if action == "update":
        candidate = _prepare_request_rule(message.get("rule"), rule_id)
        await _validate_request_rule_conditions(hass, candidate)
        _validate_request_rule_functions(
            candidate,
            configured_function_tools_from_data(subentry.data)
            if hasattr(subentry, "data")
            else [],
        )
        rule = await rules.async_update(
            rule_id, candidate, expected_revision=message.get("revision")
        )
        return {
            "rule": rule,
            "sensitive_matching_warning": rule_has_sensitive_actions(rule),
            "revision": rules.revision(),
        }
    if action == "delete":
        if message.get("confirm") is not True:
            raise HomeAssistantError("Explicit confirmation is required")
        deleted = await rules.async_delete(
            rule_id, expected_revision=message.get("revision")
        )
        return {"deleted": deleted, "revision": rules.revision()}
    if action == "duplicate":
        rule = await rules.async_duplicate(
            rule_id, expected_revision=message.get("revision")
        )
        return {
            "rule": rule,
            "sensitive_matching_warning": rule_has_sensitive_actions(rule),
            "revision": rules.revision(),
        }
    if action == "move":
        direction = message.get("direction")
        if not isinstance(direction, str):
            raise HomeAssistantError("direction is required")
        rule = await rules.async_move(
            rule_id,
            direction,
            target_rule_id=message.get("target_rule_id"),
            expected_revision=message.get("revision"),
        )
        return {
            "rule": rule,
            "sensitive_matching_warning": rule_has_sensitive_actions(rule),
            "revision": rules.revision(),
        }
    raise HomeAssistantError(f"Unknown Request Rules action: {action}")


async def async_guest_mode_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the guest mode Management section."""
    hass = request.hass
    is_admin = request.is_admin
    message = request.message
    entry = request.entry
    subentry = request.subentry
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    guest_manager = await async_get_guest_mode(hass, entry_id, subentry_id)

    if action == "get":
        legacy_policy = (
            subentry.data.get(CONF_GUEST_POLICY_VERSION) != GUEST_POLICY_VERSION
        )
        configured_tools: list[dict[str, Any]] = []
        exposed_entities = None
        if legacy_policy and is_admin:
            # Migration translation is safety-sensitive: retain the existing
            # conservative legacy projection on the admin primary response.
            configured_tools = configured_function_tools_from_data(subentry.data)
            exposed_entities = get_exposed_entities(hass)

        primary_result: dict[str, Any] = {
            "revision": persisted_config_projection(subentry).revision,
            "status": guest_manager.status(),
            "config": (
                guest_policy_editor_snapshot(
                    hass,
                    subentry.data,
                    configured_tools,
                    exposed_entities=exposed_entities,
                )
                if is_admin
                else {}
            ),
            "legacy_policy": legacy_policy,
            "migration_notice": (
                "This agent still uses the legacy Guest allow-list. Review the "
                "conservative exclusion draft below; the legacy policy remains "
                "enforced until you save."
                if legacy_policy
                else None
            ),
        }
        return primary_result

    if action == "details":
        started = perf_counter()

        phase = perf_counter()
        configured_tools = configured_function_tools_from_data(subentry.data)
        configured_tools_ms = _elapsed_ms(phase)

        phase = perf_counter()
        exposed_entities = get_exposed_entities(hass) if is_admin else None
        exposed_entities_ms = _elapsed_ms(phase)

        phase = perf_counter()
        policy = (
            resolve_guest_policy(
                hass,
                subentry.data,
                guest_manager,
                configured_tools,
                exposed_entities=exposed_entities,
            )
            if is_admin
            else resolve_guest_policy(
                hass, subentry.data, guest_manager, configured_tools
            )
        )
        policy_ms = _elapsed_ms(phase)

        details_result: dict[str, Any] = {
            "policy": policy.as_diagnostics(),
        }
        timings = {
            "configured_tools_ms": configured_tools_ms,
            "exposed_entities_ms": exposed_entities_ms,
            "policy_ms": policy_ms,
        }

        if is_admin:
            phase = perf_counter()
            library = await async_get_knowledge(hass, entry_id, subentry_id)
            knowledge_manager_ms = _elapsed_ms(phase)

            phase = perf_counter()
            groups = validate_function_groups(
                subentry.data.get(CONF_FUNCTION_GROUPS, []), configured_tools
            )
            groups_ms = _elapsed_ms(phase)

            phase = perf_counter()
            knowledge_sources = await library.async_list()
            knowledge_list_ms = _elapsed_ms(phase)

            details_result.update(
                {
                    "knowledge_sources": knowledge_sources,
                    "functions": [
                        {
                            "name": tool["spec"]["name"],
                            "description": tool["spec"].get("description", ""),
                            "enabled": function_tool_enabled(tool),
                            "unsafe_in_guest_mode": classify_tool(tool)
                            > FunctionSecurity.CONTROL,
                        }
                        for tool in configured_tools
                    ],
                    "function_groups": [
                        {
                            "id": group["id"],
                            "name": group["name"],
                            "description": group["description"],
                            "functions": group["functions"],
                        }
                        for group in groups
                    ],
                    "domains": sorted(
                        {
                            item["entity_id"].partition(".")[0]
                            for item in exposed_entities or ()
                            if isinstance(item.get("entity_id"), str)
                        }
                    ),
                }
            )
            timings.update(
                {
                    "knowledge_manager_ms": knowledge_manager_ms,
                    "groups_ms": groups_ms,
                    "knowledge_list_ms": knowledge_list_ms,
                }
            )

        details_result["_performance"] = {
            **timings,
            "total_ms": _elapsed_ms(started),
        }
        return details_result

    _require_admin(is_admin)
    if action == "save_policy":
        _require_agent_config_revision(subentry, message.get("revision"))
        updates = message.get("config")
        if not isinstance(updates, dict):
            raise HomeAssistantError("config must be an object")
        guest_fields = GUEST_V2_FIELDS | {CONF_GUEST_MODE_ENABLED}
        if set(updates) - guest_fields:
            raise HomeAssistantError("Guest policy contains unknown fields")
        updates[CONF_GUEST_POLICY_VERSION] = GUEST_POLICY_VERSION
        normalized = merge_agent_config(subentry.data, updates)
        hass.config_entries.async_update_subentry(entry, subentry, data=normalized)
        configured_tools = configured_function_tools_from_data(normalized)
        return {
            "revision": saved_agent_config_revision(
                subentry, normalized, subentry.title
            ),
            "config": guest_policy_editor_snapshot(hass, normalized, configured_tools),
        }
    if action == "update":
        return {
            "status": await guest_manager.async_update_trusted(
                active_from=message.get("active_from"),
                active_until=message.get("active_until"),
                indefinite=message.get("indefinite") is True,
            )
        }
    if action == "disable":
        return {"status": await guest_manager.async_disable_trusted()}

    return _unknown_management_action(request)


async def async_backup_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the backup Management section."""
    hass = request.hass
    is_admin = request.is_admin
    message = request.message
    entry = request.entry
    subentry = request.subentry
    action = request.message["action"]
    _require_admin(is_admin)
    if action == "create":
        return await async_create_backup(hass, entry, subentry)
    if action == "inspect":
        prepared = inspect_backup(message.get("document"), subentry.subentry_id)
        return {
            "valid": True,
            "title": prepared.title,
            "summary": prepared.summary(),
        }
    if action == "restore":
        if message.get("confirm") is not True:
            raise HomeAssistantError("Explicit confirmation is required")
        return await async_restore_backup(
            hass, entry, subentry, message.get("document")
        )

    return _unknown_management_action(request)


async def _async_save_configuration(request: _ManagementRequest) -> dict[str, Any]:
    """Normalize once, persist that candidate, and return the frontend snapshot."""
    from .management_loading_performance import (
        _agent_snapshot,
        _snapshot_normalized_configuration,
    )

    started = perf_counter()
    timings: dict[str, Any] = {}
    hass, is_admin, message = request.hass, request.is_admin, request.message
    entry, subentry = request.entry, request.subentry
    title = message.get("title")
    if title is not None and (not isinstance(title, str) or not title.strip()):
        return {"valid": False, "errors": {"title": "must not be empty"}}

    _require_admin(is_admin)
    updates = message.get("config", {})
    if not isinstance(updates, dict):
        raise HomeAssistantError("config must be an object")
    phase = perf_counter()
    if message.get("revision") is not None:
        _require_agent_config_revision(subentry, message["revision"])
    timings["stale_revision_ms"] = _elapsed_ms(phase)

    phase = perf_counter()
    projection = persisted_config_projection(subentry)
    tools_changed = CONF_FUNCTION_TOOLS in updates
    groups_changed = CONF_FUNCTION_GROUPS in updates
    repair_state = None
    validated_functions = None
    if not tools_changed:
        if (
            projection.repair_state is not None
            and projection.repair_state.invalid
            and not groups_changed
        ):
            repair_state = projection.repair_state
        elif projection.snapshot is not None:
            validated_functions = (
                projection.snapshot[CONF_FUNCTION_TOOLS],
                None if groups_changed else projection.snapshot[CONF_FUNCTION_GROUPS],
            )
        elif not groups_changed and has_unavailable_native_tool(dict(projection.data)):
            repair_state, _ = repair_state_for_projection(projection)
            if repair_state is not None and not repair_state.invalid:
                repair_state = None
    timings["function_state_lookup_ms"] = _elapsed_ms(phase)
    timings["function_tools_reused"] = (
        repair_state is not None or validated_functions is not None
    )
    timings["function_groups_reused"] = repair_state is not None or (
        validated_functions is not None and validated_functions[1] is not None
    )
    phase = perf_counter()
    if repair_state is not None:

        def merge_for_repair() -> dict[str, Any]:
            merged = merge_validated_agent_config(
                dict(subentry.data),
                updates,
                validated_functions=(repair_state.valid, repair_state.groups),
            )
            for key in (CONF_FUNCTION_TOOLS, CONF_FUNCTION_GROUPS):
                if key in subentry.data:
                    merged[key] = deepcopy(subentry.data[key])
                else:
                    merged.pop(key, None)
            return merged

        validation: dict[str, Any] = _validation_result(merge_for_repair)
    elif validated_functions is not None:
        validation = _validation_result(
            lambda: merge_validated_agent_config(
                dict(subentry.data), updates, validated_functions=validated_functions
            )
        )
    else:
        validation = _validation_result(
            lambda: merge_agent_config(subentry.data, updates)
        )
    if not validation.get("valid"):
        return validation
    timings["merge_validation_ms"] = _elapsed_ms(phase)
    phase = perf_counter()
    validation = _validation_result(
        lambda: _validated_model_request(
            validation["config"], entry.data, subentry.data
        )
    )
    if not validation.get("valid"):
        return validation
    timings["model_request_validation_ms"] = _elapsed_ms(phase)

    phase = perf_counter()
    normalized = validation["config"]
    persisted = preserve_legacy_guest_policy(dict(subentry.data), deepcopy(normalized))
    saved_title = title.strip() if isinstance(title, str) else subentry.title
    refresh_local_handling = _local_handling_config_changed(
        subentry.data, persisted, updates
    )
    timings["persistence_preparation_ms"] = _elapsed_ms(phase)
    phase = perf_counter()
    hass.config_entries.async_update_subentry(
        entry,
        subentry,
        data=persisted,
        **({"title": saved_title} if isinstance(title, str) else {}),
    )
    timings["subentry_update_ms"] = _elapsed_ms(phase)

    # merge_agent_config validated both fields before persistence. Decode the
    # normalized YAML for the editor without repeating schema validation.
    phase = perf_counter()
    response_data = persisted
    if repair_state is not None:
        response_data = dict(persisted)
        response_data[CONF_FUNCTION_TOOLS] = repair_state.valid
        response_data[CONF_FUNCTION_GROUPS] = repair_state.groups
    elif validated_functions is not None:
        response_data = dict(persisted)
        response_data[CONF_FUNCTION_TOOLS] = validated_functions[0]
    snapshot = _snapshot_normalized_configuration(response_data, validated=True)
    timings["response_snapshot_ms"] = _elapsed_ms(phase)
    phase = perf_counter()
    revision = saved_agent_config_revision(subentry, persisted, saved_title)
    timings["revision_calculation_ms"] = _elapsed_ms(phase)
    phase = perf_counter()
    capability_hits = _cached_model_capabilities.cache_info().hits
    capabilities = _configuration_model_capabilities(snapshot[CONF_CHAT_MODEL])
    timings["model_capabilities_ms"] = _elapsed_ms(phase)
    timings["model_capabilities_cache_hit"] = (
        _cached_model_capabilities.cache_info().hits > capability_hits
    )
    saved = {
        "title": saved_title,
        "config": snapshot,
        "revision": revision,
        "model_capabilities": capabilities,
    }
    if repair_state is not None:
        saved["function_repair"] = {
            "invalid_tools": deepcopy(repair_state.invalid),
            "invalid_count": len(repair_state.invalid),
            "group_issues": deepcopy(repair_state.group_issues),
            "persisted_groups": deepcopy(repair_state.raw_groups),
            "validation_error": repair_state.issue,
            "isolatable": bool(repair_state.invalid),
        }
    phase = perf_counter()
    seed_persisted_config_projection(entry, subentry, snapshot, revision)
    timings["projection_seed_ms"] = _elapsed_ms(phase)
    phase = perf_counter()
    if refresh_local_handling:
        saved["local_handling"] = local_handling_snapshot(
            hass,
            str(entry.entry_id),
            str(subentry.subentry_id),
            snapshot.get(CONF_LOCAL_INTENT_EXCLUSIONS, []),
        )
    timings["local_handling_ms"] = _elapsed_ms(phase)
    phase = perf_counter()
    agent = _agent_snapshot(hass, entry, subentry, config=snapshot, title=saved_title)
    timings["agent_snapshot_ms"] = _elapsed_ms(phase)
    phase = perf_counter()
    result = {
        "valid": True,
        "errors": {},
        **saved,
        "agent": agent,
        "_performance": timings,
    }
    timings["response_assembly_ms"] = _elapsed_ms(phase)
    timings["total_ms"] = _elapsed_ms(started)
    return result


async def async_configuration_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the configuration Management section."""
    hass = request.hass
    user_id = request.user_id
    is_admin = request.is_admin
    message = request.message
    entry = request.entry
    subentry = request.subentry
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    _require_admin(is_admin)
    if action == "save":
        return await _async_save_configuration(request)
    if action == "retention_get":
        started = perf_counter()
        projection_diagnostics: dict[str, Any] = {}
        projection = persisted_config_projection(subentry, projection_diagnostics)
        fields = (CONF_USAGE_REQUEST_RETENTION_DAYS, CONF_USAGE_RUN_RETENTION_DAYS)
        options = _cached_configuration_options()
        return {
            "title": subentry.title,
            "revision": projection.revision,
            "config": {key: deepcopy(projection.retention[key]) for key in fields},
            "options": {key: deepcopy(options[key]) for key in fields},
            "projection": "retention",
            "_performance": {
                **projection_diagnostics,
                "projection_ms": _elapsed_ms(started),
            },
        }
    if action == "get":
        started = perf_counter()
        phase = perf_counter()
        projection_diagnostics = {}
        projection = persisted_config_projection(subentry, projection_diagnostics)
        projection_ms = _elapsed_ms(phase)
        phase = perf_counter()
        defaults_cache_hit = _cached_configuration_defaults.cache_info().currsize > 0
        defaults = _configuration_defaults()
        defaults_ms = _elapsed_ms(phase)
        from .management_function_repair import safe_configuration_payload

        def repair_response(
            *,
            strict_skipped: bool,
            attempt_ms: float,
            validation_ms: float,
            state_cache_hit: bool,
        ) -> dict[str, Any]:
            repair_started = perf_counter()
            safe = safe_configuration_payload(
                hass, entry, subentry, projection=projection
            )
            safe["server_epoch"] = _MANAGEMENT_SERVER_EPOCH
            safe["_performance"] = {
                **projection_diagnostics,
                "projection_ms": projection_ms,
                "defaults_snapshot_ms": defaults_ms,
                "snapshot_attempt_ms": attempt_ms,
                "strict_snapshot_skipped_for_repair": strict_skipped,
                "repair_state_cache_hit": state_cache_hit,
                "function_validation_ms": validation_ms,
                "function_issue_check_ms": validation_ms,
                "repair_projection_ms": _elapsed_ms(repair_started),
                "total_ms": _elapsed_ms(started),
            }
            return safe

        preflight_started = perf_counter()
        repair_hint = False
        if projection.snapshot is None:
            health = peek_function_tool_health(dict(projection.data))
            repair_hint = (
                projection.repair_state is not None
                or bool(health and health.get("validation_error"))
                or has_unavailable_native_tool(dict(projection.data))
            )
        if repair_hint:
            state, state_cache_hit = repair_state_for_projection(projection)
            if state is not None:
                return repair_response(
                    strict_skipped=True,
                    attempt_ms=0.0,
                    validation_ms=_elapsed_ms(preflight_started),
                    state_cache_hit=state_cache_hit,
                )
        repair_preflight_ms = _elapsed_ms(preflight_started)
        phase = perf_counter()
        try:
            snapshot_diagnostics: dict[str, Any] = {}
            config, snapshot_cache_hit = normalized_persisted_config_snapshot(
                projection, snapshot_diagnostics, default_snapshot=defaults
            )
        except AgentConfigError:
            # Retain the normal cached fast path for valid tools. A malformed
            # Function Tool snapshot can still be served through quarantine,
            # including on cold reads before the catalogue is available.
            snapshot_attempt_ms = _elapsed_ms(phase)
            validation_started = perf_counter()
            state, state_cache_hit = repair_state_for_projection(projection)
            if state is None:
                raise
            return repair_response(
                strict_skipped=False,
                attempt_ms=snapshot_attempt_ms,
                validation_ms=_elapsed_ms(validation_started),
                state_cache_hit=state_cache_hit,
            )
        config_ms = _elapsed_ms(phase)

        phase = perf_counter()
        revision = projection.revision
        revision_ms = _elapsed_ms(phase)

        phase = perf_counter()
        options_cache_hit = _cached_configuration_options.cache_info().currsize > 0
        options = _configuration_options()
        options_ms = _elapsed_ms(phase)

        phase = perf_counter()
        capabilities = _configuration_model_capabilities(config[CONF_CHAT_MODEL])
        model_capabilities_ms = _elapsed_ms(phase)

        assembly_started = perf_counter()
        timings = {
            **projection_diagnostics,
            "projection_ms": projection_ms,
            "repair_preflight_ms": repair_preflight_ms,
            "snapshot_cache_hit": snapshot_cache_hit,
            **snapshot_diagnostics,
            "config_snapshot_ms": config_ms,
            "revision_ms": revision_ms,
            "defaults_snapshot_ms": defaults_ms,
            "defaults_cache_hit": defaults_cache_hit,
            "options_ms": options_ms,
            "options_cache_hit": options_cache_hit,
            "model_capabilities_ms": model_capabilities_ms,
        }
        payload = {
            "title": subentry.title,
            "revision": revision,
            "server_epoch": _MANAGEMENT_SERVER_EPOCH,
            "config": config,
            "defaults": defaults,
            "options": options,
            "model_capabilities": capabilities,
            "function_types": sorted(FUNCTIONS),
            "_performance": timings,
        }
        timings["response_assembly_ms"] = _elapsed_ms(assembly_started)
        timings["total_ms"] = _elapsed_ms(started)
        return payload
    if action == "live_metadata":
        requested = message.get("metadata_keys", [])
        if not isinstance(requested, list) or any(
            not isinstance(item, str) for item in requested
        ):
            raise HomeAssistantError("metadata_keys must be a list of strings")
        unknown = set(requested) - _LIVE_CONFIGURATION_METADATA
        if unknown:
            raise HomeAssistantError(
                "Unknown configuration metadata: " + ", ".join(sorted(unknown))
            )
        metadata: dict[str, Any] = {}
        if "local_handling" in requested:
            metadata["local_handling"] = local_handling_snapshot(
                hass,
                entry_id,
                subentry_id,
                subentry.data.get(CONF_LOCAL_INTENT_EXCLUSIONS, []),
            )
        if "exposed_attribute_catalog" in requested:
            metadata["exposed_attribute_catalog"] = exposed_attribute_catalog(
                hass, subentry.data
            )
        return metadata
    if action == "validate":
        updates = message.get("config", {})
        if not isinstance(updates, dict):
            raise HomeAssistantError("config must be an object")
        result = _validation_result(
            lambda: agent_config_snapshot(
                _validated_model_request(
                    merge_agent_config(subentry.data, updates),
                    entry.data,
                    subentry.data,
                )
            )
        )
        if result["valid"]:
            result["model_capabilities"] = model_capabilities(
                result["config"][CONF_CHAT_MODEL]
            )
        return result
    if action == "update":
        _require_admin(is_admin)
        updates = message.get("config")
        if not isinstance(updates, dict):
            raise HomeAssistantError("config must be an object")
        _require_agent_config_revision(subentry, message.get("revision"))
        normalized = merge_agent_config(subentry.data, updates)
        _validated_model_request(normalized, entry.data, subentry.data)
        if CONF_GUEST_POLICY_VERSION not in subentry.data:
            for key in GUEST_V2_FIELDS:
                normalized.pop(key, None)
        requested_title = message.get("title")
        saved_title = (
            validate_agent_title(requested_title)
            if requested_title is not None
            else validate_agent_title(subentry.title)
        )
        refresh_local_handling = _local_handling_config_changed(
            subentry.data, normalized, updates
        )
        hass.config_entries.async_update_subentry(
            entry, subentry, data=normalized, title=saved_title
        )
        snapshot = agent_config_snapshot(normalized)
        result = {
            "title": saved_title,
            "revision": saved_agent_config_revision(subentry, normalized, saved_title),
            "config": snapshot,
            "model_capabilities": model_capabilities(snapshot[CONF_CHAT_MODEL]),
        }
        seed_persisted_config_projection(entry, subentry, snapshot, result["revision"])
        if refresh_local_handling:
            result["local_handling"] = local_handling_snapshot(
                hass,
                entry_id,
                subentry_id,
                snapshot.get(CONF_LOCAL_INTENT_EXCLUSIONS, []),
            )
        return result
    if action == "duplicate":
        _require_admin(is_admin)
        requested_title = message.get("title")
        if requested_title is None:
            suffix = " - Copy"
            source = validate_agent_title(subentry.title)
            base = source[: MAX_AGENT_TITLE_LENGTH - len(suffix)].rstrip()
            title = validate_agent_title(f"{base}{suffix}")
        else:
            title = validate_agent_title(requested_title)
        duplicate_source = {
            key: value
            for key, value in subentry.data.items()
            if key in AGENT_CONFIG_FIELDS
        }
        duplicate = ConfigSubentry(
            data=MappingProxyType(
                preserve_legacy_guest_policy(
                    duplicate_source, normalize_agent_config(duplicate_source)
                )
            ),
            subentry_type="conversation",
            title=title,
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, duplicate)
        return {
            "status": "created",
            "entry_id": entry.entry_id,
            "subentry_id": duplicate.subentry_id,
            "title": duplicate.title,
        }
    if action == "export":
        document = _export_agent(subentry)
        return {
            "document": document,
            "json": json.dumps(document, indent=2, ensure_ascii=False),
        }
    if action == "import_preview":
        parsed = _parse_import_document(message.get("document"))
        return {
            "valid": True,
            "title": parsed["title"],
            "config": agent_config_snapshot(parsed["config"]),
            "summary": {
                "model": parsed["config"][CONF_CHAT_MODEL],
                "tools": len(
                    validate_function_tools(parsed["config"].get("functions"))
                ),
                "function_groups": len(parsed["config"].get("function_groups", [])),
                "speech_rules": len(
                    parsed["config"].get("speech_regex_replacements", [])
                ),
            },
        }
    if action == "import":
        _require_admin(is_admin)
        parsed = _parse_import_document(message.get("document"))
        mode = message.get("mode", "current")
        if mode == "current":
            if message.get("confirm") is not True:
                raise HomeAssistantError("Explicit confirmation is required")
            _require_agent_config_revision(subentry, message.get("revision"))
            hass.config_entries.async_update_subentry(
                entry, subentry, data=parsed["config"], title=parsed["title"]
            )
            from .management_loading_performance import (
                _snapshot_normalized_configuration,
            )

            snapshot = _snapshot_normalized_configuration(
                parsed["config"], validated=True
            )
            revision = saved_agent_config_revision(
                subentry, parsed["config"], parsed["title"]
            )
            seed_persisted_config_projection(entry, subentry, snapshot, revision)
            return {
                "status": "updated",
                "subentry_id": subentry.subentry_id,
                "revision": revision,
            }
        if mode != "new":
            raise HomeAssistantError("mode must be current or new")
        imported = ConfigSubentry(
            data=MappingProxyType(parsed["config"]),
            subentry_type="conversation",
            title=parsed["title"],
            unique_id=None,
        )
        hass.config_entries.async_add_subentry(entry, imported)
        return {"status": "created", "subentry_id": imported.subentry_id}
    if action == "speech_preview":
        sample = message.get("sample_text", "")
        updates = message.get("config", {})
        if not isinstance(sample, str) or not isinstance(updates, dict):
            raise HomeAssistantError("sample_text and config are invalid")
        normalized = merge_agent_config(subentry.data, updates)
        return {
            "speech_text": await async_process_speech_text(hass, sample, normalized)
        }
    if action in {"prompt_preview", "request_preview"}:
        updates = message.get("config", {})
        if not isinstance(updates, dict):
            raise HomeAssistantError("config must be an object")
        normalized = merge_agent_config(subentry.data, updates)
        return await _async_preview_effective_request(
            hass, entry, subentry, normalized, user_id
        )

    return _unknown_management_action(request)


async def async_tools_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the tools Management section."""
    hass = request.hass
    user_id = request.user_id
    is_admin = request.is_admin
    message = request.message
    entry = request.entry
    subentry = request.subentry
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    _require_admin(is_admin)
    if action in {"ha_catalog", "ha_add"}:
        from homeassistant.helpers import llm

        references = None
        if action == "ha_add":
            selected = message.get("tools")
            if not isinstance(selected, list) or len(selected) > 1000:
                raise HomeAssistantError("Select up to 1000 individual HA tools")
            try:
                references = [validate_reference(ref) for ref in selected]
            except ValueError as err:
                raise HomeAssistantError(str(err)) from err
        ha_catalog_snapshot = await async_discover(
            hass,
            llm.LLMContext(
                platform=DOMAIN,
                context=Context(user_id=user_id),
                language=hass.config.language,
                assistant="conversation",
                device_id=None,
            ),
            references,
        )
        # Discovery awaits external sources. Re-read canonical state before any
        # synchronous mutation so concurrent saves cannot be overwritten.
        latest_entry = hass.config_entries.async_get_entry(entry_id)
        if latest_entry is None or subentry_id not in latest_entry.subentries:
            raise HomeAssistantError("Agent no longer exists")
        entry = latest_entry
        subentry = entry.subentries[subentry_id]
        tools = configured_function_tools_from_data(subentry.data)
        ha_existing = {
            reference_key(tool["function"]): tool for tool in tools if is_ha_tool(tool)
        }
        if action == "ha_catalog":
            return {
                "tools": [
                    {
                        "reference": live.reference,
                        "name": live.tool.name,
                        "description": live.tool.description or "",
                        "source": live.source_label,
                        "already_added": key in ha_existing,
                    }
                    for key, live in ha_catalog_snapshot.tools.items()
                ],
                "saved": {
                    tool["spec"]["name"]: {
                        "available": key in ha_catalog_snapshot.tools,
                        "name": tool["function"]["tool_name"],
                        "source": ha_catalog_snapshot.tools[key].source_label
                        if key in ha_catalog_snapshot.tools
                        else tool["function"]["source_id"],
                        "description": ha_catalog_snapshot.tools[key].tool.description
                        or ""
                        if key in ha_catalog_snapshot.tools
                        else "Unavailable in this preview context",
                    }
                    for key, tool in ha_existing.items()
                },
                "unavailable_sources": ha_catalog_snapshot.unavailable_sources,
            }
        assert references is not None
        if any(
            reference_key(ref) not in ha_catalog_snapshot.tools for ref in references
        ):
            raise HomeAssistantError(
                "A selected HA tool is no longer available; refresh the catalogue"
            )
        groups = validate_function_groups(
            subentry.data.get(CONF_FUNCTION_GROUPS, []), tools
        )
        group_id = message.get("group_id")
        ha_target_group = next(
            (group for group in groups if group["id"] == group_id), None
        )
        if group_id and ha_target_group is None:
            raise HomeAssistantError("Function Group no longer exists")
        occupied = {tool["spec"]["name"] for tool in tools}
        for ref in references:
            key = reference_key(ref)
            if key in ha_existing:
                continue
            ha_added_tool = new_reference_tool(ref, occupied)
            tools.append(ha_added_tool)
            ha_existing[key] = ha_added_tool
            if ha_target_group is not None:
                ha_target_group["functions"].append(ha_added_tool["spec"]["name"])
        result = _persist_function_configuration(
            hass,
            entry,
            subentry,
            tools,
            groups,
            expected_revision=message.get("revision"),
        )
        result["ha_saved"] = {
            tool["spec"]["name"]: {
                "available": key in ha_catalog_snapshot.tools,
                "name": tool["function"]["tool_name"],
                "source": ha_catalog_snapshot.tools[key].source_label
                if key in ha_catalog_snapshot.tools
                else tool["function"]["source_id"],
                "description": ha_catalog_snapshot.tools[key].tool.description or ""
                if key in ha_catalog_snapshot.tools
                else "Unavailable in this preview context",
            }
            for key, tool in ha_existing.items()
        }
        return result
    if action == "validate":
        return _validation_result(lambda: validate_function_tools(message.get("tools")))
    if action == "serialize":
        return {"yaml": function_tool_yaml(message.get("tool"))}
    if action == "starter":
        return {"yaml": starter_function_tool_yaml()}
    if action == "built_in_catalog":
        configured = validate_function_tools(message.get("tools", []))
        return {
            "functions": [
                {
                    "label": preset["label"],
                    "implementation": preset["implementation"],
                    "name": preset["tool"]["spec"]["name"],
                    "description": preset["tool"]["spec"]["description"],
                    "already_configured": preset["already_configured"],
                    "yaml": function_tool_yaml(preset["tool"]),
                }
                for preset in built_in_function_catalog(configured)
            ]
        }
    if action == "validate_yaml":
        result = _validation_result(
            lambda: validate_single_function_tool(message.get("yaml"))
        )
        if result["valid"]:
            tool = result["config"]
            result.update(
                {
                    "yaml": function_tool_yaml(tool),
                    "name": tool["spec"]["name"],
                    "type": tool["function"]["type"],
                    "description": tool["spec"].get("description", ""),
                }
            )
        return result
    tools = configured_function_tools_from_data(subentry.data)
    groups = validate_function_groups(
        subentry.data.get(CONF_FUNCTION_GROUPS, []), tools
    )
    if action == "validate_current":
        return {"valid": True, "errors": {}}
    if action == "save":
        tool_candidate = message.get("tool")
        if not isinstance(tool_candidate, dict):
            raise HomeAssistantError("tool must be an object")
        saved_tool = validate_function_tools([tool_candidate])[0]
        original_name = message.get("original_name")
        if original_name is not None and not isinstance(original_name, str):
            raise HomeAssistantError("original_name must be a string")
        saved_name = saved_tool["spec"]["name"]
        existing_index = next(
            (
                index
                for index, tool in enumerate(tools)
                if tool["spec"]["name"] == original_name
            ),
            None,
        )
        if original_name is not None and existing_index is None:
            raise HomeAssistantError("The Function Tool no longer exists")
        if any(
            tool["spec"]["name"] == saved_name and index != existing_index
            for index, tool in enumerate(tools)
        ):
            raise HomeAssistantError(f"Function Tool {saved_name} already exists")
        if existing_index is None:
            tools.append(saved_tool)
            return _persist_function_configuration(
                hass,
                entry,
                subentry,
                tools,
                groups,
                expected_revision=message.get("revision"),
            )

        tools[existing_index] = saved_tool
        if original_name == saved_name:
            return _persist_function_configuration(
                hass,
                entry,
                subentry,
                tools,
                groups,
                expected_revision=message.get("revision"),
            )

        assert original_name is not None
        operation_revision = persisted_config_projection(subentry).revision
        original_tools = configured_function_tools_from_data(subentry.data)
        original_groups = validate_function_groups(
            subentry.data.get(CONF_FUNCTION_GROUPS, []), original_tools
        )
        groups = [
            {
                **group,
                "functions": [
                    saved_name if name == original_name else name
                    for name in group["functions"]
                ],
            }
            for group in groups
        ]
        rules, references = await _function_reference_state(
            hass, entry_id, subentry_id, subentry.data, original_name
        )
        rules_revision = rules.revision()
        guest_names = list(subentry.data.get(CONF_GUEST_ALLOWED_FUNCTION_NAMES, []))
        renamed_guest_names = [
            saved_name if name == original_name else name for name in guest_names
        ]
        result = _persist_function_configuration(
            hass,
            entry,
            subentry,
            tools,
            groups,
            extra_updates={CONF_GUEST_ALLOWED_FUNCTION_NAMES: renamed_guest_names},
            expected_revision=operation_revision,
        )
        try:
            renamed_rule_references = await rules.async_rename_function_reference(
                original_name,
                saved_name,
                expected_revision=rules_revision,
            )
        except Exception as err:
            try:
                _persist_function_configuration(
                    hass,
                    entry,
                    subentry,
                    original_tools,
                    original_groups,
                    extra_updates={CONF_GUEST_ALLOWED_FUNCTION_NAMES: guest_names},
                    expected_revision=result["revision"],
                )
            except HomeAssistantError as rollback_err:
                raise HomeAssistantError(
                    "Function Tool configuration changed while a related Request Rule "
                    "rename failed. The newer configuration was preserved; reload "
                    "before retrying."
                ) from rollback_err
            raise err
        result["renamed_references"] = {
            "request_rules": renamed_rule_references,
            "guest_mode": references["guest_mode"],
        }
        return result
    if action == "set_enabled":
        name = message.get("name")
        enabled = message.get("enabled")
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise HomeAssistantError("name and enabled are required")
        tool = next((item for item in tools if item["spec"]["name"] == name), None)
        if tool is None:
            raise HomeAssistantError("The Function Tool no longer exists")
        tool["enabled"] = enabled
        result = _persist_function_configuration(
            hass,
            entry,
            subentry,
            tools,
            groups,
            expected_revision=message.get("revision"),
        )
        if not enabled:
            _rules, references = await _function_reference_state(
                hass, entry_id, subentry_id, subentry.data, name
            )
            result["references"] = references
        return result
    if action == "delete":
        if message.get("confirm") is not True:
            raise HomeAssistantError("Explicit confirmation is required")
        name = message.get("name")
        if not isinstance(name, str):
            raise HomeAssistantError("name is required")
        remaining = [tool for tool in tools if tool["spec"]["name"] != name]
        if len(remaining) == len(tools):
            raise HomeAssistantError("The Function Tool no longer exists")
        operation_revision = persisted_config_projection(subentry).revision
        _rules, references = await _function_reference_state(
            hass, entry_id, subentry_id, subentry.data, name
        )
        if references["request_rules"] or references["guest_mode"]:
            raise HomeAssistantError(_function_reference_error(name, references))
        groups = [
            {
                **group,
                "functions": [item for item in group["functions"] if item != name],
            }
            for group in groups
        ]
        return _persist_function_configuration(
            hass,
            entry,
            subentry,
            remaining,
            groups,
            expected_revision=operation_revision,
        )
    if action == "save_group":
        group_candidate = message.get("group")
        if not isinstance(group_candidate, dict):
            raise HomeAssistantError("group must be an object")
        candidate = group_candidate
        candidate_functions = candidate.get("functions", [])
        if not isinstance(candidate_functions, list) or not all(
            isinstance(name, str) for name in candidate_functions
        ):
            raise HomeAssistantError("group functions must be a list of names")
        original_id = message.get("original_id")
        if original_id is not None and not isinstance(original_id, str):
            raise HomeAssistantError("original_id must be a string")
        existing = next((group for group in groups if group["id"] == original_id), None)
        if original_id is not None and existing is None:
            raise HomeAssistantError("The Function Group no longer exists")
        selected = set(candidate_functions)
        remaining_groups = [
            {
                **group,
                "functions": [
                    name for name in group["functions"] if name not in selected
                ],
            }
            for group in groups
            if group["id"] != original_id
        ]
        validated_groups = validate_function_groups(
            [*remaining_groups, candidate], tools
        )
        return _persist_function_configuration(
            hass,
            entry,
            subentry,
            tools,
            validated_groups,
            expected_revision=message.get("revision"),
            extra_updates=group_reference_updates(
                subentry.data, original_id, candidate["id"]
            )
            if isinstance(original_id, str)
            and isinstance(candidate.get("id"), str)
            and original_id != candidate["id"]
            else None,
        )
    if action == "delete_group":
        if message.get("confirm") is not True:
            raise HomeAssistantError("Explicit confirmation is required")
        group_id = message.get("group_id")
        if not isinstance(group_id, str):
            raise HomeAssistantError("group_id is required")
        remaining = [group for group in groups if group["id"] != group_id]
        if len(remaining) == len(groups):
            raise HomeAssistantError("The Function Group no longer exists")
        return _persist_function_configuration(
            hass,
            entry,
            subentry,
            tools,
            remaining,
            expected_revision=message.get("revision"),
            extra_updates=group_reference_updates(subentry.data, group_id, None),
        )

    return _unknown_management_action(request)


async def async_scopes_command(request: _ManagementRequest) -> dict[str, Any]:
    """Load scopes concurrently, only when the Management browser requests them."""
    from .management_loading_performance import async_scope_catalog

    if request.message["action"] == "catalog":
        args = (
            request.hass,
            request.user_id,
            request.is_admin,
            request.entry_id,
            request.subentry_id,
        )
        if "scope_kind" not in request.message:
            return await async_scope_catalog(*args)
        return await async_scope_catalog(
            *args,
            scope_kind=str(request.message["scope_kind"]),
        )
    return _unknown_management_action(request)


async def async_service_catalog_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the service catalog Management section."""
    hass = request.hass
    is_admin = request.is_admin
    action = request.message["action"]
    if action == "get":
        _require_admin(is_admin)
        return {"services": await service_helper.async_get_all_descriptions(hass)}

    return _unknown_management_action(request)


async def async_diagnostics_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the diagnostics Management section."""
    hass = request.hass
    entry = request.entry
    subentry = request.subentry
    action = request.message["action"]
    if action == "test_agent":
        return (await async_test_agent(hass, entry, subentry)).as_dict()

    return _unknown_management_action(request)


async def async_usage_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the usage Management section."""
    hass = request.hass
    is_admin = request.is_admin
    message = request.message
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    usage = await async_get_usage(hass, entry_id, subentry_id)
    if action == "summary":
        result = usage_summary(usage)
        if not is_admin:
            result["latest"] = None
        return result
    if action == "daily":
        return usage_daily_page(
            usage,
            start_date=str(message.get("start_date", "0000-01-01")),
            end_date=str(message.get("end_date", "9999-12-31")),
            limit=int(message.get("limit", 366)),
            offset=int(message.get("offset", 0)),
        )
    if action == "runs":
        return usage_runs_page(
            usage,
            limit=int(message.get("limit", 50)),
            offset=int(message.get("offset", 0)),
            successful=message.get("successful"),
        )
    if action == "requests":
        run_id = message.get("run_id")
        if not isinstance(run_id, str):
            raise HomeAssistantError("run_id is required")
        return usage_requests_page(
            usage,
            run_id,
            limit=int(message.get("limit", 100)),
            offset=int(message.get("offset", 0)),
        )
    if action == "breakdowns":
        return usage_breakdowns(
            usage,
            start_date=message.get("start_date"),
            end_date=message.get("end_date"),
        )
    if action == "retention":
        return {
            "request_days": usage.request_retention_days,
            "run_days": usage.run_retention_days,
        }
    if action == "clear_details":
        _require_admin(is_admin)
        return await usage.async_clear_details(confirm=message.get("confirm") is True)

    return _unknown_management_action(request)


async def async_conversations_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the conversations Management section."""
    hass = request.hass
    is_admin = request.is_admin
    message = request.message
    subentry = request.subentry
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    scope_id = _selected_scope(
        request.user_id, request.is_admin, request.message.get("scope_id")
    )
    if action in {"active", "end_active"}:
        continuity = async_get_continuity(hass, entry_id, subentry_id)
    if action == "active":
        _require_admin(is_admin)
        return {
            "active": await continuity.async_list(
                int(
                    subentry.data.get(
                        CONF_CONVERSATION_TIMEOUT_MINUTES,
                        DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
                    )
                )
            )
        }
    if action == "end_active":
        _require_admin(is_admin)
        continuity_key = message.get("continuity_key")
        if not isinstance(continuity_key, str):
            raise HomeAssistantError("continuity_key is required")
        ended = await continuity.async_end(continuity_key)
        if ended:
            function_groups = get_function_group_runtime(hass, entry_id, subentry_id)
            if function_groups is not None:
                function_groups.end(f"continuity:{continuity_key}")
            _reset_request_rule_runtime(hass, entry_id, subentry_id, continuity_key)
        return {"ended": int(ended)}
    archive = await async_get_archive(hass, entry_id, subentry_id)
    if action == "list":
        return await archive_list_page(
            archive,
            scope_id,
            limit=int(message.get("limit", 50)),
            offset=int(message.get("offset", 0)),
        )
    if action == "search":
        return await archive_search_page(
            archive,
            scope_id,
            str(message.get("query", "")),
            start_date=message.get("start_date"),
            end_date=message.get("end_date"),
            limit=int(message.get("limit", 20)),
            offset=int(message.get("offset", 0)),
        )
    if action == "get":
        return await archive_get_page(
            archive,
            scope_id,
            str(message.get("session_id", "")),
            start_turn=int(message.get("start_turn", 0)),
            limit=int(message.get("limit", 20)),
        )
    if action == "delete":
        return await archive.async_delete_session(
            scope_id, str(message.get("session_id", ""))
        )
    if action == "clear":
        return await archive.async_clear_scope(
            scope_id, confirm=message.get("confirm") is True
        )
    if action == "delete_range":
        return await archive.async_delete_date_range(
            scope_id,
            str(message.get("start_date", "")),
            str(message.get("end_date", "")),
            confirm=message.get("confirm") is True,
        )
    if action == "settings":
        return settings_snapshot(subentry.data)

    return _unknown_management_action(request)


async def _async_temporary_memories_command(
    request: _ManagementRequest,
) -> dict[str, Any]:
    """Manage complete Personal/Shared Temporary Memory without changing runtime retrieval."""
    from .temporary_memory import _valid_owner_scope_id

    hass, user_id, is_admin, message = (
        request.hass,
        request.user_id,
        request.is_admin,
        request.message,
    )
    action = message["action"]
    entry, subentry = request.entry, request.subentry
    selected_scope_id = _selected_scope(user_id, is_admin, message.get("scope_id"))
    owner = _valid_owner_scope_id(selected_scope_id)
    if owner is None:
        raise HomeAssistantError(
            "Temporary Memory can only be managed in Personal or Shared scopes"
        )
    manager = await async_get_temporary_memory(
        hass, entry.entry_id, subentry.subentry_id
    )
    manager_any: Any = manager
    if action == "temporary_list":
        records = await manager_any.async_list_owned(owner)
        return {
            "memories": [
                temporary_memory_as_dict(record, include_scope=True)
                | {"owner_scope_id": record.owner_scope_id}
                for record in records
            ],
            "scope_id": owner,
            "stats": manager.stats(),
        }
    if action == "temporary_clear":
        if message.get("confirm") is not True:
            raise HomeAssistantError("Explicit confirmation is required")
        records = await manager_any.async_list_owned(owner)
        deleted = 0
        for offset in range(0, len(records), 50):
            deleted += await manager_any.async_delete_owned(
                owner, [record.memory_id for record in records[offset : offset + 50]]
            )
        return {"deleted": deleted}
    memory_id = message.get("memory_id")
    if not isinstance(memory_id, str) or not memory_id:
        raise HomeAssistantError("memory_id is required")
    if action == "temporary_delete":
        deleted = await manager_any.async_delete_owned(owner, [memory_id])
        return {"deleted": deleted}
    content = message.get("content")
    category = message.get("category")
    expires_at = message.get("expires_at")
    if (
        (content is not None and not isinstance(content, str))
        or (category is not None and not isinstance(category, str))
        or (expires_at is not None and not isinstance(expires_at, str))
    ):
        raise HomeAssistantError(
            "content, category, and expires_at must be strings when supplied"
        )
    if content is None and category is None and expires_at is None:
        raise HomeAssistantError("at least one Temporary Memory field is required")
    try:
        record = await manager_any.async_update_owned(
            owner, memory_id, content, expires_at, category
        )
    except ValueError as err:
        raise HomeAssistantError(str(err)) from err
    return {
        "memory": temporary_memory_as_dict(record, include_scope=True)
        | {"owner_scope_id": record.owner_scope_id}
    }


async def async_memories_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the memories Management section."""
    hass = request.hass
    user_id = request.user_id
    is_admin = request.is_admin
    message = request.message
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    if action in {
        "temporary_list",
        "temporary_update",
        "temporary_delete",
        "temporary_clear",
    }:
        return await _async_temporary_memories_command(request)
    scope_id = _selected_scope(
        request.user_id, request.is_admin, request.message.get("scope_id")
    )
    if action.startswith("temporary_"):
        # Preserve initialization on unrecognized legacy temporary actions.
        await async_get_temporary_memory(hass, entry_id, subentry_id)
    memory = await async_get_memory(hass, entry_id, subentry_id)
    owner = _memory_scope(scope_id)
    if action in {"list", "search"}:
        return await async_browse_memories(
            memory, owner, scope_id, message, include_scope=is_admin
        )
    if action in {"add", "update"}:
        from .management_browser import management_memory_dict

        metadata = {
            field: message[field]
            for field in ("importance", "subject", "key", "valid_from")
            if field in message
        }
        target = scope_id
        if "target_scope_id" in message:
            target = _selected_scope(user_id, is_admin, message["target_scope_id"])
            if target != scope_id:
                if not (
                    (
                        scope_id.startswith("user:")
                        and target == SHARED_HOUSEHOLD_SCOPE_ID
                    )
                    or (
                        scope_id == SHARED_HOUSEHOLD_SCOPE_ID
                        and target.startswith("user:")
                    )
                ):
                    raise HomeAssistantError(
                        "Memory moves require Personal and Shared scopes"
                    )
                if (
                    target == SHARED_HOUSEHOLD_SCOPE_ID
                    and request.subentry.data.get(
                        CONF_SHARED_MEMORY_MODE, DEFAULT_SHARED_MEMORY_MODE
                    )
                    == SHARED_MEMORY_DISABLED
                ):
                    raise HomeAssistantError("Shared household memory is disabled")
        if action == "add":
            added = await memory.async_add(
                _memory_scope(target),
                str(message.get("content", "")),
                str(message.get("category", "general")),
                "explicit",
                **metadata,
            )
            records = await memory.async_get_many(
                [(_memory_scope(target), added["memory"]["memory_id"])],
                [_memory_scope(target)],
            )
            return {
                "status": added["status"],
                "scope_id": target,
                "memory": management_memory_dict(records[0], include_scope=is_admin),
            }
        refresh_confirmation = message.get("refresh_confirmation", False)
        if not isinstance(refresh_confirmation, bool):
            raise HomeAssistantError("refresh_confirmation must be true or false")
        record = await memory.async_update(
            owner,
            str(message.get("memory_id", "")),
            message.get("content"),
            message.get("category"),
            **metadata,
            target_user_id=_memory_scope(target),
            expected_revision=message.get("expected_revision"),
            clear_fields=message.get("clear_fields"),
            refresh_confirmation=refresh_confirmation,
        )
        return {
            "status": "updated",
            "scope_id": target,
            "memory": management_memory_dict(record, include_scope=is_admin),
        }
    if action == "delete":
        return {
            "deleted": await memory.async_delete(
                owner, [str(message.get("memory_id", ""))]
            )
        }
    if action == "clear":
        if message.get("confirm") is not True:
            raise HomeAssistantError("Explicit confirmation is required")
        return {"deleted": await memory.async_clear(owner, message.get("category"))}
    if action == "reassign_legacy":
        _require_admin(is_admin)
        target = _selected_scope(user_id, True, message.get("target_scope_id"))
        memory_ids = message.get("memory_ids")
        if not isinstance(memory_ids, list) or not all(
            isinstance(value, str) for value in memory_ids
        ):
            raise HomeAssistantError("memory_ids must be a list of strings")
        return await memory.async_reassign(
            ANONYMOUS_USER_ID, _memory_scope(target), memory_ids
        )

    return _unknown_management_action(request)


async def async_knowledge_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the knowledge Management section."""
    hass = request.hass
    message = request.message
    entry_id = request.entry_id
    subentry_id = request.subentry_id
    action = request.message["action"]
    _selected_scope(request.user_id, request.is_admin, request.message.get("scope_id"))
    library = await async_get_knowledge(hass, entry_id, subentry_id)
    if action == "list":
        from .feature_status import management_feature_status

        sources = await library.async_list()
        stats = library.stats()
        source_count = len(sources) if isinstance(sources, list) else 0
        if isinstance(stats, Mapping):
            with suppress(TypeError, ValueError):
                source_count = int(stats.get("source_count", source_count))
        return {
            "sources": sources,
            "stats": stats,
            "feature_status": management_feature_status(
                request.subentry.data, knowledge_source_count=source_count
            )["knowledge"],
        }
    if action == "set_enabled":
        from .feature_status import management_feature_status

        _require_admin(request.is_admin)
        enabled = message.get("enabled")
        if not isinstance(enabled, bool):
            raise HomeAssistantError("enabled must be a boolean")
        normalized = merge_agent_config(
            request.subentry.data, {CONF_KNOWLEDGE_ENABLED: enabled}
        )
        persisted = preserve_legacy_guest_policy(
            dict(request.subentry.data), deepcopy(normalized)
        )
        request.hass.config_entries.async_update_subentry(
            request.entry, request.subentry, data=persisted
        )
        stats = library.stats()
        source_count = 0
        if isinstance(stats, Mapping):
            with suppress(TypeError, ValueError):
                source_count = int(stats.get("source_count", 0))
        return {
            "revision": saved_agent_config_revision(
                request.subentry, persisted, request.subentry.title
            ),
            "knowledge_enabled": bool(persisted.get(CONF_KNOWLEDGE_ENABLED, False)),
            "feature_status": management_feature_status(
                persisted, knowledge_source_count=source_count
            )["knowledge"],
        }
    if action == "get":
        return {
            "source": knowledge_source_as_dict(
                await library.async_get(str(message.get("source_id", "")))
            )
        }
    if action == "create":
        from .feature_status import management_feature_status
        from .knowledge import source_summary

        knowledge_source = await library.async_create(
            message.get("title", ""),
            message.get("description", ""),
            message.get("content", ""),
            message.get("enabled", True),
        )
        return {
            "status": "created",
            "source": knowledge_source_as_dict(knowledge_source),
            "summary": source_summary(knowledge_source),
            "stats": library.stats(),
            "feature_status": management_feature_status(
                request.subentry.data, knowledge_source_count=library.total_source_count
            )["knowledge"],
        }
    if action == "update":
        from .feature_status import management_feature_status
        from .knowledge import source_summary

        knowledge_source = await library.async_update(
            str(message.get("source_id", "")),
            message.get("title"),
            message.get("description"),
            message.get("content"),
            message.get("enabled"),
            expected_revision=message.get("expected_revision"),
        )
        return {
            "status": "updated",
            "source": knowledge_source_as_dict(knowledge_source),
            "summary": source_summary(knowledge_source),
            "stats": library.stats(),
            "feature_status": management_feature_status(
                request.subentry.data, knowledge_source_count=library.total_source_count
            )["knowledge"],
        }
    if action == "delete":
        from .feature_status import management_feature_status

        if message.get("confirm") is not True:
            raise HomeAssistantError("Explicit confirmation is required")
        deleted = int(await library.async_delete(str(message.get("source_id", ""))))
        return {
            "deleted": deleted,
            "stats": library.stats(),
            "feature_status": management_feature_status(
                request.subentry.data, knowledge_source_count=library.total_source_count
            )["knowledge"],
        }

    return _unknown_management_action(request)


async def async_settings_command(request: _ManagementRequest) -> dict[str, Any]:
    """Handle the settings Management section."""
    hass = request.hass
    is_admin = request.is_admin
    message = request.message
    entry = request.entry
    subentry = request.subentry
    action = request.message["action"]
    _selected_scope(request.user_id, request.is_admin, request.message.get("scope_id"))
    if action == "update":
        _require_admin(is_admin)
        updates = message.get("settings")
        if not isinstance(updates, dict):
            raise HomeAssistantError("settings must be an object")
        normalized = _validate_settings(updates)
        hass.config_entries.async_update_subentry(
            entry, subentry, data={**subentry.data, **normalized}
        )
        return {"settings": settings_snapshot({**subentry.data, **normalized})}

    return _unknown_management_action(request)


async def async_overview_command(request: _ManagementRequest) -> dict[str, Any]:
    """Load and project the selected agent's bounded Overview."""
    from .management_loading_performance import (
        async_overview_detail,
        async_overview_primary,
        async_overview_summary,
    )

    action = request.message["action"]
    if action == "summary":
        return await async_overview_summary(
            request.hass, request.entry, request.subentry, is_admin=request.is_admin
        )
    if action == "primary":
        return await async_overview_primary(
            request.hass, request.entry, request.subentry, is_admin=request.is_admin
        )
    if action == "detail":
        kind = request.message.get("kind")
        if not isinstance(kind, str):
            raise HomeAssistantError("kind is required")
        return await async_overview_detail(
            request.hass,
            request.entry,
            request.subentry,
            is_admin=request.is_admin,
            kind=kind,
        )
    return _unknown_management_action(request)


async def async_function_repair_command(request: _ManagementRequest) -> dict[str, Any]:
    """Keep invalid Function Tool recovery at its existing dedicated boundary."""
    from .management_function_repair import async_function_repair

    return await async_function_repair(
        request.hass, request.user_id, request.is_admin, request.message
    )


_MANAGEMENT_SECTION_HANDLERS: Final = MappingProxyType(
    {
        "overview": async_overview_command,
        "function_repair": async_function_repair_command,
        "request_rules": async_request_rules_command,
        "guest_mode": async_guest_mode_command,
        "backup": async_backup_command,
        "configuration": async_configuration_command,
        "tools": async_tools_command,
        "scopes": async_scopes_command,
        "service_catalog": async_service_catalog_command,
        "diagnostics": async_diagnostics_command,
        "usage": async_usage_command,
        "conversations": async_conversations_command,
        "memories": async_memories_command,
        "knowledge": async_knowledge_command,
        "settings": async_settings_command,
    }
)


async def async_management_command(
    hass: HomeAssistant,
    user_id: str,
    is_admin: bool,
    message: dict[str, Any],
) -> dict[str, Any]:
    """Execute one Management request through a stable, explicitly owned pipeline.

    Keep maintenance outermost, including result projection. Integration-global
    authorization precedes agent lookup; section-specific authorization stays with
    its handler. No setup/feature installer replaces this function.
    """
    trace_configuration = message.get("section") == "configuration" and message.get(
        "action"
    ) in {"get", "save", "update"}
    started = perf_counter() if trace_configuration else None
    async with management_command_lease(hass, message):
        lease_ms = _elapsed_ms(started) if started is not None else None
        result = await _async_management_request(hass, user_id, is_admin, message)
    if started is not None and isinstance(result.get("_performance"), dict):
        result["_performance"]["maintenance_lease_ms"] = lease_ms
        result["_performance"]["command_total_ms"] = _elapsed_ms(started)
    return result


async def _async_management_request(
    hass: HomeAssistant,
    user_id: str,
    is_admin: bool,
    message: dict[str, Any],
) -> dict[str, Any]:
    """Validate, select one agent, route, then apply the shared result contract."""
    dispatch_started = perf_counter()
    section = message.get("section", "overview")
    action = message.get("action")
    if not isinstance(section, str) or not isinstance(action, str):
        raise HomeAssistantError("section and action must be strings")
    require_management_permission(is_admin, message)
    if section == "quiet_hours":
        return await async_quiet_hours_command(hass, is_admin, message)
    if action == "agents":
        from .management_loading_performance import async_agent_catalog

        return await async_agent_catalog(hass, user_id, is_admin)

    entry_id, subentry_id = message.get("entry_id"), message.get("subentry_id")
    if section == "memories" and action in {
        "temporary_list",
        "temporary_update",
        "temporary_delete",
        "temporary_clear",
    }:
        # Keep the Temporary Memory API's existing per-field validation errors.
        for key in ("entry_id", "subentry_id"):
            value = message.get(key)
            if not isinstance(value, str) or not value:
                raise HomeAssistantError(f"{key} is required")
    if not isinstance(entry_id, str) or not isinstance(subentry_id, str):
        raise HomeAssistantError("entry_id and subentry_id are required")
    resolution_started = perf_counter()
    entry, subentry = entry_and_agent(hass, entry_id, subentry_id)
    resolution_ms = _elapsed_ms(resolution_started)
    request = _ManagementRequest(
        hass, user_id, is_admin, message, entry_id, subentry_id, entry, subentry
    )
    handler = _MANAGEMENT_SECTION_HANDLERS.get(section)
    if handler is None:
        return _unknown_management_action(request)
    # These safety checks were outside the former quarantine/permission wrappers.
    # Non-admin requests reach their section's authorization error, never schema
    # or revision diagnostics. Rule mutations retain strict dependency validation.
    if is_admin and section == "request_rules" and action in {"create", "update"}:
        rule_id = message.get("rule_id") if action == "update" else None
        candidate = _prepare_request_rule(
            message.get("rule"), rule_id if isinstance(rule_id, str) else None
        )
        await async_validate_request_rule_functions(
            hass, candidate, _strict_configured_function_tools(subentry.data)
        )
    elif is_admin and section == "tools" and action in _TOOL_MUTATIONS:
        _require_agent_config_revision(subentry, message.get("revision"))

    configuration_action = _configuration_action(message)
    configuration_started = perf_counter() if configuration_action is not None else None
    dispatch_ms = _elapsed_ms(dispatch_started)
    with management_function_tools(section):
        result = await handler(request)
    handler_ms = (
        _elapsed_ms(configuration_started)
        if configuration_started is not None
        else None
    )

    if configuration_action is not None:
        decoration_started = perf_counter()
        result = decorate_configuration_result(
            hass,
            getattr(entry, "data", {}),
            result,
            action=configuration_action,
        )
        decoration_ms = _elapsed_ms(decoration_started)
        assert configuration_started is not None
        request_total_ms = _elapsed_ms(configuration_started)
        performance = result.get("_performance")
        if isinstance(performance, dict):
            performance["decoration_ms"] = decoration_ms
            performance["request_total_ms"] = request_total_ms
            if section == "configuration" and action in {"get", "save", "update"}:
                performance["dispatch_ms"] = dispatch_ms
                performance["agent_resolution_ms"] = resolution_ms
                performance["handler_ms"] = handler_ms
    return result


def _validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper using the shared agent configuration contract."""
    try:
        normalized = merge_agent_config({}, settings)
    except AgentConfigError as err:
        if err.field == "config":
            raise HomeAssistantError(
                str(err).replace("config: unknown fields", "Unknown settings")
            ) from err
        raise
    return {key: normalized[key] for key in settings}


@websocket_api.websocket_command(
    {
        vol.Required("type"): WS_COMMAND,
        vol.Required("action"): str,
        vol.Optional("section"): str,
        vol.Optional("entry_id"): str,
        vol.Optional("subentry_id"): str,
        vol.Optional("scope_id"): str,
        vol.Optional("scope_kind"): vol.In(["all", "archive", "memory", "temporary"]),
        vol.Optional("target_scope_id"): str,
        vol.Optional("temporary_scope_id"): str,
        vol.Optional("continuity_key"): str,
        vol.Optional("settings"): dict,
        vol.Optional("config"): dict,
        vol.Optional("tools"): vol.Any(str, list),
        vol.Optional("tool"): dict,
        vol.Optional("group"): dict,
        vol.Optional("name"): str,
        vol.Optional("original_name"): str,
        vol.Optional("original_id"): str,
        vol.Optional("group_id"): vol.Any(None, str),
        vol.Optional("rule_id"): str,
        vol.Optional("rule"): dict,
        vol.Optional("revision"): str,
        vol.Optional("direction"): str,
        vol.Optional("pin"): str,
        vol.Optional("pin_repeat"): str,
        vol.Optional("text"): str,
        vol.Optional("defaults"): dict,
        vol.Optional("wording_groups"): list,
        vol.Optional("groups"): list,
        vol.Optional("selection"): str,
        vol.Optional("rule_ids"): vol.Any(None, list),
        vol.Optional("pack"): vol.Any(str, dict),
        vol.Optional("review_token"): str,
        vol.Optional("enabled"): bool,
        vol.Optional("yaml"): str,
        vol.Optional("document"): vol.Any(str, dict),
        vol.Optional("sample_text"): str,
        vol.Optional("mode"): str,
        vol.Optional("importance"): vol.In(["low", "normal", "high"]),
        vol.Optional("subject"): str,
        vol.Optional("key"): str,
        vol.Optional("valid_from"): str,
        vol.Optional("clear_fields"): [vol.In(["subject", "key", "valid_from"])],
        vol.Optional("expected_revision"): str,
        vol.Optional("refresh_confirmation"): bool,
        vol.Optional("memory_ids"): list,
        vol.Optional("metadata_keys"): list,
        vol.Optional("memory_id"): str,
        vol.Optional("session_id"): str,
        vol.Optional("source_id"): str,
        vol.Optional("run_id"): str,
        vol.Optional("query"): str,
        vol.Optional("content"): str,
        vol.Optional("title"): str,
        vol.Optional("description"): str,
        vol.Optional("category"): str,
        vol.Optional("kind"): str,
        vol.Optional("start_date"): str,
        vol.Optional("end_date"): str,
        vol.Optional("active_from"): str,
        vol.Optional("active_until"): str,
        vol.Optional("indefinite"): bool,
        vol.Optional("limit"): int,
        vol.Optional("offset"): int,
        vol.Optional("start_turn"): int,
        vol.Optional("successful"): bool,
        vol.Optional("confirm"): bool,
    }
)
@websocket_api.async_response
async def websocket_management(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    started = perf_counter()
    try:
        result = await async_management_command(
            hass, connection.user.id, connection.user.is_admin, msg
        )
    except (HomeAssistantError, RuntimeError, ValueError) as err:
        connection.send_error(msg["id"], "invalid_request", str(err))
        return
    trace_configuration = msg.get("section") == "configuration" and msg.get(
        "action"
    ) in {"get", "save", "update"}
    if trace_configuration:
        performance = result.get("_performance")
        if isinstance(performance, dict):
            performance["websocket_pre_send_ms"] = _elapsed_ms(started)
    send_started = perf_counter() if trace_configuration else None
    connection.send_result(msg["id"], result)
    # Home Assistant serializes inside send_result. Its duration cannot be
    # inserted into a payload that has already been serialized.
    if send_started is not None and _PERFORMANCE_LOGGER.isEnabledFor(logging.DEBUG):
        _PERFORMANCE_LOGGER.debug(
            "configuration/%s send_result_ms=%.2f websocket_total_ms=%.2f",
            msg["action"],
            _elapsed_ms(send_started),
            _elapsed_ms(started),
        )


async def async_setup_management_ui(hass: HomeAssistant) -> None:
    """Register the bundled Management panel through the shared production assets."""
    setup_key = _UI_SETUP
    if hass.data.get(setup_key):
        return

    static_key = f"{setup_key}.static_paths"
    websocket_key = f"{setup_key}.websocket"
    panel_key = f"{setup_key}.panel"

    if not hass.data.get(static_key):
        await async_register_frontend_assets(hass)
        hass.data[static_key] = True
    if not hass.data.get(websocket_key):
        websocket_api.async_register_command(hass, websocket_management)
        hass.data[websocket_key] = True
    if not hass.data.get(panel_key):
        await panel_custom.async_register_panel(
            hass,
            webcomponent_name="extended-openai-management-panel",
            frontend_url_path=MANAGEMENT_PANEL_URL,
            module_url=frontend_entry_url(hass, "management"),
            sidebar_title=MANAGEMENT_PANEL_TITLE,
            sidebar_icon="mdi:robot-outline",
            require_admin=False,
        )
        hass.data[panel_key] = True

    hass.data[setup_key] = True
