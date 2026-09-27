"""Execution-time resolution for configured Function Tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from homeassistant.exceptions import HomeAssistantError

from .agent_config import function_tool_enabled
from .const import CONF_FUNCTION_GROUPS, DEFAULT_FUNCTION_GROUPS
from .exceptions import FunctionNotFound
from .function_tool_quarantine import (
    _runtime_validate_function_groups as validate_function_groups,
)

# These definitions are owned by the integration runtime rather than the user's
# persisted Function Tool catalogue. Their request-round objects are authoritative.
_INTEGRATION_TOOL_TYPES = frozenset(
    {
        "guest_mode",
        "memory",
        "temporary_memory",
        "knowledge",
        "archive",
        "conversation_lifecycle",
        "function_group_loader",
    }
)


def current_configuration_data(agent: Any) -> Any:
    """Return the live subentry mapping, retaining its object identity."""
    entry_id = getattr(getattr(agent, "entry", None), "entry_id", None)
    subentry = getattr(agent, "subentry", None)
    subentry_id = getattr(subentry, "subentry_id", None)
    if entry_id is None or subentry_id is None:
        return getattr(subentry, "data", None)
    getter = getattr(
        getattr(agent.hass, "config_entries", None), "async_get_entry", None
    )
    latest_entry = getter(entry_id) if callable(getter) else None
    latest_subentry = (
        latest_entry.subentries.get(subentry_id) if latest_entry is not None else None
    )
    return (
        latest_subentry.data
        if latest_subentry is not None
        else getattr(subentry, "data", None)
    )


def configured_function_tool_for_execution(
    agent: Any,
    tool_name: str,
) -> dict[str, Any]:
    """Return the current configured tool after live availability checks."""
    resolver = getattr(agent, "_configured_function_tools_from_data", None)
    if not callable(resolver):
        raise FunctionNotFound(tool_name)

    latest_data = current_configuration_data(agent)
    current_tools = resolver(latest_data)
    current_tool = next(
        (
            candidate
            for candidate in current_tools
            if candidate.get("spec", {}).get("name") == tool_name
        ),
        None,
    )
    if not isinstance(current_tool, dict):
        raise FunctionNotFound(tool_name)
    if not function_tool_enabled(current_tool):
        raise HomeAssistantError(f"Function Tool `{tool_name}` is disabled")

    current_groups = validate_function_groups(
        latest_data.get(CONF_FUNCTION_GROUPS, list(DEFAULT_FUNCTION_GROUPS)),
        current_tools,
    )
    current_group = next(
        (group for group in current_groups if tool_name in group.get("functions", [])),
        None,
    )
    if current_group is not None and current_group.get("enabled", True) is not True:
        raise HomeAssistantError(
            f"Function Tool `{tool_name}` is unavailable because Function Group "
            f"`{current_group['id']}` is disabled"
        )
    return current_tool


def latest_function_tool_for_execution(
    agent: Any,
    function_tool: dict[str, Any],
) -> dict[str, Any]:
    """Reject changed configured definitions before executing a provider call.

    The provider generated arguments against the advertised request snapshot. A
    later edit cannot silently substitute a different schema or implementation.
    Integration-owned runtime definitions remain authoritative for that round.
    """
    function = function_tool.get("function")
    if (
        not isinstance(function, Mapping)
        or function.get("type") in _INTEGRATION_TOOL_TYPES
    ):
        return function_tool

    tool_name = function_tool.get("spec", {}).get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return function_tool

    resolver = getattr(agent, "_configured_function_tools_from_data", None)
    if not callable(resolver):
        return function_tool

    current_tool = configured_function_tool_for_execution(agent, tool_name)
    # HA owns the callable's live schema. A stable saved reference keeps the
    # request-round projection, while the live HA tool validates its arguments.
    if function.get("type") == "ha_llm":
        if current_tool.get("function") != function:
            raise FunctionNotFound(tool_name)
        return function_tool
    if current_tool != function_tool:
        raise FunctionNotFound(tool_name)
    return function_tool
