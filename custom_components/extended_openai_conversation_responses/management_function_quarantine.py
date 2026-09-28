"""Keep unrelated management surfaces usable while Function Tools need repair."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import yaml

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .agent_config import (
    configured_function_tools_from_data as _STRICT_CONFIGURED_TOOLS,
    merge_agent_config as _STRICT_MERGE_AGENT_CONFIG,
    preserve_legacy_guest_policy,
    validate_function_groups as _STRICT_VALIDATE_FUNCTION_GROUPS,
)
from .agent_test import (
    AgentTestResult,
    TestCheck,
    _overall,
    async_test_agent as async_test_configured_agent,
)
from .const import CONF_FUNCTION_GROUPS, CONF_FUNCTION_TOOLS, DEFAULT_FUNCTION_GROUPS
from .live_subentry_updates import update_live_subentry
from .management_function_repair import (
    editable_function_tools,
    function_tools_issue,
    isolated_function_tools,
    persist_valid_function_configuration,
    repair_revision,
    safe_function_configuration as _safe_function_configuration,
)

_ALLOW_QUARANTINED_TOOLS: ContextVar[bool] = ContextVar(
    "extended_openai_management_allow_quarantined_tools", default=False
)
_QUARANTINED_FUNCTION_NAMES: ContextVar[frozenset[str]] = ContextVar(
    "extended_openai_management_quarantined_function_names", default=frozenset()
)


def _usable_function_tools(data: Any) -> list[dict[str, Any]]:
    """Return valid siblings when persisted Function Tools contain repairable errors."""
    raw = dict(data)
    tools, issue = function_tools_issue(raw)
    if issue is not None:
        _valid, invalid, _isolated_issue = isolated_function_tools(raw)
        _QUARANTINED_FUNCTION_NAMES.set(
            frozenset(
                str(item["name"])
                for item in invalid
                if isinstance(item.get("name"), str) and item["name"]
            )
        )
        return tools
    _QUARANTINED_FUNCTION_NAMES.set(frozenset())
    return _STRICT_CONFIGURED_TOOLS(data)


def _management_configured_tools(data: Any) -> list[dict[str, Any]]:
    """Use tolerant Function Tool loading only inside isolated management sections."""
    if _ALLOW_QUARANTINED_TOOLS.get():
        return _usable_function_tools(data)
    return _STRICT_CONFIGURED_TOOLS(data)


def _management_validate_function_groups(
    value: Any, function_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Ignore quarantined members without weakening normal group validation."""
    if not _ALLOW_QUARANTINED_TOOLS.get():
        return _STRICT_VALIDATE_FUNCTION_GROUPS(value, function_tools)
    quarantined = _QUARANTINED_FUNCTION_NAMES.get()
    if not quarantined:
        return _STRICT_VALIDATE_FUNCTION_GROUPS(value, function_tools)
    safe = deepcopy(value)
    if isinstance(safe, list):
        for group in safe:
            if not isinstance(group, dict) or not isinstance(
                group.get("functions"), list
            ):
                continue
            group["functions"] = [
                name for name in group["functions"] if name not in quarantined
            ]
    return _STRICT_VALIDATE_FUNCTION_GROUPS(safe, function_tools)


def _management_merge_agent_config(
    source: Any, updates: dict[str, Any]
) -> dict[str, Any]:
    """Persist unrelated section edits without rewriting repair-owned Function fields."""
    if not _ALLOW_QUARANTINED_TOOLS.get():
        return _STRICT_MERGE_AGENT_CONFIG(source, updates)

    raw = dict(source)
    _tools, issue = function_tools_issue(raw)
    if issue is None:
        return _STRICT_MERGE_AGENT_CONFIG(source, updates)

    normalized = _STRICT_MERGE_AGENT_CONFIG(_safe_function_configuration(raw), updates)
    for key in (CONF_FUNCTION_TOOLS, CONF_FUNCTION_GROUPS):
        if key in raw:
            normalized[key] = deepcopy(raw[key])
        else:
            normalized.pop(key, None)
    return normalized


def _restore_quarantined_group_members(
    groups: list[dict[str, Any]], raw_groups: Any, quarantined: frozenset[str]
) -> list[dict[str, Any]]:
    """Retain hidden invalid members in persisted groups while editing valid siblings."""
    restored = deepcopy(groups)
    if not quarantined or not isinstance(raw_groups, list):
        return restored
    by_id = {
        group.get("id"): group
        for group in raw_groups
        if isinstance(group, dict) and isinstance(group.get("functions"), list)
    }
    for group in restored:
        if not isinstance(group, dict) or not isinstance(group.get("functions"), list):
            continue
        original = by_id.get(group.get("id"))
        if not isinstance(original, dict):
            continue
        hidden = [
            name
            for name in original.get("functions", [])
            if isinstance(name, str) and name in quarantined
        ]
        for name in hidden:
            if name not in group["functions"]:
                group["functions"].append(name)
    return restored


def _tolerant_persist_function_configuration(
    hass: HomeAssistant,
    entry: Any,
    subentry: Any,
    tools: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    *,
    extra_updates: dict[str, Any] | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Persist edits to valid siblings while retaining quarantined raw tools."""
    raw = dict(subentry.data)
    _valid, invalid, issue = isolated_function_tools(raw)
    if issue is None or not invalid:
        return persist_valid_function_configuration(
            hass,
            entry,
            subentry,
            tools,
            groups,
            extra_updates=extra_updates,
            expected_revision=expected_revision,
        )

    if expected_revision is not None and expected_revision != repair_revision(subentry):
        raise HomeAssistantError(
            "Configuration changed in another tab. Reload the latest saved settings before saving."
        )

    invalid_names = frozenset(
        str(item["name"])
        for item in invalid
        if isinstance(item.get("name"), str) and item["name"]
    )
    valid_names = {
        str(tool.get("spec", {}).get("name"))
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("spec"), dict)
    }
    duplicate = sorted(name for name in invalid_names if name in valid_names)
    if duplicate:
        raise HomeAssistantError(
            f"Function Tool {duplicate[0]} already exists as a quarantined tool"
        )

    editable = editable_function_tools(raw)
    if not isinstance(editable, list):
        raise HomeAssistantError("Saved Function Tools cannot be isolated safely")
    invalid_indices = {
        int(item["index"]) for item in invalid if isinstance(item.get("index"), int)
    }
    quarantined_raw = [
        deepcopy(candidate)
        for index, candidate in enumerate(editable)
        if index in invalid_indices
    ]
    persisted_groups = _restore_quarantined_group_members(
        groups,
        raw.get(CONF_FUNCTION_GROUPS, DEFAULT_FUNCTION_GROUPS),
        invalid_names,
    )
    persisted = dict(raw)
    persisted[CONF_FUNCTION_TOOLS] = yaml.safe_dump(
        [*deepcopy(tools), *quarantined_raw],
        sort_keys=False,
        allow_unicode=True,
    )
    persisted[CONF_FUNCTION_GROUPS] = persisted_groups
    if extra_updates:
        persisted.update(deepcopy(extra_updates))
    persisted = preserve_legacy_guest_policy(raw, persisted)
    update_live_subentry(hass, entry, subentry, data=persisted)
    return {
        "functions": deepcopy(tools),
        "function_groups": deepcopy(groups),
        "revision": repair_revision(subentry),
    }


async def _tolerant_agent_test(
    hass: HomeAssistant, entry: Any, subentry: Any
) -> AgentTestResult:
    """Run provider diagnostics with valid Function Tool siblings only."""
    raw = dict(subentry.data)
    _valid, invalid, issue = isolated_function_tools(raw)
    if issue is None:
        return await async_test_configured_agent(hass, entry, subentry)

    safe_subentry = SimpleNamespace(
        data=_safe_function_configuration(raw),
        subentry_id=subentry.subentry_id,
        title=getattr(subentry, "title", ""),
    )
    result = await async_test_configured_agent(hass, entry, cast(Any, safe_subentry))
    if invalid:
        names = [
            str(item.get("name") or f"tool {int(item.get('index', 0)) + 1}")
            for item in invalid
        ]
        detail = (
            f"{len(invalid)} invalid Function Tool"
            f"{'s' if len(invalid) != 1 else ''} quarantined: "
            + ", ".join(names[:5])
            + ("…" if len(names) > 5 else "")
        )
    else:
        detail = "Invalid Function Tool configuration was quarantined for this test"
    result.checks.append(TestCheck("Function Tools", "Warning", detail))
    result.status = _overall(result.checks)
    return result


@contextmanager
def management_function_tools(section: str) -> Iterator[None]:
    """Limit tolerant reads to the three existing repair-aware Management sections."""
    if section not in {"request_rules", "guest_mode", "tools"}:
        yield
        return
    token = _ALLOW_QUARANTINED_TOOLS.set(True)
    names_token = _QUARANTINED_FUNCTION_NAMES.set(frozenset())
    try:
        yield
    finally:
        _QUARANTINED_FUNCTION_NAMES.reset(names_token)
        _ALLOW_QUARANTINED_TOOLS.reset(token)
