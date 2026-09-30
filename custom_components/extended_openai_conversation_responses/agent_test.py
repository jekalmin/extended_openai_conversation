"""Safe, minimal conversation-agent configuration test."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

from openai import OpenAIError
import yaml

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant

from .agent_config import configured_function_tools_from_data, validate_function_groups
from .const import (
    API_MODE_AUTO,
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_API_PROVIDER,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_SKILLS,
    CONF_WEB_SEARCH,
    DEFAULT_API_MODE,
    DEFAULT_API_PROVIDER,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTION_TOOLS,
    DEFAULT_FUNCTION_GROUPS,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_WEB_SEARCH,
)
from .functions import get_function
from .guest_mode import get_loaded_guest_mode, resolve_guest_policy
from .ha_llm_tools import is_ha_tool, validate_reference
from .helpers import get_api_mode, get_exposed_entities, supports_openai_hosted_tools
from .memory import async_get_memory, memory_enabled
from .model_capabilities import capability_allowed, model_capability_snapshot
from .model_catalog import model_metadata
from .provider_errors import (
    classify_config_provider_error,
    ensure_successful_responses_result,
    provider_user_message,
    request_reauthentication,
)
from .skill_availability import skill_loader_status
from .skills import SkillManager
from .usage import async_get_usage, extract_usage


@dataclass(slots=True)
class TestCheck:
    """One human-readable agent test check."""

    name: str
    status: str
    message: str


@dataclass(slots=True)
class AgentTestResult:
    """Structured overall test result."""

    status: str
    checks: list[TestCheck]
    authentication_rejected: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe response for flows and WebSocket clients."""
        return {
            "status": self.status,
            "checks": [asdict(check) for check in self.checks],
            "authentication_rejected": self.authentication_rejected,
        }

    def as_text(self) -> str:
        """Return a concise result suitable for a native Home Assistant form."""
        return "\n".join(
            [f"Overall: {self.status}"]
            + [
                f"{check.name}: {check.status} — {check.message}"
                for check in self.checks
            ]
        )


def _check(name: str, status: str, message: str) -> TestCheck:
    return TestCheck(name=name, status=status, message=message)


def _overall(checks: list[TestCheck]) -> str:
    if any(check.status == "Failed" for check in checks):
        return "Failed"
    if any(check.status == "Warning" for check in checks):
        return "Warning"
    return "Passed"


def _validate_function_schema(subentry: ConfigSubentry) -> int:
    """Validate configured tool schemas without executing any tool."""
    configured = subentry.data.get(CONF_FUNCTION_TOOLS)
    tools = yaml.safe_load(configured) if configured else DEFAULT_CONF_FUNCTION_TOOLS
    for tool in tools or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
            raise ValueError("Each function tool must contain a function mapping")
        function_config = cast(dict[str, Any], tool["function"])
        if is_ha_tool(tool):
            validate_reference(function_config)
            continue
        get_function(function_config["type"]).validate_schema(function_config)
    return len(tools or [])


async def async_test_agent(
    hass: HomeAssistant,
    entry: ConfigEntry,
    subentry: ConfigSubentry,
) -> AgentTestResult:
    """Test one agent with local checks and at most one minimal live request."""
    checks: list[TestCheck] = []
    authentication_rejected = False
    client = getattr(entry, "runtime_data", None)
    if client is None:
        checks.append(_check("Authentication", "Failed", "API client is unavailable"))
        return AgentTestResult(_overall(checks), checks)
    checks.append(_check("Authentication", "Passed", "API client is available"))

    model = subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
    configured_mode = subentry.data.get(CONF_API_MODE, DEFAULT_API_MODE)
    if configured_mode not in {
        API_MODE_AUTO,
        API_MODE_CHAT_COMPLETIONS,
        API_MODE_RESPONSES,
    }:
        checks.append(
            _check("API mode", "Failed", f"Unsupported mode: {configured_mode}")
        )
        return AgentTestResult(_overall(checks), checks)
    metadata = model_metadata(model)
    with model_capability_snapshot(model, metadata):
        api_mode = get_api_mode(configured_mode, model)
    checks.append(_check("API mode", "Passed", api_mode.replace("_", " ").title()))
    usage_provider = str(entry.data.get(CONF_API_PROVIDER, DEFAULT_API_PROVIDER))
    usage_model = str(model)
    usage_api_mode = str(api_mode)

    try:
        function_count = _validate_function_schema(subentry)
    except Exception as err:
        checks.append(_check("Configuration", "Failed", str(err)))
        return AgentTestResult(_overall(checks), checks)
    checks.append(
        _check("Configuration", "Passed", f"{function_count} tool schemas valid")
    )
    guest_mode = get_loaded_guest_mode(hass, entry.entry_id, subentry.subentry_id)
    guest_status = (
        guest_mode.status() if guest_mode is not None else {"state": "inactive"}
    )
    configured_tools = configured_function_tools_from_data(subentry.data)
    guest_policy = resolve_guest_policy(
        hass,
        subentry.data,
        guest_mode,
        configured_tools,
    )
    policy = guest_policy.as_diagnostics()
    checks.append(
        _check(
            "Guest Mode",
            "Passed",
            (
                f"{str(guest_status['state']).replace('_', ' ').title()}; "
                f"{policy['readable_entity_count'] or 0} visible entities; "
                f"{policy['configured_tool_count'] or 0} custom tools"
                if guest_policy.guest_active
                else "Inactive"
            ),
        )
    )

    try:
        entity_count = len(get_exposed_entities(hass))
    except Exception:
        entity_count = 0
    checks.append(
        _check(
            "Exposed entities",
            "Passed" if entity_count else "Warning",
            str(entity_count),
        )
    )

    selected_skills = list(subentry.data.get(CONF_SKILLS, []) or [])
    if not selected_skills:
        checks.append(_check("Skills", "Passed", "Disabled"))
    else:
        try:
            groups = validate_function_groups(
                subentry.data.get(CONF_FUNCTION_GROUPS, DEFAULT_FUNCTION_GROUPS),
                configured_tools,
            )
            loader_status = skill_loader_status(
                selected_skills,
                configured_tools,
                groups,
                max_function_calls=int(
                    subentry.data.get(
                        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                        DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                    )
                ),
            )
            if not loader_status.available:
                checks.append(
                    _check(
                        "Skills",
                        "Failed",
                        loader_status.reason or "Selected Skills are not loadable",
                    )
                )
            else:
                skill_manager = await SkillManager.async_get_instance(hass)
                installed_names = {
                    skill.name for skill in skill_manager.get_all_skills()
                }
                missing = sorted(set(selected_skills) - installed_names)
                if missing:
                    checks.append(
                        _check(
                            "Skills",
                            "Failed",
                            "Selected but not installed: " + ", ".join(missing),
                        )
                    )
                else:
                    location = (
                        f" through on-demand group `{loader_status.group_id}`"
                        if loader_status.on_demand and loader_status.group_id
                        else ""
                    )
                    checks.append(
                        _check(
                            "Skills",
                            "Passed",
                            f"{len(selected_skills)} enabled and loadable{location}",
                        )
                    )
        except Exception as err:
            checks.append(_check("Skills", "Failed", str(err)))

    if memory_enabled(subentry.data):
        try:
            memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
            checks.append(
                _check(
                    "Persistent memory",
                    "Passed",
                    f"Available ({memory.stats()['memory_count']} stored)",
                )
            )
        except Exception as err:
            checks.append(_check("Persistent memory", "Failed", type(err).__name__))
    else:
        checks.append(_check("Persistent memory", "Passed", "Disabled"))

    web_search = subentry.data.get(CONF_WEB_SEARCH, DEFAULT_WEB_SEARCH)
    with model_capability_snapshot(model, metadata):
        web_search_compatible = (
            api_mode == API_MODE_RESPONSES
            and capability_allowed(
                model,
                "web_search",
                api_mode,
                effort=subentry.data.get("reasoning_effort")
                or metadata.get("recommended_profile", {}).get("reasoning_effort"),
            )
            and supports_openai_hosted_tools(
                entry.data.get(CONF_API_PROVIDER), entry.data.get(CONF_BASE_URL)
            )
        )
    if web_search and not web_search_compatible:
        checks.append(
            _check(
                "Web Search",
                "Failed",
                "Enabled, but this API mode or provider does not support hosted OpenAI Web Search",
            )
        )
    elif not web_search:
        checks.append(_check("Web Search", "Passed", "Disabled"))

    noop_tool = {
        "type": "function",
        "name": "configuration_test_noop",
        "description": "Schema-only compatibility test; never execute this function.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    }
    usage_manager = await async_get_usage(hass, entry.entry_id, subentry.subentry_id)
    try:
        if api_mode == API_MODE_RESPONSES:
            tools: list[dict[str, Any]] = [noop_tool]
            if web_search and web_search_compatible:
                tools.insert(0, {"type": "web_search", "search_context_size": "low"})
            response = await client.responses.create(
                model=model,
                input=[{"role": "user", "content": "Reply OK."}],
                max_output_tokens=16,
                store=False,
                tools=tools,
                tool_choice="none",
            )
            ensure_successful_responses_result(response)
        else:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": "Reply OK."}],
                "stream": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            key: value
                            for key, value in noop_tool.items()
                            if key != "type"
                        },
                    }
                ],
                "tool_choice": "none",
                "max_completion_tokens": 16,
            }
            response = await client.chat.completions.create(**kwargs)
    except OpenAIError as err:
        is_authentication_failure = (
            classify_config_provider_error(err) == "invalid_auth"
        )
        if is_authentication_failure:
            # Authentication recovery is more important than diagnostics bookkeeping.
            # Request it first so a secondary usage-storage failure cannot suppress reauth.
            request_reauthentication(hass, entry, err)
            authentication_rejected = True
            await usage_manager.async_record_request(
                successful=False,
                provider=usage_provider,
                model=usage_model,
                api_mode=usage_api_mode,
            )
            authentication = next(
                check for check in checks if check.name == "Authentication"
            )
            authentication.status = "Failed"
            authentication.message = provider_user_message(err)
            checks.append(_check("Model access", "Failed", "Authentication rejected"))
            checks.append(_check("Function calling", "Failed", "Probe was rejected"))
        else:
            await usage_manager.async_record_request(
                successful=False,
                provider=usage_provider,
                model=usage_model,
                api_mode=usage_api_mode,
            )
            message = provider_user_message(err)
            checks.append(_check("Model access", "Failed", message))
            checks.append(_check("Function calling", "Failed", "Probe was rejected"))
            if web_search and web_search_compatible:
                checks.append(_check("Web Search", "Failed", message))
    except Exception as err:
        await usage_manager.async_record_request(
            successful=False,
            provider=usage_provider,
            model=usage_model,
            api_mode=usage_api_mode,
        )
        checks.append(_check("Model access", "Failed", str(err)))
        checks.append(_check("Function calling", "Failed", "Probe was rejected"))
        if web_search and web_search_compatible:
            checks.append(_check("Web Search", "Failed", str(err)))
    else:
        await usage_manager.async_record_request(
            successful=True,
            usage=extract_usage(getattr(response, "usage", None)),
            provider=usage_provider,
            model=usage_model,
            api_mode=usage_api_mode,
        )
        checks.append(
            _check("Model access", "Passed", f"Minimal {model} request succeeded")
        )
        checks.append(_check("Function calling", "Passed", "Function schema accepted"))
        if web_search and web_search_compatible:
            checks.append(_check("Web Search", "Passed", "Hosted tool schema accepted"))

    return AgentTestResult(
        _overall(checks), checks, authentication_rejected=authentication_rejected
    )
