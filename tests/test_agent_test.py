"""Tests for the safe Test agent action."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from openai import AuthenticationError, OpenAIError
import pytest
import yaml

from custom_components.extended_openai_conversation_responses import agent_test
from custom_components.extended_openai_conversation_responses.agent_test import (
    async_test_agent,
)
from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_MEMORY_MODE,
    CONF_WEB_SEARCH,
    MEMORY_MODE_MANUAL,
    MEMORY_MODE_OFF,
)


def _objects(options: dict | None = None):
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Assistant",
        data={
            CONF_CHAT_MODEL: "gpt-4.1-mini",
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_FUNCTION_TOOLS: "[]",
            CONF_MEMORY_MODE: MEMORY_MODE_OFF,
            **(options or {}),
        },
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1, total_tokens=5),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        data={},
        runtime_data=client,
    )
    hass = MagicMock()
    usage = SimpleNamespace(async_record_request=AsyncMock())
    return hass, entry, subentry, client, usage


def _usage_attribution(
    *,
    provider: str = agent_test.DEFAULT_API_PROVIDER,
    model: str = "gpt-4.1-mini",
    api_mode: str = API_MODE_CHAT_COMPLETIONS,
) -> dict[str, str]:
    return {"provider": provider, "model": model, "api_mode": api_mode}


async def _run(options: dict | None = None, *, exposed: int = 1):
    hass, entry, subentry, client, usage = _objects(options)
    entities = [SimpleNamespace()] * exposed
    with (
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.get_exposed_entities",
            return_value=entities,
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_usage",
            AsyncMock(return_value=usage),
        ),
    ):
        result = await async_test_agent(hass, entry, subentry)
    return result, client, usage


async def test_successful_agent_test() -> None:
    result, client, usage = await _run()

    assert result.status == "Passed"
    assert result.authentication_rejected is False
    assert result.as_dict()["authentication_rejected"] is False
    assert {check.name for check in result.checks} >= {
        "Authentication",
        "Model access",
        "API mode",
        "Function calling",
        "Web Search",
        "Persistent memory",
        "Exposed entities",
        "Skills",
        "Configuration",
    }
    client.chat.completions.create.assert_awaited_once()
    usage.async_record_request.assert_awaited_once_with(
        successful=True,
        usage=agent_test.extract_usage(client.chat.completions.create.return_value.usage),
        **_usage_attribution(),
    )


async def test_invalid_authentication() -> None:
    hass, entry, subentry, client, usage = _objects()
    request = MagicMock()
    response = MagicMock(status_code=401, request=request)
    authentication_error = AuthenticationError(
        "invalid key", response=response, body=None
    )
    client.chat.completions.create.side_effect = authentication_error
    reauth = MagicMock(return_value=True)
    with (
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.get_exposed_entities",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_usage",
            AsyncMock(return_value=usage),
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.request_reauthentication",
            reauth,
        ),
    ):
        result = await async_test_agent(hass, entry, subentry)

    assert result.status == "Failed"
    assert result.authentication_rejected is True
    assert result.as_dict()["authentication_rejected"] is True
    assert any(
        check.name == "Authentication" and check.status == "Failed"
        for check in result.checks
    )
    reauth.assert_called_once_with(hass, entry, authentication_error)


async def test_generic_openai_http_401_uses_runtime_reauth_rule() -> None:
    class Generic401Error(OpenAIError):
        status_code = 401

    hass, entry, subentry, client, usage = _objects()
    entry.async_start_reauth = MagicMock()
    generic_401 = Generic401Error("generic authentication rejection")
    client.chat.completions.create.side_effect = generic_401

    with (
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.get_exposed_entities",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_usage",
            AsyncMock(return_value=usage),
        ),
    ):
        result = await async_test_agent(hass, entry, subentry)

    assert result.status == "Failed"
    assert result.authentication_rejected is True
    entry.async_start_reauth.assert_called_once_with(hass)
    authentication = next(
        check for check in result.checks if check.name == "Authentication"
    )
    assert authentication.status == "Failed"
    assert any(
        check.name == "Model access" and check.message == "Authentication rejected"
        for check in result.checks
    )


async def test_reauthentication_is_requested_before_usage_recording() -> None:
    hass, entry, subentry, client, usage = _objects()
    response = MagicMock(status_code=401, request=MagicMock())
    authentication_error = AuthenticationError(
        "invalid key", response=response, body=None
    )
    client.chat.completions.create.side_effect = authentication_error
    usage.async_record_request.side_effect = RuntimeError("usage unavailable")
    reauth = MagicMock(return_value=True)

    with (
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.get_exposed_entities",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_usage",
            AsyncMock(return_value=usage),
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.request_reauthentication",
            reauth,
        ),
        pytest.raises(RuntimeError, match="usage unavailable"),
    ):
        await async_test_agent(hass, entry, subentry)

    reauth.assert_called_once_with(hass, entry, authentication_error)


async def test_unsupported_api_mode_stops_before_probe() -> None:
    hass, entry, subentry, client, _ = _objects({CONF_API_MODE: "unknown"})
    result = await async_test_agent(hass, entry, subentry)

    assert result.status == "Failed"
    assert result.authentication_rejected is False
    assert result.checks[-1].name == "API mode"
    client.chat.completions.create.assert_not_awaited()


async def test_web_search_incompatibility_is_specific_failure() -> None:
    result, _, _ = await _run({CONF_WEB_SEARCH: True})

    assert result.status == "Failed"
    web = next(check for check in result.checks if check.name == "Web Search")
    assert web.status == "Failed"
    assert "does not support" in web.message


async def test_memory_unavailable_is_reported() -> None:
    hass, entry, subentry, _, usage = _objects({CONF_MEMORY_MODE: MEMORY_MODE_MANUAL})
    with (
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.get_exposed_entities",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_usage",
            AsyncMock(return_value=usage),
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_memory",
            AsyncMock(side_effect=RuntimeError("storage unavailable")),
        ),
    ):
        result = await async_test_agent(hass, entry, subentry)

    memory = next(check for check in result.checks if check.name == "Persistent memory")
    assert result.status == "Failed"
    assert memory.status == "Failed"


async def test_no_exposed_entities_produces_partial_warning() -> None:
    result, _, _ = await _run(exposed=0)

    assert result.status == "Warning"
    entities = next(
        check for check in result.checks if check.name == "Exposed entities"
    )
    assert entities.status == "Warning"
    assert entities.message == "0"


async def test_responses_failed_status_is_reported() -> None:
    hass, entry, subentry, client, usage = _objects(
        {CONF_API_MODE: API_MODE_RESPONSES, CONF_CHAT_MODEL: "gpt-5.6"}
    )
    client.responses = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                id="resp_failed",
                status="failed",
                error=SimpleNamespace(
                    message="provider failed",
                    code="server_error",
                    type="server_error",
                ),
                usage=None,
            )
        )
    )
    with (
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.get_exposed_entities",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_usage",
            AsyncMock(return_value=usage),
        ),
    ):
        result = await async_test_agent(hass, entry, subentry)
    model = next(check for check in result.checks if check.name == "Model access")
    assert result.status == "Failed"
    assert result.authentication_rejected is False
    assert model.status == "Failed"
    assert "server_error" in model.message
    usage.async_record_request.assert_awaited_once_with(
        successful=False,
        **_usage_attribution(model="gpt-5.6", api_mode=API_MODE_RESPONSES),
    )


async def test_responses_incomplete_status_is_reported() -> None:
    hass, entry, subentry, client, usage = _objects(
        {CONF_API_MODE: API_MODE_RESPONSES, CONF_CHAT_MODEL: "gpt-5.6"}
    )
    client.responses = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                id="resp_incomplete",
                status="incomplete",
                error=None,
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                usage=None,
            )
        )
    )
    with (
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.get_exposed_entities",
            return_value=[SimpleNamespace()],
        ),
        patch(
            "custom_components.extended_openai_conversation_responses.agent_test.async_get_usage",
            AsyncMock(return_value=usage),
        ),
    ):
        result = await async_test_agent(hass, entry, subentry)
    model = next(check for check in result.checks if check.name == "Model access")
    assert result.status == "Failed"
    assert result.authentication_rejected is False
    assert model.status == "Failed"
    assert "max_output_tokens" in model.message
    usage.async_record_request.assert_awaited_once_with(
        successful=False,
        **_usage_attribution(model="gpt-5.6", api_mode=API_MODE_RESPONSES),
    )


class _CoverageUsage:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_record_request(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _CoverageCreate:
    def __init__(self, *, result: Any = None, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _coverage_entry(client: Any, **data: Any) -> SimpleNamespace:
    return SimpleNamespace(entry_id="entry-1", runtime_data=client, data=data)


def _coverage_subentry(**data: Any) -> SimpleNamespace:
    return SimpleNamespace(subentry_id="agent-1", data=data)


def _coverage_patch_common(monkeypatch: pytest.MonkeyPatch, usage: _CoverageUsage | None = None) -> _CoverageUsage:
    usage = usage or _CoverageUsage()
    monkeypatch.setattr(
        agent_test,
        "get_api_mode",
        lambda *_args: agent_test.API_MODE_RESPONSES,
    )
    monkeypatch.setattr(agent_test, "_validate_function_schema", lambda _subentry: 2)
    monkeypatch.setattr(agent_test, "get_loaded_guest_mode", lambda *args: None)
    monkeypatch.setattr(
        agent_test,
        "resolve_guest_policy",
        lambda *args: SimpleNamespace(
            guest_active=False,
            as_diagnostics=lambda: {
                "readable_entity_count": 0,
                "configured_tool_count": 0,
            },
        ),
    )
    monkeypatch.setattr(agent_test, "configured_function_tools_from_data", lambda _data: [])
    monkeypatch.setattr(agent_test, "get_exposed_entities", lambda _hass: [{"entity_id": "light.kitchen"}])
    monkeypatch.setattr(agent_test, "memory_enabled", lambda _data: False)
    monkeypatch.setattr(agent_test, "supports_openai_hosted_tools", lambda *_args: False)

    async def get_usage(*_args: Any) -> _CoverageUsage:
        return usage

    monkeypatch.setattr(agent_test, "async_get_usage", get_usage)
    return usage


def _coverage_checks(result: agent_test.AgentTestResult) -> dict[str, agent_test.TestCheck]:
    return {check.name: check for check in result.checks}


def test_result_helpers_cover_warning_failure_and_rendering() -> None:
    checks = [
        agent_test.TestCheck("One", "Passed", "ok"),
        agent_test.TestCheck("Two", "Warning", "careful"),
    ]
    result = agent_test.AgentTestResult(agent_test._overall(checks), checks)

    assert result.status == "Warning"
    assert result.as_dict() == {
        "status": "Warning",
        "checks": [
            {"name": "One", "status": "Passed", "message": "ok"},
            {"name": "Two", "status": "Warning", "message": "careful"},
        ],
        "authentication_rejected": False,
    }
    assert "Overall: Warning" in result.as_text()
    assert "Two: Warning — careful" in result.as_text()
    assert agent_test._overall([agent_test._check("x", "Failed", "bad")]) == "Failed"
    assert agent_test._overall([agent_test._check("x", "Passed", "ok")]) == "Passed"


@pytest.mark.asyncio
async def test_agent_test_fails_fast_without_client(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(None)),
        cast(Any, _coverage_subentry()),
    )

    assert result.status == "Failed"
    assert _coverage_checks(result)["Authentication"].message == "API client is unavailable"


@pytest.mark.asyncio
async def test_agent_test_rejects_unknown_api_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(SimpleNamespace())),
        cast(Any, _coverage_subentry(api_mode="future_mode")),
    )

    assert result.status == "Failed"
    assert _coverage_checks(result)["API mode"].message == "Unsupported mode: future_mode"


@pytest.mark.asyncio
async def test_agent_test_reports_configuration_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_test,
        "_validate_function_schema",
        lambda _subentry: (_ for _ in ()).throw(ValueError("bad tool schema")),
    )

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(SimpleNamespace())),
        cast(Any, _coverage_subentry()),
    )

    assert result.status == "Failed"
    assert _coverage_checks(result)["Configuration"].message == "bad tool schema"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loader", "installed", "expected_status", "expected_text"),
    [
        (SimpleNamespace(available=False, reason="loader disabled"), None, "Failed", "loader disabled"),
        (
            SimpleNamespace(available=True, on_demand=False, group_id=None),
            ["other"],
            "Failed",
            "Selected but not installed: weather",
        ),
        (
            SimpleNamespace(available=True, on_demand=True, group_id="skills"),
            ["weather"],
            "Passed",
            "1 enabled and loadable through on-demand group `skills`",
        ),
    ],
)
async def test_agent_test_skill_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    loader: Any,
    installed: list[str] | None,
    expected_status: str,
    expected_text: str,
) -> None:
    _coverage_patch_common(monkeypatch)
    monkeypatch.setattr(agent_test, "validate_function_groups", lambda *_args: {})
    monkeypatch.setattr(agent_test, "skill_loader_status", lambda *_args, **_kwargs: loader)

    if installed is not None:
        manager = SimpleNamespace(
            get_all_skills=lambda: [SimpleNamespace(name=name) for name in installed]
        )

        async def get_instance(_hass: Any) -> Any:
            return manager

        monkeypatch.setattr(agent_test.SkillManager, "async_get_instance", get_instance)

    create = _CoverageCreate(result=SimpleNamespace(usage=None))
    client = SimpleNamespace(
        responses=SimpleNamespace(create=create.create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create.create)),
    )
    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client)),
        cast(Any, _coverage_subentry(skills=["weather"])),
    )

    check = _coverage_checks(result)["Skills"]
    assert check.status == expected_status
    assert check.message == expected_text


@pytest.mark.asyncio
async def test_agent_test_reports_skill_and_memory_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _coverage_patch_common(monkeypatch)
    monkeypatch.setattr(
        agent_test,
        "validate_function_groups",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("groups broken")),
    )
    monkeypatch.setattr(agent_test, "memory_enabled", lambda _data: True)

    async def fail_memory(*_args: Any) -> Any:
        raise RuntimeError("memory broken")

    monkeypatch.setattr(agent_test, "async_get_memory", fail_memory)
    create = _CoverageCreate(result=SimpleNamespace(usage=None))
    client = SimpleNamespace(responses=SimpleNamespace(create=create.create))

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client)),
        cast(Any, _coverage_subentry(skills=["weather"])),
    )

    checks = _coverage_checks(result)
    assert checks["Skills"].message == "groups broken"
    assert checks["Persistent memory"].status == "Failed"
    assert checks["Persistent memory"].message == "RuntimeError"


@pytest.mark.asyncio
async def test_agent_test_flags_incompatible_web_search_without_probe_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _coverage_patch_common(monkeypatch)
    create = _CoverageCreate(result=SimpleNamespace(usage=None))
    client = SimpleNamespace(responses=SimpleNamespace(create=create.create))

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client, api_provider="other")),
        cast(Any, _coverage_subentry(web_search=True)),
    )

    check = _coverage_checks(result)["Web Search"]
    assert check.status == "Failed"
    assert "does not support" in check.message


@pytest.mark.asyncio
async def test_agent_test_authentication_failure_requests_reauth_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = _coverage_patch_common(monkeypatch)
    error = OpenAIError("bad auth")
    create = _CoverageCreate(error=error)
    client = SimpleNamespace(responses=SimpleNamespace(create=create.create))
    calls: list[str] = []

    monkeypatch.setattr(agent_test, "classify_config_provider_error", lambda _err: "invalid_auth")
    monkeypatch.setattr(agent_test, "provider_user_message", lambda _err: "Authentication failed")
    monkeypatch.setattr(agent_test, "request_reauthentication", lambda *_args: calls.append("reauth"))

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client)),
        cast(Any, _coverage_subentry()),
    )

    checks = _coverage_checks(result)
    assert calls == ["reauth"]
    assert result.authentication_rejected is True
    assert checks["Authentication"].status == "Failed"
    assert checks["Authentication"].message == "Authentication failed"
    assert checks["Model access"].message == "Authentication rejected"
    assert usage.calls == [
        {
            "successful": False,
            **_usage_attribution(
                model=agent_test.DEFAULT_CHAT_MODEL,
                api_mode=agent_test.API_MODE_RESPONSES,
            ),
        }
    ]


@pytest.mark.asyncio
async def test_agent_test_non_auth_provider_error_and_web_search_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = _coverage_patch_common(monkeypatch)
    monkeypatch.setattr(agent_test, "supports_openai_hosted_tools", lambda *_args: True)
    monkeypatch.setattr(agent_test, "classify_config_provider_error", lambda _err: "rate_limit")
    monkeypatch.setattr(agent_test, "provider_user_message", lambda _err: "Rate limited")
    error = OpenAIError("slow down")
    create = _CoverageCreate(error=error)
    client = SimpleNamespace(responses=SimpleNamespace(create=create.create))

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client)),
        cast(Any, _coverage_subentry(web_search=True, reasoning_effort="low")),
    )

    checks = _coverage_checks(result)
    assert result.authentication_rejected is False
    assert checks["Model access"].message == "Rate limited"
    assert checks["Function calling"].message == "Probe was rejected"
    assert checks["Web Search"].message == "Rate limited"
    assert usage.calls == [
        {
            "successful": False,
            **_usage_attribution(
                model=agent_test.DEFAULT_CHAT_MODEL,
                api_mode=agent_test.API_MODE_RESPONSES,
            ),
        }
    ]


@pytest.mark.asyncio
async def test_agent_test_unexpected_probe_error_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = _coverage_patch_common(monkeypatch)
    create = _CoverageCreate(error=RuntimeError("transport exploded"))
    client = SimpleNamespace(responses=SimpleNamespace(create=create.create))

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client)),
        cast(Any, _coverage_subentry()),
    )

    checks = _coverage_checks(result)
    assert checks["Model access"].message == "transport exploded"
    assert checks["Function calling"].status == "Failed"
    assert usage.calls == [
        {
            "successful": False,
            **_usage_attribution(
                model=agent_test.DEFAULT_CHAT_MODEL,
                api_mode=agent_test.API_MODE_RESPONSES,
            ),
        }
    ]


@pytest.mark.asyncio
async def test_agent_test_successful_responses_probe_records_usage_and_web_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = _coverage_patch_common(monkeypatch)
    monkeypatch.setattr(agent_test, "supports_openai_hosted_tools", lambda *_args: True)
    monkeypatch.setattr(agent_test, "ensure_successful_responses_result", lambda response: None)
    monkeypatch.setattr(agent_test, "extract_usage", lambda raw: {"input_tokens": 3} if raw == "usage" else {})
    create = _CoverageCreate(result=SimpleNamespace(usage="usage"))
    client = SimpleNamespace(responses=SimpleNamespace(create=create.create))

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client)),
        cast(Any, _coverage_subentry(web_search=True, reasoning_effort="low")),
    )

    checks = _coverage_checks(result)
    assert checks["Model access"].status == "Passed"
    assert checks["Function calling"].status == "Passed"
    assert checks["Web Search"].status == "Passed"
    assert create.calls[0]["tools"][0]["type"] == "web_search"
    assert create.calls[0]["tool_choice"] == "none"
    assert usage.calls == [
        {
            "successful": True,
            "usage": {"input_tokens": 3},
            **_usage_attribution(
                model=agent_test.DEFAULT_CHAT_MODEL,
                api_mode=agent_test.API_MODE_RESPONSES,
            ),
        }
    ]


@pytest.mark.asyncio
async def test_agent_test_usage_attribution_uses_configured_provider_model_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = _coverage_patch_common(monkeypatch)
    monkeypatch.setattr(
        agent_test,
        "get_api_mode",
        lambda *_args: agent_test.API_MODE_CHAT_COMPLETIONS,
    )
    monkeypatch.setattr(agent_test, "extract_usage", lambda _raw: {})
    create = _CoverageCreate(result=SimpleNamespace(usage=None))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create.create))
    )

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client, api_provider="azure")),
        cast(
            Any,
            _coverage_subentry(
                api_mode=agent_test.API_MODE_CHAT_COMPLETIONS,
                chat_model="diagnostic-model",
            ),
        ),
    )

    assert result.status == "Passed"
    assert usage.calls == [
        {
            "successful": True,
            "usage": {},
            "provider": "azure",
            "model": "diagnostic-model",
            "api_mode": agent_test.API_MODE_CHAT_COMPLETIONS,
        }
    ]


@pytest.mark.asyncio
async def test_agent_test_chat_completions_probe_uses_wrapped_function_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _coverage_patch_common(monkeypatch)
    monkeypatch.setattr(agent_test, "get_api_mode", lambda *_args: agent_test.API_MODE_CHAT_COMPLETIONS)
    monkeypatch.setattr(agent_test, "extract_usage", lambda _raw: {})
    create = _CoverageCreate(result=SimpleNamespace(usage=None))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create.create)))

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()),
        cast(Any, _coverage_entry(client)),
        cast(Any, _coverage_subentry(api_mode=agent_test.API_MODE_CHAT_COMPLETIONS)),
    )

    assert result.status == "Passed"
    kwargs = create.calls[0]
    assert kwargs["stream"] is False
    assert kwargs["tools"][0]["type"] == "function"
    assert kwargs["tools"][0]["function"]["name"] == "configuration_test_noop"
    assert "type" not in kwargs["tools"][0]["function"]


class _ResidualUsage:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def async_record_request(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _ResidualCreate:
    def __init__(self, *, result: Any = None, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _residual_subentry_with_tools(tools: list[dict[str, Any]]) -> Any:
    return SimpleNamespace(data={agent_test.CONF_FUNCTION_TOOLS: yaml.safe_dump(tools)})


def test_validate_function_schema_rejects_missing_function_mapping() -> None:
    subentry = SimpleNamespace(
        data={agent_test.CONF_FUNCTION_TOOLS: yaml.safe_dump([{"type": "function"}])}
    )

    with pytest.raises(ValueError, match="function mapping"):
        agent_test._validate_function_schema(cast(Any, subentry))


def test_validate_function_schema_dispatches_ha_tool_to_reference_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function_config = {"name": "ha_tool", "type": "native"}
    subentry = _residual_subentry_with_tools([{"type": "function", "function": function_config}])
    validate = MagicMock()

    monkeypatch.setattr(agent_test, "is_ha_tool", lambda _tool: True)
    monkeypatch.setattr(agent_test, "validate_reference", validate)
    monkeypatch.setattr(
        agent_test,
        "get_function",
        lambda _kind: pytest.fail("HA tools must not use custom function validation"),
    )

    assert agent_test._validate_function_schema(cast(Any, subentry)) == 1
    validate.assert_called_once_with(function_config)


def test_validate_function_schema_dispatches_custom_tool_by_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function_config = {"name": "custom", "type": "template"}
    subentry = _residual_subentry_with_tools([{"type": "function", "function": function_config}])
    validate = MagicMock()
    get_function = MagicMock(return_value=SimpleNamespace(validate_schema=validate))

    monkeypatch.setattr(agent_test, "is_ha_tool", lambda _tool: False)
    monkeypatch.setattr(agent_test, "get_function", get_function)

    assert agent_test._validate_function_schema(cast(Any, subentry)) == 1
    get_function.assert_called_once_with("template")
    validate.assert_called_once_with(function_config)


async def _residual_run_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    web_search: bool = False,
    probe_error: BaseException | None = None,
    memory_is_enabled: bool = False,
) -> tuple[agent_test.AgentTestResult, _ResidualUsage]:
    usage = _ResidualUsage()
    create = _ResidualCreate(result=SimpleNamespace(usage=None), error=probe_error)
    client = SimpleNamespace(responses=SimpleNamespace(create=create.create))
    entry = SimpleNamespace(entry_id="entry-1", runtime_data=client, data={})
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        data={
            agent_test.CONF_API_MODE: agent_test.API_MODE_RESPONSES,
            agent_test.CONF_CHAT_MODEL: "gpt-5.6",
            agent_test.CONF_FUNCTION_TOOLS: "[]",
            agent_test.CONF_WEB_SEARCH: web_search,
        },
    )

    monkeypatch.setattr(agent_test, "_validate_function_schema", lambda _subentry: 0)
    monkeypatch.setattr(agent_test, "get_api_mode", lambda *_args: agent_test.API_MODE_RESPONSES)
    monkeypatch.setattr(
        agent_test,
        "get_loaded_guest_mode",
        lambda *_args: SimpleNamespace(status=lambda: {"state": "active"}),
    )
    monkeypatch.setattr(agent_test, "configured_function_tools_from_data", lambda _data: [])
    monkeypatch.setattr(
        agent_test,
        "resolve_guest_policy",
        lambda *_args: SimpleNamespace(
            guest_active=True,
            as_diagnostics=lambda: {
                "readable_entity_count": 3,
                "configured_tool_count": 2,
            },
        ),
    )
    monkeypatch.setattr(
        agent_test,
        "get_exposed_entities",
        lambda _hass: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )
    monkeypatch.setattr(agent_test, "memory_enabled", lambda _data: memory_is_enabled)
    monkeypatch.setattr(agent_test, "supports_openai_hosted_tools", lambda *_args: True)
    monkeypatch.setattr(agent_test, "ensure_successful_responses_result", lambda _response: None)
    monkeypatch.setattr(agent_test, "extract_usage", lambda _usage: {})

    if memory_is_enabled:
        async def get_memory(*_args: Any) -> Any:
            return SimpleNamespace(stats=lambda: {"memory_count": 4})

        monkeypatch.setattr(agent_test, "async_get_memory", get_memory)

    async def get_usage(*_args: Any) -> _ResidualUsage:
        return usage

    monkeypatch.setattr(agent_test, "async_get_usage", get_usage)

    result = await agent_test.async_test_agent(
        cast(Any, SimpleNamespace()), cast(Any, entry), cast(Any, subentry)
    )
    return result, usage


@pytest.mark.asyncio
async def test_guest_active_message_and_exposed_entity_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _usage = await _residual_run_agent(monkeypatch)
    checks = {check.name: check for check in result.checks}

    assert checks["Guest Mode"].status == "Passed"
    assert checks["Guest Mode"].message == "Active; 3 visible entities; 2 custom tools"
    assert checks["Exposed entities"].status == "Warning"
    assert checks["Exposed entities"].message == "0"


@pytest.mark.asyncio
async def test_successful_memory_check_reports_stored_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _usage = await _residual_run_agent(monkeypatch, memory_is_enabled=True)
    memory = next(check for check in result.checks if check.name == "Persistent memory")

    assert memory.status == "Passed"
    assert memory.message == "Available (4 stored)"


@pytest.mark.asyncio
async def test_unexpected_probe_error_marks_compatible_web_search_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, usage = await _residual_run_agent(
        monkeypatch,
        web_search=True,
        probe_error=RuntimeError("probe transport failed"),
    )
    checks = {check.name: check for check in result.checks}

    assert checks["Model access"].status == "Failed"
    assert checks["Web Search"].status == "Failed"
    assert checks["Web Search"].message == "probe transport failed"
    assert usage.calls == [
        {
            "successful": False,
            **_usage_attribution(
                model="gpt-5.6",
                api_mode=agent_test.API_MODE_RESPONSES,
            ),
        }
    ]
