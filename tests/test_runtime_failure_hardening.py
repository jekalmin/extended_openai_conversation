"""Regression tests for runtime failure containment."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from openai import OpenAIError
import httpx

import pytest

from custom_components.extended_openai_conversation_responses import runtime_failure_hardening as hardening
from custom_components.extended_openai_conversation_responses.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation_responses.entity import (
    ExtendedOpenAIBaseLLMEntity,
)
from homeassistant.exceptions import HomeAssistantError


class FakeUsage:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def mark_current_run_failed(self, error_type: str) -> None:
        self.failed.append(error_type)


class FakeConversationEntity:
    def __init__(self) -> None:
        self._usage = FakeUsage()
        self.events: list[tuple[str, str | None]] = []

    def _get_exposed_entities(self):
        raise HomeAssistantError("broken request preparation")

    def _fire_conversation_finished(
        self, _user_input, _chat_log, *, status: str, error_type: str | None = None
    ) -> None:
        self.events.append((status, error_type))


class FakeArchiveEntity:
    _async_dispatch_function_tool = (
        ExtendedOpenAIAgentEntity._async_dispatch_function_tool
    )

    def _effective_guest_policy(self):
        return SimpleNamespace(guest_active=False)

    async def _async_execute_archive_tool(self, _operation: str, _arguments: dict):
        raise OSError("archive store unavailable")

    def _tool_result(self, _tool_input, result):
        return result


@pytest.mark.asyncio
async def test_request_preparation_home_assistant_error_returns_assist_error() -> None:
    entity = FakeConversationEntity()
    user_input = SimpleNamespace(
        language="en",
        conversation_id="conversation-id",
        as_llm_context=lambda _domain: SimpleNamespace(),
    )

    result = await ExtendedOpenAIAgentEntity._async_handle_message(
        entity, user_input, SimpleNamespace()
    )

    assert result.conversation_id == "conversation-id"
    assert entity._usage.failed == ["HomeAssistantError"]
    assert entity.events == [("error", "HomeAssistantError")]


@pytest.mark.asyncio
async def test_unexpected_archive_failure_is_labeled_as_archive() -> None:
    entity = FakeArchiveEntity()
    tool = {"function": {"type": "archive", "operation": "search"}}
    tool_input = SimpleNamespace(id="call", tool_name="archive_search", tool_args={})

    result = await ExtendedOpenAIAgentEntity._execute_function_tool(
        entity, tool, tool_input, None, []
    )

    assert result == {
        "status": "unavailable",
        "error": "Conversation Archive is temporarily unavailable",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_id,expected_id", [(None, "call_late_123"), ("original", "original")]
)
async def test_chat_stream_repairs_tool_call_id_received_in_later_delta(
    initial_id, expected_id
) -> None:
    entity = SimpleNamespace(subentry=SimpleNamespace(data={}))
    chat_log = SimpleNamespace(async_trace=lambda _trace: None)

    async def stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=initial_id,
                                function=SimpleNamespace(
                                    name="late_id_tool", arguments='{"value":'
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_late_123",
                                function=SimpleNamespace(name=None, arguments="1}"),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )

    output = [
        item
        async for item in ExtendedOpenAIBaseLLMEntity._transform_chat_stream(
            entity, chat_log, stream()
        )
    ]
    tool_payload = next(item for item in output if item.get("tool_calls"))
    tool_call = tool_payload["tool_calls"][0]

    assert tool_call.id == expected_id
    assert tool_call.tool_name == "late_id_tool"
    assert tool_call.tool_args == json.loads('{"value":1}')


class _Usage:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def mark_current_run_failed(self, error_type: str) -> None:
        self.failed.append(error_type)


class _ConversationEntity:
    def __init__(self) -> None:
        self.hass = SimpleNamespace()
        self.entry = SimpleNamespace()
        self._usage = _Usage()
        self.finished: list[tuple[str, str | None]] = []

    def _fire_conversation_finished(
        self, _user_input, _chat_log, *, status: str, error_type: str | None = None
    ) -> None:
        self.finished.append((status, error_type))


class _ArchiveEntity:
    _async_dispatch_function_tool = (
        ExtendedOpenAIAgentEntity._async_dispatch_function_tool
    )

    def __init__(self, *, guest_active: bool = False, allowed: bool = True) -> None:
        self.guest_active = guest_active
        self.allowed = allowed
        self.archive_calls: list[tuple[str, dict]] = []

    def _effective_guest_policy(self):
        return SimpleNamespace(guest_active=self.guest_active)

    def _guest_integration_allowed(self, integration: str, operation: str) -> bool:
        assert integration == "archive"
        return self.allowed

    async def _async_execute_archive_tool(self, operation: str, arguments: dict):
        self.archive_calls.append((operation, arguments))
        return {"status": "ok"}

    def _tool_result(self, _tool_input, result):
        return result


def test_openai_conversation_error_uses_provider_failure_path(monkeypatch) -> None:
    entity = _ConversationEntity()
    user_input = SimpleNamespace(language="en", conversation_id="conversation-1")
    error = OpenAIError("provider unavailable")
    reauth = Mock()
    record = Mock()
    log = Mock()
    monkeypatch.setattr(hardening, "request_reauthentication", reauth)
    monkeypatch.setattr(hardening, "record_current_provider_failure", record)
    monkeypatch.setattr(hardening, "log_provider_failure", log)
    monkeypatch.setattr(hardening, "provider_user_message", lambda _err: "try again")

    result = hardening._conversation_error_result(
        entity, user_input, SimpleNamespace(), error
    )

    assert result.conversation_id == "conversation-1"
    assert entity._usage.failed == ["OpenAIError"]
    assert entity.finished == [("error", "OpenAIError")]
    reauth.assert_called_once_with(entity.hass, entity.entry, error)
    record.assert_called_once_with(error)
    log.assert_called_once()


def test_conversation_error_helper_preserves_phase_logger_and_reset_id(
    monkeypatch,
) -> None:
    entity = _ConversationEntity()
    user_input = SimpleNamespace(language="en", conversation_id="conversation-1")
    error = OpenAIError("provider unavailable")
    log = Mock()
    logger = Mock()
    monkeypatch.setattr(hardening, "request_reauthentication", Mock())
    monkeypatch.setattr(hardening, "record_current_provider_failure", Mock())
    monkeypatch.setattr(hardening, "log_provider_failure", log)
    monkeypatch.setattr(hardening, "provider_user_message", lambda _err: "try again")

    result = hardening._conversation_error_result(
        entity,
        user_input,
        SimpleNamespace(),
        error,
        logger=logger,
        provider_log_message="OpenAI conversation request failed",
        conversation_id=None,
    )

    assert result.conversation_id is None
    log.assert_called_once_with(logger, "OpenAI conversation request failed", error)
    assert entity.finished == [("error", "OpenAIError")]


async def test_archive_wrapper_delegates_non_archive_and_blocks_disallowed_guest(
    monkeypatch,
) -> None:
    original = AsyncMock(return_value="delegated")
    original._extended_openai_archive_failure_label = False
    installed = ExtendedOpenAIAgentEntity._execute_function_tool
    assert ExtendedOpenAIAgentEntity._execute_function_tool is installed

    entity = _ArchiveEntity(guest_active=True, allowed=False)
    tool_input = SimpleNamespace(tool_args={"query": "hello"})

    blocked = await installed(
        entity,
        {"function": {"type": "archive", "operation": "search"}},
        tool_input,
        None,
        [],
    )

    assert blocked == {
        "status": "error",
        "error": "This capability is unavailable in Guest Mode.",
    }
    assert entity.archive_calls == []


async def test_archive_wrapper_maps_value_error_without_mislabeling_as_unavailable(
    monkeypatch,
) -> None:
    original = AsyncMock(return_value="unused")
    original._extended_openai_archive_failure_label = False

    entity = _ArchiveEntity()

    async def invalid_archive(_operation: str, _arguments: dict):
        raise ValueError("bad archive arguments")

    entity._async_execute_archive_tool = invalid_archive
    result = await ExtendedOpenAIAgentEntity._execute_function_tool(
        entity,
        {"function": {"type": "archive", "operation": "search"}},
        SimpleNamespace(tool_args={}),
        None,
        [],
    )

    assert result == {"status": "error", "error": "bad archive arguments"}


def test_httpx_request_error_uses_provider_transport_path_without_usage_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finished: list[tuple[str, str | None]] = []
    entity = SimpleNamespace(
        hass=SimpleNamespace(),
        entry=SimpleNamespace(),
        _fire_conversation_finished=lambda _input, _log, *, status, error_type=None: finished.append(
            (status, error_type)
        ),
    )
    user_input = SimpleNamespace(language="en", conversation_id="conversation-httpx")
    error = httpx.ConnectError(
        "offline",
        request=httpx.Request("POST", "https://api.openai.invalid/v1/responses"),
    )
    reauth = Mock()
    record = Mock()
    log = Mock()
    monkeypatch.setattr(hardening, "request_reauthentication", reauth)
    monkeypatch.setattr(hardening, "record_current_provider_failure", record)
    monkeypatch.setattr(hardening, "log_provider_failure", log)
    monkeypatch.setattr(hardening, "provider_user_message", lambda _err: "temporarily unavailable")

    result = hardening._conversation_error_result(
        entity,
        user_input,
        SimpleNamespace(),
        error,
        provider_log_message="provider stream failed",
    )

    assert result.conversation_id == "conversation-httpx"
    assert finished == [("error", "ProviderTransportError")]
    converted = reauth.call_args.args[2]
    assert isinstance(converted, hardening.ProviderTransportError)
    assert isinstance(converted.__cause__, httpx.ConnectError)
    record.assert_called_once_with(converted)
    log.assert_called_once_with(
        hardening._LOGGER,
        "provider stream failed",
        converted,
    )


def test_home_assistant_error_uses_generic_path_without_provider_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _ConversationEntity()
    error = HomeAssistantError("local configuration invalid")
    logger = Mock()
    reauth = Mock()
    record = Mock()
    provider_log = Mock()
    monkeypatch.setattr(hardening, "request_reauthentication", reauth)
    monkeypatch.setattr(hardening, "record_current_provider_failure", record)
    monkeypatch.setattr(hardening, "log_provider_failure", provider_log)

    result = hardening._conversation_error_result(
        entity,
        SimpleNamespace(language="en", conversation_id="local-id"),
        SimpleNamespace(),
        error,
        logger=logger,
    )

    assert result.conversation_id == "local-id"
    assert entity._usage.failed == ["HomeAssistantError"]
    assert entity.finished == [("error", "HomeAssistantError")]
    reauth.assert_not_called()
    record.assert_not_called()
    provider_log.assert_not_called()
    logger.error.assert_called_once()
