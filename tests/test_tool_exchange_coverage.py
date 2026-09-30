"""Focused residual coverage for protocol-safe Function Tool exchanges."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.extended_openai_conversation_responses import tool_exchange
from custom_components.extended_openai_conversation_responses.exceptions import (
    FunctionNotFound,
)
from custom_components.extended_openai_conversation_responses.function_call_budget import (
    FunctionCallBudget,
)
from custom_components.extended_openai_conversation_responses.function_tool_recovery import (
    CorrectableToolFailure,
    ToolRecoveryState,
)
from custom_components.extended_openai_conversation_responses.ha_tool_result_compat import (
    tool_result_data,
)
from homeassistant.exceptions import HomeAssistantError


@dataclass
class FakeAssistantContent:
    tool_calls: list[object] | None = None


@dataclass
class FakeToolResultContent:
    agent_id: str
    tool_call_id: str
    tool_name: str
    tool_result: object


class FakeChatLog:
    def __init__(self, content=None) -> None:
        self.content = list(content or [])
        self.added: list[object] = []

    def async_add_assistant_content_without_tools(self, content) -> None:
        self.added.append(content)
        self.content.append(content)


def _call(call_id: str, name: str = "demo", args=None):
    return SimpleNamespace(
        id=call_id,
        tool_name=name,
        tool_args={} if args is None else args,
        external=True,
    )


def _tool(name: str = "demo", **extra):
    value = {"spec": {"name": name}, "function": {"type": "script"}}
    value.update(extra)
    return value


def _patch_content_types(monkeypatch) -> None:
    monkeypatch.setattr(
        tool_exchange.conversation, "AssistantContent", FakeAssistantContent
    )
    monkeypatch.setattr(
        tool_exchange.conversation, "ToolResultContent", FakeToolResultContent
    )


def test_error_text_handles_interruption_empty_detail_and_bounded_output() -> None:
    assert (
        tool_exchange._error_text(None)
        == "Tool execution was interrupted before completion"
    )
    assert tool_exchange._error_text(RuntimeError()) == "RuntimeError"

    text = tool_exchange._error_text(ValueError("x" * 1000))
    assert len(text) == tool_exchange._MAX_ERROR_TEXT
    assert text.startswith("ValueError: ")
    assert text.endswith("…")


def test_retained_tool_calls_since_returns_only_new_assistant_calls(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    old = FakeAssistantContent([_call("old")])
    new = FakeAssistantContent([_call("a"), _call("b")])
    chat_log = FakeChatLog([old, object(), new, FakeAssistantContent(None)])

    calls = tool_exchange.retained_tool_calls_since(chat_log, {id(old)})

    assert [call.id for call in calls] == ["a", "b"]


def test_append_unresolved_results_preserves_completed_and_marks_failure_then_skips(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    calls = [_call("done", "first"), _call("bad", "second"), _call("later", "third")]
    chat_log = FakeChatLog(
        [
            FakeAssistantContent(calls),
            FakeToolResultContent("agent", "done", "first", {"result": "ok"}),
        ]
    )

    tool_exchange.append_unresolved_tool_results(
        chat_log,
        "agent",
        calls,
        failed_call_id="bad",
        error=RuntimeError("boom"),
    )

    assert [item.tool_call_id for item in chat_log.added] == ["bad", "later"]
    assert tool_result_data(chat_log.added[0]) == {
        "result": {"status": "error", "error": "RuntimeError: boom"}
    }
    assert tool_result_data(chat_log.added[1])["result"]["status"] == "skipped"
    assert "`second` failed" in tool_result_data(chat_log.added[1])["result"]["error"]


def test_append_unresolved_uses_first_unresolved_for_round_failure_and_is_idempotent(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    calls = [_call("a", "alpha"), _call("b", "beta")]
    chat_log = FakeChatLog([FakeAssistantContent(calls)])

    tool_exchange.append_unresolved_tool_results(
        chat_log, "agent", calls, failed_call_id="not-retained", error=None
    )
    tool_exchange.append_unresolved_tool_results(chat_log, "agent", calls)

    assert len(chat_log.added) == 2
    assert chat_log.added[0].tool_call_id == "a"
    assert tool_result_data(chat_log.added[0])["result"]["status"] == "error"
    assert tool_result_data(chat_log.added[1])["result"]["status"] == "skipped"


def test_append_unresolved_is_noop_without_calls_or_unresolved_calls(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    chat_log = FakeChatLog()
    tool_exchange.append_unresolved_tool_results(chat_log, "agent", [])
    assert chat_log.added == []

    call = _call("done")
    chat_log = FakeChatLog(
        [
            FakeAssistantContent([call]),
            FakeToolResultContent("agent", "done", "demo", {}),
        ]
    )
    tool_exchange.append_unresolved_tool_results(chat_log, "agent", [call])
    assert chat_log.added == []


def test_index_tools_keeps_first_valid_named_definition() -> None:
    first = _tool("same", marker="first")
    second = _tool("same", marker="second")
    unnamed = {"spec": {"name": 123}}

    indexed = tool_exchange._index_tools([first, second, unnamed, {}])

    assert indexed == {"same": first}


def test_resolve_current_tool_rejects_removed_or_edited_tool(monkeypatch) -> None:
    call = _call("call", "demo")
    request = _tool("demo", marker="request")
    current = _tool("demo", marker="current")
    latest = Mock(return_value={"resolved": True})
    monkeypatch.setattr(tool_exchange, "latest_function_tool_for_execution", latest)

    entity = object()
    with pytest.raises(FunctionNotFound):
        tool_exchange._resolve_current_tool(
            entity,
            call,
            {"demo": request},
            lambda: [current],
        )
    latest.assert_not_called()

    result = tool_exchange._resolve_current_tool(
        entity, call, {"demo": request}, lambda: [request]
    )
    assert result == {"resolved": True}
    latest.assert_called_once_with(entity, request)

    with pytest.raises(FunctionNotFound):
        tool_exchange._resolve_current_tool(entity, call, {}, None)
    with pytest.raises(FunctionNotFound):
        tool_exchange._resolve_current_tool(entity, call, {"demo": request}, lambda: [])


@pytest.mark.asyncio
async def test_validate_recoverable_call_consumes_malformed_marker_without_validation(
    monkeypatch,
) -> None:
    state = ToolRecoveryState(enabled=True)
    state.remember_malformed("call", "{bad")
    validate = AsyncMock()
    monkeypatch.setattr(tool_exchange, "async_validate_function_arguments", validate)

    result = await tool_exchange._async_validate_recoverable_call(
        SimpleNamespace(hass=object()), _tool(), _call("call"), state
    )

    assert isinstance(result, CorrectableToolFailure)
    assert result.code == "invalid_json"
    validate.assert_not_awaited()
    assert state.malformed_calls == {}


@pytest.mark.asyncio
async def test_validate_recoverable_call_bypasses_ha_tools(monkeypatch) -> None:
    call = _call("call", args={"entity_id": "light.kitchen"})
    validate = AsyncMock()
    monkeypatch.setattr(tool_exchange, "is_ha_tool", lambda _tool: True)
    monkeypatch.setattr(tool_exchange, "async_validate_function_arguments", validate)

    result = await tool_exchange._async_validate_recoverable_call(
        SimpleNamespace(hass=object()), _tool(), call, ToolRecoveryState(enabled=True)
    )

    assert result is call
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_validate_recoverable_call_returns_correctable_failure_or_coerced_input(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tool_exchange, "is_ha_tool", lambda _tool: False)
    entity = SimpleNamespace(hass=object())
    call = _call("call", args={"count": "2"})
    state = ToolRecoveryState(enabled=True)

    monkeypatch.setattr(
        tool_exchange,
        "async_validate_function_arguments",
        AsyncMock(side_effect=HomeAssistantError("invalid arguments")),
    )
    failure = await tool_exchange._async_validate_recoverable_call(
        entity, _tool(), call, state
    )
    assert isinstance(failure, CorrectableToolFailure)
    assert failure.code == "invalid_arguments"

    monkeypatch.setattr(
        tool_exchange,
        "async_validate_function_arguments",
        AsyncMock(return_value={"count": 2}),
    )
    validated = await tool_exchange._async_validate_recoverable_call(
        entity, _tool(), call, state
    )
    assert validated.id == call.id
    assert validated.tool_name == call.tool_name
    assert validated.tool_args == {"count": 2}
    assert validated.external is True


def test_append_recovery_honors_independent_recovery_cap(monkeypatch) -> None:
    _patch_content_types(monkeypatch)
    failure = CorrectableToolFailure(
        code="invalid_arguments",
        stage="pre_dispatch_validation",
        message="fix it",
        original=HomeAssistantError("fix it"),
    )
    state = ToolRecoveryState(enabled=True, limit=1)
    chat_log = FakeChatLog()
    entity = SimpleNamespace(entity_id="agent")
    call = _call("call")

    assert (
        tool_exchange._append_recovery(chat_log, entity, call, failure, state) is True
    )
    assert state.used == 1
    assert (
        tool_exchange._append_recovery(chat_log, entity, call, failure, state) is False
    )
    assert len(chat_log.added) == 1


@pytest.mark.asyncio
async def test_execute_tool_exchange_empty_batch_is_noop() -> None:
    budget = FunctionCallBudget(1)
    entity = SimpleNamespace(_execute_function_tool=AsyncMock())

    await tool_exchange.async_execute_tool_exchange(
        entity, FakeChatLog(), [], [], budget, None, []
    )

    entity._execute_function_tool.assert_not_awaited()
    assert budget.used == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["identical_recreated", "max_tokens", "prompt", "function_tools"]
)
async def test_delete_recreate_identical_tool_cannot_rebind_outstanding_call(
    monkeypatch,
    change,
) -> None:
    _patch_content_types(monkeypatch)
    old_data = {"function_tools": "same bytes"}
    new_data = {"function_tools": "same bytes"}
    if change != "identical_recreated":
        new_data[change] = "changed while provider was pending"
    assert old_data is not new_data
    subentry = SimpleNamespace(data=new_data)
    entry = SimpleNamespace(subentries={"agent": subentry})
    call = _call("call-1")
    chat_log = FakeChatLog([FakeAssistantContent([call])])
    execute = AsyncMock()
    entity = SimpleNamespace(
        entity_id="agent",
        entry=SimpleNamespace(entry_id="entry"),
        subentry=SimpleNamespace(subentry_id="agent", data=old_data),
        hass=SimpleNamespace(
            config_entries=SimpleNamespace(async_get_entry=lambda _: entry)
        ),
        _execute_function_tool=execute,
    )

    with pytest.raises(HomeAssistantError, match=r"configuration changed.*stopped"):
        await tool_exchange.async_execute_tool_exchange(
            entity,
            chat_log,
            [call],
            [_tool()],
            FunctionCallBudget(2),
            None,
            [],
            request_config_data=old_data,
        )

    execute.assert_not_awaited()
    assert chat_log.added[0].tool_call_id == "call-1"


async def test_failed_tool_result_remains_ambiguous_with_fresh_call_id(monkeypatch):
    """A FunctionNotFound result alone does not certify that execution never began."""
    from custom_components.extended_openai_conversation_responses.tool_replay_guard import (
        clear_unacknowledged_calls,
        remember_unacknowledged_calls,
        was_unacknowledged_equivalent,
    )

    _patch_content_types(monkeypatch)
    original = _call("old-id")
    log = FakeChatLog(
        [
            FakeAssistantContent([original]),
            FakeToolResultContent(
                "agent", "old-id", "demo", {"error": "function does not exist"}
            ),
        ]
    )
    log.conversation_id = "ambiguous"
    execute = AsyncMock()
    entity = SimpleNamespace(entity_id="agent", _execute_function_tool=execute)
    remember_unacknowledged_calls(entity, log, set())
    retry = _call("fresh-provider-id")
    assert was_unacknowledged_equivalent(entity, log, retry)
    with pytest.raises(HomeAssistantError, match="Start a new conversation"):
        await tool_exchange.async_execute_tool_exchange(
            entity, log, [retry], [_tool()], FunctionCallBudget(2), None, []
        )
    execute.assert_not_awaited()
    fresh = FakeChatLog()
    fresh.conversation_id = "fresh"
    assert not was_unacknowledged_equivalent(entity, fresh, retry)
    clear_unacknowledged_calls(entity, "ambiguous")
    assert not was_unacknowledged_equivalent(entity, log, retry)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_id", ["", "x" * 1025, "bad\nline"])
async def test_invalid_provider_call_id_never_executes(monkeypatch, call_id) -> None:
    _patch_content_types(monkeypatch)
    call = _call(call_id)
    execute = AsyncMock()
    entity = SimpleNamespace(entity_id="agent", _execute_function_tool=execute)
    chat_log = FakeChatLog([FakeAssistantContent([call])])
    with pytest.raises(HomeAssistantError, match="invalid tool call id"):
        await tool_exchange.async_execute_tool_exchange(
            entity, chat_log, [call], [_tool()], FunctionCallBudget(2), None, []
        )
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_replayed_completed_call_id_never_executes(monkeypatch) -> None:
    _patch_content_types(monkeypatch)
    call = _call("call-1")
    execute = AsyncMock()
    entity = SimpleNamespace(entity_id="agent", _execute_function_tool=execute)
    chat_log = FakeChatLog(
        [
            FakeAssistantContent([call]),
            FakeToolResultContent("agent", "call-1", "demo", {"result": "done"}),
            FakeAssistantContent([call]),
        ]
    )
    with pytest.raises(HomeAssistantError, match="repeated a completed"):
        await tool_exchange.async_execute_tool_exchange(
            entity, chat_log, [call], [_tool()], FunctionCallBudget(2), None, []
        )
    execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("second_args", [{}, {"different": True}])
async def test_duplicate_id_in_one_provider_batch_never_executes(
    monkeypatch, second_args
) -> None:
    _patch_content_types(monkeypatch)
    calls = [_call("same", args={}), _call("same", args=second_args)]
    execute = AsyncMock()
    entity = SimpleNamespace(entity_id="agent", _execute_function_tool=execute)
    chat_log = FakeChatLog([FakeAssistantContent(calls)])
    with pytest.raises(HomeAssistantError, match="duplicate tool call id"):
        await tool_exchange.async_execute_tool_exchange(
            entity, chat_log, calls, [_tool()], FunctionCallBudget(2), None, []
        )
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_serial_exchange_failure_closes_current_and_later_retained_calls(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    calls = [_call("a", "alpha"), _call("b", "beta")]
    chat_log = FakeChatLog([FakeAssistantContent(calls)])
    entity = SimpleNamespace(entity_id="agent", _execute_function_tool=AsyncMock())
    monkeypatch.setattr(tool_exchange, "resolve_parallel_safe_batch", lambda *_: None)

    with pytest.raises(FunctionNotFound):
        await tool_exchange.async_execute_tool_exchange(
            entity,
            chat_log,
            calls,
            [_tool("beta")],
            FunctionCallBudget(5),
            None,
            [],
        )

    assert [item.tool_call_id for item in chat_log.added] == ["a", "b"]
    assert tool_result_data(chat_log.added[0])["result"]["status"] == "error"
    assert tool_result_data(chat_log.added[1])["result"]["status"] == "skipped"
    entity._execute_function_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_parallel_exchange_budget_failure_marks_the_first_refused_call(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    calls = [_call("a", "alpha"), _call("b", "beta")]
    tools = [_tool("alpha"), _tool("beta")]
    chat_log = FakeChatLog([FakeAssistantContent(calls)])
    entity = SimpleNamespace(entity_id="agent")
    monkeypatch.setattr(
        tool_exchange,
        "latest_function_tool_for_execution",
        lambda _entity, candidate: candidate,
    )
    monkeypatch.setattr(
        tool_exchange,
        "resolve_parallel_safe_batch",
        lambda pending, indexed: [(indexed[call.tool_name], call) for call in pending],
    )

    with pytest.raises(HomeAssistantError, match="Function call limit"):
        await tool_exchange.async_execute_tool_exchange(
            entity,
            chat_log,
            calls,
            tools,
            FunctionCallBudget(1),
            None,
            [],
        )

    assert [item.tool_call_id for item in chat_log.added] == ["b", "a"] or [
        item.tool_call_id for item in chat_log.added
    ] == ["a", "b"]
    by_id = {item.tool_call_id: item for item in chat_log.added}
    assert tool_result_data(by_id["b"])["result"]["status"] == "error"
    assert tool_result_data(by_id["a"])["result"]["status"] == "skipped"


@pytest.mark.asyncio
async def test_parallel_exchange_records_success_and_failure_then_raises_first_error(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    calls = [_call("a", "alpha"), _call("b", "beta")]
    tools = [_tool("alpha"), _tool("beta")]
    chat_log = FakeChatLog([FakeAssistantContent(calls)])
    success = FakeToolResultContent("agent", "a", "alpha", {"result": "ok"})
    failure = RuntimeError("beta failed")
    entity = SimpleNamespace(entity_id="agent")
    monkeypatch.setattr(
        tool_exchange,
        "latest_function_tool_for_execution",
        lambda _entity, candidate: candidate,
    )
    monkeypatch.setattr(
        tool_exchange,
        "resolve_parallel_safe_batch",
        lambda pending, indexed: [(indexed[call.tool_name], call) for call in pending],
    )
    monkeypatch.setattr(
        tool_exchange,
        "async_execute_parallel_safe_batch_outcomes",
        AsyncMock(return_value=[success, failure]),
    )

    with pytest.raises(RuntimeError, match="beta failed"):
        await tool_exchange.async_execute_tool_exchange(
            entity,
            chat_log,
            calls,
            tools,
            FunctionCallBudget(5),
            None,
            [],
        )

    by_id = {item.tool_call_id: item for item in chat_log.added}
    assert by_id["a"] is success
    assert tool_result_data(by_id["b"])["result"] == {
        "status": "error",
        "error": "RuntimeError: beta failed",
    }


@pytest.mark.asyncio
async def test_recovery_parallel_validation_exhaustion_closes_exchange(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    call = _call("a", "alpha")
    chat_log = FakeChatLog([FakeAssistantContent([call])])
    entity = SimpleNamespace(entity_id="agent")
    tool = _tool("alpha")
    failure_error = HomeAssistantError("bad arguments")
    failure = CorrectableToolFailure(
        code="invalid_arguments",
        stage="pre_dispatch_validation",
        message="bad arguments",
        original=failure_error,
    )
    state = ToolRecoveryState(enabled=True, limit=0)
    monkeypatch.setattr(
        tool_exchange,
        "latest_function_tool_for_execution",
        lambda _entity, candidate: candidate,
    )
    monkeypatch.setattr(
        tool_exchange,
        "resolve_parallel_safe_batch",
        lambda pending, indexed: [(indexed[call.tool_name], call) for call in pending],
    )
    monkeypatch.setattr(
        tool_exchange,
        "_async_validate_recoverable_call",
        AsyncMock(return_value=failure),
    )

    with pytest.raises(HomeAssistantError, match="bad arguments"):
        await tool_exchange.async_execute_tool_exchange(
            entity,
            chat_log,
            [call],
            [tool],
            FunctionCallBudget(5),
            None,
            [],
            recovery_state=state,
        )

    assert len(chat_log.added) == 1
    assert tool_result_data(chat_log.added[0])["result"]["status"] == "error"


@pytest.mark.asyncio
async def test_recovery_parallel_mixes_correctable_success_and_runtime_failure(
    monkeypatch,
) -> None:
    _patch_content_types(monkeypatch)
    calls = [_call("recover", "recover"), _call("ok", "ok"), _call("bad", "bad")]
    tools = [_tool(call.tool_name) for call in calls]
    chat_log = FakeChatLog([FakeAssistantContent(calls)])
    entity = SimpleNamespace(entity_id="agent")
    state = ToolRecoveryState(enabled=True, limit=2)
    correctable = CorrectableToolFailure(
        code="invalid_arguments",
        stage="pre_dispatch_validation",
        message="repair this",
        original=HomeAssistantError("repair this"),
    )
    success = FakeToolResultContent("agent", "ok", "ok", {"result": "ok"})
    runtime_error = RuntimeError("runtime bad")

    monkeypatch.setattr(
        tool_exchange,
        "latest_function_tool_for_execution",
        lambda _entity, candidate: candidate,
    )
    monkeypatch.setattr(
        tool_exchange,
        "resolve_parallel_safe_batch",
        lambda pending, indexed: [(indexed[call.tool_name], call) for call in pending],
    )

    async def validate(_entity, _tool_value, call, _state):
        return correctable if call.id == "recover" else call

    monkeypatch.setattr(tool_exchange, "_async_validate_recoverable_call", validate)
    monkeypatch.setattr(
        tool_exchange,
        "async_execute_parallel_safe_batch_outcomes",
        AsyncMock(return_value=[success, runtime_error]),
    )

    with pytest.raises(RuntimeError, match="runtime bad"):
        await tool_exchange.async_execute_tool_exchange(
            entity,
            chat_log,
            calls,
            tools,
            FunctionCallBudget(5),
            None,
            [],
            recovery_state=state,
        )

    by_id = {item.tool_call_id: item for item in chat_log.added}
    assert (
        tool_result_data(by_id["recover"])["result"]["reason"]
        == "correctable_tool_error"
    )
    assert by_id["ok"] is success
    assert tool_result_data(by_id["bad"])["result"]["status"] == "error"
    assert state.used == 1
