"""Regression coverage for protocol-valid retained Function Tool exchanges."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
)
from custom_components.extended_openai_conversation_responses.entity import (
    ExtendedOpenAIBaseLLMEntity,
    _convert_content_to_param,
    _convert_content_to_responses_param,
)
from custom_components.extended_openai_conversation_responses.exceptions import (
    FunctionNotFound,
    ParseArgumentsFailed,
)
from custom_components.extended_openai_conversation_responses.function_call_budget import (
    FunctionCallBudget,
)
from custom_components.extended_openai_conversation_responses.ha_tool_result_compat import (
    tool_result_data,
)
from custom_components.extended_openai_conversation_responses.provider_errors import (
    ProviderStreamError,
)
from custom_components.extended_openai_conversation_responses.tool_exchange import (
    append_unresolved_tool_results,
    async_execute_tool_exchange,
)
from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm


class FakeStream:
    """Async iterator over fake Responses events."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self.events:
            yield event


def _event(event_type: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, **kwargs)


def _completed_event() -> SimpleNamespace:
    return _event(
        "response.completed",
        response=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5)
        ),
    )


def _function_call_stream(
    calls: list[tuple[str, str, dict[str, Any]]],
) -> FakeStream:
    events: list[Any] = []
    for call_id, name, arguments in calls:
        item = SimpleNamespace(
            type="function_call",
            call_id=call_id,
            name=name,
            arguments=json.dumps(arguments),
        )
        events.extend(
            [
                _event("response.output_item.added", item=item),
                _event("response.output_item.done", item=item),
            ]
        )
    events.append(_completed_event())
    return FakeStream(events)


def _failed_stream(message: str = "provider failed") -> FakeStream:
    return FakeStream(
        [
            _event(
                "response.failed",
                response=SimpleNamespace(
                    id="resp-failed",
                    error=SimpleNamespace(
                        message=message,
                        code="server_error",
                        type="server_error",
                    ),
                ),
            )
        ]
    )


def _partial_tool_then_failed_stream() -> FakeStream:
    item = SimpleNamespace(
        type="function_call",
        call_id="partial-1",
        name="first",
        arguments="{}",
    )
    return FakeStream(
        [
            _event("response.output_item.added", item=item),
            _event("response.output_item.done", item=item),
            _event(
                "response.failed",
                response=SimpleNamespace(
                    id="resp-partial",
                    error=SimpleNamespace(
                        message="stream aborted",
                        code="server_error",
                        type="server_error",
                    ),
                ),
            ),
        ]
    )


def _malformed_tool_stream() -> FakeStream:
    item = SimpleNamespace(
        type="function_call",
        call_id="broken-1",
        name="first",
        arguments="{",
    )
    return FakeStream(
        [
            _event("response.output_item.added", item=item),
            _event("response.output_item.done", item=item),
        ]
    )


def _final_stream(text: str = "Done") -> FakeStream:
    return FakeStream(
        [
            _event(
                "response.output_item.added",
                item=SimpleNamespace(type="message"),
            ),
            _event("response.output_text.delta", delta=text),
            _completed_event(),
        ]
    )


def _tool(
    name: str,
    *,
    function_type: str = "service",
    operation: str | None = None,
    native_name: str | None = None,
) -> dict[str, Any]:
    function: dict[str, Any] = {"type": function_type}
    if function_type == "service":
        function["service"] = f"test.{name}"
    if operation is not None:
        function["operation"] = operation
    if native_name is not None:
        function["name"] = native_name
    return {
        "spec": {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": function,
    }


def _entity(hass: Any, streams: list[FakeStream], *, limit: int = 2) -> Any:
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(side_effect=streams))
    )
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.entry = SimpleNamespace(
        runtime_data=client,
        data={},
        entry_id="entry-1",
    )
    entity.subentry = SimpleNamespace(
        subentry_id="agent-1",
        data={
            CONF_CHAT_MODEL: "gpt-5.6-luna",
            CONF_API_MODE: API_MODE_RESPONSES,
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: limit,
        },
    )
    entity.hass = hass
    entity.entity_id = "conversation.test"
    entity._usage = None
    return entity


def _chat_log(hass: Any) -> conversation.ChatLog:
    chat_log = conversation.ChatLog(hass, "conversation-id")
    chat_log.content[0] = conversation.SystemContent(content="Be helpful")
    chat_log.async_add_user_content(conversation.UserContent(content="Do the work"))
    return chat_log


def _call(call_id: str, name: str) -> llm.ToolInput:
    return llm.ToolInput(
        id=call_id,
        tool_name=name,
        tool_args={},
        external=True,
    )


def _result(entity: Any, tool_input: llm.ToolInput, value: str = "ok") -> Any:
    return conversation.ToolResultContent(
        agent_id=entity.entity_id,
        tool_call_id=tool_input.id,
        tool_name=tool_input.tool_name,
        tool_result={"result": value},
    )


def _retained_results(chat_log: conversation.ChatLog) -> list[Any]:
    return [
        content
        for content in chat_log.content
        if isinstance(content, conversation.ToolResultContent)
    ]


async def test_serial_budget_failure_is_protocol_valid_on_following_turn(hass) -> None:
    """A successful first side effect is not retried and every call gets one output."""
    first = _tool("first")
    second = _tool("second")
    entity = _entity(
        hass,
        [
            _function_call_stream([("call-1", "first", {}), ("call-2", "second", {})]),
            _final_stream("Recovered"),
        ],
        limit=1,
    )
    executed: list[str] = []

    async def execute(
        _function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        _llm_context: Any,
        _exposed_entities: list[dict[str, Any]],
    ) -> Any:
        executed.append(tool_input.tool_name)
        return _result(entity, tool_input)

    entity._execute_function_tool = AsyncMock(side_effect=execute)
    chat_log = _chat_log(hass)

    with pytest.raises(HomeAssistantError, match="Function call limit of 1 reached"):
        await entity._async_handle_chat_log(chat_log, [first, second], [])

    assert executed == ["first"]
    results = _retained_results(chat_log)
    assert [result.tool_call_id for result in results] == ["call-1", "call-2"]
    assert tool_result_data(results[0]) == {"result": "ok"}
    assert tool_result_data(results[1])["result"]["status"] == "error"

    chat_log.async_add_user_content(conversation.UserContent(content="Continue"))
    await entity._async_handle_chat_log(chat_log, [first, second], [])

    assert executed == ["first"]
    next_input = entity._client.responses.create.await_args_list[1].kwargs["input"]
    calls = [item for item in next_input if item.get("type") == "function_call"]
    outputs = [
        item for item in next_input if item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in calls] == ["call-1", "call-2"]
    assert [item["call_id"] for item in outputs] == ["call-1", "call-2"]
    assert len({item["call_id"] for item in outputs}) == 2
    assert chat_log.unresponded_tool_results is False


async def test_partial_provider_stream_drops_uncommitted_tool_call_without_execution(
    hass,
) -> None:
    """A stream failure must not create an orphan output for an uncommitted call."""
    entity = _entity(hass, [_partial_tool_then_failed_stream()])
    entity._execute_function_tool = AsyncMock()
    chat_log = _chat_log(hass)

    with pytest.raises(ProviderStreamError):
        await entity._async_handle_chat_log(chat_log, [_tool("first")], [])

    entity._execute_function_tool.assert_not_awaited()
    assert _retained_results(chat_log) == []
    assert not any(
        isinstance(content, conversation.AssistantContent) and content.tool_calls
        for content in chat_log.content
    )
    assert chat_log.unresponded_tool_results is False


async def test_provider_failure_after_execution_never_retries_side_effect(hass) -> None:
    entity = _entity(
        hass,
        [_function_call_stream([("call-1", "first", {})]), _failed_stream()],
    )
    executed = 0

    async def execute(
        _function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        _llm_context: Any,
        _exposed_entities: list[dict[str, Any]],
    ) -> Any:
        nonlocal executed
        executed += 1
        return _result(entity, tool_input)

    entity._execute_function_tool = AsyncMock(side_effect=execute)
    chat_log = _chat_log(hass)

    with pytest.raises(ProviderStreamError):
        await entity._async_handle_chat_log(chat_log, [_tool("first")], [])

    assert executed == 1
    results = _retained_results(chat_log)
    assert len(results) == 1
    assert results[0].tool_call_id == "call-1"
    assert tool_result_data(results[0]) == {"result": "ok"}


async def test_cancellation_closes_current_and_skips_later_serial_call(hass) -> None:
    entity = _entity(
        hass,
        [_function_call_stream([("call-1", "first", {}), ("call-2", "second", {})])],
        limit=2,
    )
    started = asyncio.Event()
    executed: list[str] = []

    async def execute(
        _function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        _llm_context: Any,
        _exposed_entities: list[dict[str, Any]],
    ) -> Any:
        executed.append(tool_input.tool_name)
        started.set()
        await asyncio.Event().wait()
        return _result(entity, tool_input)

    entity._execute_function_tool = AsyncMock(side_effect=execute)
    chat_log = _chat_log(hass)
    task = asyncio.create_task(
        entity._async_handle_chat_log(chat_log, [_tool("first"), _tool("second")], [])
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert executed == ["first"]
    results = _retained_results(chat_log)
    assert [result.tool_call_id for result in results] == ["call-1", "call-2"]
    assert tool_result_data(results[0])["result"]["status"] == "error"
    assert tool_result_data(results[1])["result"]["status"] == "skipped"


async def test_current_effective_tool_disappearance_fails_before_dispatch(hass) -> None:
    tool = _tool("first")
    entity = _entity(hass, [_function_call_stream([("call-1", "first", {})])])
    entity._execute_function_tool = AsyncMock()
    chat_log = _chat_log(hass)
    factory_calls = 0

    def current_tools() -> list[dict[str, Any]]:
        nonlocal factory_calls
        factory_calls += 1
        return [tool] if factory_calls == 1 else []

    with pytest.raises(FunctionNotFound):
        await entity._async_handle_chat_log(
            chat_log,
            [tool],
            [],
            function_tools_factory=current_tools,
        )

    entity._execute_function_tool.assert_not_awaited()
    results = _retained_results(chat_log)
    assert len(results) == 1
    assert tool_result_data(results[0])["result"]["status"] == "error"


async def test_unknown_provider_tool_is_closed_without_dispatch(hass) -> None:
    entity = _entity(hass, [_function_call_stream([("call-x", "missing", {})])])
    entity._execute_function_tool = AsyncMock()
    chat_log = _chat_log(hass)

    with pytest.raises(FunctionNotFound):
        await entity._async_handle_chat_log(chat_log, [_tool("known")], [])

    entity._execute_function_tool.assert_not_awaited()
    results = _retained_results(chat_log)
    assert len(results) == 1
    assert results[0].tool_call_id == "call-x"
    assert tool_result_data(results[0])["result"]["status"] == "error"


async def test_parallel_batch_rejects_edited_definition_before_execution(hass) -> None:
    first_old = _tool("first", function_type="native", native_name="get_history")
    second_old = _tool("second", function_type="native", native_name="get_statistics")
    first_current = _tool(
        "first", function_type="native", native_name="execute_service"
    )
    second_current = second_old
    entity = _entity(
        hass,
        [
            _function_call_stream([("call-1", "first", {}), ("call-2", "second", {})]),
            _final_stream(),
        ],
    )
    factory_calls = 0
    concurrent = 0
    maximum_concurrent = 0

    def current_tools() -> list[dict[str, Any]]:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return [first_old, second_old]
        return [first_current, second_current]

    async def execute(
        _function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        _llm_context: Any,
        _exposed_entities: list[dict[str, Any]],
    ) -> Any:
        nonlocal concurrent, maximum_concurrent
        concurrent += 1
        maximum_concurrent = max(maximum_concurrent, concurrent)
        await asyncio.sleep(0)
        concurrent -= 1
        return _result(entity, tool_input)

    entity._execute_function_tool = AsyncMock(side_effect=execute)
    with pytest.raises(FunctionNotFound):
        await entity._async_handle_chat_log(
            _chat_log(hass),
            [first_old, second_old],
            [],
            function_tools_factory=current_tools,
        )

    assert maximum_concurrent == 0
    entity._execute_function_tool.assert_not_awaited()


async def test_parallel_failure_preserves_successful_sibling_result(hass) -> None:
    first_tool = _tool("first", function_type="knowledge", operation="search")
    second_tool = _tool("second", function_type="knowledge", operation="list")
    first_call = _call("call-1", "first")
    second_call = _call("call-2", "second")
    chat_log = _chat_log(hass)
    chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(
            agent_id="conversation.test",
            tool_calls=[first_call, second_call],
        )
    )
    entity = SimpleNamespace(entity_id="conversation.test")

    async def execute(
        _function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        _llm_context: Any,
        _exposed_entities: list[dict[str, Any]],
    ) -> Any:
        await asyncio.sleep(0)
        if tool_input.id == "call-1":
            raise RuntimeError("read failed")
        return _result(entity, tool_input, "second-ok")

    entity._execute_function_tool = execute

    with pytest.raises(RuntimeError, match="read failed"):
        await async_execute_tool_exchange(
            entity,
            chat_log,
            [first_call, second_call],
            [first_tool, second_tool],
            FunctionCallBudget(2),
            None,
            [],
        )

    results = _retained_results(chat_log)
    assert [result.tool_call_id for result in results] == ["call-1", "call-2"]
    assert tool_result_data(results[0])["result"]["status"] == "error"
    assert tool_result_data(results[1]) == {"result": "second-ok"}


async def test_iteration_exhaustion_leaves_each_round_call_closed(
    hass, monkeypatch
) -> None:
    import custom_components.extended_openai_conversation_responses.entity as entity_module

    monkeypatch.setattr(entity_module, "MAX_TOOL_ITERATIONS", 2)
    entity = _entity(
        hass,
        [
            _function_call_stream([("call-1", "first", {})]),
            _function_call_stream([("call-2", "first", {})]),
        ],
        limit=5,
    )

    async def execute(
        _function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        _llm_context: Any,
        _exposed_entities: list[dict[str, Any]],
    ) -> Any:
        return _result(entity, tool_input)

    entity._execute_function_tool = AsyncMock(side_effect=execute)
    chat_log = _chat_log(hass)

    with pytest.raises(HomeAssistantError, match="safety limit of 2 requests"):
        await entity._async_handle_chat_log(chat_log, [_tool("first")], [])

    results = _retained_results(chat_log)
    assert [result.tool_call_id for result in results] == ["call-1", "call-2"]
    assert all(tool_result_data(result) == {"result": "ok"} for result in results)


async def test_malformed_arguments_do_not_leave_a_retained_tool_call(hass) -> None:
    entity = _entity(hass, [_malformed_tool_stream()])
    entity._execute_function_tool = AsyncMock()
    chat_log = _chat_log(hass)

    with pytest.raises(ParseArgumentsFailed):
        await entity._async_handle_chat_log(chat_log, [_tool("first")], [])

    assert not any(
        isinstance(content, conversation.AssistantContent) and content.tool_calls
        for content in chat_log.content
    )
    assert _retained_results(chat_log) == []
    entity._execute_function_tool.assert_not_awaited()


class _NativeReasoning:
    type = "reasoning"

    def model_dump(self, *, exclude_none: bool = True) -> dict[str, Any]:
        return {"type": "reasoning", "id": "reason-1", "summary": []}


def test_failed_outputs_are_valid_in_both_provider_conversion_formats(hass) -> None:
    chat_log = _chat_log(hass)
    first_call = _call("call-1", "first")
    second_call = _call("call-2", "second")
    chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(
            agent_id="conversation.test",
            tool_calls=[first_call, second_call],
            native=_NativeReasoning(),
        )
    )
    chat_log.async_add_assistant_content_without_tools(
        conversation.ToolResultContent(
            agent_id="conversation.test",
            tool_call_id="call-1",
            tool_name="first",
            tool_result={"result": "ok"},
        )
    )
    append_unresolved_tool_results(
        chat_log,
        "conversation.test",
        [first_call, second_call],
        failed_call_id="call-2",
        error=RuntimeError("budget exhausted"),
    )

    chat_messages = _convert_content_to_param(chat_log.content)
    chat_tool_calls = [
        call
        for message in chat_messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    chat_outputs = [
        message for message in chat_messages if message.get("role") == "tool"
    ]
    assert [call["id"] for call in chat_tool_calls] == ["call-1", "call-2"]
    assert [message["tool_call_id"] for message in chat_outputs] == [
        "call-1",
        "call-2",
    ]

    response_items = _convert_content_to_responses_param(chat_log.content)
    assert any(item.get("type") == "reasoning" for item in response_items)
    response_calls = [
        item for item in response_items if item.get("type") == "function_call"
    ]
    response_outputs = [
        item for item in response_items if item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in response_calls] == ["call-1", "call-2"]
    assert [item["call_id"] for item in response_outputs] == ["call-1", "call-2"]
