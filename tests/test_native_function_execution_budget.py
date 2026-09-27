"""Regression coverage for native Function Tool execution budgeting."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    FUNCTION_GROUP_LOADER_TOOL_NAME,
)
from custom_components.extended_openai_conversation_responses.entity import (
    MAX_TOOL_ITERATIONS,
    ExtendedOpenAIBaseLLMEntity,
)
from custom_components.extended_openai_conversation_responses.exceptions import (
    FunctionNotFound,
)
from custom_components.extended_openai_conversation_responses.function_call_budget import (
    FunctionCallBudget,
)
from custom_components.extended_openai_conversation_responses.function_tool_resolution import (
    latest_function_tool_for_execution,
)
from custom_components.extended_openai_conversation_responses.provider_loop import (
    MAX_PROVIDER_REQUESTS,
)
from custom_components.extended_openai_conversation_responses.request import (
    CONTINUE_CONVERSATION_TOOL_NAME,
)
from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError


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


def _tool(name: str, function: dict[str, Any]) -> dict[str, Any]:
    return {
        "spec": {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": function,
    }


def _entity(hass: Any, streams: list[FakeStream], *, limit: int) -> Any:
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


def _executor(entity: Any) -> AsyncMock:
    async def execute(
        _function_tool: dict[str, Any],
        tool_input: Any,
        _llm_context: Any,
        _exposed_entities: list[dict[str, Any]],
    ) -> conversation.ToolResultContent:
        return conversation.ToolResultContent(
            agent_id=entity.entity_id,
            tool_call_id=tool_input.id,
            tool_name=tool_input.tool_name,
            tool_result={"result": "ok"},
        )

    return AsyncMock(side_effect=execute)


def test_function_call_budget_counts_individual_calls() -> None:
    budget = FunctionCallBudget(limit=2)

    budget.claim("first")
    budget.claim("second")

    assert budget.used == 2
    assert budget.exhausted is True
    with pytest.raises(HomeAssistantError, match="Function call limit of 2 reached"):
        budget.claim("third")
    assert budget.used == 2


def test_parallel_budget_reservation_is_atomic() -> None:
    budget = FunctionCallBudget(limit=2, used=1)

    with pytest.raises(
        HomeAssistantError,
        match=r"Function call limit of 2 reached; refusing to execute additional tool `third`",
    ):
        budget.claim_many(("second", "third"))

    assert budget.used == 1
    assert budget.remaining == 1
    assert budget.exhausted is False


def test_function_call_budget_unlimited_and_empty_batch_edges() -> None:
    unlimited = FunctionCallBudget(limit=-1)

    assert unlimited.remaining is None
    assert unlimited.exhausted is False

    unlimited.claim_many(("first_tool", "second_tool"))

    assert unlimited.used == 2
    assert unlimited.remaining is None
    assert unlimited.exhausted is False

    exhausted = FunctionCallBudget(limit=0)
    assert exhausted.exhausted is True

    exhausted.claim_many(())

    assert exhausted.used == 0
    assert exhausted.remaining == 0


def test_provider_safety_ceiling_is_separate_from_function_budget() -> None:
    assert MAX_TOOL_ITERATIONS == MAX_PROVIDER_REQUESTS
    assert MAX_PROVIDER_REQUESTS > 30


async def test_one_parallel_batch_cannot_overshoot_remaining_budget(hass) -> None:
    tools = [
        _tool("memory_one", {"type": "memory", "operation": "search"}),
        _tool("memory_two", {"type": "memory", "operation": "search"}),
    ]
    entity = _entity(
        hass,
        [
            _function_call_stream(
                [
                    ("call-1", "memory_one", {}),
                    ("call-2", "memory_two", {}),
                ]
            )
        ],
        limit=1,
    )
    entity._execute_function_tool = _executor(entity)

    with pytest.raises(HomeAssistantError, match="Function call limit of 1 reached"):
        await entity._async_handle_chat_log(_chat_log(hass), tools, [])

    entity._execute_function_tool.assert_not_awaited()


async def test_serial_over_budget_tool_is_not_called(hass) -> None:
    tools = [
        _tool("first", {"type": "service", "service": "light.turn_on"}),
        _tool("second", {"type": "service", "service": "light.turn_off"}),
    ]
    entity = _entity(
        hass,
        [
            _function_call_stream(
                [
                    ("call-1", "first", {}),
                    ("call-2", "second", {}),
                ]
            )
        ],
        limit=1,
    )
    entity._execute_function_tool = _executor(entity)

    with pytest.raises(HomeAssistantError, match="Function call limit of 1 reached"):
        await entity._async_handle_chat_log(_chat_log(hass), tools, [])

    assert entity._execute_function_tool.await_count == 1
    assert entity._execute_function_tool.await_args_list[0].args[1].tool_name == "first"


async def test_budget_above_twenty_allows_more_than_twenty_calls(hass) -> None:
    tool = _tool("do_work", {"type": "service", "service": "light.turn_on"})
    streams = [
        _function_call_stream([(f"call-{index}", "do_work", {})]) for index in range(21)
    ]
    streams.append(_final_stream())
    entity = _entity(hass, streams, limit=25)
    entity._execute_function_tool = _executor(entity)

    await entity._async_handle_chat_log(_chat_log(hass), [tool], [])

    assert entity._execute_function_tool.await_count == 21
    assert entity._client.responses.create.await_count == 22


async def test_zero_budget_stops_advertising_ordinary_functions(hass) -> None:
    tool = _tool("do_work", {"type": "service", "service": "light.turn_on"})
    entity = _entity(hass, [_final_stream()], limit=0)
    entity._execute_function_tool = _executor(entity)

    await entity._async_handle_chat_log(_chat_log(hass), [tool], [])

    kwargs = entity._client.responses.create.await_args.kwargs
    assert "tools" not in kwargs
    entity._execute_function_tool.assert_not_awaited()


async def test_zero_budget_does_not_consume_conditional_finalizer(hass) -> None:
    entity = _entity(
        hass,
        [
            _function_call_stream(
                [
                    (
                        "final-1",
                        CONTINUE_CONVERSATION_TOOL_NAME,
                        {"response": "Finished", "continue_conversation": False},
                    )
                ]
            )
        ],
        limit=0,
    )
    entity._execute_function_tool = _executor(entity)

    result = await entity._async_handle_chat_log(
        _chat_log(hass),
        [],
        [],
        conditional_continue=True,
    )

    assert result is False
    entity._execute_function_tool.assert_not_awaited()


async def test_zero_budget_does_not_consume_function_group_loader(hass) -> None:
    loader = _tool(
        FUNCTION_GROUP_LOADER_TOOL_NAME,
        {"type": "function_group_loader"},
    )
    entity = _entity(
        hass,
        [
            _function_call_stream(
                [("loader-1", FUNCTION_GROUP_LOADER_TOOL_NAME, {"groups": ["extra"]})]
            ),
            _final_stream(),
        ],
        limit=0,
    )
    entity._execute_function_tool = _executor(entity)
    load_group = Mock(return_value={"status": "loaded"})

    await entity._async_handle_chat_log(
        _chat_log(hass),
        [loader],
        [],
        function_group_loader=load_group,
    )

    load_group.assert_called_once_with(["extra"])
    entity._execute_function_tool.assert_not_awaited()


async def test_provider_loop_exhaustion_is_explicit(hass, monkeypatch) -> None:
    import custom_components.extended_openai_conversation_responses.entity as entity_module

    monkeypatch.setattr(entity_module, "MAX_TOOL_ITERATIONS", 2)
    tool = _tool("do_work", {"type": "service", "service": "light.turn_on"})
    entity = _entity(
        hass,
        [
            _function_call_stream([("call-1", "do_work", {})]),
            _function_call_stream([("call-2", "do_work", {})]),
        ],
        limit=10,
    )
    entity._execute_function_tool = _executor(entity)

    with pytest.raises(
        HomeAssistantError,
        match=r"Provider tool loop exceeded the safety limit of 2 requests",
    ):
        await entity._async_handle_chat_log(_chat_log(hass), [tool], [])

    assert entity._execute_function_tool.await_count == 2


async def test_tool_edit_between_request_and_execution_rejects_stale_call(
    hass,
) -> None:
    stale = _tool("notify", {"type": "service", "service": "notify.old"})
    current = _tool("notify", {"type": "service", "service": "notify.current"})
    entity = _entity(
        hass,
        [
            _function_call_stream([("call-1", "notify", {})]),
            _final_stream(),
        ],
        limit=2,
    )
    latest_data = {"revision": 2}
    latest_subentry = SimpleNamespace(data=latest_data)
    latest_entry = SimpleNamespace(subentries={"agent-1": latest_subentry})
    entity.hass = SimpleNamespace(
        config_entries=SimpleNamespace(async_get_entry=Mock(return_value=latest_entry))
    )
    entity._configured_function_tools_from_data = Mock(return_value=[current])
    entity._execute_function_tool = _executor(entity)

    with pytest.raises(FunctionNotFound):
        await entity._async_handle_chat_log(_chat_log(hass), [stale], [])

    entity._execute_function_tool.assert_not_awaited()
    entity._configured_function_tools_from_data.assert_called_with(latest_data)


def test_edited_definition_is_rejected_before_execution() -> None:
    stale = _tool("notify", {"type": "service", "service": "notify.old"})
    current = _tool("notify", {"type": "service", "service": "notify.current"})
    latest_data = {"revision": 2}
    latest_subentry = SimpleNamespace(data=latest_data)
    latest_entry = SimpleNamespace(subentries={"agent-1": latest_subentry})
    agent = SimpleNamespace(
        hass=SimpleNamespace(
            config_entries=SimpleNamespace(
                async_get_entry=Mock(return_value=latest_entry)
            )
        ),
        entry=SimpleNamespace(entry_id="entry-1"),
        subentry=SimpleNamespace(subentry_id="agent-1", data={"revision": 1}),
        _configured_function_tools_from_data=Mock(return_value=[current]),
    )

    with pytest.raises(FunctionNotFound):
        latest_function_tool_for_execution(agent, stale)
    agent._configured_function_tools_from_data.assert_called_once_with(latest_data)


def test_deleted_definition_fails_closed_before_execution() -> None:
    stale = _tool("notify", {"type": "service", "service": "notify.old"})
    latest_data = {"revision": 2}
    latest_subentry = SimpleNamespace(data=latest_data)
    latest_entry = SimpleNamespace(subentries={"agent-1": latest_subentry})
    agent = SimpleNamespace(
        hass=SimpleNamespace(
            config_entries=SimpleNamespace(
                async_get_entry=Mock(return_value=latest_entry)
            )
        ),
        entry=SimpleNamespace(entry_id="entry-1"),
        subentry=SimpleNamespace(subentry_id="agent-1", data={"revision": 1}),
        _configured_function_tools_from_data=Mock(return_value=[]),
    )

    with pytest.raises(FunctionNotFound):
        latest_function_tool_for_execution(agent, stale)
