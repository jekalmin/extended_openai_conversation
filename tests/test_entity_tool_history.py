"""Regression tests for interrupted and failed custom tool calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.extended_openai_conversation import entity
from homeassistant.components import conversation
from homeassistant.exceptions import ServiceNotFound
from homeassistant.helpers import llm


def tool_call(call_id: str, name: str = "execute_services") -> llm.ToolInput:
    return llm.ToolInput(id=call_id, tool_name=name, tool_args={}, external=True)


def assistant(*calls: llm.ToolInput) -> conversation.AssistantContent:
    return conversation.AssistantContent(
        agent_id="conversation.test", content="", tool_calls=list(calls)
    )


def tool_result(call_id: str) -> conversation.ToolResultContent:
    return entity._make_tool_result_content(
        "conversation.test", call_id, "execute_services", {"result": "ok"}
    )


def assert_valid_messages(content, shorten=False) -> None:
    """Every tool call must be answered before the next conversation turn."""
    pending = set()
    for message in entity._convert_content_to_param(content, shorten):
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
        else:
            assert not pending
            pending = {call["id"] for call in message.get("tool_calls", [])}
    assert not pending


class FakeChatLog:
    """Minimal chat log; all tool execution and API I/O are mocked."""

    def __init__(self, content, calls):
        self.content = content
        self.batches = iter([[assistant(*calls)], [assistant()]])

    async def async_add_delta_content_stream(self, *args):
        for content in next(self.batches):
            self.content.append(content)
            yield content

    def async_add_assistant_content_without_tools(self, content):
        self.content.append(content)

    @property
    def unresponded_tool_results(self):
        return self.content[-1].role == "tool_result"


async def handle_log(log, execute):
    async def create(**kwargs):
        # Check the actual messages passed to the API, including the first request.
        pending = set()
        for message in kwargs["messages"]:
            if message["role"] == "tool":
                assert message["tool_call_id"] in pending
                pending.remove(message["tool_call_id"])
            else:
                assert not pending
                pending = {call["id"] for call in message.get("tool_calls", [])}
        assert not pending

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=create))
        )
    )
    instance = SimpleNamespace(
        entity_id="conversation.test",
        subentry=SimpleNamespace(data={}),
        _client=client,
        _transform_stream=lambda *args: None,
        _execute_function_tool=execute,
    )
    config = dict(
        supports_max_completion_tokens=False,
        supports_max_tokens=False,
        supports_top_p=False,
        supports_temperature=False,
    )
    with patch.object(entity, "get_model_config", return_value=config):
        await entity.ExtendedOpenAIBaseLLMEntity._async_handle_chat_log(
            instance, log, [{"spec": {"name": "execute_services"}}], []
        )
    return client.chat.completions.create


@pytest.mark.asyncio
async def test_failed_service_answers_all_calls() -> None:
    execute = AsyncMock(side_effect=ServiceNotFound("cover", "close"))
    log = FakeChatLog([], [tool_call("a"), tool_call("b")])
    create = await handle_log(log, execute)
    assert execute.await_count == 2
    assert create.await_count == 2
    results = [item for item in log.content if item.role == "tool_result"]
    assert [entity._tool_result_data(item)["error"] for item in results] == [
        "ServiceNotFound",
        "ServiceNotFound",
    ]
    assert_valid_messages(log.content)


@pytest.mark.asyncio
async def test_unknown_tool_gets_error_result() -> None:
    execute = AsyncMock()
    log = FakeChatLog([], [tool_call("a", "unknown_tool")])
    await handle_log(log, execute)
    execute.assert_not_awaited()
    assert entity._tool_result_data(log.content[-2])["error"] == "FunctionNotFound"
    assert_valid_messages(log.content)


@pytest.mark.asyncio
async def test_successful_result_is_preserved() -> None:
    result = tool_result("a")
    log = FakeChatLog([], [tool_call("a")])
    await handle_log(log, AsyncMock(return_value=result))
    assert log.content[-2] is result
    assert_valid_messages(log.content)


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    log = FakeChatLog([], [tool_call("a")])
    with pytest.raises(asyncio.CancelledError):
        await handle_log(log, AsyncMock(side_effect=asyncio.CancelledError))
