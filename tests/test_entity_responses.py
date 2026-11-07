"""Tests for the optional OpenAI Responses API mode."""

from collections.abc import AsyncIterator
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseError,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
import pytest
import voluptuous as vol

from custom_components.extended_openai_conversation import entity as entity_module
from custom_components.extended_openai_conversation.entity import (
    ExtendedOpenAIBaseLLMEntity,
    _convert_content_to_responses_input,
    _format_responses_tool,
    _make_tool_result_content,
    _shorten_tool_call_id,
)
from custom_components.extended_openai_conversation.exceptions import (
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from custom_components.extended_openai_conversation.functions import get_function
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

FUNCTION_TOOL = {
    "spec": {
        "name": "get_state",
        "description": "Get the state of an entity.",
        "parameters": {
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
            "required": ["entity_id"],
        },
    },
    "function": {"type": "template", "value_template": "{{ entity_id }} is on"},
}


def _make_entity(hass, client: Any, **options: Any) -> ExtendedOpenAIBaseLLMEntity:
    """Create an entity with the given client and subentry options."""
    entry = MagicMock()
    entry.runtime_data = client
    subentry = ConfigSubentry(
        data=options, subentry_type="conversation", title="Test", unique_id=None
    )
    entity = ExtendedOpenAIBaseLLMEntity(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    return entity


def _make_chat_log(hass, *content: conversation.Content) -> conversation.ChatLog:
    """Create a chat log with a system prompt followed by the given content."""
    return conversation.ChatLog(
        hass,
        "conversation-id",
        content=[
            conversation.SystemContent(content="You are a voice assistant."),
            *content,
        ],
    )


def _function_tools() -> list[dict[str, Any]]:
    """Return validated function tools."""
    function = get_function(FUNCTION_TOOL["function"]["type"])
    return [
        {
            "spec": FUNCTION_TOOL["spec"],
            "function": function.validate_schema(FUNCTION_TOOL["function"]),
        }
    ]


async def _stream(*events: Any) -> AsyncIterator[Any]:
    for event in events:
        yield event


def _text_delta(delta: str) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta", delta=delta
    )


def _function_call_done(
    call_id: str, name: str, arguments: str
) -> ResponseOutputItemDoneEvent:
    return ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        item=ResponseFunctionToolCall.model_construct(
            type="function_call", call_id=call_id, name=name, arguments=arguments
        ),
    )


def _completed(total_tokens: int = 15) -> ResponseCompletedEvent:
    return ResponseCompletedEvent.model_construct(
        type="response.completed",
        response=Response.model_construct(
            usage=ResponseUsage.model_construct(
                input_tokens=10, output_tokens=5, total_tokens=total_tokens
            )
        ),
    )


def test_convert_content_to_responses_input() -> None:
    """Chat log content is converted to instructions and input items."""
    tool_call = llm.ToolInput(
        id="call_1", tool_name="get_state", tool_args={"entity_id": "light.kitchen"}
    )
    content = [
        conversation.SystemContent(content="You are a voice assistant."),
        conversation.UserContent(content="Is the kitchen light on?"),
        conversation.AssistantContent(
            agent_id="conversation.test",
            content="Let me check.",
            tool_calls=[tool_call],
        ),
        _make_tool_result_content(
            "conversation.test", "call_1", "get_state", {"result": "on"}
        ),
        conversation.AssistantContent(
            agent_id="conversation.test", content="It is on."
        ),
    ]

    instructions, items = _convert_content_to_responses_input(content)

    assert instructions == "You are a voice assistant."
    assert items == [
        {"type": "message", "role": "user", "content": "Is the kitchen light on?"},
        {"type": "message", "role": "assistant", "content": "Let me check."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_state",
            "arguments": '{"entity_id": "light.kitchen"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"result":"on"}',
        },
        {"type": "message", "role": "assistant", "content": "It is on."},
    ]


def test_convert_content_to_responses_input_shortens_call_ids() -> None:
    """Shortened call ids match between function calls and their outputs."""
    content = [
        conversation.AssistantContent(
            agent_id="conversation.test",
            tool_calls=[
                llm.ToolInput(id="call_1", tool_name="get_state", tool_args={})
            ],
        ),
        _make_tool_result_content(
            "conversation.test", "call_1", "get_state", {"result": "on"}
        ),
    ]

    instructions, items = _convert_content_to_responses_input(content, True)

    assert instructions is None
    assert [item["call_id"] for item in items] == [_shorten_tool_call_id("call_1")] * 2


def test_format_responses_tool() -> None:
    """Function specs become flat, non-strict Responses API tools."""
    assert _format_responses_tool(FUNCTION_TOOL["spec"]) == {
        "type": "function",
        "strict": False,
        **FUNCTION_TOOL["spec"],
    }
    assert _format_responses_tool({"name": "now", "strict": True}) == {
        "type": "function",
        "name": "now",
        "parameters": {"type": "object", "properties": {}},
        "strict": True,
    }


async def test_responses_function_call_round_trip(hass) -> None:
    """A function call is executed and its output is sent back by call_id."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=[
            _stream(
                _function_call_done(
                    "call_1", "get_state", '{"entity_id": "light.kitchen"}'
                ),
                _completed(),
            ),
            _stream(
                _text_delta("The kitchen "), _text_delta("light is on."), _completed()
            ),
        ]
    )
    entity = _make_entity(
        hass,
        client,
        chat_model="gpt-6.1-sol",
        api_mode="responses",
        reasoning_effort="low",
        max_tokens=2000,
    )
    chat_log = _make_chat_log(
        hass, conversation.UserContent(content="Is the kitchen light on?")
    )

    await entity._async_handle_chat_log(chat_log, _function_tools(), [])

    client.chat.completions.create.assert_not_called()
    first, second = (call.kwargs for call in client.responses.create.call_args_list)
    assert first == {
        "model": "gpt-6.1-sol",
        "stream": True,
        "store": False,
        "max_output_tokens": 2000,
        "reasoning": {"effort": "low"},
        "instructions": "You are a voice assistant.",
        "input": [
            {"type": "message", "role": "user", "content": "Is the kitchen light on?"}
        ],
        "tools": [{"type": "function", "strict": False, **FUNCTION_TOOL["spec"]}],
        "tool_choice": "auto",
    }
    assert second["input"][1:] == [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_state",
            "arguments": '{"entity_id": "light.kitchen"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"result":"light.kitchen is on"}',
        },
    ]

    assistant, tool_result, answer = chat_log.content[2:]
    assert assistant.tool_calls[0].id == "call_1"
    assert assistant.tool_calls[0].tool_args == {"entity_id": "light.kitchen"}
    assert tool_result.tool_call_id == "call_1"
    assert answer.content == "The kitchen light is on."


async def test_responses_incomplete_max_output_tokens(hass) -> None:
    """Running out of output tokens raises like the Chat Completions length limit."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=_stream(
            _text_delta("The kitchen"),
            ResponseIncompleteEvent.model_construct(
                type="response.incomplete",
                response=Response.model_construct(
                    usage=None,
                    incomplete_details=IncompleteDetails(reason="max_output_tokens"),
                ),
            ),
        )
    )
    entity = _make_entity(hass, client, api_mode="responses", max_tokens=16)
    chat_log = _make_chat_log(hass, conversation.UserContent(content="Hi"))

    with pytest.raises(TokenLengthExceededError):
        await entity._async_handle_chat_log(chat_log, [], [])


@pytest.mark.parametrize(
    ("event", "error"),
    [
        (
            ResponseIncompleteEvent.model_construct(
                type="response.incomplete",
                response=Response.model_construct(
                    usage=None,
                    incomplete_details=IncompleteDetails(reason="content_filter"),
                ),
            ),
            "OpenAI response incomplete: content_filter",
        ),
        (
            ResponseFailedEvent.model_construct(
                type="response.failed",
                response=Response.model_construct(
                    error=ResponseError.model_construct(message="server error")
                ),
            ),
            "OpenAI response failed: server error",
        ),
        (
            ResponseErrorEvent.model_construct(type="error", message="rate limited"),
            "OpenAI response error: rate limited",
        ),
    ],
    ids=["incomplete", "failed", "error"],
)
async def test_responses_error_events(hass, event: Any, error: str) -> None:
    """Failed or incomplete responses raise an error."""
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_stream(event))
    entity = _make_entity(hass, client, api_mode="responses")
    chat_log = _make_chat_log(hass, conversation.UserContent(content="Hi"))

    with pytest.raises(HomeAssistantError, match=error):
        await entity._async_handle_chat_log(chat_log, [], [])


async def test_responses_invalid_function_arguments(hass) -> None:
    """Malformed function call arguments raise ParseArgumentsFailed."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=_stream(_function_call_done("call_1", "get_state", "{invalid"))
    )
    entity = _make_entity(hass, client, api_mode="responses")
    chat_log = _make_chat_log(hass, conversation.UserContent(content="Hi"))

    with pytest.raises(ParseArgumentsFailed):
        await entity._async_handle_chat_log(chat_log, _function_tools(), [])


async def test_responses_usage_truncates_history(hass) -> None:
    """Usage above the context threshold clears the conversation history."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=_stream(_text_delta("Hello"), _completed(total_tokens=100))
    )
    entity = _make_entity(hass, client, api_mode="responses", context_threshold=50)
    chat_log = _make_chat_log(
        hass,
        conversation.UserContent(content="Old question"),
        conversation.AssistantContent(agent_id="conversation.test", content="Old"),
        conversation.UserContent(content="New question"),
    )

    await entity._async_handle_chat_log(chat_log, [], [])

    assert [content.content for content in chat_log.content] == [
        "You are a voice assistant.",
        "New question",
        "Hello",
    ]


async def test_responses_structured_output(hass) -> None:
    """Structured output is sent as a Responses API text format."""
    client = MagicMock()
    client.responses.create = AsyncMock(
        return_value=_stream(_text_delta('{"answer": "yes"}'), _completed())
    )
    entity = _make_entity(hass, client, api_mode="responses")
    chat_log = _make_chat_log(hass, conversation.UserContent(content="Hi"))
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    with patch.object(entity_module, "_format_structured_output", return_value=schema):
        await entity._async_handle_chat_log(
            chat_log,
            [],
            [],
            structure_name="Test Task",
            structure=vol.Schema({vol.Required("answer"): str}),
        )

    assert client.responses.create.call_args.kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "test_task",
            "strict": True,
            "schema": schema,
        }
    }
    assert "response_format" not in client.responses.create.call_args.kwargs


async def test_chat_completions_is_default(hass) -> None:
    """Without an API mode the Chat Completions request is unchanged."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_stream(
            ChatCompletionChunk.model_construct(
                choices=[
                    Choice.model_construct(
                        index=0,
                        delta=ChoiceDelta.model_construct(
                            content="Hi", tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        )
    )
    entity = _make_entity(hass, client, chat_model="gpt-4o-mini")
    chat_log = _make_chat_log(hass, conversation.UserContent(content="Hi"))

    await entity._async_handle_chat_log(chat_log, _function_tools(), [])

    client.responses.create.assert_not_called()
    assert client.chat.completions.create.call_args.kwargs == {
        "model": "gpt-4o-mini",
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 500,
        "top_p": 1,
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": "You are a voice assistant."},
            {"role": "user", "content": "Hi"},
        ],
        "tools": [{"type": "function", "function": FUNCTION_TOOL["spec"]}],
        "tool_choice": "auto",
    }
    assert chat_log.content[-1].content == "Hi"


async def test_chat_completions_gpt6_parameters(hass) -> None:
    """GPT-6 models get reasoning parameters on Chat Completions."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_stream())
    entity = _make_entity(
        hass, client, chat_model="gpt-6-luna", reasoning_effort="none"
    )
    chat_log = _make_chat_log(hass, conversation.UserContent(content="Hi"))

    await entity._async_handle_chat_log(chat_log, [], [])

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 500
    assert kwargs["reasoning_effort"] == "none"
    assert not {"max_tokens", "temperature", "top_p", "service_tier"} & kwargs.keys()


@pytest.mark.parametrize(
    ("client_factory", "expected_url"),
    [
        (
            lambda http_client: AsyncOpenAI(
                api_key="test-key",
                base_url="https://api.example.com/v1",
                http_client=http_client,
            ),
            "https://api.example.com/v1/responses",
        ),
        (
            lambda http_client: AsyncAzureOpenAI(
                api_key="test-key",
                azure_endpoint="https://example.openai.azure.com",
                api_version="2025-04-01-preview",
                http_client=http_client,
            ),
            "https://example.openai.azure.com/openai/responses?api-version=2025-04-01-preview",
        ),
    ],
    ids=["openai", "azure"],
)
async def test_responses_stream_with_sdk_clients(
    hass, client_factory, expected_url
) -> None:
    """OpenAI and Azure clients stream Responses API events end to end."""
    events = [
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
            "sequence_number": 1,
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": {
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
            },
        },
    ]
    body = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=body
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        entity = _make_entity(
            hass,
            client_factory(http_client),
            chat_model="gpt-6.1-sol",
            api_mode="responses",
        )
        chat_log = _make_chat_log(hass, conversation.UserContent(content="Hi"))

        await entity._async_handle_chat_log(chat_log, [], [])

    assert str(requests[0].url) == expected_url
    assert json.loads(requests[0].content)["store"] is False
    assert chat_log.content[-1].content == "Hello"
