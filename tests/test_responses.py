"""Responses transport tests using real HA chat logs and mocked API streams."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
import probatio
import pytest
import voluptuous as vol

from custom_components.extended_openai_conversation import entity, responses
from custom_components.extended_openai_conversation.const import (
    RESPONSES_MODEL_REASONING_EFFORT_OPTIONS,
)
from custom_components.extended_openai_conversation.exceptions import (
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from custom_components.extended_openai_conversation.helpers import (
    get_model_config,
    get_reasoning_effort_options,
)
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import DATA_CHAT_LOGS
from homeassistant.exceptions import HomeAssistantError

MODEL = "gpt-6-luna"
AGENT = "conversation.test"
TOOLS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Execute services",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "native"},
    }
]


class Stream:
    """Closeable stream that may break before a terminal event."""

    def __init__(self, events):
        self.events = events
        self.close = AsyncMock()

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, BaseException):
                raise event
            yield event


def call(call_id="call_a", name="execute_services", arguments="{}"):
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}",
        call_id=call_id,
        type="function_call",
        name=name,
        arguments=arguments,
        status="completed",
    )


def reasoning():
    return ResponseReasoningItem(
        id="rs_test",
        type="reasoning",
        summary=[],
        encrypted_content="encrypted-test",
    )


def completed(*output):
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            output=list(output),
            usage=SimpleNamespace(
                input_tokens=40,
                output_tokens=20,
                total_tokens=60,
            ),
        ),
    )


def text(value="Gotowe."):
    return SimpleNamespace(type="response.output_text.delta", delta=value)


def make_log(hass, name="test"):
    return conversation.ChatLog(
        hass,
        name,
        content=[
            conversation.SystemContent(content="Odpowiadaj po polsku."),
            conversation.UserContent(content="Jaka jest temperatura?"),
        ],
    )


def make_entity(streams, options=None):
    instance = entity.ExtendedOpenAIBaseLLMEntity(
        SimpleNamespace(),
        SimpleNamespace(
            subentry_id="test",
            title="Test",
            data={
                "chat_model": MODEL,
                "max_tokens": 3000,
                **(options or {}),
            },
        ),
    )
    instance.entity_id = AGENT
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(side_effect=streams))
    )
    return instance, client


async def run(instance, client, log, tools=None, **kwargs):
    with patch.object(
        type(instance), "_client", new_callable=lambda: property(lambda self: client)
    ):
        await instance._async_handle_chat_log(log, tools or [], [], **kwargs)


@pytest.mark.parametrize(
    "model,effort",
    [
        (model, effort)
        for model, levels in RESPONSES_MODEL_REASONING_EFFORT_OPTIONS.items()
        for effort in levels
    ],
)
def test_parameters(model, effort):
    params = responses.request_parameters(
        model,
        {
            "reasoning_effort": effort,
            "max_tokens": 3000,
            "service_tier": "auto",
            "temperature": 0.8,
            "top_p": 0.6,
        },
        TOOLS,
    )
    assert params["reasoning"] == {"effort": effort}
    assert params["model"] == model
    assert params["max_output_tokens"] == 3000
    assert params["service_tier"] == "auto"
    assert params["store"] is False
    assert params["include"] == ["reasoning.encrypted_content"]
    assert (
        not {"max_tokens", "max_completion_tokens", "temperature", "top_p"}
        & params.keys()
    )
    assert params["tools"][0]["name"] == "execute_services"
    assert params["tools"][0]["strict"] is False
    assert "strict" not in TOOLS[0]["spec"]


def test_model_routing_and_levels():
    for model, levels in RESPONSES_MODEL_REASONING_EFFORT_OPTIONS.items():
        assert get_model_config(model)["uses_responses"]
        assert get_reasoning_effort_options(model) == levels
        assert get_model_config(model.upper())["uses_responses"]
        assert get_reasoning_effort_options(model.upper()) == levels
    for model in ("gpt-5-mini", "gpt-4o", "gpt-6.1-unknown", "custom-model"):
        assert not get_model_config(model).get("uses_responses")
        assert get_reasoning_effort_options(model) == ["low", "medium", "high"]


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
def test_none_is_rejected_for_models_that_require_reasoning(model):
    assert "none" not in get_reasoning_effort_options(model)
    with pytest.raises(HomeAssistantError, match="Unsupported reasoning effort"):
        responses.request_parameters(model, {"reasoning_effort": "none"}, TOOLS)


def test_bad_effort_and_extra_body_are_explicit_errors():
    for options in ({"reasoning_effort": "invalid"}, {"extra_body": '{"store":true}'}):
        with pytest.raises(HomeAssistantError):
            responses.request_parameters(MODEL, options, [])


async def test_real_chatlog_stream_and_native_history(hass):
    log = make_log(hass)
    stats = {}
    stream = Stream([text("Cześć"), text("!"), completed(reasoning())])
    added = [
        item
        async for item in log.async_add_delta_content_stream(
            AGENT,
            responses.transform_stream(stream, MODEL, 3000, stats),
        )
    ]
    assert added[-1].content == "Cześć!"
    assert added[-1].native["output"][0]["encrypted_content"] == "encrypted-test"
    assert "native" not in added[-1].as_dict()
    before = deepcopy(log.content[-1].native)
    log.content[0] = conversation.SystemContent(content="Nowy stan domu")
    log.content.append(conversation.UserContent(content="A w kuchni?"))
    items = responses.response_input(log, MODEL)
    assert items[0]["content"] == "Nowy stan domu"
    assert items[2]["type"] == "reasoning"
    assert items[-1]["content"] == "A w kuchni?"
    assert log.content[-2].native == before
    assert stats == {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}


@pytest.mark.parametrize("model", RESPONSES_MODEL_REASONING_EFFORT_OPTIONS)
async def test_functions_and_error_results_share_existing_executor(hass, model):
    first = Stream([completed(reasoning(), call(), call("call_b"))])
    second = Stream([text(), completed()])
    instance, client = make_entity([first, second], {"chat_model": model})
    log = make_log(hass)

    async def execute(config, tool, *unused):
        if tool.id == "call_a":
            raise HomeAssistantError("Service unavailable")
        return entity._make_tool_result_content(
            AGENT, tool.id, tool.tool_name, {"result": "ok"}
        )

    with patch.object(instance, "_execute_function_tool", side_effect=execute) as mock:
        await run(instance, client, log, TOOLS)
    assert mock.call_count == 2
    request = client.responses.create.call_args_list[1].kwargs
    assert request["model"] == model
    outputs = [
        item for item in request["input"] if item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in outputs] == ["call_a", "call_b"]
    assert "Service unavailable" in outputs[0]["output"]
    assert '"ok"' in outputs[1]["output"]
    assert first.close.await_count == second.close.await_count == 1
    assert log.content[-1].content == "Gotowe."


async def test_fragmented_arguments_only_execute_after_completed(hass):
    partial = SimpleNamespace(
        type="response.function_call_arguments.delta", delta='{"x":'
    )
    stream = Stream([partial, completed(call(arguments='{"x":1}'))])
    deltas = [
        item async for item in responses.transform_stream(stream, MODEL, 3000, {})
    ]
    calls = deltas[-1]["tool_calls"]
    assert len(calls) == 1 and calls[0].tool_args == {"x": 1}
    assert not any("tool_calls" in item for item in deltas[:-1])


@pytest.mark.parametrize("failure", [RuntimeError("broken"), asyncio.CancelledError()])
async def test_interrupted_stream_does_not_execute_actions(hass, failure):
    stream = Stream(
        [SimpleNamespace(type="response.output_item.done", item=call()), failure]
    )
    instance, client = make_entity([stream])
    with (
        patch.object(
            instance, "_execute_function_tool", new_callable=AsyncMock
        ) as execute,
        pytest.raises(type(failure)),
    ):
        await run(instance, client, make_log(hass), TOOLS)
    execute.assert_not_awaited()
    stream.close.assert_awaited_once()


async def test_mid_execution_cancellation_is_not_replayed(hass):
    instance, client = make_entity([Stream([completed(call(), call("call_b"))])])
    log = make_log(hass)

    async def execute(config, tool, *unused):
        if tool.id == "call_b":
            raise asyncio.CancelledError
        return entity._make_tool_result_content(
            AGENT, tool.id, tool.tool_name, {"result": "ok"}
        )

    with (
        patch.object(instance, "_execute_function_tool", side_effect=execute),
        pytest.raises(asyncio.CancelledError),
    ):
        await run(instance, client, log, TOOLS)
    log.content.append(conversation.UserContent(content="Co się stało?"))
    inputs = responses.response_input(log, MODEL)
    results = [item for item in inputs if item.get("type") == "function_call_output"]
    assert len(results) == 2
    assert "ok" in results[0]["output"]
    assert "interrupted_tool_call" in results[1]["output"]
    # Recovery conversion never invokes a tool.
    assert responses.response_input(log, MODEL) == inputs


async def test_call_limit_and_unknown_tool(hass):
    instance, client = make_entity(
        [
            Stream([completed(call(name="missing"), call("call_b"))]),
            Stream([text(), completed()]),
        ],
        {"max_function_calls_per_conversation": 1},
    )
    log = make_log(hass)
    with patch.object(
        instance, "_execute_function_tool", new_callable=AsyncMock
    ) as execute:
        await run(instance, client, log, TOOLS)
    execute.assert_not_awaited()
    assert client.responses.create.call_args_list[1].kwargs["tool_choice"] == "none"
    results = [
        entity._tool_result_data(item)
        for item in log.content
        if item.role == "tool_result"
    ]
    assert results[0]["error"] == "FunctionNotFound"
    assert "Maximum function" in results[1]["message"]


@pytest.mark.parametrize(
    "events,error",
    [
        ([], HomeAssistantError),
        (
            [
                SimpleNamespace(
                    type="response.incomplete",
                    response=SimpleNamespace(
                        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    ),
                )
            ],
            TokenLengthExceededError,
        ),
        (
            [
                SimpleNamespace(
                    type="response.failed", response=SimpleNamespace(error="failure")
                )
            ],
            HomeAssistantError,
        ),
        ([completed(call(arguments="{"))], ParseArgumentsFailed),
    ],
)
async def test_terminal_errors(events, error):
    with pytest.raises(error):
        _ = [
            delta
            async for delta in responses.transform_stream(
                Stream(events), MODEL, 3000, {}
            )
        ]


@pytest.mark.parametrize(
    "schema",
    [
        vol.Schema({vol.Required("value"): int}),
        probatio.Schema({probatio.Required("value"): int}),
    ],
)
async def test_ai_task_structured_response(hass, schema):
    instance, client = make_entity([Stream([text('{"value":1}'), completed()])])
    await run(
        instance,
        client,
        make_log(hass),
        structure_name="Task",
        structure=schema,
    )
    params = client.responses.create.call_args.kwargs
    assert params["text"]["format"]["type"] == "json_schema"
    assert params["text"]["format"]["strict"] is True
    assert "response_format" not in params
    assert "tools" not in params


async def test_session_isolation_clear_and_reload(hass):
    first, second = make_log(hass, "first"), make_log(hass, "second")
    async for _ in first.async_add_delta_content_stream(
        AGENT,
        responses.transform_stream(
            Stream([text(), completed(reasoning())]),
            MODEL,
            3000,
            {},
        ),
    ):
        pass
    assert not any(
        item.get("type") == "reasoning"
        for item in responses.response_input(second, MODEL)
    )
    hass.data[DATA_CHAT_LOGS] = {"first": first, "second": second}
    responses.clear_native_history(hass, {AGENT})
    assert first.content[-1].native is None
    assert first.content[-1].content == "Gotowe."
    first.content[:] = first.content[:2]
    assert not any(
        item.get("type") == "reasoning"
        for item in responses.response_input(first, MODEL)
    )
