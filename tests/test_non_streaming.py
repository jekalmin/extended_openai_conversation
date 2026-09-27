"""Completed-response adapters for non-streaming OpenAI models."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.extended_openai_conversation_responses.non_streaming import (
    completed_chat_chunks,
    completed_responses_events,
)


async def _collect(iterator):
    return [item async for item in iterator]


@pytest.mark.asyncio
async def test_completed_responses_replays_text_annotations_refusal_and_items() -> None:
    annotation = SimpleNamespace(type="url_citation", url="https://example.invalid")
    message = SimpleNamespace(
        type="message",
        content=[
            SimpleNamespace(
                type="output_text",
                text="Complete answer",
                annotations=[annotation],
            ),
            SimpleNamespace(type="refusal", refusal="Cannot do that"),
            SimpleNamespace(type="other", value="ignored"),
        ],
    )
    function_call = SimpleNamespace(type="function_call", name="demo")
    response = SimpleNamespace(
        output=[message, function_call],
        status="completed",
    )

    events = await _collect(completed_responses_events(response))

    assert [event.type for event in events] == [
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.annotation.added",
        "response.refusal.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[1].delta == "Complete answer"
    assert (events[1].output_index, events[1].content_index) == (0, 0)
    assert events[2].annotation is annotation
    assert (events[2].output_index, events[2].content_index) == (0, 0)
    assert events[3].refusal == "Cannot do that"
    assert (events[3].output_index, events[3].content_index) == (0, 1)
    assert events[4].item is message
    assert events[6].item is function_call
    assert events[-1].response is response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "event_type"),
    [
        ("completed", "response.completed"),
        ("incomplete", "response.incomplete"),
        ("failed", "response.failed"),
        ("unexpected", "response.failed"),
    ],
)
async def test_completed_responses_maps_terminal_status_and_empty_text(
    status: str, event_type: str
) -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="", annotations=[]),
                    SimpleNamespace(type="refusal"),
                ],
            )
        ],
        status=status,
    )

    events = await _collect(completed_responses_events(response))

    assert [event.type for event in events] == [
        "response.output_item.added",
        "response.refusal.done",
        "response.output_item.done",
        event_type,
    ]
    assert events[1].refusal == ""


@pytest.mark.asyncio
async def test_completed_responses_defaults_missing_output_and_status() -> None:
    response = SimpleNamespace()

    events = await _collect(completed_responses_events(response))

    assert len(events) == 1
    assert events[0].type == "response.completed"
    assert events[0].response is response


@pytest.mark.asyncio
async def test_completed_chat_chunks_preserve_content_refusal_tools_finish_and_usage() -> None:
    first_function = SimpleNamespace(name="first", arguments='{"value":1}')
    second_function = SimpleNamespace(name="second", arguments="{}")
    choice = SimpleNamespace(
        message=SimpleNamespace(
            content="Finished",
            refusal=None,
            tool_calls=[
                SimpleNamespace(id="call-1", function=first_function),
                SimpleNamespace(id="call-2", function=second_function),
            ],
        ),
        finish_reason="tool_calls",
    )
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    response = SimpleNamespace(choices=[choice], usage=usage)

    chunks = await _collect(completed_chat_chunks(response))

    assert len(chunks) == 2
    completed = chunks[0]
    assert completed.usage is None
    assert len(completed.choices) == 1
    emitted = completed.choices[0]
    assert emitted.finish_reason == "tool_calls"
    assert emitted.delta.content == "Finished"
    assert emitted.delta.refusal is None
    assert [
        (tool.index, tool.id, tool.function)
        for tool in emitted.delta.tool_calls
    ] == [
        (0, "call-1", first_function),
        (1, "call-2", second_function),
    ]
    assert chunks[1].choices == []
    assert chunks[1].usage is usage


@pytest.mark.asyncio
async def test_completed_chat_chunks_handle_missing_tools_usage_and_empty_choices() -> None:
    choice = SimpleNamespace(
        message=SimpleNamespace(content=None, refusal="No"),
        finish_reason="stop",
    )

    chunks = await _collect(
        completed_chat_chunks(SimpleNamespace(choices=[choice], usage=None))
    )
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.tool_calls == []
    assert chunks[0].choices[0].delta.content is None
    assert chunks[0].choices[0].delta.refusal == "No"
    assert chunks[0].choices[0].finish_reason == "stop"

    assert await _collect(completed_chat_chunks(SimpleNamespace())) == []
