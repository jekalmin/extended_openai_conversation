"""Tests for tool result handling across Home Assistant versions."""

from custom_components.extended_openai_conversation.entity import (
    _convert_content_to_param,
    _make_tool_result_content,
    _tool_result_data,
)
from homeassistant.components import conversation


def test_make_tool_result_content_round_trip() -> None:
    """A built tool result exposes its data on the running HA version."""
    content = _make_tool_result_content(
        agent_id="conversation.test",
        tool_call_id="call_1",
        tool_name="execute_services",
        data={"result": "ok"},
    )

    assert isinstance(content, conversation.ToolResultContent)
    assert content.tool_call_id == "call_1"
    assert content.tool_name == "execute_services"
    assert _tool_result_data(content) == {"result": "ok"}


def test_tool_result_converted_to_tool_message() -> None:
    """A tool result in the chat log becomes an OpenAI tool message."""
    content = _make_tool_result_content(
        agent_id="conversation.test",
        tool_call_id="call_1",
        tool_name="execute_services",
        data={"result": "ok"},
    )

    messages = _convert_content_to_param([content])

    assert messages == [
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result":"ok"}'}
    ]
