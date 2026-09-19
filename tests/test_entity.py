"""Tests for entity.py message serialization helpers."""

from custom_components.extended_openai_conversation.entity import (
    _convert_content_to_param,
)
from homeassistant.components import conversation
from homeassistant.helpers import llm


def test_assistant_tool_call_without_text_has_content_key():
    """Assistant messages with tool_calls but no text must still include "content"."""
    content = conversation.AssistantContent(
        agent_id="test_agent",
        content=None,
        tool_calls=[
            llm.ToolInput(
                id="call_1",
                tool_name="turn_on_light",
                tool_args={"entity_id": "light.living_room"},
            )
        ],
    )

    messages = _convert_content_to_param([content])

    assert len(messages) == 1
    assert messages[0]["content"] is None


def test_assistant_text_only_content_unchanged():
    """Assistant messages that already carry text keep their content unchanged."""
    content = conversation.AssistantContent(
        agent_id="test_agent",
        content="Hello there",
    )

    messages = _convert_content_to_param([content])

    assert len(messages) == 1
    assert messages[0]["content"] == "Hello there"
