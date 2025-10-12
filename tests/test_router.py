import asyncio
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

import test_streaming as streaming_setup
from openai.types.chat.chat_completion import ChatCompletion

SPEC = importlib.util.spec_from_file_location(
    "router",
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "extended_openai_conversation"
    / "router.py",
)
router = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = router
SPEC.loader.exec_module(router)

classify_intent = router.classify_intent


def test_router_detects_memory_write():
    decision = classify_intent(
        "Remember that I like tea",
        write_pattern=r"^(remember|save|note that|add to memory)\b",
        search_pattern=r"^(what's my|what is my|do you remember|what did I say about)\b",
        force_tools=True,
    )

    assert decision.forced_tool == "memory.write"
    assert decision.chat_tool_choice["function"]["name"] == "memory.write"
    assert decision.responses_tool_choice["name"] == "memory.write"
    assert "I like tea" in decision.normalized_text


def test_router_detects_memory_search():
    decision = classify_intent(
        "Do you remember my favorite movie?",
        write_pattern=r"^(remember|save|note that|add to memory)\b",
        search_pattern=r"^(what's my|what is my|do you remember|what did I say about)\b",
        force_tools=True,
    )

    assert decision.forced_tool == "memory.search"
    assert decision.chat_tool_choice["function"]["name"] == "memory.search"
    assert decision.responses_tool_choice["name"] == "memory.search"
    assert "favorite movie" in decision.normalized_text


def test_force_write_tool_choice_shapes():
    decision = classify_intent(
        "Remember that the garage code is 1234",
        write_pattern=r"^(remember|save|note that|add to memory)\b",
        search_pattern=r"^(what's my|what is my|do you remember|what did I say about)\b",
        force_tools=True,
    )

    assert decision.chat_tool_choice == {
        "type": "function",
        "function": {"name": "memory.write"},
    }
    assert decision.responses_tool_choice == {
        "type": "tool",
        "name": "memory.write",
    }


def test_force_search_tool_choice_shapes():
    decision = classify_intent(
        "Do you remember my thermostat preference?",
        write_pattern=r"^(remember|save|note that|add to memory)\b",
        search_pattern=r"^(what's my|what is my|do you remember|what did I say about)\b",
        force_tools=True,
    )

    assert decision.chat_tool_choice == {
        "type": "function",
        "function": {"name": "memory.search"},
    }
    assert decision.responses_tool_choice == {
        "type": "tool",
        "name": "memory.search",
    }


def test_router_respects_force_toggle():
    decision = classify_intent(
        "Remember that I like tea",
        write_pattern=r"^(remember|save|note that|add to memory)\b",
        search_pattern=r"^(what's my|what is my|do you remember|what did I say about)\b",
        force_tools=False,
    )

    assert decision.forced_tool is None
    assert decision.chat_tool_choice == "auto"
    assert decision.responses_tool_choice == "auto"


def test_chat_tool_choice_uses_string_none_when_threshold_met(monkeypatch):
    conversation = streaming_setup.conversation
    const_module = streaming_setup.const_module

    class CapturingCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return ChatCompletion.model_validate(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )

    captured = CapturingCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=captured)
            self.responses = SimpleNamespace(create=lambda **kwargs: None)

        def platform_headers(self):
            return {}

    monkeypatch.setattr(conversation, "AsyncOpenAI", lambda **kwargs: FakeClient())
    monkeypatch.setattr(conversation, "AsyncAzureOpenAI", lambda **kwargs: FakeClient())
    monkeypatch.setattr(conversation, "get_async_client", lambda hass: object())
    monkeypatch.setattr(conversation, "detect_chat_tools_support", lambda method: True)
    monkeypatch.setattr(
        conversation, "detect_responses_max_tokens_param", lambda method: None
    )

    entry = streaming_setup.FakeEntry()
    entry.options.update(
        {
            streaming_setup.CONF_CHAT_MODEL: "gpt-4o-mini",
            streaming_setup.CONF_USE_TOOLS: True,
            streaming_setup.CONF_USE_RESPONSES_API: False,
            streaming_setup.CONF_ENABLE_STREAMING: False,
            const_module.CONF_FUNCTIONS: "[]",
        }
    )

    agent = conversation.OpenAIAgent(streaming_setup.FakeHass(), entry)

    user_input = SimpleNamespace(
        conversation_id="conv-test",
        text="hello",
        language="en",
        context=SimpleNamespace(user_id="user1"),
        device_id=None,
    )
    messages = [
        {"role": "system", "content": "Base"},
        {"role": "user", "content": "Hello"},
    ]

    async def run_query():
        await agent.query(
            user_input,
            messages,
            [],
            conversation.DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )

    asyncio.run(run_query())

    assert captured.kwargs is not None
    assert captured.kwargs.get("tool_choice") == "none"


def test_responses_tool_choice_stays_none_after_cap(monkeypatch):
    conversation = streaming_setup.conversation
    const_module = streaming_setup.const_module

    class CapturingResponses:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return streaming_setup.FakeResponse()

    captured = CapturingResponses()

    class FakeClient:
        def __init__(self):
            self.responses = SimpleNamespace(create=captured.create)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: pytest.fail(
                        "Chat path should not be used"
                    )
                )
            )

        def platform_headers(self):
            return {}

    monkeypatch.setattr(conversation, "AsyncOpenAI", lambda **kwargs: FakeClient())
    monkeypatch.setattr(conversation, "AsyncAzureOpenAI", lambda **kwargs: FakeClient())
    monkeypatch.setattr(conversation, "get_async_client", lambda hass: object())
    monkeypatch.setattr(
        conversation, "detect_chat_tools_support", lambda method: True
    )
    monkeypatch.setattr(
        conversation, "detect_responses_max_tokens_param", lambda method: None
    )

    entry = streaming_setup.FakeEntry()
    entry.options.update(
        {
            streaming_setup.CONF_CHAT_MODEL: "o3-test",
            streaming_setup.CONF_USE_RESPONSES_API: True,
            streaming_setup.CONF_ENABLE_STREAMING: False,
            streaming_setup.CONF_USE_TOOLS: True,
            const_module.CONF_FUNCTIONS: "[]",
        }
    )

    agent = conversation.OpenAIAgent(streaming_setup.FakeHass(), entry)

    user_input = SimpleNamespace(
        conversation_id="conv-test",
        text="hello",
        language="en",
        context=SimpleNamespace(user_id="user1"),
        device_id=None,
    )
    messages = [
        {"role": "system", "content": "Base"},
        {"role": "user", "content": "Hello"},
    ]

    async def run_query():
        await agent.query(
            user_input,
            messages,
            [],
            conversation.DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION + 1,
        )

    asyncio.run(run_query())

    assert captured.kwargs is not None
    assert captured.kwargs.get("tool_choice") == "none"
