import asyncio
from types import SimpleNamespace

import test_streaming as streaming_setup
from openai.types.chat.chat_completion import ChatCompletion


def _build_user_input(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        conversation_id="conv-test",
        text=text,
        language="en",
        context=SimpleNamespace(user_id="user1"),
        device_id=None,
    )


async def _run_query(agent, user_input=None):
    if user_input is None:
        user_input = _build_user_input()
    messages = [
        {"role": "system", "content": "Base"},
        {"role": "user", "content": user_input.text},
    ]
    await agent.query(user_input, messages, [], 0)


def test_param_mapping_responses_reasoning(monkeypatch):
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
                completions=SimpleNamespace(create=lambda **kwargs: None)
            )

        def platform_headers(self):
            return {}

    monkeypatch.setattr(conversation, "AsyncOpenAI", lambda **kwargs: FakeClient())
    monkeypatch.setattr(conversation, "AsyncAzureOpenAI", lambda **kwargs: FakeClient())
    monkeypatch.setattr(conversation, "get_async_client", lambda hass: object())

    entry = streaming_setup.FakeEntry()
    entry.options.update(
        {
            streaming_setup.CONF_CHAT_MODEL: "gpt-5.1-thinking",
            streaming_setup.CONF_USE_RESPONSES_API: True,
            streaming_setup.CONF_ENABLE_STREAMING: False,
            const_module.CONF_FUNCTIONS: "[]",
            const_module.CONF_MAX_COMPLETION_TOKENS: 55,
        }
    )

    agent = conversation.OpenAIAgent(streaming_setup.FakeHass(), entry)
    asyncio.run(_run_query(agent))

    assert captured.kwargs is not None
    assert captured.kwargs.get("max_output_tokens") == 55
    assert "max_tokens" not in captured.kwargs


def test_param_mapping_chat_reasoning(monkeypatch):
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
                    "model": "gpt-5.1-thinking",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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

    entry = streaming_setup.FakeEntry()
    entry.options.update(
        {
            streaming_setup.CONF_CHAT_MODEL: "gpt-5.1-thinking",
            const_module.CONF_MODEL_STRATEGY: const_module.MODEL_STRATEGY_FORCE_CHAT,
            const_module.CONF_MAX_COMPLETION_TOKENS: 77,
            streaming_setup.CONF_ENABLE_STREAMING: False,
            const_module.CONF_FUNCTIONS: "[]",
        }
    )

    agent = conversation.OpenAIAgent(streaming_setup.FakeHass(), entry)
    asyncio.run(_run_query(agent))

    assert captured.kwargs is not None
    assert captured.kwargs.get("max_completion_tokens") == 77
    assert "max_tokens" not in captured.kwargs


def test_param_mapping_chat_non_reasoning(monkeypatch):
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
                    "model": kwargs.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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

    entry = streaming_setup.FakeEntry()
    entry.options.update(
        {
            streaming_setup.CONF_CHAT_MODEL: "gpt-4o-mini",
            const_module.CONF_MAX_TOKENS: 88,
            streaming_setup.CONF_ENABLE_STREAMING: False,
            const_module.CONF_FUNCTIONS: "[]",
        }
    )

    agent = conversation.OpenAIAgent(streaming_setup.FakeHass(), entry)
    asyncio.run(_run_query(agent))

    assert captured.kwargs is not None
    assert captured.kwargs.get("max_tokens") == 88
    assert "max_completion_tokens" not in captured.kwargs
