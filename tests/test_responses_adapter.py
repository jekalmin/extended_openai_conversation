import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "responses_adapter",
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "extended_openai_conversation"
    / "responses_adapter.py",
)
responses_adapter = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(responses_adapter)

model_is_reasoning = responses_adapter.model_is_reasoning
responses_to_chat_like = responses_adapter.responses_to_chat_like


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def test_model_is_reasoning_matches_prefix_and_aliases():
    assert model_is_reasoning("gpt-5.5-preview")
    assert model_is_reasoning("gpt-4.1")
    assert model_is_reasoning("o3-mini")
    assert model_is_reasoning("custom-thinking-prototype")
    assert model_is_reasoning("o4-mini")
    assert not model_is_reasoning("gpt-4o")


def test_responses_to_chat_like_text_only():
    payload = {
        "id": "resp_123",
        "object": "response",
        "created": 1,
        "model": "gpt-5.1-mini",
        "output": [
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello from GPT-5"},
                ],
            }
        ],
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }

    response = FakeResponse(payload)
    normalized = responses_to_chat_like(response)
    assert normalized["choices"][0]["message"]["content"] == "Hello from GPT-5"
    assert "tool_calls" not in normalized["choices"][0]["message"]
    assert normalized["choices"][0]["finish_reason"] == "stop"
    assert normalized["usage"]["prompt_tokens"] == 20
    assert normalized["usage"]["completion_tokens"] == 10
    assert normalized["usage"]["total_tokens"] == 30


def test_responses_to_chat_like_with_tool_call():
    payload = {
        "id": "resp_456",
        "object": "response",
        "created": 2,
        "model": "gpt-5-thinking",
        "output": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "id": "toolu_1",
                            "name": "execute_services",
                            "arguments": {"domain": "rest_command", "service": "memory_write"},
                        },
                    }
                ],
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    normalized = responses_to_chat_like(FakeResponse(payload))

    tool_calls = normalized["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "execute_services"
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert arguments["domain"] == "rest_command"
    assert normalized["choices"][0]["finish_reason"] == "tool_calls"
    assert normalized["choices"][0]["message"]["content"] == ""


def test_responses_to_chat_like_with_multiple_tool_calls():
    payload = {
        "id": "resp_multi",
        "object": "response",
        "created": 3,
        "model": "o3-mini",
        "output": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "id": "call_1",
                            "name": "turn_on_light",
                            "arguments": {"entity_id": "light.kitchen"},
                        },
                    },
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "id": "call_2",
                            "name": "turn_on_light",
                            "arguments": {"entity_id": "light.patio"},
                        },
                    },
                ],
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }

    normalized = responses_to_chat_like(FakeResponse(payload))

    tool_calls = normalized["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    first_args = json.loads(tool_calls[0]["function"]["arguments"])
    second_args = json.loads(tool_calls[1]["function"]["arguments"])
    assert first_args["entity_id"] == "light.kitchen"
    assert second_args["entity_id"] == "light.patio"
