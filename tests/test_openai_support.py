import importlib.util
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location(
    "openai_support",
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "extended_openai_conversation"
    / "openai_support.py",
)
openai_support = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(openai_support)

detect_responses_max_tokens_param = openai_support.detect_responses_max_tokens_param
detect_chat_tools_support = openai_support.detect_chat_tools_support


async def _responses_create_with_output(
    *,
    model: str,
    input: list[dict[str, Any]],
    max_output_tokens: int | None = None,
) -> None:
    return None


async def _responses_create_without_tokens(
    *,
    model: str,
    input: list[dict[str, Any]],
) -> None:
    return None


async def _chat_create_with_tools(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> None:
    return None


async def _chat_create_without_tools(
    *,
    model: str,
    messages: list[dict[str, Any]],
) -> None:
    return None


def test_detect_responses_max_tokens_param_handles_missing_param():
    assert (
        detect_responses_max_tokens_param(_responses_create_with_output) ==
        "max_output_tokens"
    )
    assert detect_responses_max_tokens_param(_responses_create_without_tokens) is None


def test_detect_chat_tools_support():
    assert detect_chat_tools_support(_chat_create_with_tools)
    assert not detect_chat_tools_support(_chat_create_without_tools)
