"""Real Home Assistant acceptance tests for the AI Task runtime path."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry, MockUser
import voluptuous as vol

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_SKIP_AUTHENTICATION,
    CONFIG_ENTRY_VERSION,
    DEFAULT_AI_TASK_OPTIONS,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DOMAIN,
)
from homeassistant.components import ai_task, media_source
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, llm


class FakeStream:
    """Small async iterator matching the OpenAI streaming boundary."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield chunk


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = "stop",
    tool_calls: list[Any] | None = None,
) -> Any:
    """Build the Chat Completions fields consumed by the integration."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    refusal=None,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _tool_delta(index: int, name: str, arguments: str, call_id: str) -> Any:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


Outcome = str | BaseException | Callable[[dict[str, Any]], list[Any]]


class FakeCompletions:
    """Deterministic fake at the provider SDK boundary."""

    def __init__(self, outcomes: list[Outcome]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("Unexpected provider request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return FakeStream(outcome(kwargs))
        return FakeStream([_chunk(content=outcome)])


class FakeClient:
    """Only the Chat Completions SDK surface used by these tests."""

    def __init__(self, outcomes: list[Outcome]) -> None:
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


class ContextProbeTool(llm.Tool):
    """Caller-owned HA LLM tool that records the authoritative request context."""

    name = "context_probe"
    description = "Return the supplied value and record the Home Assistant context."
    parameters = vol.Schema({vol.Required("value"): str})

    def __init__(self) -> None:
        self.calls: list[tuple[llm.ToolInput, llm.LLMContext]] = []

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        self.calls.append((tool_input, llm_context))
        return {"echo": tool_input.tool_args["value"]}


class CallerAPI(llm.API):
    """Minimal caller-supplied HA LLM API for the genuine AI Task path."""

    tools: list[llm.Tool]

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt="Caller API prompt.",
            llm_context=llm_context,
            tools=list(self.tools),
        )


def _entry(api_mode: str = API_MODE_CHAT_COMPLETIONS) -> MockConfigEntry:
    """Build one local-only integration entry with one AI Task subentry."""
    options = dict(DEFAULT_AI_TASK_OPTIONS)
    options[CONF_API_MODE] = api_mode
    return MockConfigEntry(
        domain=DOMAIN,
        title="AI Task Runtime Acceptance",
        data={
            CONF_API_KEY: "sk-ai-task-acceptance",
            CONF_SKIP_AUTHENTICATION: True,
        },
        version=CONFIG_ENTRY_VERSION,
        subentries_data=[
            {
                "data": options,
                "subentry_type": "ai_task_data",
                "title": "AI Task Runtime",
                "unique_id": None,
            }
        ],
    )


async def _setup_entry(
    hass: HomeAssistant,
    outcomes: list[Outcome],
) -> tuple[MockConfigEntry, str, FakeClient]:
    """Load the real integration and replace only its provider SDK client."""
    entry = _entry()
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED

    subentry = next(
        item
        for item in entry.subentries.values()
        if item.subentry_type == "ai_task_data"
    )
    entity_id = er.async_get(hass).async_get_entity_id(
        ai_task.DOMAIN, DOMAIN, subentry.subentry_id
    )
    assert entity_id is not None

    client = FakeClient(outcomes)
    entry.runtime_data = client
    return entry, entity_id, client


def _flatten_text(messages: list[Any]) -> str:
    """Return request text without depending on exact provider message nesting."""
    return str(messages)


@pytest.mark.asyncio
async def test_generate_data_uses_real_ha_runtime_and_does_not_leak_previous_task(
    hass: HomeAssistant,
) -> None:
    """Public HA AI Task calls reach the entity and each fresh task has fresh input."""
    _entry_obj, entity_id, client = await _setup_entry(
        hass, ["First result", "Second result"]
    )

    first = await ai_task.async_generate_data(
        hass,
        task_name="First Task",
        entity_id=entity_id,
        instructions="FIRST_TASK_MARKER",
    )
    second = await ai_task.async_generate_data(
        hass,
        task_name="Second Task",
        entity_id=entity_id,
        instructions="SECOND_TASK_MARKER",
    )

    assert first.data == "First result"
    assert second.data == "Second result"
    assert len(client.completions.calls) == 2
    first_messages = _flatten_text(client.completions.calls[0]["messages"])
    second_messages = _flatten_text(client.completions.calls[1]["messages"])
    assert "FIRST_TASK_MARKER" in first_messages
    assert "SECOND_TASK_MARKER" in second_messages
    assert "FIRST_TASK_MARKER" not in second_messages


@pytest.mark.asyncio
async def test_generate_data_structured_output_flows_through_home_assistant(
    hass: HomeAssistant,
) -> None:
    """HA's public task API returns parsed structured data and requests JSON schema."""
    _entry_obj, entity_id, client = await _setup_entry(
        hass, ['{"answer":"ready","count":2}']
    )
    structure = vol.Schema(
        {
            vol.Required("answer"): str,
            vol.Required("count"): int,
        }
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="Structured Runtime Task",
        entity_id=entity_id,
        instructions="Return the requested fields.",
        structure=structure,
    )

    assert result.data == {"answer": "ready", "count": 2}
    request = client.completions.calls[0]
    response_format = request["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == {"answer", "count"}


@pytest.mark.asyncio
async def test_generate_data_image_attachment_reaches_provider_request(
    hass: HomeAssistant,
    tmp_path: Path,
) -> None:
    """HA resolves a local image attachment and the integration sends its bytes."""
    _entry_obj, entity_id, client = await _setup_entry(hass, ["Image inspected"])
    image_path = tmp_path / "task-image.png"
    image_path.write_bytes(b"deterministic-image-bytes")

    with patch(
        "homeassistant.components.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url="http://example.invalid/task-image.png",
            mime_type="image/png",
            path=image_path,
        ),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Attachment Runtime Task",
            entity_id=entity_id,
            instructions="Inspect the attached image.",
            attachments=[
                {
                    "media_content_id": "media-source://local/task-image.png",
                    "media_content_type": "image/png",
                }
            ],
        )

    assert result.data == "Image inspected"
    user_message = next(
        message
        for message in reversed(client.completions.calls[0]["messages"])
        if message.get("role") == "user"
    )
    assert isinstance(user_message["content"], list)
    assert user_message["content"][0] == {
        "type": "text",
        "text": "Inspect the attached image.",
    }
    image_part = user_message["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_caller_llm_tool_executes_with_the_original_home_assistant_context(
    hass: HomeAssistant,
) -> None:
    """Caller-supplied HA tools execute through the AI Task exchange with its Context."""

    def request_tool(kwargs: dict[str, Any]) -> list[Any]:
        tools = kwargs["tools"]
        assert len(tools) == 1
        alias = tools[0]["function"]["name"]
        return [
            _chunk(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    _tool_delta(0, alias, '{"value":"from-model"}', "call-1")
                ],
            )
        ]

    _entry_obj, entity_id, client = await _setup_entry(
        hass, [request_tool, "Tool round trip complete"]
    )
    tool = ContextProbeTool()
    caller_api = CallerAPI(hass=hass, id="runtime-probe", name="Runtime Probe")
    caller_api.tools = [tool]
    user = MockUser(id="ai-task-runtime-user", name="AI Task Runtime User")
    user.add_to_hass(hass)
    context = Context(user_id=user.id)

    result = await ai_task.async_generate_data(
        hass,
        task_name="Tool Runtime Task",
        entity_id=entity_id,
        instructions="Use the available tool once.",
        llm_api=caller_api,
        context=context,
    )

    assert result.data == "Tool round trip complete"
    assert len(client.completions.calls) == 2
    assert len(tool.calls) == 1
    tool_input, llm_context = tool.calls[0]
    assert tool_input.tool_name == "context_probe"
    assert tool_input.tool_args == {"value": "from-model"}
    assert llm_context.context is context
    assert llm_context.context.user_id == user.id
    second_messages = _flatten_text(client.completions.calls[1]["messages"])
    assert "from-model" in second_messages


@pytest.mark.asyncio
async def test_ai_task_tool_budget_rejects_an_over_limit_provider_batch(
    hass: HomeAssistant,
) -> None:
    """The AI Task runtime applies the same request-local tool execution budget."""

    def over_budget(kwargs: dict[str, Any]) -> list[Any]:
        alias = kwargs["tools"][0]["function"]["name"]
        count = DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION + 1
        return [
            _chunk(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    _tool_delta(
                        index,
                        alias,
                        '{"value":"budget"}',
                        f"call-{index}",
                    )
                    for index in range(count)
                ],
            )
        ]

    _entry_obj, entity_id, _client = await _setup_entry(hass, [over_budget])
    tool = ContextProbeTool()
    caller_api = CallerAPI(hass=hass, id="budget-probe", name="Budget Probe")
    caller_api.tools = [tool]

    with pytest.raises(
        HomeAssistantError,
        match=(
            rf"Function call limit of {DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION} "
            r"reached"
        ),
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Budget Runtime Task",
            entity_id=entity_id,
            instructions="Attempt too many tool calls.",
            llm_api=caller_api,
            context=Context(user_id="budget-user"),
        )

    assert tool.calls == []


@pytest.mark.asyncio
async def test_malformed_provider_stream_fails_through_public_ai_task_api(
    hass: HomeAssistant,
) -> None:
    """A partial provider stream cannot be mistaken for a successful AI Task result."""

    def incomplete_stream(_kwargs: dict[str, Any]) -> list[Any]:
        return [_chunk(content="partial", finish_reason=None)]

    _entry_obj, entity_id, _client = await _setup_entry(hass, [incomplete_stream])

    with pytest.raises(
        HomeAssistantError,
        match="stream ended before a terminal finish reason",
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Malformed Runtime Task",
            entity_id=entity_id,
            instructions="This provider response is intentionally incomplete.",
        )
