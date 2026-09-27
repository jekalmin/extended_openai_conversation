"""Long-lived public AI Task and conversation use under one real HA entry."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry, MockUser

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_SKIP_AUTHENTICATION,
    CONFIG_ENTRY_VERSION,
    DEFAULT_AI_TASK_OPTIONS,
    DOMAIN,
)
from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from tests_real_ha.test_ai_task_runtime import (
    CallerAPI,
    ContextProbeTool,
    FakeClient,
    FakeStream,
    _chunk,
    _tool_delta,
)
from tests_stress.conftest import record

_MARKER = re.compile(r"TASK_MARKER_[0-9]{4}")


def _subentry(kind: str, title: str, model: str) -> dict[str, Any]:
    options = dict(DEFAULT_AI_TASK_OPTIONS)
    options[CONF_API_MODE] = API_MODE_CHAT_COMPLETIONS
    options[CONF_CHAT_MODEL] = model
    return {
        "data": options,
        "subentry_type": kind,
        "title": title,
        "unique_id": None,
    }


@pytest.mark.asyncio
async def test_mixed_ai_tasks_remain_request_isolated_after_concurrency_and_reload(
    hass: HomeAssistant,
    stress_scale: int,
    stress_trace: list[dict],
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Mixed AI Task household",
        data={CONF_API_KEY: "sk-mixed-ai-task", CONF_SKIP_AUTHENTICATION: True},
        version=CONFIG_ENTRY_VERSION,
        subentries_data=[
            _subentry("ai_task_data", "Fast task", "gpt-5-mini"),
            _subentry("ai_task_data", "Detailed task", "gpt-5.6"),
            _subentry("conversation", "Household conversation", "gpt-5-mini"),
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED

    registry = er.async_get(hass)
    task_entities = {
        subentry.title: er.async_get(hass).async_get_entity_id(
            ai_task.DOMAIN, DOMAIN, subentry.subentry_id
        )
        for subentry in entry.subentries.values()
        if subentry.subentry_type == "ai_task_data"
    }
    assert len(task_entities) == 2
    assert all(task_entities.values())
    assert (
        len(
            [
                row
                for row in er.async_entries_for_config_entry(registry, entry.entry_id)
                if row.domain == ai_task.DOMAIN
            ]
        )
        == 2
    )
    assert conversation.async_get_agent(hass, entry.entry_id) is not None

    provider_calls: list[dict[str, Any]] = []
    failure_marker = "TASK_MARKER_9999"
    failed_once = False
    fast_model = "gpt-5-mini"

    async def create(**kwargs: Any) -> FakeStream:
        nonlocal failed_once
        provider_calls.append(kwargs)
        text = str(kwargs["messages"])
        markers = set(_MARKER.findall(text))
        if "CONVERSATION_MARKER" in text:
            assert not markers
            return FakeStream([_chunk(content="Conversation remains available")])
        assert len(markers) == 1, f"AI Task request leaked another task: {markers}"
        marker = next(iter(markers))
        expected_model = fast_model if int(marker[-4:]) % 2 == 0 else "gpt-5.6"
        assert kwargs["model"] == expected_model
        if marker == "TASK_MARKER_9001":
            assert len(kwargs["tools"]) == 1
            if any(message["role"] == "tool" for message in kwargs["messages"]):
                assert "from-enhanced-task" in text
                return FakeStream([_chunk(content="Tool task completed")])
            alias = kwargs["tools"][0]["function"]["name"]
            return FakeStream(
                [
                    _chunk(
                        content=None,
                        finish_reason="tool_calls",
                        tool_calls=[
                            _tool_delta(
                                0, alias, '{"value":"from-enhanced-task"}', "task-call"
                            )
                        ],
                    )
                ]
            )
        if marker == failure_marker and not failed_once:
            failed_once = True
            raise RuntimeError("deterministic provider failure")
        return FakeStream([_chunk(content=f"Result {marker}")])

    def install_client() -> None:
        client = FakeClient([])
        client.completions.create = create
        entry.runtime_data = client

    install_client()

    async def task(index: int) -> None:
        marker = f"TASK_MARKER_{index:04d}"
        title = "Fast task" if index % 2 == 0 else "Detailed task"
        result = await ai_task.async_generate_data(
            hass,
            task_name=f"Mixed task {index}",
            entity_id=task_entities[title],
            instructions=f"Reply to {marker} only.",
        )
        assert result.data == f"Result {marker}"

    sequential = 20 * stress_scale
    concurrent = 8 * stress_scale
    for index in range(sequential):
        await task(index)
    await asyncio.gather(
        *(task(index) for index in range(sequential, sequential + concurrent))
    )

    result = await conversation.async_converse(
        hass=hass,
        text="CONVERSATION_MARKER",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
    )
    assert (
        "Conversation remains available"
        in result.response.as_dict()["speech"]["plain"]["speech"]
    )

    # The next task reads changed configuration; existing task history stays absent.
    fast_subentry = next(
        subentry
        for subentry in entry.subentries.values()
        if subentry.title == "Fast task"
    )
    hass.config_entries.async_update_subentry(
        entry,
        fast_subentry,
        data={**fast_subentry.data, CONF_CHAT_MODEL: "gpt-5.6"},
    )
    await hass.async_block_till_done()
    # HA may reload the parent entry after a subentry edit, replacing the SDK
    # client. Reinstall only the deterministic provider seam on that new client.
    install_client()
    fast_model = "gpt-5.6"
    await task(sequential + concurrent + 2)

    with pytest.raises(Exception, match="deterministic provider failure"):
        await ai_task.async_generate_data(
            hass,
            task_name="Failed task",
            entity_id=task_entities["Detailed task"],
            instructions=failure_marker,
        )
    await task(sequential + concurrent + 4)

    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    install_client()
    for index in range(sequential + concurrent + 5, sequential + concurrent + 9):
        await task(index)

    # Mix a caller-owned HA tool into the same long-lived two-agent sequence.
    tool = ContextProbeTool()
    caller_api = CallerAPI(hass=hass, id="enhanced-task", name="Enhanced Task")
    caller_api.tools = [tool]
    user = MockUser(id="enhanced-task-user", name="Enhanced Task User")
    user.add_to_hass(hass)
    context = Context(user_id=user.id)
    tool_result = await ai_task.async_generate_data(
        hass,
        task_name="Caller tool after prolonged use",
        entity_id=task_entities["Detailed task"],
        instructions="Use your tool for TASK_MARKER_9001.",
        llm_api=caller_api,
        context=context,
    )
    assert tool_result.data == "Tool task completed"
    assert len(tool.calls) == 1
    assert tool.calls[0][1].context is context
    assert len(provider_calls) == sequential + concurrent + 10
    record(
        stress_trace,
        "summary",
        layer="real-ha + fake-sdk-client",
        ai_task_turns=sequential + concurrent + 7,
        ai_task_concurrent=concurrent,
        ai_task_agents=2,
        ai_task_provider_failures=1,
        public_turns=sequential + concurrent + 8,
        provider_requests=len(provider_calls),
        actual_tool_executions=1,
    )
