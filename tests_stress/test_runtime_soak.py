"""Long, seeded public-API lifecycle journey on genuine Home Assistant."""

from __future__ import annotations

import asyncio
import gc
import random

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry, MockUser

from custom_components.extended_openai_conversation_responses.const import (
    CONF_SKIP_AUTHENTICATION,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
)
from custom_components.extended_openai_conversation_responses.conversation import (
    ExtendedOpenAIAgentEntity,
)
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from tests_stress.conftest import record

_MANAGER_REGISTRIES = (
    "memory_managers",
    "temporary_memory_managers",
    "knowledge_managers",
    "archive_managers",
    "usage_managers",
    "guest_mode_managers",
    "request_rule_managers",
    "request_rule_runtimes",
    "continuity_managers",
)


def _resource_footprint(hass: HomeAssistant) -> dict[str, int]:
    """Count EOAI-owned registries that should stabilize after warm setup."""
    counts = {
        name: len(hass.data.get(f"{DOMAIN}.{name}", {})) for name in _MANAGER_REGISTRIES
    }
    counts["registered_services"] = len(hass.services.async_services().get(DOMAIN, {}))
    return counts


def _eoai_task_count() -> int:
    """Count live tasks whose coroutine body belongs to the integration."""
    return sum(
        "custom_components/extended_openai_conversation_responses/"
        in getattr(
            getattr(task.get_coro(), "cr_code", None), "co_filename", ""
        ).replace("\\", "/")
        for task in asyncio.all_tasks()
        if not task.done()
    )


def _entry(number: int) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title=f"Nightly agent {number}",
        data={CONF_API_KEY: "sk-local-stress", CONF_SKIP_AUTHENTICATION: True},
        version=CONFIG_ENTRY_VERSION,
        subentries_data=[
            {
                "data": {},
                "subentry_type": "conversation",
                "title": f"Agent {number}",
                "unique_id": None,
            }
        ],
    )


def _entity_ids(hass: HomeAssistant, entry: MockConfigEntry) -> set[str]:
    registry = er.async_get(hass)
    return {
        item.entity_id
        for item in er.async_entries_for_config_entry(registry, entry.entry_id)
    }


@pytest.mark.asyncio
async def test_seeded_multi_entry_runtime_soak(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_seed: int,
    stress_scale: int,
    stress_trace: list[dict],
) -> None:
    rng = random.Random(stress_seed ^ 0xC07E)
    for user in range(6):
        MockUser(id=f"nightly-user-{user}", name=f"Nightly user {user}").add_to_hass(
            hass
        )
    entries = [_entry(index) for index in range(2)]
    for entry in entries:
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    baseline = {entry.entry_id: _entity_ids(hass, entry) for entry in entries}
    assert all(baseline.values())

    # Each conversation uses a permanent marker. The provider seam inspects the
    # actual ChatLog passed by the public Assist API on every request.
    conversations: dict[tuple[int, int], str] = {}
    markers = {
        (agent, user): f"private-agent-{agent}-user-{user}"
        for agent in range(2)
        for user in range(6)
    }
    calls = 0

    def install_model(agent_index: int) -> None:
        agent = conversation.async_get_agent(hass, entries[agent_index].entry_id)
        assert isinstance(agent, ExtendedOpenAIAgentEntity)

        async def model(log: conversation.ChatLog, **kwargs) -> None:
            del kwargs
            nonlocal calls
            parts = [
                item.content
                for item in log.content
                if isinstance(getattr(item, "content", None), str)
            ]
            current = parts[-1]
            marker = next(value for value in markers.values() if value in current)
            joined = "\n".join(parts)
            assert all(
                other == marker or other not in joined for other in markers.values()
            ), (marker, joined)
            calls += 1
            log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=agent.entity_id, content=f"ok:{current}"
                )
            )

        monkeypatch.setattr(agent, "_async_handle_chat_log", model)

    async def converse(agent_index: int, user_index: int, number: int) -> None:
        key = (agent_index, user_index)
        marker = markers[key]
        message = f"turn {number} {marker}"
        result = await conversation.async_converse(
            hass=hass,
            text=message,
            conversation_id=conversations.get(key),
            context=Context(user_id=f"nightly-user-{user_index}"),
            language="en",
            agent_id=entries[agent_index].entry_id,
        )
        assert result.response.as_dict()["speech"]["plain"]["speech"] == f"ok:{message}"
        assert result.conversation_id
        conversations[key] = result.conversation_id

    for index in range(2):
        install_model(index)
    for index in range(2):
        await converse(index, 0, -1)
    warm_resources = _resource_footprint(hass)
    gc.collect()
    warm_tasks = _eoai_task_count()
    for number in range(120 * stress_scale):
        agent_index = rng.randrange(2)
        roll = rng.random()
        if roll < 0.77:
            user_index = rng.randrange(6)
            record(stress_trace, "conversation", agent=agent_index, user=user_index)
            await converse(agent_index, user_index, number)
        elif roll < 0.9:
            users = rng.sample(range(6), 4)
            record(
                stress_trace, "concurrent_conversations", agent=agent_index, users=users
            )
            await asyncio.gather(
                *(converse(agent_index, user, number) for user in users)
            )
        else:
            entry = entries[agent_index]
            record(stress_trace, "unload_setup", agent=agent_index)
            assert await hass.config_entries.async_unload(entry.entry_id)
            await hass.async_block_till_done()
            assert entry.state is ConfigEntryState.NOT_LOADED
            assert conversation.async_get_agent(hass, entry.entry_id) is None
            assert await hass.config_entries.async_setup(entry.entry_id)
            await hass.async_block_till_done()
            assert entry.state is ConfigEntryState.LOADED
            install_model(agent_index)
            # A reload may discard ChatLogs; start fresh sessions after it.
            for key in list(conversations):
                if key[0] == agent_index:
                    del conversations[key]
        assert all(entry.state is ConfigEntryState.LOADED for entry in entries)
        assert all(
            _entity_ids(hass, entry) == baseline[entry.entry_id] for entry in entries
        )
        assert all(
            conversation.async_get_agent(hass, entry.entry_id) is not None
            for entry in entries
        )
        current_resources = _resource_footprint(hass)
        assert all(
            current_resources[name] <= baseline_count
            for name, baseline_count in warm_resources.items()
        ), (warm_resources, current_resources)
        if number % 12 == 0:
            await hass.async_block_till_done()
            gc.collect()
            assert _eoai_task_count() <= warm_tasks + 2

    for entry in entries:
        assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    assert all(
        conversation.async_get_agent(hass, entry.entry_id) is None for entry in entries
    )
    assert calls >= 120 * stress_scale
    record(
        stress_trace,
        "summary",
        conversation_turns=calls,
        lifecycle_cycles=sum(
            item["operation"] == "unload_setup" for item in stress_trace
        ),
        agents=2,
        users=6,
        warm_resource_counts=warm_resources,
        warm_eoai_tasks=warm_tasks,
    )
