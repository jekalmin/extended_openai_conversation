"""Large valid Home Assistant installation with management and public paths."""

from __future__ import annotations

import logging
from time import perf_counter

from pytest_homeassistant_custom_component.common import MockConfigEntry, MockUser

from custom_components.extended_openai_conversation_responses import backup
from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_KNOWLEDGE_ENABLED,
    CONF_MEMORY_MODE,
    CONF_SKIP_AUTHENTICATION,
    CONFIG_ENTRY_VERSION,
    DOMAIN,
    MEMORY_MODE_MANUAL,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    MAX_SOURCES_PER_AGENT,
    async_get_knowledge,
)
from custom_components.extended_openai_conversation_responses.memory import (
    async_get_memory,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    async_get_request_rules,
)
from homeassistant.components import conversation
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from tests_stress.conftest import record
from tests_stress.health import HealthChecks, assert_enhanced_health
from tests_stress.test_function_groups_state_machine import _tool


async def test_large_installation_survives_setup_management_backup_and_assist(
    hass: HomeAssistant,
    monkeypatch,
    stress_scale: int,
    stress_trace: list[dict],
) -> None:
    # The HA test Store mock logs the entire accumulated payload after each
    # write. At this scale that becomes quadratic CI output, not diagnostics.
    logging.getLogger("pytest_homeassistant_custom_component.common").setLevel(
        logging.WARNING
    )
    agents = 10 if stress_scale == 1 else 20
    rule_count = 100 * stress_scale
    tool_count = 60 * stress_scale
    group_count = 20 * stress_scale
    memory_count = 500 * stress_scale
    knowledge_count = 150 * stress_scale
    MockUser(
        id="large-owner", name="Large installation owner", is_owner=True
    ).add_to_hass(hass)
    tools = [_tool(f"large-tool-{number}") for number in range(tool_count)]
    groups = [
        {
            "id": f"large-group-{number}",
            "name": f"Large group {number}",
            "description": "Scale fixture",
            "loading_mode": "always" if number % 2 == 0 else "on_demand",
            "functions": [f"large-tool-{number * 3 + offset}" for offset in range(3)],
            "enabled": True,
        }
        for number in range(group_count)
    ]
    primary_group_count = min(group_count, 50)
    primary_tool_count = primary_group_count * 3
    entries = []
    setup_started = perf_counter()
    for number in range(agents):
        options = {CONF_MEMORY_MODE: MEMORY_MODE_MANUAL, CONF_KNOWLEDGE_ENABLED: True}
        if number == 0:
            options |= {
                CONF_FUNCTION_TOOLS: tools[:primary_tool_count],
                CONF_FUNCTION_GROUPS: groups[:primary_group_count],
            }
        elif number == 1 and group_count > primary_group_count:
            options |= {
                CONF_FUNCTION_TOOLS: tools[primary_tool_count:],
                CONF_FUNCTION_GROUPS: groups[primary_group_count:],
            }
        entry = MockConfigEntry(
            domain=DOMAIN,
            title=f"Scale provider {number}",
            data={CONF_API_KEY: "sk-local", CONF_SKIP_AUTHENTICATION: True},
            version=CONFIG_ENTRY_VERSION,
            subentries_data=[
                {
                    "data": options,
                    "subentry_type": "conversation",
                    "title": f"Scale agent {number}",
                    "unique_id": None,
                }
            ],
        )
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        entries.append(entry)
    await hass.async_block_till_done()
    setup_seconds = round(perf_counter() - setup_started, 3)
    primary = entries[0]
    subentry = next(iter(primary.subentries.values()))
    assert len(subentry.data[CONF_FUNCTION_GROUPS]) == primary_group_count
    if group_count > primary_group_count:
        second_subentry = next(iter(entries[1].subentries.values()))
        assert len(second_subentry.data[CONF_FUNCTION_GROUPS]) == group_count - primary_group_count
        assert len(second_subentry.data[CONF_FUNCTION_TOOLS]) == tool_count - primary_tool_count
    memory = await async_get_memory(hass, primary.entry_id, subentry.subentry_id)
    knowledge = await async_get_knowledge(hass, primary.entry_id, subentry.subentry_id)
    rules = await async_get_request_rules(hass, primary.entry_id, subentry.subentry_id)
    population_started = perf_counter()
    for number in range(memory_count):
        assert (
            await memory.async_add(
                "large-owner",
                f"Large memory {number} 🎯",
                "scale",
                "explicit",
                key=f"large-{number}",
            )
        )["status"] == "created"
    # Heavy has 600 sources; EOAI correctly limits one agent to 500. Keep all
    # 600 sources by distributing the overflow to a second real agent.
    primary_knowledge_count = min(knowledge_count, MAX_SOURCES_PER_AGENT)
    overflow_knowledge_count = knowledge_count - primary_knowledge_count
    overflow_knowledge = None
    if overflow_knowledge_count:
        secondary = entries[1]
        secondary_subentry = next(iter(secondary.subentries.values()))
        overflow_knowledge = await async_get_knowledge(
            hass, secondary.entry_id, secondary_subentry.subentry_id
        )
    for number in range(knowledge_count):
        target = knowledge if number < primary_knowledge_count else overflow_knowledge
        assert target is not None
        await target.async_create(
            f"Large source {number}",
            f"Description {number}",
            f"Knowledge body {number} 東京",
        )
    for number in range(rule_count):
        await rules.async_create(
            {
                "name": f"Large rule {number}",
                "phrases": [f"large command {number}"],
                "match_type": "equals",
                "action_type": "local_action",
                "action": {"actions": [{"action": "script.turn_on"}]},
            }
        )
    population_seconds = round(perf_counter() - population_started, 3)
    counts = await assert_enhanced_health(
        hass,
        primary,
        subentry,
        HealthChecks(
            backup=True,
            memory_users=("large-owner",),
            knowledge=True,
            request_rules=True,
        ),
    )
    assert counts["memory_records"] == memory_count
    assert counts["knowledge_sources"] == primary_knowledge_count
    if overflow_knowledge is not None:
        assert len((await overflow_knowledge.async_backup_data())["sources"]) == overflow_knowledge_count
    assert counts["request_rules"] == rule_count
    snapshot_started = perf_counter()
    snapshot = await backup.async_collect_backup_snapshot(hass, primary, subentry)
    assert backup.inspect_backup(snapshot, subentry.subentry_id)
    snapshot_seconds = round(perf_counter() - snapshot_started, 3)
    assert await hass.config_entries.async_reload(primary.entry_id)
    await hass.async_block_till_done()
    assert (await backup.async_collect_backup_snapshot(hass, primary, subentry))[
        "memories"
    ] == snapshot["memories"]
    if overflow_knowledge is not None:
        secondary = entries[1]
        secondary_subentry = next(iter(secondary.subentries.values()))
        assert await hass.config_entries.async_reload(secondary.entry_id)
        await hass.async_block_till_done()
        reloaded_overflow = await async_get_knowledge(
            hass, secondary.entry_id, secondary_subentry.subentry_id
        )
        assert len((await reloaded_overflow.async_backup_data())["sources"]) == overflow_knowledge_count
    model_calls = 0
    public_turn_ms = []
    for entry in entries:
        agent = conversation.async_get_agent(hass, entry.entry_id)
        assert agent is not None

        async def model(log, *, entity_id=agent.entity_id, **kwargs):
            nonlocal model_calls
            del kwargs
            model_calls += 1
            log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=entity_id, content="scale healthy"
                )
            )

        monkeypatch.setattr(agent, "_async_handle_chat_log", model)
        turn_started = perf_counter()
        result = await conversation.async_converse(
            hass=hass,
            text="scale public probe",
            conversation_id=None,
            context=Context(user_id="large-owner"),
            language="en",
            agent_id=entry.entry_id,
        )
        public_turn_ms.append(round((perf_counter() - turn_started) * 1000, 2))
        assert result.response.as_dict()["speech"]["plain"]["speech"] == "scale healthy"
        assert model_calls == len(public_turn_ms), (
            "One Assist turn must invoke its model handler once"
        )
    record(
        stress_trace,
        "summary",
        layer="real-ha",
        agents=agents,
        request_rules=rule_count,
        function_tools=tool_count,
        function_groups=group_count,
        memory_records=memory_count,
        knowledge_sources=knowledge_count,
        public_turns=agents,
        model_calls=model_calls,
        public_turn_ms=public_turn_ms,
        public_turn_max_ms=max(public_turn_ms),
        setup_seconds=setup_seconds,
        population_seconds=population_seconds,
        snapshot_seconds=snapshot_seconds,
        reloads=1,
    )
