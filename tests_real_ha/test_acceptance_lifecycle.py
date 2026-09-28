"""High-value acceptance tests using Home Assistant's real runtime fixtures."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import yaml

from custom_components.extended_openai_conversation_responses import request_rules
from custom_components.extended_openai_conversation_responses.agent_config import (
    configured_function_tools_from_data,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_REASONING_EFFORT,
    CONF_SKIP_AUTHENTICATION,
    CONFIG_ENTRY_VERSION,
    DEFAULT_AI_TASK_OPTIONS,
    DOMAIN,
    SERVICE_PROCESS,
)
from custom_components.extended_openai_conversation_responses.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation_responses.local_intents import (
    CONF_LOCAL_INTENTS_ENABLED,
)
from custom_components.extended_openai_conversation_responses.resource_limits import (
    MAX_NATIVE_SERVICE_ACTIONS,
)
from custom_components.extended_openai_conversation_responses.template import (
    DATA_TEMPLATE_MANAGER,
)
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_API_KEY, STATE_UNAVAILABLE
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er


def _subentry(
    subentry_type: str,
    title: str,
    data: dict | None = None,
) -> dict:
    """Return storage-shaped subentry data for MockConfigEntry."""
    return {
        "data": data or {},
        "subentry_type": subentry_type,
        "title": title,
        "unique_id": None,
    }


def _make_entry(
    title: str = "Acceptance",
    *,
    include_ai_task: bool = True,
    local_intents: bool = False,
    conversation_options: dict | None = None,
    base_url: str | None = None,
) -> MockConfigEntry:
    """Create a current-version entry that cannot make an authentication request."""
    conversation_data = dict(conversation_options or {})
    # Acceptance fixtures commonly expose tools through Chat Completions. Keep
    # their default request on a provider-supported reasoning effort.
    if conversation_data.get(CONF_CHAT_MODEL) == "gpt-5.6":
        conversation_data.setdefault(CONF_REASONING_EFFORT, "none")
    if local_intents:
        conversation_data[CONF_LOCAL_INTENTS_ENABLED] = True

    subentries = [
        _subentry("conversation", f"{title} Conversation", conversation_data),
    ]
    if include_ai_task:
        subentries.append(
            _subentry(
                "ai_task_data",
                f"{title} AI Task",
                dict(DEFAULT_AI_TASK_OPTIONS),
            )
        )

    return MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data={
            CONF_API_KEY: "sk-acceptance-test",
            # The acceptance suite exercises the real HA and integration lifecycle,
            # but must never need an external OpenAI request merely to load an entry.
            CONF_SKIP_AUTHENTICATION: True,
            **({CONF_BASE_URL: base_url} if base_url else {}),
        },
        version=CONFIG_ENTRY_VERSION,
        subentries_data=subentries,
    )


async def _setup_entry(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    """Add and set up one entry through Home Assistant's config-entry manager."""
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED


def _registry_entries(hass: HomeAssistant, entry: MockConfigEntry):
    """Return the real entity-registry rows created for an entry."""
    return er.async_entries_for_config_entry(er.async_get(hass), entry.entry_id)


def _conversation_subentry(entry: MockConfigEntry):
    return next(
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == "conversation"
    )


def _legacy_execute_service_tool() -> dict:
    """Return one persisted historical stock execute_service Function Tool."""
    return {
        "spec": {
            "name": "legacy_service_action",
            "description": "My customised legacy HA service tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay": {
                        "type": "object",
                        "description": "Time to wait before execution",
                        "properties": {
                            "hours": {"type": "integer", "minimum": 0},
                            "minutes": {"type": "integer", "minimum": 0},
                            "seconds": {"type": "integer", "minimum": 0},
                        },
                    },
                    "list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "description": "The domain of the service.",
                                },
                                "service": {
                                    "type": "string",
                                    "description": "The service to be called",
                                },
                                "service_data": {
                                    "type": "object",
                                    "description": (
                                        "The service data object to indicate what to control."
                                    ),
                                    "properties": {
                                        "entity_id": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": (
                                                    "The entity_id retrieved from available "
                                                    "devices. It must start with domain, "
                                                    "followed by dot character."
                                                ),
                                            },
                                        },
                                        "area_id": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": (
                                                    "The id retrieved from areas. You can "
                                                    "specify only area_id without entity_id "
                                                    "to act on all entities in that area"
                                                ),
                                            },
                                        },
                                    },
                                },
                            },
                            "required": ["domain", "service", "service_data"],
                        },
                    },
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
        "enabled": True,
    }


@pytest.mark.asyncio
async def test_real_ha_setup_loads_platforms_agent_and_runtime(
    hass: HomeAssistant,
) -> None:
    """Set up the assembled integration through HA, not direct platform calls."""
    entry = _make_entry()
    await _setup_entry(hass, entry)

    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert isinstance(agent, ExtendedOpenAIAgentEntity)
    assert entry.runtime_data is not None
    assert hass.services.has_service(DOMAIN, SERVICE_PROCESS)
    assert DATA_TEMPLATE_MANAGER in hass.data[DOMAIN]

    rows = _registry_entries(hass, entry)
    domains = {row.domain for row in rows}
    assert {"conversation", "ai_task", "sensor"}.issubset(domains)

    conversation_subentry = _conversation_subentry(entry)
    conversation_id = conversation_subentry.subentry_id
    expected_sensor_unique_ids = {
        f"{conversation_id}_usage",
        f"{conversation_id}_usage_today",
        f"{conversation_id}_usage_month",
        f"{conversation_id}_last_response_usage",
        f"{conversation_id}_guest_mode",
    }
    assert expected_sensor_unique_ids.issubset({row.unique_id for row in rows})

    conversation_rows = [row for row in rows if row.domain == "conversation"]
    assert len(conversation_rows) == 1
    assert conversation_rows[0].config_subentry_id == conversation_id

    ai_task_subentry = next(
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == "ai_task_data"
    )
    ai_task_rows = [row for row in rows if row.domain == "ai_task"]
    assert len(ai_task_rows) == 1
    assert ai_task_rows[0].config_subentry_id == ai_task_subentry.subentry_id

    guest_mode_row = next(
        row for row in rows if row.unique_id == f"{conversation_id}_guest_mode"
    )
    guest_mode_state = hass.states.get(guest_mode_row.entity_id)
    assert guest_mode_state is not None
    assert guest_mode_state.state != STATE_UNAVAILABLE


@pytest.mark.asyncio
async def test_real_ha_unload_reload_cleans_and_recreates_runtime(
    hass: HomeAssistant,
) -> None:
    """Exercise entity teardown, agent teardown and template ownership via HA."""
    entry = _make_entry(include_ai_task=False)
    await _setup_entry(hass, entry)

    rows_before = _registry_entries(hass, entry)
    entity_ids_before = {row.entity_id for row in rows_before}
    conversation_id = _conversation_subentry(entry).subentry_id
    guest_mode_entity_id = next(
        row.entity_id
        for row in rows_before
        if row.unique_id == f"{conversation_id}_guest_mode"
    )
    template_manager_before = hass.data[DOMAIN][DATA_TEMPLATE_MANAGER]

    assert conversation.async_get_agent(hass, entry.entry_id) is not None
    manager_before = conversation.async_get_agent(
        hass, entry.entry_id
    )._request_rules
    assert hass.states.get(guest_mode_entity_id) is not None

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.NOT_LOADED
    assert conversation.async_get_agent(hass, entry.entry_id) is None
    assert (entry.entry_id, conversation_id) not in hass.data.get(
        request_rules._MANAGERS, {}
    )
    unloaded_guest_mode = hass.states.get(guest_mode_entity_id)
    assert unloaded_guest_mode is not None
    # Registry-backed entities intentionally remain represented by HA as unavailable
    # after unload.  This is HA's public lifecycle contract; absence is not required.
    assert unloaded_guest_mode.state == STATE_UNAVAILABLE
    assert DATA_TEMPLATE_MANAGER not in hass.data.get(DOMAIN, {})

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    assert conversation.async_get_agent(hass, entry.entry_id) is not None
    manager_after = conversation.async_get_agent(hass, entry.entry_id)._request_rules
    assert manager_after is not manager_before
    reloaded_guest_mode = hass.states.get(guest_mode_entity_id)
    assert reloaded_guest_mode is not None
    assert reloaded_guest_mode.state != STATE_UNAVAILABLE
    assert hass.data[DOMAIN][DATA_TEMPLATE_MANAGER] is not template_manager_before
    assert {
        row.entity_id for row in _registry_entries(hass, entry)
    } == entity_ids_before



@pytest.mark.asyncio
async def test_real_ha_legacy_entry_migrates_before_platform_setup(
    hass: HomeAssistant,
) -> None:
    """Exercise the v1-to-current migration inside HA's actual setup lifecycle."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Legacy Acceptance",
        data={
            CONF_API_KEY: "sk-acceptance-test",
            CONF_SKIP_AUTHENTICATION: True,
        },
        options={"chat_model": "gpt-4o-mini"},
        version=1,
    )

    await _setup_entry(hass, entry)

    assert entry.version == CONFIG_ENTRY_VERSION
    assert dict(entry.options) == {}
    assert {subentry.subentry_type for subentry in entry.subentries.values()} == {
        "conversation",
        "ai_task_data",
    }
    assert conversation.async_get_agent(hass, entry.entry_id) is not None
    assert {"conversation", "ai_task", "sensor"}.issubset(
        {row.domain for row in _registry_entries(hass, entry)}
    )


@pytest.mark.asyncio
async def test_real_ha_stale_native_tool_schema_migrates_and_survives_reload(
    hass: HomeAssistant,
) -> None:
    """Recover a recognized stale native Function Tool through HA startup."""
    legacy_tool = _legacy_execute_service_tool()
    original_tool = deepcopy(legacy_tool)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Stale Native Tool Acceptance",
        data={
            CONF_API_KEY: "sk-acceptance-test",
            CONF_SKIP_AUTHENTICATION: True,
        },
        version=CONFIG_ENTRY_VERSION,
        subentries_data=[
            _subentry(
                "conversation",
                "Stale Native Tool Conversation",
                {
                    CONF_FUNCTION_TOOLS: yaml.safe_dump(
                        [legacy_tool], sort_keys=False, allow_unicode=True
                    )
                },
            )
        ],
    )

    await _setup_entry(hass, entry)

    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert isinstance(agent, ExtendedOpenAIAgentEntity)

    conversation_subentry = _conversation_subentry(entry)
    migrated_yaml = conversation_subentry.data[CONF_FUNCTION_TOOLS]
    migrated_tools = yaml.safe_load(migrated_yaml)
    assert isinstance(migrated_tools, list)
    assert len(migrated_tools) == 1
    migrated = migrated_tools[0]

    assert migrated["spec"]["name"] == original_tool["spec"]["name"]
    assert migrated["spec"]["description"] == original_tool["spec"]["description"]
    assert migrated["enabled"] == original_tool["enabled"]
    assert migrated["function"] == original_tool["function"]

    parameters = migrated["spec"]["parameters"]
    assert migrated["spec"]["strict"] is False
    assert parameters["required"] == ["list"]
    list_schema = parameters["properties"]["list"]
    assert list_schema["maxItems"] == MAX_NATIVE_SERVICE_ACTIONS
    service_data = list_schema["items"]["properties"]["service_data"]
    assert service_data["additionalProperties"] is True
    for concept in ("device_id", "floor_id", "label_id", "brightness_pct"):
        assert concept in service_data["description"]

    configured_tools = configured_function_tools_from_data(agent.subentry.data)
    configured_tool = next(
        tool for tool in configured_tools if tool["spec"]["name"] == "legacy_service_action"
    )
    assert configured_tool["spec"]["strict"] is False
    assert configured_tool["spec"]["parameters"] == parameters
    assert configured_tool["function"]["name"] == "execute_service"

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    after_first_setup = _conversation_subentry(entry).data[CONF_FUNCTION_TOOLS]

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    reloaded_agent = conversation.async_get_agent(hass, entry.entry_id)
    assert isinstance(reloaded_agent, ExtendedOpenAIAgentEntity)
    after_second_setup = _conversation_subentry(entry).data[CONF_FUNCTION_TOOLS]
    assert after_second_setup == after_first_setup


@pytest.mark.asyncio
async def test_real_ha_public_conversation_api_can_complete_locally(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Join HA's public Conversation API to the integration without provider I/O."""
    entry = _make_entry(include_ai_task=False, local_intents=True)
    await _setup_entry(hass, entry)

    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert isinstance(agent, ExtendedOpenAIAgentEntity)

    provider_path = AsyncMock(
        side_effect=AssertionError(
            "A built-in local intent unexpectedly fell through to the provider path"
        )
    )
    monkeypatch.setattr(agent, "_async_handle_message_with_ha_tools", provider_path)

    result = await conversation.async_converse(
        hass=hass,
        text="what time is it",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
    )

    provider_path.assert_not_awaited()
    response = result.response.as_dict()
    assert response["speech"]["plain"]["speech"]
    assert result.conversation_id is not None


@pytest.mark.asyncio
async def test_real_ha_separate_calls_resume_history_and_preserve_continue_signal(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two public Assist calls share device history and return the listening flag."""
    from custom_components.extended_openai_conversation_responses.const import (
        CONF_CONTINUE_CONVERSATION,
        CONF_CONVERSATION_CONTINUITY,
        CONTINUE_CONVERSATION_ALWAYS,
        CONVERSATION_CONTINUITY_DEVICE,
    )

    entry = _make_entry(
        include_ai_task=False,
        conversation_options={
            CONF_CONVERSATION_CONTINUITY: CONVERSATION_CONTINUITY_DEVICE,
            CONF_CONTINUE_CONVERSATION: CONTINUE_CONVERSATION_ALWAYS,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert isinstance(agent, ExtendedOpenAIAgentEntity)
    seen = []

    async def model(log, **kwargs):
        seen.append([item.content for item in log.content])
        log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=agent.entity_id, content="The mug is blue."
            )
        )
        return None

    monkeypatch.setattr(agent, "_async_handle_chat_log", model)
    first = await conversation.async_converse(
        hass=hass,
        text="Remember my blue mug",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
        device_id="kitchen",
    )
    second = await conversation.async_converse(
        hass=hass,
        text="What colour is it?",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
        device_id="kitchen",
    )
    assert first.conversation_id == second.conversation_id
    assert "Remember my blue mug" in seen[0]
    assert seen[1][-3:] == [
        "Remember my blue mug",
        "The mug is blue.",
        "What colour is it?",
    ]
    for result in (first, second):
        assert result.continue_conversation is True
        assert result.as_dict()["continue_conversation"] is True
        assert (
            result.response.as_dict()["speech"]["plain"]["speech"] == "The mug is blue."
        )
