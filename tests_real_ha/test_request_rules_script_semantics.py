"""Request Rules executed by Home Assistant's real script engine."""

import asyncio
from unittest.mock import AsyncMock

from custom_components.extended_openai_conversation_responses import services
from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_TOOLS,
    DOMAIN,
    SERVICE_CALL_FUNCTION,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    _ACTIVE_FUNCTION_RESULTS,
)
from homeassistant.components import conversation
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError
from tests_real_ha.test_cross_feature_acceptance import (
    _agent,
    _provider,
    _rule,
    _say,
    _speech,
)


def _local(
    actions,
    *,
    phrase="run rule",
    match_type="equals",
    success="Done",
    failure="Failed safely",
):
    return _rule(
        "local_action",
        {
            "actions": actions,
            "success_response": success,
            "failure_response": failure,
        },
        match_type=match_type,
        phrase=phrase,
    )


def _record_action(message):
    return {"action": "rule_probe.record", "data": {"message": message}}


async def test_variables_delay_and_action_keep_one_ha_script_context(hass, monkeypatch):
    agent = await _agent(hass)
    _provider(monkeypatch, agent, [])
    calls = []

    async def record(call):
        calls.append((call.data["message"], call.context.user_id))

    hass.services.async_register("rule_probe", "record", record)
    owner = await hass.auth.async_create_user("Request Rule Owner")
    await agent._request_rules.async_create(
        _local(
            [
                {"variables": {"marker": "from variables"}},
                {"delay": {"milliseconds": 10}},
                _record_action("{{ marker }}"),
            ]
        )
    )
    result = await conversation.async_converse(
        hass=hass,
        text="run rule",
        conversation_id=None,
        context=Context(user_id=owner.id),
        language="en",
        agent_id=agent.entry.entry_id,
    )
    assert _speech(result) == "Done"
    assert calls == [("from variables", owner.id)]
    await hass.async_block_till_done()


async def test_wait_template_blocks_then_resumes_once_with_variables(hass, monkeypatch):
    agent = await _agent(hass)
    _provider(monkeypatch, agent, [])
    entered = asyncio.Event()
    calls = []

    async def enter(_call):
        entered.set()

    async def record(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "enter", enter)
    hass.services.async_register("rule_probe", "record", record)
    hass.states.async_set("sensor.rule_gate", "closed")
    await agent._request_rules.async_create(
        _local(
            [
                {"variables": {"marker": "kept"}},
                {"action": "rule_probe.enter"},
                {
                    "wait_template": "{{ is_state('sensor.rule_gate', 'open') }}",
                    "timeout": "00:00:05",
                    "continue_on_timeout": False,
                },
                _record_action("{{ marker }}"),
            ]
        )
    )
    running = asyncio.create_task(_say(hass, agent, "run rule"))
    await asyncio.wait_for(entered.wait(), 2)
    assert not running.done()
    assert calls == []
    hass.states.async_set("sensor.rule_gate", "open")
    assert _speech(await asyncio.wait_for(running, 2)) == "Done"
    assert calls == ["kept"]


async def test_concurrent_rules_keep_ha_variables_request_local(hass, monkeypatch):
    agent = await _agent(hass)
    _provider(monkeypatch, agent, [])
    calls = []

    async def record(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "record", record)
    await agent._request_rules.async_create(
        _local(
            [
                {"variables": {"label": "{{ marker }}"}},
                {"delay": {"milliseconds": 10}},
                _record_action("{{ label }}"),
            ],
            phrase="run {marker}",
            match_type="sentence_pattern",
        )
    )
    results = await asyncio.wait_for(
        asyncio.gather(
            _say(hass, agent, "run alpha"),
            _say(hass, agent, "run beta"),
        ),
        3,
    )
    assert [_speech(result) for result in results] == ["Done", "Done"]
    assert sorted(calls) == ["alpha", "beta"]


async def test_function_capture_preserves_ha_variables_and_survives_reload(
    hass, monkeypatch
):
    tool = {
        "spec": {
            "name": "rule_battery",
            "description": "Return a deterministic battery reading.",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "template", "value_template": '{"level": 62}'},
    }
    agent = await _agent(hass, **{CONF_FUNCTION_TOOLS: [tool]})
    _provider(monkeypatch, agent, [])
    function_calls = AsyncMock(wraps=services.async_call_active_function)
    monkeypatch.setattr(services, "async_call_active_function", function_calls)
    calls = []

    async def record(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "record", record)
    await agent._request_rules.async_create(
        _local(
            [
                {"variables": {"marker": "kitchen"}},
                {
                    "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
                    "data": {
                        "function": "rule_battery",
                        "arguments": {},
                        "result_alias": "battery",
                    },
                },
                _record_action("{{ marker }}:{battery.level}"),
            ],
            success="Battery {battery.level}",
        )
    )
    assert _speech(await _say(hass, agent, "run rule")) == "Battery 62"
    assert calls == ["kitchen:62"]
    assert function_calls.await_count == 1
    assert _ACTIVE_FUNCTION_RESULTS.get() is None

    entry_id = agent.entry.entry_id
    assert await hass.config_entries.async_reload(entry_id)
    agent = conversation.async_get_agent(hass, entry_id)
    _provider(monkeypatch, agent, [])
    assert _speech(await _say(hass, agent, "run rule")) == "Battery 62"
    assert calls == ["kitchen:62", "kitchen:62"]
    assert function_calls.await_count == 2
    assert _ACTIVE_FUNCTION_RESULTS.get() is None
    await agent._request_rules.async_create(
        _local(
            [
                {
                    "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
                    "data": {
                        "function": "rule_battery",
                        "arguments": {},
                        "result_alias": "battery",
                    },
                },
                _record_action("{battery.missing}"),
            ],
            phrase="missing battery field",
        )
    )
    assert _speech(await _say(hass, agent, "missing battery field")) == "Failed safely"
    assert calls == ["kitchen:62", "kitchen:62"]
    assert function_calls.await_count == 3
    assert _ACTIVE_FUNCTION_RESULTS.get() is None


async def test_wait_timeout_stops_actions_and_next_request_works(hass, monkeypatch):
    agent = await _agent(hass)
    _provider(monkeypatch, agent, [])
    calls = []

    async def record(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "record", record)
    hass.states.async_set("sensor.rule_gate", "closed")
    await agent._request_rules.async_create(
        _local(
            [
                {
                    "wait_template": "{{ is_state('sensor.rule_gate', 'open') }}",
                    "timeout": {"milliseconds": 10},
                    "continue_on_timeout": False,
                },
                _record_action("late"),
            ],
            phrase="timeout rule",
        )
    )
    await agent._request_rules.async_create(
        _local([_record_action("healthy")], phrase="healthy rule")
    )
    assert _speech(await _say(hass, agent, "timeout rule")) == "Failed safely"
    assert calls == []
    assert _speech(await _say(hass, agent, "healthy rule")) == "Done"
    assert calls == ["healthy"]
    await agent._request_rules.async_create(
        _local(
            [
                {
                    "wait_template": "{{ is_state('sensor.rule_gate', 'open') }}",
                    "timeout": {"milliseconds": 10},
                    "continue_on_timeout": True,
                },
                _record_action("after nonfatal timeout"),
            ],
            phrase="nonfatal timeout rule",
        )
    )
    assert _speech(await _say(hass, agent, "nonfatal timeout rule")) == "Done"
    assert calls == ["healthy", "after nonfatal timeout"]


async def test_failing_ha_action_stops_without_replaying_previous_steps(
    hass, monkeypatch
):
    agent = await _agent(hass)
    _provider(monkeypatch, agent, [])
    calls = []

    async def record(call):
        calls.append(call.data["message"])

    async def fail(_call):
        raise HomeAssistantError("deterministic service failure")

    hass.services.async_register("rule_probe", "record", record)
    hass.services.async_register("rule_probe", "fail", fail)
    await agent._request_rules.async_create(
        _local(
            [
                _record_action("before"),
                {"action": "rule_probe.fail"},
                _record_action("after"),
            ],
            phrase="failing rule",
        )
    )
    await agent._request_rules.async_create(
        _local([_record_action("healthy")], phrase="healthy rule")
    )
    assert _speech(await _say(hass, agent, "failing rule")) == "Failed safely"
    assert calls == ["before"]
    assert _speech(await _say(hass, agent, "healthy rule")) == "Done"
    assert calls == ["before", "healthy"]
