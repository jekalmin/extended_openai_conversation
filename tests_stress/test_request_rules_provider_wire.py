"""Generated Request Rule cases across public Assist and SDK transport."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from custom_components.extended_openai_conversation_responses import (
    request_rules as rules_module,
)
from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_REASONING_EFFORT,
    DOMAIN,
    SERVICE_CALL_FUNCTION,
)
from custom_components.extended_openai_conversation_responses.request_rule_match_preview import (
    request_rule_match_preview,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
    async_evaluate_rule,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_cross_feature_acceptance import (
    _agent as _cross_feature_agent,
    _provider as _cross_feature_provider,
    _say as _cross_feature_say,
    _speech as _cross_feature_speech,
)
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _install_wire,
    _responses_sse_text,
    _speech,
)
from tests_real_ha.test_request_rules_script_semantics import _local, _record_action
from tests_stress.conftest import record
from tests_stress.test_request_rules_matrix import CLASSIFIED_MATCHERS

MATCHERS = sorted(CLASSIFIED_MATCHERS)
API_MODES = [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES]


async def test_referenced_function_recreation_cannot_rebind_inflight_rule(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A same-name Function replacement cannot satisfy an old local rule turn."""
    name = "nightly_rule_function"
    tool = {
        "spec": {
            "name": name,
            "description": "Return the reference generation",
            "parameters": {"type": "object", "properties": {}},
        },
        "function": {"type": "template", "value_template": "original generation"},
        "enabled": True,
    }
    agent = await _cross_feature_agent(hass, **{CONF_FUNCTION_TOOLS: [tool]})
    assert agent is not None
    old_wire = _cross_feature_provider(monkeypatch, agent, [])
    calls = []

    async def record_call(call):
        calls.append(call)

    hass.services.async_register("rule_probe", "record", record_call)
    await agent._request_rules.async_create(_local([
        {
            "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
            "data": {"function": name, "arguments": {}},
        },
        _record_action("executed"),
    ]))
    entered, release = asyncio.Event(), asyncio.Event()
    validate = rules_module.async_validate_actions_config

    async def paused_validate(*args, **kwargs):
        result = await validate(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(rules_module, "async_validate_actions_config", paused_validate)
    pending = asyncio.create_task(_cross_feature_say(hass, agent, "run rule"))
    await asyncio.wait_for(entered.wait(), timeout=10)
    entry = agent.entry

    def replace(tools: list[dict]) -> None:
        subentry = next(item for item in entry.subentries.values() if item.subentry_type == "conversation")
        hass.config_entries.async_update_subentry(
            entry, subentry, data={**subentry.data, CONF_FUNCTION_TOOLS: tools},
        )

    before = agent.subentry.data
    replace([])
    replace([tool])
    after = next(item for item in entry.subentries.values() if item.subentry_type == "conversation").data
    assert after == before and after is not before
    release.set()
    stale = await asyncio.wait_for(pending, timeout=10)
    assert stale.response.error_code is not None or stale.response.as_dict()["speech"]["plain"]["speech"] != "Done"
    assert not calls
    assert not old_wire

    await hass.async_block_till_done()
    new_agent = conversation.async_get_agent(hass, entry.entry_id)
    assert new_agent is not None and new_agent is not agent
    fresh_wire = _cross_feature_provider(monkeypatch, new_agent, [])
    fresh = await _cross_feature_say(hass, new_agent, "run rule")
    assert _cross_feature_speech(fresh) == "Done"
    assert len(calls) == 1
    assert not fresh_wire
    record(stress_trace, "summary", same_name_recreation=True, stale_function_calls=0)


@pytest.mark.parametrize("phase", ["matching", "validation"])
@pytest.mark.parametrize("mutation", ["delete_recreate_same", "rule_aba"])
async def test_inflight_rule_identity_rejects_replacement_and_aba(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    phase: str,
    mutation: str,
) -> None:
    """A matched rule cannot run after its committed identity has changed."""
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    rules = agent._request_rules
    runtime = agent._request_rule_runtime
    assert rules is not None and runtime is not None
    calls = []

    async def turn_off(call):
        calls.append(call)

    hass.services.async_register("light", "turn_off", turn_off)
    hass.states.async_set("light.enhanced_rule", "on")
    hass.states.async_set("light.replacement_rule", "on")
    original = await rules.async_create({
        **_rule(
            "equals",
            "local_action",
            {
                "actions": [{
                    "domain": "light", "service": "turn_off",
                    "target": {"entity_id": ["light.enhanced_rule"]}, "data": {},
                }],
                "success_response": "Original rule executed",
            },
        ),
        "id": "nightly-aba-rule",
    })
    before = rules.revision()
    assert not rules._has_continuation
    entered, release = asyncio.Event(), asyncio.Event()
    if phase == "matching":
        match = rules.async_match

        async def paused_match(*args, **kwargs):
            result = await match(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(rules, "async_match", paused_match)
    else:
        validate = rules_module.async_validate_actions_config

        async def paused_validate(*args, **kwargs):
            result = await validate(*args, **kwargs)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(rules_module, "async_validate_actions_config", paused_validate)

    pending = asyncio.create_task(async_evaluate_rule(
        hass, rules, runtime, "think deeply", "nightly-aba-session",
    ))
    await asyncio.wait_for(entered.wait(), timeout=10)
    if mutation == "delete_recreate_same":
        await rules.async_delete(original["id"])
        await rules.async_create(original)
    else:
        changed = {
            **original,
            "action": {
                **original["action"],
                "actions": [{
                    "domain": "light", "service": "turn_off",
                    "target": {"entity_id": ["light.replacement_rule"]}, "data": {},
                }],
                "success_response": "Replacement rule must not execute",
            },
        }
        await rules.async_update(original["id"], changed)
        await rules.async_update(original["id"], original)
    assert rules.revision() != before
    assert rules.snapshot()["rules"] == [original]
    release.set()
    with pytest.raises(HomeAssistantError, match="changed during matching"):
        await asyncio.wait_for(pending, timeout=10)
    assert calls == []

    wire = _install_wire(monkeypatch, agent, [])
    fresh = await _say(hass, agent, "think deeply")
    assert _speech(fresh) == "Original rule executed"
    assert len(calls) == 1
    assert not wire.requests
    record(stress_trace, "summary", phase=phase, mutation=mutation, stale_service_calls=0)


async def _agent(hass: HomeAssistant, api_mode: str):
    entry = _make_entry(
        "Enhanced Rule Wire",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: api_mode,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "medium",
            CONF_FUNCTION_TOOLS: [],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    return agent


async def _say(hass: HomeAssistant, agent: Any, text: str, conversation_id=None):
    return await conversation.async_converse(
        hass=hass,
        text=text,
        conversation_id=conversation_id,
        context=Context(),
        language="en",
        agent_id=agent.entry.entry_id,
    )


def _rule(match_type: str, action_type: str, action: dict) -> dict:
    return {
        "name": f"Enhanced {match_type} {action_type}",
        "enabled": True,
        "phrases": ["think deeply"],
        "match_type": match_type,
        "action_type": action_type,
        "action": action,
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": 0,
    }


@pytest.mark.parametrize("match_type", MATCHERS)
@pytest.mark.parametrize("scope", ["request", "conversation"])
@pytest.mark.parametrize("api_mode", API_MODES)
async def test_matcher_route_scope_reaches_wire_and_next_turn(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    match_type: str,
    scope: str,
    api_mode: str,
    stress_trace: list[dict],
) -> None:
    agent = await _agent(hass, api_mode)
    created = await agent._request_rules.async_create(
        _rule(
            match_type,
            "model_routing",
            {
                "model": "gpt-6-astra",
                "reasoning_effort": "xhigh",
                "scope": scope,
                "reset": False,
                "continue_to_ai": True,
                "success_response": "Route selected",
            },
        )
    )
    preview = request_rule_match_preview(agent._request_rules.match("think deeply"))
    assert preview["matched"]
    assert preview["rule"]["id"] == created["id"]
    reply = (
        _chat_sse_text if api_mode == API_MODE_CHAT_COMPLETIONS else _responses_sse_text
    )
    wire = _install_wire(monkeypatch, agent, [reply("routed"), reply("followup")])
    first = await _say(hass, agent, "think deeply")
    assert _speech(first) == "routed"
    second = await _say(hass, agent, "ordinary request", first.conversation_id)
    assert _speech(second) == "followup"
    assert len(wire.requests) == 2
    assert wire.requests[0]["body"]["model"] == "gpt-6-astra"
    assert wire.requests[1]["body"]["model"] == (
        "gpt-6-astra" if scope == "conversation" else "gpt-5.6"
    )
    record(
        stress_trace,
        "summary",
        layer="provider-wire",
        matcher=match_type,
        scope=scope,
        api_mode=api_mode,
        public_turns=2,
        provider_requests=2,
    )


@pytest.mark.parametrize("match_type", MATCHERS)
async def test_matcher_local_action_calls_ha_without_provider(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    match_type: str,
    stress_trace: list[dict],
) -> None:
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    calls = []

    async def turn_off(call):
        calls.append(call)

    hass.services.async_register("light", "turn_off", turn_off)
    hass.states.async_set("light.enhanced_rule", "on")
    await agent._request_rules.async_create(
        _rule(
            match_type,
            "local_action",
            {
                "actions": [
                    {
                        "domain": "light",
                        "service": "turn_off",
                        "target": {"entity_id": ["light.enhanced_rule"]},
                        "data": {},
                    }
                ],
                "success_response": "Local action complete",
            },
        )
    )
    wire = _install_wire(monkeypatch, agent, [])
    result = await _say(hass, agent, "think deeply")
    assert _speech(result) == "Local action complete"
    assert len(calls) == 1
    assert not wire.requests
    record(
        stress_trace,
        "summary",
        layer="real-ha",
        matcher=match_type,
        public_turns=1,
        ha_service_calls=1,
        provider_requests=0,
    )


async def test_continue_matching_skips_false_condition_and_reaches_later_rule(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """UI-authored continuation semantics execute later eligible rules in order."""
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    calls: list[str] = []

    async def record_call(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "record", record_call)
    hass.states.async_set("input_boolean.chain_allowed", "off")

    first = _local(
        [_record_action("first")],
        phrase="run chain",
        success="First complete",
    )
    first["continue_matching"] = True

    skipped = _local(
        [_record_action("conditional")],
        phrase="run chain",
        success="Conditional complete",
    )
    skipped["continue_matching"] = True
    skipped["conditions"] = [
        {
            "condition": "state",
            "entity_id": "input_boolean.chain_allowed",
            "state": "on",
        }
    ]

    final = _local(
        [_record_action("final")],
        phrase="run chain",
        success="Final complete",
    )

    for item in (first, skipped, final):
        await agent._request_rules.async_create(item)

    wire = _install_wire(monkeypatch, agent, [])
    result = await _say(hass, agent, "run chain")
    assert _speech(result) == "Final complete"
    assert calls == ["first", "final"]
    assert not wire.requests
    record(
        stress_trace,
        "summary",
        scenario="continue_matching_condition_skip",
        executed=calls,
        provider_requests=0,
    )


async def test_continue_matching_stops_on_provider_handoff(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A Continue-to-AI handoff is terminal even when Continue Matching is enabled."""
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    calls: list[str] = []

    async def record_call(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "record", record_call)
    first = _local(
        [_record_action("first")],
        phrase="handoff chain",
        success="Unused",
    )
    first["continue_matching"] = True
    first["action"]["continue_to_ai"] = True

    later = _local(
        [_record_action("later")],
        phrase="handoff chain",
        success="Later should not run",
    )
    await agent._request_rules.async_create(first)
    await agent._request_rules.async_create(later)

    wire = _install_wire(monkeypatch, agent, [_chat_sse_text("Provider response")])
    result = await _say(hass, agent, "handoff chain")
    assert _speech(result) == "Provider response"
    assert calls == ["first"]
    assert len(wire.requests) == 1
    record(
        stress_trace,
        "summary",
        scenario="continue_matching_provider_handoff",
        executed=calls,
        provider_requests=1,
    )


@pytest.mark.parametrize(
    ("terminal_action", "expected_response", "scenario"),
    [
        (
            [{"set_conversation_response": "Terminal response"}],
            "Terminal response",
            "conversation_response",
        ),
        (
            [{"stop": "finished"}],
            "First complete",
            "stop",
        ),
    ],
)
async def test_continue_matching_stops_on_terminal_local_outcome(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    terminal_action: list[dict],
    expected_response: str,
    scenario: str,
) -> None:
    """Conversation response and successful Stop prevent later rule execution."""
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    calls: list[str] = []

    async def record_call(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "record", record_call)
    first = _local(
        terminal_action,
        phrase="terminal chain",
        success="First complete",
    )
    first["continue_matching"] = True
    later = _local(
        [_record_action("later")],
        phrase="terminal chain",
        success="Later should not run",
    )
    await agent._request_rules.async_create(first)
    await agent._request_rules.async_create(later)

    wire = _install_wire(monkeypatch, agent, [])
    result = await _say(hass, agent, "terminal chain")
    assert _speech(result) == expected_response
    assert calls == []
    assert not wire.requests
    record(
        stress_trace,
        "summary",
        scenario=f"continue_matching_{scenario}",
        executed=calls,
        provider_requests=0,
    )


async def test_continue_matching_stops_on_local_failure(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A failed local action returns the failure response and stops matching."""
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    calls: list[str] = []

    async def fail(_call):
        raise HomeAssistantError("nightly failure")

    async def record_call(call):
        calls.append(call.data["message"])

    hass.services.async_register("rule_probe", "fail", fail)
    hass.services.async_register("rule_probe", "record", record_call)
    first = _local(
        [{"action": "rule_probe.fail"}],
        phrase="failure chain",
        success="Should not succeed",
        failure="Failed safely",
    )
    first["continue_matching"] = True
    later = _local(
        [_record_action("later")],
        phrase="failure chain",
        success="Later should not run",
    )
    await agent._request_rules.async_create(first)
    await agent._request_rules.async_create(later)

    wire = _install_wire(monkeypatch, agent, [])
    result = await _say(hass, agent, "failure chain")
    assert _speech(result) == "Failed safely"
    assert calls == []
    assert not wire.requests
    record(
        stress_trace,
        "summary",
        scenario="continue_matching_failure",
        executed=calls,
        provider_requests=0,
    )
