"""Confirmed management requests exercise real HA and the SDK provider wire."""

import asyncio

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    async_management_command,
)
from tests_real_ha.test_provider_wire_e2e import _agent, _chat_sse_text, _install_wire


@pytest.mark.parametrize("matched", [False, True])
async def test_live_rule_request_reaches_provider(hass, monkeypatch, matched):
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    if matched:
        await agent._request_rules.async_create(
            {
                "name": "AI route",
                "phrases": ["hello"],
                "match_type": "equals",
                "action_type": "model_routing",
                "action": {
                    "model": "gpt-5.6",
                    "scope": "request",
                    "continue_to_ai": True,
                },
            }
        )
    wire = _install_wire(monkeypatch, agent, [_chat_sse_text("Live provider response")])
    owner = await hass.auth.async_create_user(
        "Live request admin", group_ids=["system-admin"]
    )
    result = await async_management_command(
        hass,
        owner.id,
        True,
        {
            "section": "request_rules",
            "action": "test",
            "confirm": True,
            "entry_id": agent.entry.entry_id,
            "subentry_id": agent.subentry.subentry_id,
            "text": "hello",
        },
    )
    assert result["response"] == "Live provider response"
    assert result["conversation_id"]
    assert len(wire.requests) == 1


@pytest.mark.parametrize("executing", [False, True])
async def test_native_rule_edit_during_wait_keeps_original_outcome(
    hass, monkeypatch, executing
):
    from tests_real_ha.test_cross_feature_acceptance import _say, _speech
    from tests_real_ha.test_request_rules_script_semantics import _local, _record_action

    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    wire = _install_wire(monkeypatch, agent, [])
    entered, release = asyncio.Event(), asyncio.Event()
    markers = []

    async def wait(_call):
        entered.set()
        await release.wait()

    async def record(call):
        markers.append(call.data["message"])

    hass.services.async_register("rule_probe", "wait", wait)
    hass.services.async_register("rule_probe", "record", record)
    original = await agent._request_rules.async_create(
        _local(
            [
                {"action": "rule_probe.wait"},
                _record_action("original marker"),
            ]
        )
    )
    other = await agent._request_rules.async_create(
        _local([_record_action("unrelated")], phrase="other")
    )
    pending = asyncio.create_task(_say(hass, agent, "run rule"))
    await asyncio.wait_for(entered.wait(), 5)
    target = original if executing else other
    await agent._request_rules.async_update(
        target["id"],
        {
            **target,
            "action": {
                **target["action"],
                "actions": [_record_action("replacement marker")],
                "success_response": "Future requests only",
            },
        },
    )
    release.set()
    result = await asyncio.wait_for(pending, 5)
    assert _speech(result) == "Done"
    assert markers == ["original marker"]
    assert wire.requests == []
