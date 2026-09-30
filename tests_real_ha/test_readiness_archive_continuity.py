"""Immediate live configuration saves and first-turn HA archive identity."""

import asyncio

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_ARCHIVE_ENABLED,
    CONF_CONVERSATION_CONTINUITY,
    CONVERSATION_CONTINUITY_HA_DEFAULT,
    CONVERSATION_CONTINUITY_USER,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    async_management_command,
)
from homeassistant.components import conversation
from homeassistant.core import Context
from tests_real_ha.test_cross_feature_acceptance import _agent
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _install_wire,
    _responses_sse_text,
)


async def test_request_waits_for_readiness_and_failed_initialization_is_handled(
    hass, monkeypatch
):
    agent = await _agent(hass)
    wire = _install_wire(monkeypatch, agent, [_chat_sse_text("Ready")])
    entered = asyncio.Event()

    class ObservedReady(asyncio.Event):
        async def wait(self):
            entered.set()
            return await super().wait()

    agent._agent_ready = ObservedReady()
    task = asyncio.create_task(
        conversation.async_converse(
            hass=hass,
            text="Wait for initialization",
            conversation_id=None,
            context=Context(),
            language="en",
            agent_id=agent.entry.entry_id,
        )
    )
    await entered.wait()
    assert not task.done()
    assert not wire.requests
    agent._agent_ready.set()
    result = await task
    assert result.response.error_code is None
    assert len(wire.requests) == 1
    agent._agent_initialization_failed = True
    result = await conversation.async_converse(
        hass=hass,
        text="Unavailable initialization",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=agent.entry.entry_id,
    )
    assert result.response.error_code is not None
    assert "not ready" in result.response.as_dict()["speech"]["plain"]["speech"]
    assert len(wire.requests) == 1


async def test_immediate_save_process_repeated_api_switches_keep_ready_agent(
    hass, monkeypatch
):
    agent = await _agent(hass)
    owner = await hass.auth.async_create_user("Save owner", group_ids=["system-admin"])
    modes = [API_MODE_RESPONSES, API_MODE_CHAT_COMPLETIONS] * 5
    replies = [
        _responses_sse_text("Ready response")
        if mode == API_MODE_RESPONSES
        else _chat_sse_text("Ready response")
        for mode in modes
    ]
    wire = _install_wire(monkeypatch, agent, replies)
    for mode in modes:
        snapshot = await async_management_command(
            hass,
            owner.id,
            True,
            {
                "section": "configuration",
                "action": "get",
                "entry_id": agent.entry.entry_id,
                "subentry_id": agent.subentry.subentry_id,
            },
        )
        await async_management_command(
            hass,
            owner.id,
            True,
            {
                "section": "configuration",
                "action": "update",
                "entry_id": agent.entry.entry_id,
                "subentry_id": agent.subentry.subentry_id,
                "revision": snapshot["revision"],
                "config": {CONF_API_MODE: mode},
            },
        )
        result = await conversation.async_converse(
            hass=hass,
            text="Immediately after Save",
            conversation_id=None,
            context=Context(user_id=owner.id),
            language="en",
            agent_id=agent.entry.entry_id,
        )
        assert result.response.error_code is None
        assert (
            result.response.as_dict()["speech"]["plain"]["speech"] == "Ready response"
        )
        await hass.async_block_till_done()
        assert conversation.async_get_agent(hass, agent.entry.entry_id) is agent
    assert [request["path"] for request in wire.requests] == [
        "/v1/responses" if mode == API_MODE_RESPONSES else "/v1/chat/completions"
        for mode in modes
    ]


@pytest.mark.parametrize(
    "continuity", [CONVERSATION_CONTINUITY_HA_DEFAULT, CONVERSATION_CONTINUITY_USER]
)
async def test_first_turn_and_two_continuations_share_one_archive(
    hass, monkeypatch, continuity
):
    agent = await _agent(
        hass, **{CONF_ARCHIVE_ENABLED: True, CONF_CONVERSATION_CONTINUITY: continuity}
    )
    owner = await hass.auth.async_create_user("Archive owner")
    _install_wire(
        monkeypatch, agent, [_chat_sse_text(f"Response {index}") for index in range(3)]
    )
    conversation_id = None
    for index in range(3):
        result = await conversation.async_converse(
            hass=hass,
            text=f"Turn {index}",
            conversation_id=conversation_id,
            context=Context(user_id=owner.id),
            language="en",
            agent_id=agent.entry.entry_id,
        )
        assert result.response.error_code is None
        if conversation_id is not None:
            assert result.conversation_id == conversation_id
        conversation_id = result.conversation_id
        assert conversation_id
    sessions = await agent._archive.async_list_sessions(f"user:{owner.id}")
    assert len(sessions["sessions"]) == 1
    session = sessions["sessions"][0]
    assert session["home_assistant_conversation_id"] == conversation_id
    assert session["turn_count"] == 3
    detail = await agent._archive.async_get(f"user:{owner.id}", session["session_id"])
    assert [turn["user_text"] for turn in detail["turns"]] == [
        "Turn 0",
        "Turn 1",
        "Turn 2",
    ]
