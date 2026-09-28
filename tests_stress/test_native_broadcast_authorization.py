"""Native Broadcast authorization through real HA and the provider wire."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses.built_in_functions import (
    BUILT_IN_FUNCTION_PRESETS,
)
from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_GUEST_MODE_ENABLED,
)
from custom_components.extended_openai_conversation_responses.intercom import (
    ANNOUNCE_FEATURE,
    async_get_intercom,
)
from homeassistant.auth.models import Group
from homeassistant.auth.permissions.const import CAT_ENTITIES, POLICY_CONTROL, POLICY_READ
from homeassistant.auth.permissions.entities import ENTITY_ENTITY_IDS
from homeassistant.components import conversation
from homeassistant.const import ATTR_FRIENDLY_NAME, ATTR_SUPPORTED_FEATURES
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_knowledge_provider_wire_e2e import (
    _chat_sse_text,
    _chat_sse_tool_call,
)
from tests_real_ha.test_provider_wire_e2e import _install_wire, _speech

_SATELLITE = "assist_satellite.broadcast_auth_target"
_USER_ID = "broadcast-auth-user"
_CALL_ID = "call-native-broadcast-auth"
_ARGUMENTS = {"destination": "Auth target", "message": "Dinner is ready"}


def _send_broadcast_tool() -> dict[str, Any]:
    preset = next(
        item
        for item in BUILT_IN_FUNCTION_PRESETS
        if item["implementation"] == "send_broadcast"
    )
    tool = deepcopy(preset["tool"])
    tool["enabled"] = True
    return tool


def _user(*, can_control: bool) -> MockUser:
    entity_policy = {POLICY_READ: True}
    if can_control:
        entity_policy[POLICY_CONTROL] = True
    return MockUser(
        id=_USER_ID,
        name="Broadcast authorization user",
        is_owner=False,
        groups=[
            Group(
                id=f"broadcast-auth-{can_control}",
                name="Broadcast authorization",
                policy={
                    CAT_ENTITIES: {
                        ENTITY_ENTITY_IDS: {_SATELLITE: entity_policy}
                    }
                },
            )
        ],
    )


async def _say(hass: HomeAssistant, entry_id: str):
    return await conversation.async_converse(
        hass=hass,
        text="Send the announcement",
        conversation_id=None,
        context=Context(user_id=_USER_ID),
        language="en",
        agent_id=entry_id,
    )


def _result_from_wire(request: dict[str, Any]) -> Any:
    tool_message = next(
        item
        for item in request["messages"]
        if item.get("role") == "tool" and item.get("tool_call_id") == _CALL_ID
    )
    return json.loads(tool_message["content"])["result"]


@pytest.mark.parametrize(
    ("can_control", "guest_mode", "expected_announcement"),
    [(False, False, False), (True, False, True), (True, True, False)],
    ids=["ha-control-denied", "ha-control-allowed", "guest-mode-denied"],
)
async def test_native_broadcast_obeys_ha_control_and_guest_policy(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    can_control: bool,
    guest_mode: bool,
    expected_announcement: bool,
) -> None:
    """Denied callers cannot queue or deliver through the model-facing tool."""
    from custom_components.extended_openai_conversation_responses import intercom

    monkeypatch.setattr(intercom, "IDLE_STABILITY_SECONDS", 0)
    monkeypatch.setattr(intercom, "async_call_later", lambda *_args, **_kwargs: None)
    hass.states.async_set(
        _SATELLITE,
        "idle",
        {
            ATTR_FRIENDLY_NAME: "Auth target",
            ATTR_SUPPORTED_FEATURES: ANNOUNCE_FEATURE,
        },
    )
    user = _user(can_control=can_control)
    user.add_to_hass(hass)

    entry = _make_entry(
        "Native Broadcast Authorization",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [_send_broadcast_tool()],
            CONF_GUEST_MODE_ENABLED: True,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    if guest_mode:
        assert agent._guest_mode is not None
        await agent._guest_mode.async_update_trusted(indefinite=True)

    manager = await async_get_intercom(hass)
    await manager.async_set_enabled(True)
    announcements: list[dict[str, Any]] = []

    async def announce(call: Any) -> None:
        announcements.append(dict(call.data))

    hass.services.async_register("assist_satellite", "announce", announce)
    wire_responses = (
        [_chat_sse_text("Done")]
        if guest_mode
        else [
            _chat_sse_tool_call(_CALL_ID, "send_broadcast", _ARGUMENTS),
            _chat_sse_text("Done"),
        ]
    )
    wire = _install_wire(monkeypatch, agent, wire_responses)

    result = await _say(hass, entry.entry_id)
    await hass.async_block_till_done()

    assert _speech(result) == "Done"
    assert len(wire.requests) == (1 if guest_mode else 2)
    if guest_mode:
        assert all(
            tool.get("function", {}).get("name") != "send_broadcast"
            for tool in wire.requests[0]["body"]["tools"]
        )
        assert manager.history() == []
        assert announcements == []
        return

    tool_result = _result_from_wire(wire.requests[1]["body"])
    if expected_announcement:
        assert len(manager.history()) == 1
        assert announcements == [
            {"message": "Dinner is ready", "entity_id": _SATELLITE}
        ]
        assert tool_result["success"] is True
    else:
        assert manager.history() == []
        assert announcements == []
        assert "permission" in json.dumps(tool_result).casefold()
