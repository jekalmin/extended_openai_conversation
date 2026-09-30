"""Request Rule snapshot boundaries prevent misleading effect retries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation_responses import (
    management_ui,
    request_rules as rr,
)
from homeassistant.exceptions import HomeAssistantError
from tests.test_request_rules_management import MemoryStore, local_rule


@pytest.mark.parametrize("mutation", ["unrelated", "executing"])
@pytest.mark.parametrize("fails", [False, True])
async def test_running_script_keeps_snapshot_outcome(
    hass, monkeypatch, mutation, fails
):
    rules = rr.RequestRules(MemoryStore())
    await rules.async_initialize()
    original = await rules.async_create(local_rule())
    other = await rules.async_create(local_rule(rule_id="other", name="Other"))
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    class Script:
        def __init__(self, _hass, actions, *_args, **_kwargs):
            self.actions = actions

        async def async_run(self, *_args):
            entered.set()
            await release.wait()
            effects.append("original marker")
            if fails:
                raise HomeAssistantError("later action failed")
            completion = self.actions[-1]["variables"].as_dict()
            return SimpleNamespace(variables=completion, conversation_response=None)

        async def async_unload(self):
            pass

    monkeypatch.setattr(rr, "Script", Script)
    monkeypatch.setattr(
        rr,
        "async_validate_actions_config",
        AsyncMock(side_effect=lambda _hass, actions: actions),
    )
    pending = asyncio.create_task(
        rr.async_evaluate_rule(
            hass, rules, rr.RequestRuleRuntime(), "good night", "session"
        )
    )
    await asyncio.wait_for(entered.wait(), 2)
    target = other if mutation == "unrelated" else original
    await rules.async_update(
        target["id"],
        {
            **target,
            "name": "Edited during delay",
            "action": {**target["action"], "success_response": "Future requests only"},
        },
    )
    release.set()
    result = await asyncio.wait_for(pending, 2)
    assert result.successful is not fails
    assert result.response == ("Failed safely" if fails else "Done")
    assert effects == ["original marker"]
    assert "retry" not in result.response.lower()


async def test_revision_rejects_before_script_starts(hass, monkeypatch):
    rules = rr.RequestRules(MemoryStore())
    await rules.async_initialize()
    original = await rules.async_create(local_rule())
    constructed = []

    async def validate(_hass, actions):
        await rules.async_update(
            original["id"], {**original, "name": "Changed before execution"}
        )
        return actions

    monkeypatch.setattr(rr, "async_validate_actions_config", validate)
    monkeypatch.setattr(
        rr, "Script", lambda *_args, **_kwargs: constructed.append(True)
    )
    with pytest.raises(HomeAssistantError, match="changed during matching"):
        await rr.async_evaluate_rule(
            hass, rules, rr.RequestRuleRuntime(), "good night", "session"
        )
    assert constructed == []


async def test_live_management_uses_normal_service_with_owner_context(
    hass, monkeypatch
):
    monkeypatch.setattr(
        management_ui, "entry_and_agent", lambda *_: (object(), object())
    )
    monkeypatch.setattr(
        management_ui, "_conversation_entity_id", lambda *_: "conversation.selected"
    )
    hass.services.async_call.return_value = {
        "response": "Real response",
        "conversation_id": "real-cid",
    }
    message = {
        "section": "request_rules",
        "action": "test",
        "entry_id": "entry",
        "subentry_id": "agent",
        "text": "hello",
    }
    with pytest.raises(HomeAssistantError, match="Confirm the live request"):
        await management_ui.async_management_command(hass, "admin", True, message)
    hass.services.async_call.assert_not_awaited()
    result = await management_ui.async_management_command(
        hass, "admin", True, {**message, "confirm": True}
    )
    assert result["conversation_id"] == "real-cid"
    args, kwargs = hass.services.async_call.call_args
    assert args == (
        rr.DOMAIN,
        "process",
        {"agent_id": "conversation.selected", "text": "hello"},
    )
    assert kwargs["context"].user_id == "admin"
    assert kwargs["blocking"] and kwargs["return_response"]
