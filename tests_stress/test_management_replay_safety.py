"""Nightly replay safety for genuine HA Management WebSocket mutations."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    CONF_CONVERSATION_TIMEOUT_MINUTES,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    WS_COMMAND,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
)
from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant
from tests_real_ha.test_management_backend_acceptance import (
    _admin_client,
    _entry,
    _fresh_reload,
    _management_call,
    _management_response,
    _setup_entry,
)
from tests_stress.conftest import record


async def _commit_without_ack(
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entry: Any,
    section: str,
    action: str,
    **payload: Any,
) -> dict[str, Any]:
    """Let HA commit, then drop only the successful WebSocket reply."""
    committed = asyncio.Event()
    result: dict[str, Any] = {}
    original = websocket_api.ActiveConnection.send_result

    def drop_result(connection: Any, message_id: int, value: Any = None) -> None:
        del connection, message_id
        assert isinstance(value, dict)
        result.update(value)
        committed.set()

    with monkeypatch.context() as patch:
        patch.setattr(websocket_api.ActiveConnection, "send_result", drop_result)
        subentry = next(iter(entry.subentries.values()))
        await client.send_json_auto_id(
            {
                "type": WS_COMMAND,
                "entry_id": entry.entry_id,
                "subentry_id": subentry.subentry_id,
                "section": section,
                "action": action,
                **payload,
            }
        )
        await asyncio.wait_for(committed.wait(), timeout=10)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.receive_json(), timeout=0.2)
    assert websocket_api.ActiveConnection.send_result is original
    return result


@pytest.mark.asyncio
async def test_replayed_management_mutations_do_not_double_apply(
    hass: HomeAssistant,
    hass_ws_client: Any,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A lost reply after a durable commit cannot create a second mutation."""
    entry = _entry("Management replay safety")
    await _setup_entry(hass, entry)
    client = await _admin_client(
        hass,
        hass_ws_client,
        user_id="management-replay-admin",
        name="Management Replay Admin",
    )

    # Revision-protected configuration writes: the first exact mutation commits;
    # replaying the same stale revision must be rejected without changing state.
    before = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    config_payload = {
        "revision": before["revision"],
        "title": "Replay winner",
        "config": {CONF_CONVERSATION_TIMEOUT_MINUTES: 37},
    }
    first = await _commit_without_ack(
        client,
        monkeypatch,
        entry=entry,
        section="configuration",
        action="update",
        **config_payload,
    )
    assert first["valid"] is True
    replay = await _management_response(
        client,
        entry=entry,
        section="configuration",
        action="update",
        **config_payload,
    )
    assert replay["success"] is False
    assert "changed in another tab" in replay["error"]["message"].lower()
    authoritative = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    assert authoritative["title"] == "Replay winner"
    assert authoritative["config"][CONF_CONVERSATION_TIMEOUT_MINUTES] == 37

    # Request Rule creation carries a stable caller-supplied identity and store
    # revision. Replaying the same create cannot produce a second rule.
    rules_before = await _management_call(
        client, entry=entry, section="request_rules", action="list"
    )
    rule_payload = {
        "revision": rules_before["revision"],
        "rule": {
            "id": "replay-safe-rule",
            "name": "Replay-safe rule",
            "enabled": True,
            "phrases": ["replay safe command"],
            "match_type": "equals",
            "action_type": "model_routing",
            "action": {
                "model": "gpt-5-mini",
                "reasoning_effort": "medium",
                "scope": "request",
                "reset": False,
                "continue_to_ai": True,
                "success_response": "Replay route selected",
            },
            "matching_behavior": "defaults",
            "matching": dict(DEFAULT_MATCHING),
            "order": len(rules_before["rules"]),
        },
    }
    created = await _commit_without_ack(
        client,
        monkeypatch,
        entry=entry,
        section="request_rules",
        action="create",
        **rule_payload,
    )
    assert created["rule"]["id"] == "replay-safe-rule"
    replayed_rule = await _management_response(
        client,
        entry=entry,
        section="request_rules",
        action="create",
        **rule_payload,
    )
    assert replayed_rule["success"] is False
    rules_after = await _management_call(
        client, entry=entry, section="request_rules", action="list"
    )
    assert sum(rule["id"] == "replay-safe-rule" for rule in rules_after["rules"]) == 1

    # Memory add is deliberately duplicate-aware. An identical retry should
    # resolve to the original record instead of manufacturing a second fact.
    memory_payload = {
        "content": "Exact replay should remain one durable memory.",
        "category": "replay",
    }
    added = await _commit_without_ack(
        client, monkeypatch, entry=entry, section="memories", action="add", **memory_payload
    )
    replayed_memory = await _management_call(
        client, entry=entry, section="memories", action="add", **memory_payload
    )
    assert added["status"] == "created"
    assert replayed_memory["status"] == "duplicate"
    assert replayed_memory["memory"]["memory_id"] == added["memory"]["memory_id"]
    listed = await _management_call(
        client, entry=entry, section="memories", action="list"
    )
    matching = [
        item
        for item in listed["memories"]
        if item["content"] == memory_payload["content"]
    ]
    assert len(matching) == 1

    await _fresh_reload(hass, entry)
    reloaded_config = await _management_call(
        client, entry=entry, section="configuration", action="get"
    )
    assert reloaded_config["title"] == "Replay winner"
    assert reloaded_config["config"][CONF_CONVERSATION_TIMEOUT_MINUTES] == 37
    reloaded_rules = await _management_call(
        client, entry=entry, section="request_rules", action="list"
    )
    assert sum(rule["id"] == "replay-safe-rule" for rule in reloaded_rules["rules"]) == 1
    reloaded_memories = await _management_call(
        client, entry=entry, section="memories", action="list"
    )
    assert sum(item["content"] == memory_payload["content"] for item in reloaded_memories["memories"]) == 1

    # Replaying a destructive mutation after the object is already gone is
    # harmless and must not affect unrelated records.
    deleted = await _management_call(
        client,
        entry=entry,
        section="memories",
        action="delete",
        memory_id=added["memory"]["memory_id"],
    )
    replayed_delete = await _management_call(
        client,
        entry=entry,
        section="memories",
        action="delete",
        memory_id=added["memory"]["memory_id"],
    )
    assert deleted["deleted"] == 1
    assert replayed_delete["deleted"] == 0

    record(
        stress_trace,
        "summary",
        layer="Real HA WebSocket",
        replayed_management_mutations=4,
        rejected_stale_replays=2,
        deduplicated_replays=1,
        harmless_delete_replays=1,
    )
