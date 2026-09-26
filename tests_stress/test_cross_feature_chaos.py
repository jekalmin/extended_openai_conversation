"""Seeded valid mutations across independent durable agent stores and reloads."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from datetime import timedelta
import random

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry, MockUser

from custom_components.extended_openai_conversation_responses import backup
from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_TOOLS,
    CONF_KNOWLEDGE_ENABLED,
    CONF_MEMORY_MODE,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPORARY_MEMORY,
    CONFIG_ENTRY_VERSION,
    DEFAULT_CONF_FUNCTION_TOOLS,
    DOMAIN,
    MEMORY_MODE_MANUAL,
)
from custom_components.extended_openai_conversation_responses.guest_mode import (
    async_get_guest_mode,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    async_get_knowledge,
)
from custom_components.extended_openai_conversation_responses.memory import (
    async_get_memory,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    async_get_request_rules,
)
from custom_components.extended_openai_conversation_responses.temporary_memory import (
    async_get_temporary_memory,
)
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.const import CONF_API_KEY
from homeassistant.core import Context, HomeAssistant
from homeassistant.util import dt as dt_util
from tests_stress.conftest import record
from tests_stress.health import HealthChecks, assert_enhanced_health


def _semantic(snapshot: dict) -> dict:
    return {key: value for key, value in snapshot.items() if key != "created_at"}


@pytest.mark.asyncio
async def test_seeded_cross_store_chaos_preserves_valid_agent_state(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_seed: int,
    stress_scale: int,
    stress_trace: list[dict],
) -> None:
    rng = random.Random(stress_seed ^ 0xC4A05)
    for number in range(4):
        MockUser(id=f"chaos-user-{number}", name=f"Chaos user {number}").add_to_hass(
            hass
        )
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Chaos agent",
        data={CONF_API_KEY: "sk-local", CONF_SKIP_AUTHENTICATION: True},
        version=CONFIG_ENTRY_VERSION,
        subentries_data=[
            {
                "data": {
                    CONF_MEMORY_MODE: MEMORY_MODE_MANUAL,
                    CONF_KNOWLEDGE_ENABLED: True,
                    CONF_TEMPORARY_MEMORY: "balanced",
                },
                "subentry_type": "conversation",
                "title": "Chaos conversation",
                "unique_id": None,
            }
        ],
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    subentry = next(
        item
        for item in entry.subentries.values()
        if item.subentry_type == "conversation"
    )
    checkpoints: list[dict] = []
    turns = 0
    conversations: dict[str, str] = {}

    async def managers():
        return (
            await async_get_memory(hass, entry.entry_id, subentry.subentry_id),
            await async_get_knowledge(hass, entry.entry_id, subentry.subentry_id),
            await async_get_request_rules(hass, entry.entry_id, subentry.subentry_id),
        )

    for step in range(90 * stress_scale):
        memory, knowledge, rules = await managers()
        operation = rng.choices(
            (
                "memory_add",
                "memory_delete",
                "knowledge_create",
                "knowledge_delete",
                "rule_create",
                "rule_delete",
                "temporary_add",
                "temporary_delete",
                "guest_toggle",
                "config_edit",
                "exposure_toggle",
                "checkpoint",
                "restore",
                "reload",
                "conversation_turn",
                "cancel_turn",
                "tool_toggle",
            ),
            weights=(16, 6, 10, 5, 8, 4, 8, 3, 8, 4, 4, 7, 4, 5, 10, 3, 4),
            k=1,
        )[0]
        record(
            stress_trace,
            "sequence_choice",
            seed=stress_seed,
            step=step,
            choice=operation,
        )
        users = [f"chaos-user-{number}" for number in range(4)]
        if operation == "memory_add":
            user = rng.choice(users)
            record(stress_trace, operation, step=step, user=user)
            await memory.async_add(
                user,
                f"chaos marker {user} operation {step}",
                "acceptance",
                "explicit",
                key=f"chaos-{step}",
            )
        elif operation == "memory_delete":
            user = rng.choice(users)
            items = await memory.async_list(user, limit=50)
            if items:
                record(
                    stress_trace, operation, step=step, user=user, id=items[0].memory_id
                )
                assert await memory.async_delete(user, [items[0].memory_id]) == 1
        elif operation == "knowledge_create":
            record(stress_trace, operation, step=step)
            await knowledge.async_create(
                f"Chaos title {step % 3}",
                "valid description",
                f"Knowledge marker {step} 東京",
            )
        elif operation == "knowledge_delete":
            items = await knowledge.async_list()
            if items:
                selected = rng.choice(items)
                record(stress_trace, operation, step=step, id=selected["source_id"])
                assert await knowledge.async_delete(selected["source_id"])
        elif operation == "rule_create":
            record(stress_trace, operation, step=step)
            await rules.async_create(
                {
                    "name": f"Chaos rule {step}",
                    "phrases": [f"local command {step}"],
                    "match_type": "equals",
                    "action_type": "local_action",
                    "action": {"actions": [{"action": "script.turn_on"}]},
                }
            )
        elif operation == "rule_delete":
            items = rules.snapshot()["rules"]
            if items:
                selected = rng.choice(items)
                record(stress_trace, operation, step=step, id=selected["id"])
                assert await rules.async_delete(selected["id"])
        elif operation == "temporary_add":
            temporary = await async_get_temporary_memory(
                hass, entry.entry_id, subentry.subentry_id
            )
            user = rng.choice(users)
            await temporary.async_add(
                f"user:{user}",
                f"Temporary chaos marker {step}",
                (dt_util.utcnow() + timedelta(hours=1)).isoformat(),
                "acceptance",
                owner_scope_id=f"user:{user}",
            )
            record(stress_trace, operation, step=step, user=user)
        elif operation == "temporary_delete":
            temporary = await async_get_temporary_memory(
                hass, entry.entry_id, subentry.subentry_id
            )
            user = rng.choice(users)
            items = await temporary.async_active(
                f"user:{user}", owner_scope_id=f"user:{user}"
            )
            if items:
                await temporary.async_delete(
                    f"user:{user}", [items[0].memory_id], owner_scope_id=f"user:{user}"
                )
                record(stress_trace, operation, step=step, user=user)
        elif operation == "guest_toggle":
            guest = await async_get_guest_mode(
                hass, entry.entry_id, subentry.subentry_id
            )
            assert guest is not None
            if guest.is_active():
                await guest.async_disable_trusted()
            else:
                await guest.async_update_trusted(indefinite=True)
            record(stress_trace, operation, step=step, active=guest.is_active())
        elif operation == "config_edit":
            options = dict(subentry.data)
            options["max_tokens"] = 600 + step
            hass.config_entries.async_update_subentry(entry, subentry, data=options)
            await hass.async_block_till_done()
            assert await hass.config_entries.async_reload(entry.entry_id)
            await hass.async_block_till_done()
            record(stress_trace, operation, step=step, max_tokens=options["max_tokens"])
        elif operation == "tool_toggle":
            options = dict(subentry.data)
            tool = deepcopy(DEFAULT_CONF_FUNCTION_TOOLS[0])
            tool["enabled"] = not bool(
                (options.get(CONF_FUNCTION_TOOLS) or [{}])[0].get("enabled", False)
            )
            options[CONF_FUNCTION_TOOLS] = [tool]
            hass.config_entries.async_update_subentry(entry, subentry, data=options)
            await hass.async_block_till_done()
            assert await hass.config_entries.async_reload(entry.entry_id)
            await hass.async_block_till_done()
            assert subentry.data[CONF_FUNCTION_TOOLS][0]["enabled"] is tool["enabled"]
            conversations.clear()
            record(stress_trace, operation, step=step, enabled=tool["enabled"])
        elif operation == "exposure_toggle":
            entity_id = "light.chaos_probe"
            exposed = bool(step % 2)
            if exposed:
                hass.states.async_set(entity_id, "on")
            else:
                hass.states.async_remove(entity_id)
            async_expose_entity(hass, conversation.DOMAIN, entity_id, exposed)
            record(stress_trace, operation, step=step, exposed=exposed)
        elif operation == "checkpoint":
            checkpoints.append(
                await backup.async_collect_backup_snapshot(hass, entry, subentry)
            )
            record(stress_trace, operation, step=step, checkpoint=len(checkpoints) - 1)
        elif operation == "restore" and checkpoints:
            checkpoint = rng.choice(checkpoints)
            record(
                stress_trace,
                operation,
                step=step,
                checkpoint=checkpoints.index(checkpoint),
            )
            assert (
                await backup.async_restore_backup(hass, entry, subentry, checkpoint)
            )["status"] == "restored"
            await hass.async_block_till_done()
            assert _semantic(
                await backup.async_collect_backup_snapshot(hass, entry, subentry)
            ) == _semantic(checkpoint)
            conversations.clear()
        elif operation == "reload":
            record(stress_trace, operation, step=step)
            assert await hass.config_entries.async_reload(entry.entry_id)
            await hass.async_block_till_done()
            conversations.clear()

        agent = conversation.async_get_agent(hass, entry.entry_id)
        assert agent is not None

        async def model(
            log: conversation.ChatLog,
            _agent_entity_id: str = agent.entity_id,
            _step: int = step,
            _operation: str = operation,
            **kwargs,
        ) -> None:
            del kwargs
            contents = [
                item.content
                for item in log.content
                if isinstance(getattr(item, "content", None), str)
            ]
            markers = [f"private-chaos-{number}-" for number in range(4)]
            present = [
                marker for marker in markers if any(marker in text for text in contents)
            ]
            assert len(present) <= 1, (stress_seed, _step, _operation, present)
            log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=_agent_entity_id, content="chaos healthy"
                )
            )

        monkeypatch.setattr(agent, "_async_handle_chat_log", model)
        if operation == "conversation_turn":
            user_index = rng.randrange(4)
            user = users[user_index]
            result = await conversation.async_converse(
                hass=hass,
                text=f"private-chaos-{user_index}-turn-{step}",
                conversation_id=conversations.get(user),
                context=Context(user_id=user),
                language="en",
                agent_id=entry.entry_id,
            )
            assert (
                result.response.as_dict()["speech"]["plain"]["speech"]
                == "chaos healthy"
            )
            assert result.conversation_id
            conversations[user] = result.conversation_id
            record(
                stress_trace,
                operation,
                step=step,
                user=user,
                conversation_id=result.conversation_id,
            )
        elif operation == "cancel_turn":
            entered = asyncio.Event()
            release = asyncio.Event()

            async def blocked_model(
                log: conversation.ChatLog,
                _entered: asyncio.Event = entered,
                _release: asyncio.Event = release,
                **kwargs,
            ) -> None:
                _entered.set()
                await _release.wait()
                await model(log, **kwargs)

            monkeypatch.setattr(agent, "_async_handle_chat_log", blocked_model)
            task = asyncio.create_task(
                conversation.async_converse(
                    hass=hass,
                    text=f"cancel-chaos-{step}",
                    conversation_id=None,
                    context=Context(user_id=users[step % len(users)]),
                    language="en",
                    agent_id=entry.entry_id,
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=10)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                monkeypatch.setattr(agent, "_async_handle_chat_log", model)
            record(stress_trace, operation, step=step)
        await assert_enhanced_health(
            hass,
            entry,
            subentry,
            HealthChecks(
                backup=True,
                memory_users=tuple(users),
                knowledge=True,
                request_rules=True,
                public_probe=True,
                probe_user=users[step % len(users)],
                probe_text=f"probe {step}",
                expected_speech="chaos healthy",
            ),
        )
        turns += 1
    record(
        stress_trace,
        "summary",
        chaos_operations=90 * stress_scale,
        checkpoints=len(checkpoints),
        public_conversation_turns=turns,
    )
