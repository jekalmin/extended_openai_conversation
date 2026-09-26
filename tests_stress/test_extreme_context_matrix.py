"""Combined pathological but valid context through public Assist and SDK wire."""

from __future__ import annotations

from datetime import timedelta
import json
import logging
import random
from time import perf_counter
import unicodedata

import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses import backup
from custom_components.extended_openai_conversation_responses.agent_config import (
    configured_function_tools_from_data,
)
from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_ARCHIVE_ENABLED,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_EXPOSED_ENTITIES_ENABLED,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_KNOWLEDGE_ENABLED,
    CONF_MEMORY_AUTO_RETRIEVE_LIMIT,
    CONF_MEMORY_MODE,
    CONF_PROMPT,
    CONF_TEMPORARY_MEMORY,
    CONTEXT_TRUNCATE_KEEP_RECENT,
    MEMORY_MODE_AUTOMATIC,
    TEMPORARY_MEMORY_EAGER,
)
from custom_components.extended_openai_conversation_responses.conversation_archive import (
    async_get_archive,
)
from custom_components.extended_openai_conversation_responses.exposed_attributes import (
    CONF_EXPOSED_ENTITY_ATTRIBUTES,
)
from custom_components.extended_openai_conversation_responses.function_execution import (
    validate_function_schema,
)
from custom_components.extended_openai_conversation_responses.knowledge import (
    async_get_knowledge,
)
from custom_components.extended_openai_conversation_responses.memory import (
    async_get_memory,
)
from custom_components.extended_openai_conversation_responses.model_payload import (
    RETRIEVED_DATA_SAFETY,
)
from custom_components.extended_openai_conversation_responses.scope import user_scope
from custom_components.extended_openai_conversation_responses.temporary_memory import (
    async_get_temporary_memory,
)
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_knowledge_provider_wire_e2e import (
    _chat_sse_tool_call,
    _chat_tool_result,
)
from tests_real_ha.test_provider_wire_e2e import _chat_sse_text, _install_wire, _speech
from tests_stress.conftest import record

_TIERS = (
    # name, tools, groups, exposed entities, memory, temporary, knowledge,
    # archived turns, user-prompt characters, per-tool schema description chars.
    ("broad", 24, 6, 60, 24, 12, 12, 16, 4_000, 350),
    ("pressure", 80, 20, 240, 80, 35, 50, 48, 20_000, 1_200),
    ("edge", 160, 40, 600, 180, 80, 120, 96, 45_000, 2_200),
)


async def test_bounded_unicode_and_nested_schema_round_trip_on_public_wire(
    hass: HomeAssistant, monkeypatch, stress_seed: int, stress_trace: list[dict]
) -> None:
    """Valid Unicode identities and nested schemas survive wire and backup boundaries."""
    owner = "unicode-owner"
    other = "unicode-other"
    for user in (owner, other):
        MockUser(id=user, name=user, is_owner=True).add_to_hass(hass)
    schema: dict = {"type": "string", "description": "leaf"}
    for depth in range(12):
        schema = {
            "type": "object",
            "properties": {f"level_{depth}": schema},
            "additionalProperties": False,
        }
    tool = _tool("unicode_boundary", 3000)
    tool["spec"]["description"] = "Unicode schema 🔧 مرحبا " * 120
    tool["spec"]["parameters"] = schema
    assert validate_function_schema(schema) == ()
    entry = _make_entry(
        "Unicode boundaries",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [tool],
            CONF_MEMORY_MODE: MEMORY_MODE_AUTOMATIC,
            CONF_KNOWLEDGE_ENABLED: True,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    memory = await async_get_memory(hass, entry.entry_id, agent.subentry.subentry_id)
    knowledge = await async_get_knowledge(
        hass, entry.entry_id, agent.subentry.subentry_id
    )
    private = f"OTHER-PRIVATE-{stress_seed}"
    await memory.async_add(
        owner, "unicode owner note " + "🧠" * 2000, "nightly", "explicit"
    )
    await memory.async_add(other, private, "nightly", "explicit")
    await knowledge.async_create("Unicode source", "valid edge", "مرحبا 🌍 " * 1000)
    values = [
        unicodedata.normalize("NFC", "cafe\u0301"),
        unicodedata.normalize("NFD", "café"),
        "👩\u200d🔬",
        "مرحبا بالعالم",
        "A\u200bB",
        "  boundary\t\ntext  ",
    ]
    assert values[0] != values[1]
    wire = _install_wire(
        monkeypatch, agent, [_chat_sse_text("Unicode accepted") for _ in values]
    )
    for index, value in enumerate(values):
        text = f"edge {index} {value}"
        result = await _say(hass, entry.entry_id, owner, text)
        assert _speech(result) == "Unicode accepted"
        serialized = json.dumps(wire.requests[index]["body"], ensure_ascii=False)
        assert text in serialized
        assert private not in serialized
        assert "Unicode schema 🔧 مرحبا" in serialized
    snapshot = await backup.async_collect_backup_snapshot(hass, entry, agent.subentry)
    assert (
        backup.inspect_backup(snapshot, agent.subentry.subentry_id).config[
            CONF_FUNCTION_TOOLS
        ][0]["spec"]["parameters"]
        == schema
    )
    record(
        stress_trace, "unicode_nested_round_trip", forms=len(values), schema_depth=12
    )


def _profile(tier: int, stress_scale: int) -> tuple:
    (
        name,
        tools,
        groups,
        entities,
        memories,
        temporary,
        knowledge,
        turns,
        prompt,
        schema,
    ) = _TIERS[tier]
    if tier == 2 and stress_scale > 1:
        return name, 240, 48, 900, 360, 95, 240, 160, 60_000, 2_800
    return (
        name,
        tools,
        groups,
        entities,
        memories,
        temporary,
        knowledge,
        turns,
        prompt,
        schema,
    )


def _tool(name: str, description_chars: int, *, oversized: bool = False) -> dict:
    per_property = 6_000 if oversized else description_chars // 5
    return {
        "spec": {
            "name": name,
            "description": "Configured scale schema "
            + "d" * (30_000 if oversized else description_chars),
            "parameters": {
                "type": "object",
                "properties": {
                    f"field_{number}": {
                        "type": "string",
                        "description": f"Field {number} " + "p" * per_property,
                    }
                    for number in range(5)
                },
                "additionalProperties": False,
            },
        },
        "function": {"type": "template", "value_template": "scale result"},
        "enabled": True,
    }


def _groups(names: list[str], count: int) -> list[dict]:
    width = len(names) // count
    return [
        {
            "id": f"scale_{number}",
            "name": f"Scale {number}",
            "description": "Deterministic context-scale group",
            "loading_mode": "always" if number < min(count, 18) else "on_demand",
            "functions": names[number * width : (number + 1) * width],
            "enabled": True,
        }
        for number in range(count)
    ]


def _text_with_usage(text: str, input_tokens: int) -> bytes:
    ordinary = _chat_sse_text(text).decode().removesuffix("data: [DONE]\n\n")
    usage = {
        "id": "chatcmpl-context-usage",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5.6",
        "choices": [],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": 20,
            "total_tokens": input_tokens + 20,
        },
    }
    return (ordinary + f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n").encode()


async def _say(
    hass: HomeAssistant,
    entry_id: str,
    user_id: str,
    text: str,
    conversation_id: str | None = None,
):
    return await conversation.async_converse(
        hass=hass,
        text=text,
        conversation_id=conversation_id,
        context=Context(user_id=user_id),
        language="en",
        agent_id=entry_id,
    )


@pytest.mark.parametrize("tier", range(len(_TIERS)), ids=[item[0] for item in _TIERS])
async def test_combined_context_boundary_tiers_use_real_assist_and_provider_wire(
    hass: HomeAssistant,
    monkeypatch,
    stress_seed: int,
    stress_scale: int,
    stress_trace: list[dict],
    tier: int,
) -> None:
    logging.getLogger("pytest_homeassistant_custom_component.common").setLevel(
        logging.WARNING
    )
    (
        name,
        tool_count,
        group_count,
        entity_count,
        memory_count,
        temp_count,
        knowledge_count,
        archive_count,
        prompt_chars,
        schema_chars,
    ) = _profile(tier, stress_scale)
    rng = random.Random(stress_seed ^ (0xC0A7E + tier))
    owner = f"context-owner-{tier}"
    other = f"context-other-{tier}"
    for user in (owner, other):
        MockUser(id=user, name=user, is_owner=True).add_to_hass(hass)

    registry = er.async_get(hass)
    selected_attributes: dict[str, list[str]] = {}
    entity_order = list(range(entity_count))
    rng.shuffle(entity_order)
    for number in entity_order:
        item = registry.async_get_or_create(
            "sensor",
            "eoai_context_scale",
            f"{tier}-{number}",
            suggested_object_id=f"context_scale_{tier}_{number:04d}",
        )
        hass.states.async_set(
            item.entity_id,
            "ready",
            {"pressure_note": f"ATTR-{tier}-{number:04d}-" + "a" * 900},
        )
        async_expose_entity(hass, conversation.DOMAIN, item.entity_id, True)
        if number < min(entity_count, 100):
            selected_attributes[f"registry:{item.id}"] = ["pressure_note"]

    tools = [
        _tool(
            f"scale_tool_{tier}_{number:03d}",
            schema_chars,
            oversized=number == 0 and tier == 2,
        )
        for number in range(tool_count)
    ]
    assert all(
        validate_function_schema(tool["spec"]["parameters"]) == () for tool in tools
    )
    groups = _groups([tool["spec"]["name"] for tool in tools], group_count)
    mandatory_marker = f"AUTHORITATIVE-USER-PROMPT-{tier}-{stress_seed}"
    prompt = (
        mandatory_marker + "\n" + "configured prompt context " * (prompt_chars // 26)
    )
    options = {
        CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
        CONF_CHAT_MODEL: "gpt-5.6",
        CONF_FUNCTION_TOOLS: tools,
        CONF_FUNCTION_GROUPS: groups,
        CONF_EXPOSED_ENTITIES_ENABLED: True,
        CONF_EXPOSED_ENTITY_ATTRIBUTES: selected_attributes,
        CONF_MEMORY_MODE: MEMORY_MODE_AUTOMATIC,
        CONF_MEMORY_AUTO_RETRIEVE_LIMIT: 10,
        CONF_TEMPORARY_MEMORY: TEMPORARY_MEMORY_EAGER,
        CONF_KNOWLEDGE_ENABLED: True,
        CONF_ARCHIVE_ENABLED: True,
        CONF_CONTEXT_THRESHOLD: 1_000,
        CONF_CONTEXT_TRUNCATE_STRATEGY: CONTEXT_TRUNCATE_KEEP_RECENT,
        CONF_PROMPT: prompt,
    }
    entry = _make_entry(
        f"Combined context {name}",
        include_ai_task=False,
        conversation_options=options,
    )
    started = perf_counter()
    await _setup_entry(hass, entry)
    subentry = next(iter(entry.subentries.values()))
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    assert len(configured_function_tools_from_data(subentry.data)) == tool_count
    memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    temporary = await async_get_temporary_memory(
        hass, entry.entry_id, subentry.subentry_id
    )
    knowledge = await async_get_knowledge(hass, entry.entry_id, subentry.subentry_id)
    archive = await async_get_archive(hass, entry.entry_id, subentry.subentry_id)

    owner_memory_marker = f"OWNER-MEMORY-{tier}-{stress_seed}"
    other_memory_marker = f"OTHER-PRIVATE-MEMORY-{tier}-{stress_seed}"
    for number in range(memory_count):
        await memory.async_add(
            owner,
            f"quasar calibration {number} {owner_memory_marker if number == 0 else 'ordinary'}",
            "nightly",
            "explicit",
            key=f"scale-{tier}-{number}",
        )
    await memory.async_add(other, other_memory_marker, "nightly", "explicit")
    expiry = (dt_util.utcnow() + timedelta(hours=3)).isoformat()
    for number in range(temp_count):
        await temporary.async_add(
            f"user:{owner}",
            f"temporary scale note {number} for quasar calibration",
            expiry,
            "nightly",
            owner_scope_id=f"user:{owner}",
        )
    relevant_marker = f"KNOWLEDGE-SCALE-{tier}-{stress_seed}"
    for number in range(knowledge_count):
        body = (
            f"Quasar calibration source {number}. {relevant_marker}. "
            + "reference " * 100
            if number < 10
            else f"Unrelated source {number}. " + "reference " * 100
        )
        if number == 0 and tier == 2:
            body += "large reference " * 4_000
        await knowledge.async_create(
            f"Quasar calibration {number}" if number < 10 else f"Reference {number}",
            f"Context scale source {number}",
            body,
        )
    scope = user_scope(owner, source="authenticated_user")
    session = await archive.async_begin_session(
        f"context-scale-{tier}",
        scope,
        f"historical-{tier}",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    assert session is not None
    for number in range(archive_count):
        await archive.async_record_turn(
            session.session_id,
            run_id=f"context-run-{tier}-{number}",
            user_text=f"archived quasar context {number}",
            assistant_text=f"historical response {number}",
            successful=True,
        )
    assert archive.stats()["turn_count"] == archive_count
    assert (await archive.async_search(scope.scope_id, "archived quasar"))["results"]
    population_seconds = round(perf_counter() - started, 3)
    record(
        stress_trace,
        "population",
        tier=name,
        tools=tool_count,
        groups=group_count,
        entities=entity_count,
        memories=memory_count,
        temporary=temp_count,
        knowledge=knowledge_count,
        archive_turns=archive_count,
        population_seconds=population_seconds,
    )

    observed_tokens = 12_000 + tier * 24_000
    warm_turns = 8 if stress_scale == 1 else 16
    wire = _install_wire(
        monkeypatch,
        agent,
        [
            _chat_sse_tool_call(
                "context-search",
                "knowledge_search",
                {"query": "quasar calibration", "limit": 10},
            ),
            _text_with_usage("Knowledge retrieval complete", 400),
            *[_text_with_usage("Warm history turn", 400) for _ in range(warm_turns)],
            _text_with_usage("Recent turn complete", observed_tokens),
            _text_with_usage("History compacted", observed_tokens),
        ],
    )
    first_marker = f"EARLIEST-USER-TURN-{tier}-{stress_seed}"
    first = await _say(
        hass, entry.entry_id, owner, f"Find quasar calibration {first_marker}"
    )
    assert _speech(first) == "Knowledge retrieval complete"
    assert len(wire.requests) == 2
    initial = json.dumps(wire.requests[0]["body"], ensure_ascii=False)
    result = json.dumps(
        _chat_tool_result(wire.requests[1]["body"], "context-search"),
        ensure_ascii=False,
    )
    assert mandatory_marker in initial
    assert RETRIEVED_DATA_SAFETY in initial
    assert first_marker in initial
    assert other_memory_marker not in initial
    assert relevant_marker not in initial  # Knowledge remains on demand.
    assert relevant_marker in result
    selected_attribute_count = initial.count("ATTR-")
    assert 0 < selected_attribute_count < min(entity_count, 100)
    functions = [
        item["function"]
        for item in wire.requests[0]["body"].get("tools", [])
        if item.get("type") == "function"
    ]
    assert len(functions) <= 128
    assert any(item["name"] == tools[0]["spec"]["name"] for item in functions)
    for item in functions:
        validate_function_schema(item["parameters"])

    recent_marker = f"RECENT-USER-TURN-{tier}-{stress_seed}"
    conversation_id = first.conversation_id
    for number in range(warm_turns):
        warm = await _say(
            hass,
            entry.entry_id,
            owner,
            f"Historical context turn {number} for quasar calibration",
            conversation_id,
        )
        assert _speech(warm) == "Warm history turn"
        conversation_id = warm.conversation_id
    second = await _say(hass, entry.entry_id, owner, recent_marker, conversation_id)
    assert _speech(second) == "Recent turn complete"
    third = await _say(
        hass,
        entry.entry_id,
        owner,
        "Confirm the latest context",
        second.conversation_id,
    )
    assert _speech(third) == "History compacted"
    latest = json.dumps(wire.requests[-1]["body"], ensure_ascii=False)
    assert mandatory_marker in latest and RETRIEVED_DATA_SAFETY in latest
    assert recent_marker in latest
    assert first_marker not in latest
    assert other_memory_marker not in latest
    request_bytes = [
        len(json.dumps(item["body"], ensure_ascii=False).encode())
        for item in wire.requests
    ]
    assert max(request_bytes) < 8_000_000
    assert len(wire.requests) == warm_turns + 4

    rejected = _install_wire(
        monkeypatch,
        agent,
        [
            (
                400,
                {
                    "error": {
                        "message": "maximum context length exceeded",
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                    }
                },
            ),
            _chat_sse_text("Recovered after context limit"),
        ],
    )
    failed = await _say(
        hass, entry.entry_id, owner, "Request beyond provider context limit"
    )
    assert failed.response.error_code is not None
    assert failed.response.as_dict()["speech"]["plain"]["speech"]
    recovered = await _say(
        hass, entry.entry_id, other, "Independent recovery after rejection"
    )
    assert _speech(recovered) == "Recovered after context limit"
    other_body = json.dumps(rejected.requests[-1]["body"], ensure_ascii=False)
    assert owner_memory_marker not in other_body
    assert first_marker not in other_body
    record(
        stress_trace,
        "summary",
        layer="provider-wire",
        tier=name,
        provider_requests=len(wire.requests) + len(rejected.requests),
        public_turns=warm_turns + 5,
        request_bytes=request_bytes,
        selected_tools=len(functions),
        context_threshold=1_000,
        provider_rejections=1,
        recoveries=1,
        elapsed_seconds=round(perf_counter() - started, 3),
    )
