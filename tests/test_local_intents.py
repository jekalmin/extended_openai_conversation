"""Tests for optional Home Assistant local intent routing."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from homeassistant.components import conversation
from homeassistant.helpers import intent as ha_intent

from custom_components.extended_openai_conversation_responses import local_intents
from custom_components.extended_openai_conversation_responses.local_intents import (
    CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI,
    CONF_LOCAL_INTENT_EXCLUSIONS,
    CONF_LOCAL_INTENTS_ENABLED,
    async_try_handle_local_intent,
    conflicting_assist_pipelines,
    registered_intent_catalog,
    should_handle_locally,
)


def _recognize(intent_name: str, entities: dict[str, Any] | None = None):
    return cast(
        Any,
        SimpleNamespace(
            intent=SimpleNamespace(name=intent_name),
            entities=entities or {},
        ),
    )


def test_intent_filter_allows_unexcluded_intent() -> None:
    assert should_handle_locally(_recognize("HassTurnOff"), [], False)


def test_intent_filter_rejects_explicit_exclusion() -> None:
    assert not should_handle_locally(
        _recognize("HassBroadcast"), ["HassBroadcast"], False
    )


def test_delayed_command_can_be_sent_to_ai_without_redirecting_normal_timer() -> None:
    delayed = _recognize("HassStartTimer", {"conversation_command": object()})
    ordinary = _recognize("HassStartTimer", {"minutes": object()})

    assert not should_handle_locally(delayed, [], True)
    assert should_handle_locally(ordinary, [], True)


@pytest.mark.asyncio
async def test_disabled_local_handling_does_not_call_home_assistant(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    async def fake_handle(*args, **kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)
    result = await async_try_handle_local_intent(
        hass,
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        {CONF_LOCAL_INTENTS_ENABLED: False},
        guest_active=False,
    )

    assert result is None
    assert not called


@pytest.mark.asyncio
async def test_guest_mode_keeps_existing_policy_path(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    async def fake_handle(*args, **kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)
    result = await async_try_handle_local_intent(
        hass,
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        {CONF_LOCAL_INTENTS_ENABLED: True},
        guest_active=True,
    )

    assert result is None
    assert not called


@pytest.mark.asyncio
async def test_home_assistant_response_is_preserved_and_intent_is_reported(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = ha_intent.IntentResponse(language="en")
    response.async_set_speech("Done")

    async def fake_handle(hass, user_input, chat_log, *, intent_filter=None):
        assert intent_filter is not None
        assert not intent_filter(_recognize("HassTurnOn"))
        return response

    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)
    result = await async_try_handle_local_intent(
        hass,
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        {CONF_LOCAL_INTENTS_ENABLED: True},
        guest_active=False,
    )

    assert result is not None
    assert result.intent_name == "HassTurnOn"
    assert result.response is response


@pytest.mark.asyncio
async def test_excluded_match_falls_through(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_handle(hass, user_input, chat_log, *, intent_filter=None):
        assert intent_filter is not None
        assert intent_filter(_recognize("HassBroadcast"))
        return None

    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)
    result = await async_try_handle_local_intent(
        hass,
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        {
            CONF_LOCAL_INTENTS_ENABLED: True,
            CONF_LOCAL_INTENT_EXCLUSIONS: ["HassBroadcast"],
        },
        guest_active=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_unresolved_targeted_broadcast_blocks_greedy_whole_home_intent(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_targeted_match(*args, **kwargs):
        return None

    async def fake_handle(hass, user_input, chat_log, *, intent_filter=None):
        assert intent_filter is not None
        assert intent_filter(_recognize("HassBroadcast"))
        assert not intent_filter(_recognize("HassTurnOn"))
        return None

    monkeypatch.setattr(local_intents, "_async_try_targeted_broadcast", no_targeted_match)
    monkeypatch.setattr(
        local_intents, "is_targeted_broadcast_request", lambda _text: True
    )
    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)

    result = await async_try_handle_local_intent(
        hass,
        cast(Any, SimpleNamespace(text="Broadcast to Granny's room that dinner is ready")),
        cast(Any, SimpleNamespace()),
        {CONF_LOCAL_INTENTS_ENABLED: True},
        guest_active=False,
    )

    assert result is None


@pytest.mark.asyncio
async def test_disabled_broadcast_does_not_run_targeted_local_route(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SimpleNamespace(enabled=False)

    async def fake_get_manager(_hass):
        return manager

    async def fake_handle(hass, user_input, chat_log, *, intent_filter=None):
        assert intent_filter is not None
        assert intent_filter(_recognize("HassBroadcast"))
        return None

    monkeypatch.setattr(local_intents, "async_get_intercom", fake_get_manager)
    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)

    result = await async_try_handle_local_intent(
        hass,
        cast(Any, SimpleNamespace(text="Broadcast to kitchen that dinner is ready")),
        cast(Any, SimpleNamespace()),
        {CONF_LOCAL_INTENTS_ENABLED: True},
        guest_active=False,
    )

    assert result is None


def test_registered_intent_catalog_is_live_and_keeps_saved_missing_choices(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = [
        SimpleNamespace(intent_type="HassTurnOn"),
        SimpleNamespace(intent_type="HassNewFutureIntent"),
    ]
    monkeypatch.setattr(ha_intent, "async_get", lambda hass: handlers)

    catalog = registered_intent_catalog(hass, ["HassOldRemovedIntent"])
    by_name = {item["intent"]: item for item in catalog}

    assert by_name["HassTurnOn"]["available"] is True
    assert by_name["HassNewFutureIntent"]["available"] is True
    assert by_name["HassOldRemovedIntent"]["available"] is False
    assert by_name["HassNewFutureIntent"]["label"] == "New Future Intent"


def test_pipeline_conflicts_are_limited_to_this_agent(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        local_intents,
        "_conversation_entity_id",
        lambda hass, entry_id, subentry_id: "conversation.extended_openai",
    )
    monkeypatch.setattr(
        local_intents,
        "_get_assist_pipelines",
        lambda hass: [
            SimpleNamespace(
                id="one",
                name="Kitchen Assist",
                conversation_engine="conversation.extended_openai",
                prefer_local_intents=True,
            ),
            SimpleNamespace(
                id="two",
                name="Bedroom Assist",
                conversation_engine="conversation.extended_openai",
                prefer_local_intents=False,
            ),
            SimpleNamespace(
                id="three",
                name="Other agent",
                conversation_engine="conversation.home_assistant",
                prefer_local_intents=True,
            ),
        ],
    )

    assert conflicting_assist_pipelines(hass, "entry", "agent") == [
        {"id": "one", "name": "Kitchen Assist"}
    ]


@pytest.mark.asyncio
async def test_delayed_command_setting_is_applied_to_home_assistant_filter(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_handle(hass, user_input, chat_log, *, intent_filter=None):
        assert intent_filter is not None
        assert intent_filter(
            _recognize("HassStartTimer", {"conversation_command": object()})
        )
        assert not intent_filter(_recognize("HassStartTimer", {"minutes": object()}))
        return None

    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)
    assert (
        await async_try_handle_local_intent(
            hass,
            cast(Any, SimpleNamespace()),
            cast(Any, SimpleNamespace()),
            {
                CONF_LOCAL_INTENTS_ENABLED: True,
                CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI: True,
            },
            guest_active=False,
        )
        is None
    )


@pytest.mark.asyncio
async def test_targeted_broadcast_skips_non_text_without_loading_manager(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    async def fake_get_manager(_hass):
        nonlocal called
        called = True
        return SimpleNamespace(enabled=True)

    monkeypatch.setattr(local_intents, "async_get_intercom", fake_get_manager)

    result = await local_intents._async_try_targeted_broadcast(
        hass, cast(Any, SimpleNamespace(text=None))
    )

    assert result is None
    assert not called


@pytest.mark.asyncio
async def test_targeted_broadcast_enabled_manager_can_have_no_parse_match(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SimpleNamespace(enabled=True)

    async def fake_get_manager(_hass):
        return manager

    monkeypatch.setattr(local_intents, "async_get_intercom", fake_get_manager)
    monkeypatch.setattr(
        local_intents, "parse_targeted_broadcast", lambda _text, _manager: None
    )

    assert (
        await local_intents._async_try_targeted_broadcast(
            hass, cast(Any, SimpleNamespace(text="turn on the kitchen"))
        )
        is None
    )


@pytest.mark.asyncio
async def test_targeted_broadcast_queues_resolved_target_with_origin_context(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def async_send(message: str, **kwargs: Any) -> None:
        calls.append((message, kwargs))

    manager = SimpleNamespace(enabled=True, async_send=async_send)

    async def fake_get_manager(_hass):
        return manager

    monkeypatch.setattr(local_intents, "async_get_intercom", fake_get_manager)
    monkeypatch.setattr(
        local_intents,
        "parse_targeted_broadcast",
        lambda _text, _manager: (
            {"entity_ids": ["assist_satellite.kitchen"]},
            "Dinner is ready",
        ),
    )

    result = await local_intents._async_try_targeted_broadcast(
        hass,
        cast(
            Any,
            SimpleNamespace(
                text="Broadcast to kitchen that dinner is ready",
                language="en-IE",
                satellite_id="assist_satellite.hall",
                device_id="device-1",
            ),
        ),
    )

    assert result is not None
    assert result.intent_name == "ExtendedBroadcast"
    assert calls == [
        (
            "Dinner is ready",
            {
                "entity_ids": ["assist_satellite.kitchen"],
                "origin_entity_id": "assist_satellite.hall",
                "origin_device_id": "device-1",
                "source": "local_voice",
            },
        )
    ]


@pytest.mark.asyncio
async def test_missing_home_assistant_intent_handler_falls_through(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_targeted_match(*args, **kwargs):
        return None

    monkeypatch.setattr(local_intents, "_async_try_targeted_broadcast", no_targeted_match)
    monkeypatch.setattr(conversation, "async_handle_intents", None)

    assert (
        await async_try_handle_local_intent(
            hass,
            cast(Any, SimpleNamespace(text="turn on the kitchen")),
            cast(Any, SimpleNamespace()),
            {CONF_LOCAL_INTENTS_ENABLED: True},
            guest_active=False,
        )
        is None
    )


@pytest.mark.asyncio
async def test_response_without_filter_match_reports_unknown_intent(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = ha_intent.IntentResponse(language="en")

    async def no_targeted_match(*args, **kwargs):
        return None

    async def fake_handle(hass, user_input, chat_log, *, intent_filter=None):
        return response

    monkeypatch.setattr(local_intents, "_async_try_targeted_broadcast", no_targeted_match)
    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)

    result = await async_try_handle_local_intent(
        hass,
        cast(Any, SimpleNamespace(text="hello")),
        cast(Any, SimpleNamespace()),
        {CONF_LOCAL_INTENTS_ENABLED: True},
        guest_active=False,
    )

    assert result is not None
    assert result.response is response
    assert result.intent_name == "unknown"


def test_registered_intent_catalog_ignores_invalid_names(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    handlers = [
        SimpleNamespace(intent_type="HassTurnOn"),
        SimpleNamespace(intent_type=""),
        SimpleNamespace(intent_type=None),
        SimpleNamespace(intent_type=123),
    ]
    monkeypatch.setattr(ha_intent, "async_get", lambda hass: handlers)

    catalog = registered_intent_catalog(hass, ["", "   ", 123, "HassMissing"])

    assert [item["intent"] for item in catalog] == ["HassMissing", "HassTurnOn"]
    assert [item["available"] for item in catalog] == [False, True]


def test_conversation_entity_resolution_handles_registry_compatibility(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = object()
    monkeypatch.setattr(local_intents.er, "async_get", lambda _hass: registry)

    def unsupported(_registry, _entry_id):
        raise AttributeError

    monkeypatch.setattr(local_intents.er, "async_entries_for_config_entry", unsupported)
    assert local_intents._conversation_entity_id(hass, "entry", "agent") is None


def test_conversation_entity_resolution_selects_exact_agent(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = object()
    monkeypatch.setattr(local_intents.er, "async_get", lambda _hass: registry)
    monkeypatch.setattr(
        local_intents.er,
        "async_entries_for_config_entry",
        lambda _registry, _entry_id: [
            SimpleNamespace(
                config_subentry_id="other",
                domain="conversation",
                entity_id="conversation.other",
            ),
            SimpleNamespace(
                config_subentry_id="agent",
                domain="sensor",
                entity_id="sensor.agent",
            ),
            SimpleNamespace(
                config_subentry_id="agent",
                domain="conversation",
                entity_id="conversation.agent",
            ),
        ],
    )

    assert (
        local_intents._conversation_entity_id(hass, "entry", "agent")
        == "conversation.agent"
    )


def test_get_assist_pipelines_returns_snapshot_list(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = SimpleNamespace(id="one")
    fake_assist_pipeline = SimpleNamespace(
        async_get_pipelines=lambda _hass: (item for item in [pipeline])
    )
    components = sys.modules["homeassistant.components"]
    monkeypatch.setitem(
        sys.modules, "homeassistant.components.assist_pipeline", fake_assist_pipeline
    )
    monkeypatch.setattr(
        components, "assist_pipeline", fake_assist_pipeline, raising=False
    )

    assert local_intents._get_assist_pipelines(hass) == [pipeline]


def test_pipeline_conflicts_stop_when_agent_entity_is_missing(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        local_intents,
        "_conversation_entity_id",
        lambda hass, entry_id, subentry_id: None,
    )

    assert conflicting_assist_pipelines(hass, "entry", "agent") == []


@pytest.mark.parametrize("error", [ImportError, KeyError, RuntimeError])
def test_pipeline_conflicts_tolerate_unavailable_pipeline_state(
    hass, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    monkeypatch.setattr(
        local_intents,
        "_conversation_entity_id",
        lambda hass, entry_id, subentry_id: "conversation.agent",
    )

    def fail(_hass):
        raise error("unavailable")

    monkeypatch.setattr(local_intents, "_get_assist_pipelines", fail)

    assert conflicting_assist_pipelines(hass, "entry", "agent") == []


@pytest.mark.asyncio
async def test_resolved_targeted_broadcast_short_circuits_ha_intent_engine(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resolved targeted broadcast must not fall through to HassBroadcast."""
    response = ha_intent.IntentResponse(language="en-IE")
    response.async_set_speech("Broadcast queued.")
    targeted = local_intents.LocalIntentResult(
        response=response,
        intent_name="ExtendedBroadcast",
    )
    handle_calls = 0

    async def fake_targeted(_hass: Any, _user_input: Any):
        return targeted

    async def fake_handle(*_args: Any, **_kwargs: Any):
        nonlocal handle_calls
        handle_calls += 1
        raise AssertionError("resolved targeted broadcast fell through to HA intents")

    monkeypatch.setattr(local_intents, "_async_try_targeted_broadcast", fake_targeted)
    monkeypatch.setattr(conversation, "async_handle_intents", fake_handle)

    result = await local_intents.async_try_handle_local_intent(
        hass,
        cast(
            Any,
            SimpleNamespace(
                text="Broadcast to the kitchen that dinner is ready",
                language="en-IE",
            ),
        ),
        cast(Any, SimpleNamespace()),
        {local_intents.CONF_LOCAL_INTENTS_ENABLED: True},
        guest_active=False,
    )

    assert result is targeted
    assert result.intent_name == "ExtendedBroadcast"
    assert result.response is response
    assert handle_calls == 0


def test_conversation_entity_resolution_returns_none_when_no_agent_matches(
    hass, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = object()
    monkeypatch.setattr(local_intents.er, "async_get", lambda _hass: registry)
    monkeypatch.setattr(
        local_intents.er,
        "async_entries_for_config_entry",
        lambda _registry, _entry_id: [
            SimpleNamespace(
                config_subentry_id="other",
                domain="conversation",
                entity_id="conversation.other",
            ),
            SimpleNamespace(
                config_subentry_id="agent",
                domain="sensor",
                entity_id="sensor.agent",
            ),
        ],
    )

    assert local_intents._conversation_entity_id(hass, "entry", "agent") is None


def test_local_handling_snapshot_tolerates_lightweight_hass_standin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_catalog(_hass, _configured):
        raise TypeError("not a real HomeAssistant registry key")

    monkeypatch.setattr(local_intents, "registered_intent_catalog", broken_catalog)

    snapshot = local_intents.local_handling_snapshot(
        SimpleNamespace(),
        "entry",
        "agent",
        ["HassTurnOn", "", "HassFutureIntent"],
    )

    assert snapshot["pipeline_conflicts"] == []
    assert [item["intent"] for item in snapshot["intents"]] == [
        "HassFutureIntent",
        "HassTurnOn",
    ]
    assert all(item["available"] is False for item in snapshot["intents"])
    assert isinstance(snapshot["supported"], bool)
