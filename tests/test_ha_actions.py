"""Compact previous-state serialization and shared HA action execution tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol

from custom_components.extended_openai_conversation_responses import ha_actions
from custom_components.extended_openai_conversation_responses.ha_actions import (
    async_call_ha_action,
    serialize_reversible_state,
)
from homeassistant.core import Context, State
from homeassistant.exceptions import ServiceNotFound


def _state(entity_id: str, value: str, **attributes) -> State:
    return State(entity_id, value, attributes)


def test_light_state_is_compact_and_omits_null_attributes() -> None:
    assert serialize_reversible_state(_state("light.kitchen", "off")) == {
        "state": "off"
    }
    assert serialize_reversible_state(
        _state(
            "light.kitchen",
            "on",
            brightness=92,
            effect=None,
            friendly_name="Kitchen",
            supported_features=44,
        )
    ) == {"state": "on", "brightness": 92}


def test_light_uses_only_active_colour_representation() -> None:
    attributes = {
        "brightness": 180,
        "color_temp_kelvin": 2700,
        "rgb_color": (255, 80, 20),
        "hs_color": (15, 92),
        "xy_color": (0.6, 0.3),
    }
    assert serialize_reversible_state(
        _state("light.kitchen", "on", color_mode="color_temp", **attributes)
    ) == {
        "state": "on",
        "brightness": 180,
        "color_temp_kelvin": 2700,
    }
    assert serialize_reversible_state(
        _state("light.kitchen", "on", color_mode="rgb", **attributes)
    ) == {
        "state": "on",
        "brightness": 180,
        "rgb_color": (255, 80, 20),
    }


def test_light_without_colour_mode_uses_one_available_representation() -> None:
    assert serialize_reversible_state(
        _state(
            "light.kitchen",
            "on",
            rgb_color=(1, 2, 3),
            hs_color=(4, 5),
            xy_color=(0.1, 0.2),
        )
    ) == {"state": "on", "rgb_color": (1, 2, 3)}

    assert serialize_reversible_state(
        _state(
            "light.kitchen",
            "on",
            color_mode="brightness",
            rgb_color=(1, 2, 3),
        )
    ) == {"state": "on"}


def test_climate_excludes_measurements_and_capability_metadata() -> None:
    assert serialize_reversible_state(
        _state(
            "climate.office",
            "heat",
            temperature=21,
            target_temp_high=None,
            target_temp_low=None,
            fan_mode="auto",
            preset_mode="eco",
            swing_mode=None,
            humidity=45,
            current_temperature=18.2,
            current_humidity=51,
            min_temp=7,
            max_temp=35,
            supported_features=401,
            friendly_name="Office",
        )
    ) == {
        "state": "heat",
        "temperature": 21,
        "fan_mode": "auto",
        "preset_mode": "eco",
        "humidity": 45,
    }


def test_cover_uses_semantic_position_names() -> None:
    assert serialize_reversible_state(
        _state(
            "cover.blind",
            "open",
            current_position=62,
            current_tilt_position=14,
            friendly_name="Blind",
        )
    ) == {"state": "open", "position": 62, "tilt_position": 14}


def test_media_player_excludes_playing_media_metadata() -> None:
    assert serialize_reversible_state(
        _state(
            "media_player.lounge",
            "playing",
            volume_level=0.35,
            is_volume_muted=False,
            source="TV",
            sound_mode="Movie",
            repeat="off",
            shuffle=False,
            media_title="Private title",
            media_content_id="https://example.invalid/private",
            entity_picture="/api/media_player_proxy/example",
        )
    ) == {
        "state": "playing",
        "volume_level": 0.35,
        "is_volume_muted": False,
        "source": "TV",
        "sound_mode": "Movie",
        "repeat": "off",
        "shuffle": False,
    }


@pytest.mark.parametrize(
    "entity_id",
    [
        "switch.plug",
        "lock.front_door",
        "number.volume",
        "select.profile",
        "input_number.limit",
        "input_select.mode",
        "vacuum.downstairs",
    ],
)
def test_state_only_domains_do_not_leak_attributes(entity_id: str) -> None:
    assert serialize_reversible_state(
        _state(entity_id, "example", friendly_name="Private", custom="secret")
    ) == {"state": "example"}


@pytest.mark.parametrize(
    "entity_id",
    [
        "button.restart",
        "input_button.doorbell",
        "script.goodnight",
        "scene.relax",
        "timer.laundry",
        "update.home_assistant",
    ],
)
def test_non_reversible_entity_domains_have_no_snapshot(entity_id: str) -> None:
    assert serialize_reversible_state(_state(entity_id, "on", custom="secret")) is None


def _hass_with_states(states: dict[str, State], events: list[str] | None = None):
    event_log = events if events is not None else []
    services = SimpleNamespace(
        has_service=lambda _domain, _service: True,
        async_call=AsyncMock(side_effect=lambda **_kwargs: event_log.append("execute")),
    )

    def get_state(entity_id: str):
        event_log.append(f"capture:{entity_id}")
        return states.get(entity_id)

    return SimpleNamespace(
        services=services,
        states=SimpleNamespace(get=get_state),
    )


async def test_multiple_entities_are_captured_before_execution(monkeypatch) -> None:
    events: list[str] = []
    hass = _hass_with_states(
        {
            "light.island": _state("light.island", "off"),
            "light.sink": _state("light.sink", "on", brightness=71),
            "switch.unrelated": _state("switch.unrelated", "on"),
        },
        events,
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.ha_actions.target_helpers.async_extract_referenced_entity_ids",
        lambda _hass, _selection: SimpleNamespace(
            referenced={"light.island"},
            indirectly_referenced={"light.sink", "switch.unrelated"},
        ),
    )

    result = await async_call_ha_action(
        hass, "light", "turn_on", target={"area_id": "kitchen"}
    )

    assert result == {
        "light.island": {"state": "off"},
        "light.sink": {"state": "on", "brightness": 71},
    }
    assert events == ["capture:light.island", "capture:light.sink", "execute"]


@pytest.mark.parametrize(
    ("domain", "service", "entity_id"),
    [
        ("button", "press", "button.restart"),
        ("script", "turn_on", "script.goodnight"),
        ("scene", "turn_on", "scene.relax"),
        ("notify", "send_message", "notify.family"),
        ("automation", "trigger", "automation.arrive_home"),
    ],
)
async def test_non_reversible_actions_do_not_read_state(
    monkeypatch, domain: str, service: str, entity_id: str
) -> None:
    events: list[str] = []
    hass = _hass_with_states({entity_id: _state(entity_id, "on")}, events)
    resolver = MagicMock(
        return_value=SimpleNamespace(
            referenced={entity_id}, indirectly_referenced=set()
        )
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.ha_actions.target_helpers.async_extract_referenced_entity_ids",
        resolver,
    )

    assert (
        await async_call_ha_action(
            hass, domain, service, target={"entity_id": entity_id}
        )
        == {}
    )
    resolver.assert_called_once()
    assert events == ["execute"]


async def test_capture_happens_immediately_before_call(
    monkeypatch,
) -> None:
    events: list[str] = []
    hass = _hass_with_states(
        {"lock.front_door": _state("lock.front_door", "locked")}, events
    )

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.ha_actions.target_helpers.async_extract_referenced_entity_ids",
        lambda _hass, _selection: SimpleNamespace(
            referenced={"lock.front_door"}, indirectly_referenced=set()
        ),
    )

    result = await async_call_ha_action(
        hass,
        "lock",
        "unlock",
        target={"entity_id": "lock.front_door"},
    )

    assert result == {"lock.front_door": {"state": "locked"}}
    assert events == ["capture:lock.front_door", "execute"]


# Consolidated HA action execution/targeting regressions.
@pytest.mark.asyncio
async def test_unchecked_action_rejects_missing_service() -> None:
    hass = SimpleNamespace(
        services=SimpleNamespace(
            has_service=lambda _domain, _service: False,
            async_call=AsyncMock(),
        )
    )

    with pytest.raises(ServiceNotFound):
        await ha_actions._async_call_ha_action_unchecked(
            hass, "light", "missing", data={"brightness": 10}
        )

    hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_actions_runs_in_order_with_blocking_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    context = Context(user_id="user-1")

    async def fake_call(
        hass: object,
        domain: str,
        service: str,
        **kwargs: Any,
    ) -> dict[str, dict[str, Any]]:
        calls.append({"domain": domain, "service": service, **kwargs})
        return {f"{domain}.entity": {"state": service}}

    monkeypatch.setattr(ha_actions, "async_call_ha_action", fake_call)

    result = await ha_actions.async_execute_ha_actions(
        SimpleNamespace(),
        [
            {
                "domain": "light",
                "service": "turn_on",
                "data": {"brightness": 50},
                "target": {"entity_id": "light.kitchen"},
            },
            {"domain": "switch", "service": "turn_off"},
        ],
        context=context,
    )

    assert result == [
        {"light.entity": {"state": "turn_on"}},
        {"switch.entity": {"state": "turn_off"}},
    ]
    assert calls == [
        {
            "domain": "light",
            "service": "turn_on",
            "data": {"brightness": 50},
            "target": {"entity_id": "light.kitchen"},
            "blocking": True,
            "context": context,
        },
        {
            "domain": "switch",
            "service": "turn_off",
            "data": {},
            "target": {},
            "blocking": True,
            "context": context,
        },
    ]


def test_capture_returns_empty_when_hass_has_no_state_machine() -> None:
    assert (
        ha_actions._capture_previous_state(
            SimpleNamespace(),
            "light",
            "turn_on",
            None,
            {"entity_id": "light.kitchen"},
        )
        == {}
    )


def test_capture_skips_missing_or_non_state_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hass = SimpleNamespace(states=SimpleNamespace(get=lambda _entity_id: object()))
    monkeypatch.setattr(
        ha_actions,
        "_resolve_target_entity_ids",
        lambda *_args: {"light.kitchen"},
    )

    assert (
        ha_actions._capture_previous_state(
            hass,
            "light",
            "turn_on",
            None,
            {"entity_id": "light.kitchen"},
        )
        == {}
    )


def test_homeassistant_capture_skips_non_reversible_entity_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state("button.restart", "unknown")
    hass = SimpleNamespace(states=SimpleNamespace(get=lambda _entity_id: state))
    monkeypatch.setattr(
        ha_actions,
        "_resolve_target_entity_ids",
        lambda *_args: {"button.restart"},
    )

    assert (
        ha_actions._capture_previous_state(
            hass,
            "homeassistant",
            "turn_on",
            None,
            {"entity_id": "button.restart"},
        )
        == {}
    )


def test_resolve_target_ids_short_circuits_empty_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = MagicMock()
    monkeypatch.setattr(
        ha_actions.target_helpers,
        "async_extract_referenced_entity_ids",
        resolver,
    )

    assert ha_actions._resolve_target_entity_ids(SimpleNamespace(), {}, {}) == set()
    resolver.assert_not_called()


def test_target_selection_merges_scalar_and_list_values() -> None:
    assert ha_actions._target_selection(
        {
            "entity_id": "light.one",
            "area_id": ["kitchen", "hall"],
            "device_id": None,
        },
        {
            "entity_id": ["light.two", "light.three"],
            "area_id": "office",
            "label_id": "important",
        },
    ) == {
        "entity_id": ["light.one", "light.two", "light.three"],
        "area_id": ["kitchen", "hall", "office"],
        "label_id": ["important"],
    }


@pytest.mark.asyncio
async def test_unchecked_action_translates_service_schema_rejection() -> None:
    services = SimpleNamespace(
        has_service=lambda _domain, _service: True,
        async_call=AsyncMock(side_effect=vol.Invalid("invalid service payload")),
    )
    hass = SimpleNamespace(services=services)

    with pytest.raises(
        ha_actions.HomeAssistantError, match="invalid service payload"
    ) as raised:
        await ha_actions._async_call_ha_action_unchecked(
            hass,
            "light",
            "turn_on",
            data={"brightness": "not-a-number"},
            blocking=True,
        )

    assert isinstance(raised.value.__cause__, vol.Invalid)
    services.async_call.assert_awaited_once()


@pytest.mark.parametrize(
    ("entity_id", "attributes", "expected"),
    [
        (
            "fan.office",
            {
                "percentage": 60,
                "preset_mode": "sleep",
                "direction": "forward",
                "oscillating": True,
                "friendly_name": "Private",
            },
            {
                "state": "on",
                "percentage": 60,
                "preset_mode": "sleep",
                "direction": "forward",
                "oscillating": True,
            },
        ),
        (
            "humidifier.bedroom",
            {"humidity": 45, "mode": "auto", "current_humidity": 51},
            {"state": "on", "humidity": 45, "mode": "auto"},
        ),
        (
            "water_heater.tank",
            {"temperature": 55, "operation_mode": "eco", "away_mode": False},
            {
                "state": "eco",
                "temperature": 55,
                "operation_mode": "eco",
                "away_mode": False,
            },
        ),
        (
            "valve.radiator",
            {"current_position": 37, "friendly_name": "Radiator"},
            {"state": "open", "position": 37},
        ),
    ],
)
def test_additional_reversible_domains_keep_only_control_state(
    entity_id: str,
    attributes: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    state = "eco" if entity_id.startswith("water_heater.") else "open" if entity_id.startswith("valve.") else "on"
    assert serialize_reversible_state(_state(entity_id, state, **attributes)) == expected


def test_target_identity_comparison_uses_object_generation() -> None:
    entry = SimpleNamespace(device_id="device-1")
    device = object()
    state = _state("light.kitchen", "on")
    identity = (("light.kitchen", entry, device, state),)

    assert ha_actions._same_target_identity(identity, identity)

    replacement_state = _state("light.kitchen", "on")
    changed_state = (("light.kitchen", entry, device, replacement_state),)
    assert not ha_actions._same_target_identity(changed_state, identity)

    replacement_entry = SimpleNamespace(device_id="device-1")
    changed_entry = (("light.kitchen", replacement_entry, device, state),)
    assert not ha_actions._same_target_identity(changed_entry, identity)

    assert not ha_actions._same_target_identity((), identity)
