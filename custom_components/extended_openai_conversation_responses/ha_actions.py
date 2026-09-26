"""Shared Home Assistant action execution seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import voluptuous as vol

try:
    from probatio.error import Invalid as ProbatioInvalid
except ImportError:  # pragma: no cover - compatibility with older HA releases
    _SERVICE_SCHEMA_ERRORS: tuple[type[Exception], ...] = (vol.Invalid,)
else:
    _SERVICE_SCHEMA_ERRORS = (vol.Invalid, ProbatioInvalid)

from homeassistant.const import (
    ATTR_AREA_ID,
    ATTR_DEVICE_ID,
    ATTR_ENTITY_ID,
    ATTR_FLOOR_ID,
    ATTR_LABEL_ID,
)
from homeassistant.core import Context, HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    target as target_helpers,
)

from .ha_permissions import async_require_control_permission, get_active_ha_context


async def async_call_ha_action(
    hass: HomeAssistant,
    domain: str,
    service: str,
    *,
    data: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    blocking: bool = False,
    context: Context | None = None,
) -> dict[str, dict[str, Any]]:
    """Call one HA action through the integration's common authorization seam.

    Both model-driven native service calls and administrator-configured local
    Request Rules pass through this backend-enforced policy seam.
    """
    context = context or get_active_ha_context()
    entity_ids = _resolve_target_entity_ids(hass, data, target)
    target_identity = _target_identity(hass, entity_ids)
    service_identity = _service_identity(hass, domain, service)
    await async_require_control_permission(hass, entity_ids, context=context)

    # A visible entity_id can be reused by a different registry entry. Registry
    # entries are replaced on updates, so identity also catches A -> B -> A
    # membership changes that a second textual resolution would miss.
    selection = _target_selection(data, target)
    indirect = any(
        key in selection
        for key in (ATTR_DEVICE_ID, ATTR_AREA_ID, ATTR_FLOOR_ID, ATTR_LABEL_ID)
    )
    if (
        (indirect and _resolve_target_entity_ids(hass, data, target) != entity_ids)
        or not _same_target_identity(
            _target_identity(hass, entity_ids), target_identity
        )
        or _service_identity(hass, domain, service) is not service_identity
    ):
        raise HomeAssistantError(
            "Home Assistant target changed while authorization was in progress; "
            "please retry"
        )

    return await _async_call_ha_action_unchecked(
        hass,
        domain,
        service,
        data=data,
        target=target,
        blocking=blocking,
        context=context,
    )


async def _async_call_ha_action_unchecked(
    hass: HomeAssistant,
    domain: str,
    service: str,
    *,
    data: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    blocking: bool = False,
    context: Context | None = None,
) -> dict[str, dict[str, Any]]:
    """Call an action after the caller completed policy enforcement."""
    if not hass.services.has_service(domain, service):
        raise ServiceNotFound(domain, service)
    previous_state = _capture_previous_state(hass, domain, service, data, target)
    kwargs: dict[str, Any] = {"service_data": dict(data or {})}
    if target:
        kwargs["target"] = dict(target)
    if blocking:
        kwargs["blocking"] = True
    if context is not None:
        kwargs["context"] = context
    try:
        await hass.services.async_call(domain=domain, service=service, **kwargs)
    except _SERVICE_SCHEMA_ERRORS as err:
        # Home Assistant has transitioned service schemas from voluptuous to
        # probatio. Keep the integration's action boundary stable across both: a
        # schema rejection is an ordinary Home Assistant action failure that the
        # caller can surface to the model/user, not an uncaught conversation error.
        raise HomeAssistantError(str(err)) from err
    return previous_state


async def async_execute_ha_actions(
    hass: HomeAssistant,
    actions: Sequence[Mapping[str, Any]],
    *,
    context: Context | None = None,
) -> list[dict[str, dict[str, Any]]]:
    """Execute an authorized sequence, stopping at the first failure."""
    results: list[dict[str, dict[str, Any]]] = []
    for action in actions:
        results.append(
            await async_call_ha_action(
                hass,
                str(action["domain"]),
                str(action["service"]),
                data=action.get("data", {}),
                target=action.get("target", {}),
                blocking=True,
                context=context,
            )
        )
    return results


_TARGET_KEYS = (
    ATTR_ENTITY_ID,
    ATTR_DEVICE_ID,
    ATTR_AREA_ID,
    ATTR_FLOOR_ID,
    ATTR_LABEL_ID,
)

_ATTRIBUTE_MAP: dict[str, tuple[tuple[str, str], ...]] = {
    "fan": tuple(
        (name, name)
        for name in ("percentage", "preset_mode", "direction", "oscillating")
    ),
    "climate": tuple(
        (name, name)
        for name in (
            "temperature",
            "target_temp_high",
            "target_temp_low",
            "fan_mode",
            "preset_mode",
            "swing_mode",
            "humidity",
        )
    ),
    "cover": (
        ("current_position", "position"),
        ("current_tilt_position", "tilt_position"),
    ),
    "valve": (("current_position", "position"),),
    "media_player": tuple(
        (name, name)
        for name in (
            "volume_level",
            "is_volume_muted",
            "source",
            "sound_mode",
            "repeat",
            "shuffle",
        )
    ),
    "humidifier": (("humidity", "humidity"), ("mode", "mode")),
    "water_heater": tuple(
        (name, name) for name in ("temperature", "operation_mode", "away_mode")
    ),
}

_STATE_ONLY_DOMAINS = {
    "alarm_control_panel",
    "automation",
    "counter",
    "input_boolean",
    "input_number",
    "input_select",
    "input_text",
    "lock",
    "number",
    "select",
    "switch",
    "text",
}

_NON_REVERSIBLE_DOMAINS = {
    "button",
    "event",
    "input_button",
    "notify",
    "scene",
    "script",
    "stt",
    "timer",
    "tts",
    "update",
}

_AUTOMATION_STATE_SERVICES = {"turn_on", "turn_off", "toggle"}


def serialize_reversible_state(state: State) -> dict[str, Any] | None:
    """Serialize only the control state useful for restoring an entity."""
    domain = state.entity_id.partition(".")[0]
    if domain in _NON_REVERSIBLE_DOMAINS:
        return None

    result: dict[str, Any] = {"state": state.state}
    attributes = state.attributes
    if domain == "light":
        _copy_attribute(attributes, result, "brightness")
        _copy_light_color(attributes, result)
        _copy_attribute(attributes, result, "effect")
    elif domain in _ATTRIBUTE_MAP:
        for source, destination in _ATTRIBUTE_MAP[domain]:
            _copy_attribute(attributes, result, source, destination)
    elif domain not in _STATE_ONLY_DOMAINS:
        # Unknown stateful domains get the conservative state-only fallback.
        return result
    return result


def _copy_light_color(attributes: Mapping[str, Any], result: dict[str, Any]) -> None:
    """Copy one colour representation that matches the light's active mode."""
    color_mode = attributes.get("color_mode")
    color_mode_key = color_mode if isinstance(color_mode, str) else ""
    preferred: str | None = {
        "color_temp": "color_temp_kelvin",
        "hs": "hs_color",
        "xy": "xy_color",
        "rgb": "rgb_color",
        "rgbw": "rgb_color",
        "rgbww": "rgb_color",
    }.get(color_mode_key)
    if color_mode_key:
        if preferred is not None:
            _copy_attribute(attributes, result, preferred)
        return
    for attribute in ("rgb_color", "hs_color", "xy_color", "color_temp_kelvin"):
        if attributes.get(attribute) is not None:
            result[attribute] = attributes[attribute]
            return


def _copy_attribute(
    attributes: Mapping[str, Any],
    result: dict[str, Any],
    source: str,
    destination: str | None = None,
) -> None:
    value = attributes.get(source)
    if value is not None:
        result[destination or source] = value


def _capture_previous_state(
    hass: HomeAssistant,
    action_domain: str,
    service: str,
    data: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Resolve action targets and capture their state immediately before the call."""
    if not _action_supports_previous_state(action_domain, service):
        return {}
    if not hasattr(hass, "states"):
        return {}
    entity_ids = _resolve_target_entity_ids(hass, data, target)
    result: dict[str, dict[str, Any]] = {}
    for entity_id in sorted(entity_ids):
        entity_domain = entity_id.partition(".")[0]
        if action_domain != "homeassistant" and entity_domain != action_domain:
            continue
        state = hass.states.get(entity_id)
        if not isinstance(state, State):
            continue
        serialized = serialize_reversible_state(state)
        if serialized is not None:
            result[entity_id] = serialized
    return result


def _resolve_target_entity_ids(
    hass: HomeAssistant,
    data: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
) -> set[str]:
    """Resolve direct and indirect entity selectors for authorization."""
    selection = _target_selection(data, target)
    if not selection:
        return set()
    referenced = target_helpers.async_extract_referenced_entity_ids(
        hass, target_helpers.TargetSelection(selection)
    )
    return set(referenced.referenced | referenced.indirectly_referenced)


def _target_identity(
    hass: HomeAssistant, entity_ids: set[str]
) -> tuple[tuple[str, Any, Any, Any], ...]:
    """Capture registry/runtime owners, including device membership generations."""
    if not entity_ids:
        return ()
    entities = er.async_get(hass)
    devices = dr.async_get(hass)
    identities: list[tuple[str, Any, Any, Any]] = []
    for entity_id in sorted(entity_ids):
        entry = entities.async_get(entity_id)
        device = (
            devices.async_get(entry.device_id)
            if entry is not None and entry.device_id
            else None
        )
        identities.append(
            (
                entity_id,
                entry,
                device,
                hass.states.get(entity_id),
            )
        )
    return tuple(identities)


def _same_target_identity(
    current: tuple[tuple[str, Any, Any, Any], ...],
    previous: tuple[tuple[str, Any, Any, Any], ...],
) -> bool:
    """Compare ownership by object generation rather than entry field equality."""
    return len(current) == len(previous) and all(
        left[0] == right[0]
        and all(now is then for now, then in zip(left[1:], right[1:], strict=True))
        for left, right in zip(current, previous, strict=True)
    )


def _service_identity(hass: HomeAssistant, domain: str, service: str) -> Any:
    """Capture the current HA service owner across the authorization await."""
    return hass.services.async_services_for_domain(domain).get(service)


def _target_selection(
    data: Mapping[str, Any] | None, target: Mapping[str, Any] | None
) -> dict[str, Any]:
    selection: dict[str, Any] = {}
    for key in _TARGET_KEYS:
        values: list[Any] = []
        for source in (data, target):
            if source is None or source.get(key) is None:
                continue
            value = source[key]
            values.extend(value if isinstance(value, list) else [value])
        if values:
            selection[key] = values
    return selection


def _action_supports_previous_state(domain: str, service: str) -> bool:
    if domain in _NON_REVERSIBLE_DOMAINS:
        return False
    return domain != "automation" or service in _AUTOMATION_STATE_SERVICES
