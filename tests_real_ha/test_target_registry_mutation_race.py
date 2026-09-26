"""Real-HA acceptance for indirect target registry mutation during authorization."""

from __future__ import annotations

import asyncio

import pytest

from custom_components.extended_openai_conversation_responses import ha_actions
from homeassistant.const import ATTR_AREA_ID
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)
from pytest_homeassistant_custom_component.common import MockConfigEntry

_WAIT_TIMEOUT = 10
_DOMAIN = "registry_race_test"
_SERVICE = "mark"


@pytest.mark.parametrize("return_to_original_area", [False, True])
@pytest.mark.asyncio
async def test_registry_reassignment_between_resolution_and_dispatch_fails_closed(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    return_to_original_area: bool,
) -> None:
    """An indirect target may not change membership after authorization starts."""
    area_registry = ar.async_get(hass)
    device_registry = dr.async_get(hass)
    entity_registry = er.async_get(hass)

    config_entry = MockConfigEntry(domain=_DOMAIN)
    config_entry.add_to_hass(hass)

    area_a = area_registry.async_create("Registry race A")
    area_b = area_registry.async_create("Registry race B")
    device = device_registry.async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("registry_race_test", "device")},
        name="Registry race device",
    )
    device = device_registry.async_update_device(device.id, area_id=area_a.id)
    entity = entity_registry.async_get_or_create(
        domain="light",
        platform="registry_race_test",
        unique_id="registry-race-light",
        suggested_object_id="registry_race_light",
        device_id=device.id,
    )
    entity_id = entity.entity_id
    hass.states.async_set(entity_id, "off")
    await hass.async_block_till_done()

    service_calls: list[ServiceCall] = []

    async def service_handler(call: ServiceCall) -> None:
        service_calls.append(call)

    hass.services.async_register(_DOMAIN, _SERVICE, service_handler)

    permission_entered = asyncio.Event()
    allow_permission = asyncio.Event()
    permission_calls = 0

    async def gated_permission(
        _hass: HomeAssistant,
        entity_ids: set[str],
        *,
        context=None,
    ) -> None:
        nonlocal permission_calls
        del _hass, context
        permission_calls += 1
        assert entity_ids == {entity_id}
        if permission_calls == 1:
            permission_entered.set()
            await allow_permission.wait()

    monkeypatch.setattr(
        ha_actions,
        "async_require_control_permission",
        gated_permission,
    )

    action_task = asyncio.create_task(
        ha_actions.async_call_ha_action(
            hass,
            _DOMAIN,
            _SERVICE,
            data={ATTR_AREA_ID: area_a.id},
            blocking=True,
        )
    )
    await asyncio.wait_for(permission_entered.wait(), timeout=_WAIT_TIMEOUT)

    # The first target resolution and authorization input are now fixed to the
    # entity in area A. Move the device while the permission await is suspended;
    # the same area selector no longer resolves to that authorized entity.
    device_registry.async_update_device(device.id, area_id=area_b.id)
    if return_to_original_area:
        device_registry.async_update_device(device.id, area_id=area_a.id)
    await hass.async_block_till_done()
    allow_permission.set()

    with pytest.raises(HomeAssistantError, match="target changed.*retry"):
        await asyncio.wait_for(action_task, timeout=_WAIT_TIMEOUT)

    assert service_calls == []
    assert permission_calls == 1

    # A fresh action using the current registry assignment remains healthy. The
    # protection rejects only the stale in-flight resolution, not later requests.
    await ha_actions.async_call_ha_action(
        hass,
        _DOMAIN,
        _SERVICE,
        data={ATTR_AREA_ID: area_a.id if return_to_original_area else area_b.id},
        blocking=True,
    )
    assert len(service_calls) == 1
    assert service_calls[0].data[ATTR_AREA_ID] == (
        area_a.id if return_to_original_area else area_b.id
    )
    assert permission_calls == 2


@pytest.mark.parametrize("replace_registry_entry", [False, True])
@pytest.mark.asyncio
async def test_recreated_entity_with_same_id_cannot_inherit_authorization(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    replace_registry_entry: bool,
) -> None:
    """Registry identity, not the visible entity_id, owns the authorization."""
    registry = er.async_get(hass)
    original = registry.async_get_or_create(
        domain="light",
        platform=_DOMAIN,
        unique_id="original-light",
        suggested_object_id="reused_light",
    )
    hass.states.async_set(original.entity_id, "off")
    calls: list[ServiceCall] = []
    hass.services.async_register(_DOMAIN, _SERVICE, calls.append)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def gated_permission(_hass, entity_ids, *, context=None):
        assert entity_ids == {original.entity_id}
        entered.set()
        await resume.wait()

    monkeypatch.setattr(
        ha_actions, "async_require_control_permission", gated_permission
    )
    action = asyncio.create_task(
        ha_actions.async_call_ha_action(
            hass,
            _DOMAIN,
            _SERVICE,
            data={ATTR_ENTITY_ID: original.entity_id},
            blocking=True,
        )
    )
    await asyncio.wait_for(entered.wait(), _WAIT_TIMEOUT)
    hass.states.async_remove(original.entity_id)
    if replace_registry_entry:
        registry.async_remove(original.entity_id)
        replacement = registry.async_get_or_create(
            domain="light",
            platform=_DOMAIN,
            unique_id="replacement-light",
            suggested_object_id="reused_light",
        )
    else:
        replacement = original
    assert replacement.entity_id == original.entity_id
    assert (replacement.id != original.id) is replace_registry_entry
    hass.states.async_set(replacement.entity_id, "off")
    await hass.async_block_till_done()
    resume.set()
    with pytest.raises(HomeAssistantError, match="target changed.*retry"):
        await asyncio.wait_for(action, _WAIT_TIMEOUT)
    assert calls == []

    await ha_actions.async_call_ha_action(
        hass,
        _DOMAIN,
        _SERVICE,
        data={ATTR_ENTITY_ID: replacement.entity_id},
        blocking=True,
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_service_reload_does_not_dispatch_stale_request(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reloaded dependency cannot receive a request authorized for its predecessor."""
    entity_id = "light.reload_target"
    hass.states.async_set(entity_id, "off")
    old_calls: list[ServiceCall] = []
    new_calls: list[ServiceCall] = []
    hass.services.async_register(_DOMAIN, _SERVICE, old_calls.append)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def gated_permission(_hass, entity_ids, *, context=None):
        assert entity_ids == {entity_id}
        entered.set()
        await resume.wait()

    monkeypatch.setattr(
        ha_actions, "async_require_control_permission", gated_permission
    )
    action = asyncio.create_task(
        ha_actions.async_call_ha_action(
            hass,
            _DOMAIN,
            _SERVICE,
            data={ATTR_ENTITY_ID: entity_id},
            blocking=True,
        )
    )
    await asyncio.wait_for(entered.wait(), _WAIT_TIMEOUT)
    hass.services.async_remove(_DOMAIN, _SERVICE)
    hass.services.async_register(_DOMAIN, _SERVICE, new_calls.append)
    resume.set()
    with pytest.raises(HomeAssistantError, match="target changed.*retry"):
        await asyncio.wait_for(action, _WAIT_TIMEOUT)
    assert old_calls == new_calls == []

    await ha_actions.async_call_ha_action(
        hass,
        _DOMAIN,
        _SERVICE,
        data={ATTR_ENTITY_ID: entity_id},
        blocking=True,
    )
    assert len(new_calls) == 1
