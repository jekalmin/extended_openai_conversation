"""Additional behavioral coverage for Home Assistant native tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pytest
import yaml

from custom_components.extended_openai_conversation_responses.exceptions import (
    CallServiceError,
    NativeNotFound,
)
from custom_components.extended_openai_conversation_responses.functions import (
    native as native_module,
)
from custom_components.extended_openai_conversation_responses.functions.native import (
    NativeFunction,
)
from homeassistant.core import Context, State
from homeassistant.exceptions import HomeAssistantError


def test_automation_parser_accepts_single_item_list_and_rejects_bad_yaml() -> None:
    parsed = native_module._parse_automation_config("- alias: Morning\n  actions: []\n")
    assert parsed["alias"] == "Morning"
    assert len(parsed["id"]) == 32

    with pytest.raises(HomeAssistantError, match="YAML is invalid"):
        native_module._parse_automation_config("[")
    with pytest.raises(HomeAssistantError, match="one YAML mapping"):
        native_module._parse_automation_config("plain text")


def test_atomic_write_without_precondition_and_restore_new_file(tmp_path: Path) -> None:
    path = tmp_path / "automations.yaml"

    native_module._atomic_write_text(path, "first")
    assert path.read_text(encoding="utf-8") == "first"

    native_module._restore_automation_file(path, None, "first")
    assert not path.exists()


def test_append_rejects_invalid_or_non_list_existing_yaml(tmp_path: Path) -> None:
    path = tmp_path / "automations.yaml"
    path.write_text("[", encoding="utf-8")
    with pytest.raises(
        HomeAssistantError, match="Existing automations YAML is invalid"
    ):
        native_module._append_automation_atomic(path, {"alias": "New"})

    path.write_text("alias: Existing\n", encoding="utf-8")
    with pytest.raises(HomeAssistantError, match="must contain a list"):
        native_module._append_automation_atomic(path, {"alias": "New"})


def test_append_repairs_nonstandard_list_and_missing_newline(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "automations.yaml"
    path.write_text("[{alias: Existing}]", encoding="utf-8")
    real_safe_load = yaml.safe_load
    calls = 0

    def fail_combined_parse(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise yaml.YAMLError("cannot concatenate flow-style YAML")
        return real_safe_load(value)

    monkeypatch.setattr(native_module.yaml, "safe_load", fail_combined_parse)
    native_module._append_automation_atomic(path, {"alias": "New"})

    assert [item["alias"] for item in real_safe_load(path.read_text())] == [
        "Existing",
        "New",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "execute_service_single",
        "send_broadcast",
        "add_automation",
        "get_energy",
        "get_user_from_user_id",
    ],
)
async def test_execute_dispatches_remaining_native_tools(name: str) -> None:
    function = NativeFunction()
    delegated = AsyncMock(return_value={"name": name})
    setattr(function, name, delegated)
    hass = SimpleNamespace()
    arguments = {"value": 1}
    context = SimpleNamespace()
    exposed = [{"entity_id": "light.kitchen"}]

    assert await function.execute(
        hass, {"name": name}, arguments, context, exposed
    ) == {"name": name}
    delegated.assert_awaited_once_with(
        hass, {"name": name}, arguments, context, exposed
    )


async def test_execute_rejects_unknown_native_tool() -> None:
    with pytest.raises(NativeNotFound):
        await NativeFunction().execute(
            SimpleNamespace(), {"name": "unknown"}, {}, None, []
        )


async def test_broadcast_resolves_destination_and_passes_origin(
    hass, monkeypatch
) -> None:
    manager = SimpleNamespace(
        resolve_named_target=Mock(
            return_value={"name": "Kitchen", "entity_id": "assist_satellite.kitchen"}
        ),
        resolve_targets=Mock(return_value=["assist_satellite.kitchen"]),
        async_send=AsyncMock(
            return_value={"id": "message-1", "targets": ["kitchen"], "deliveries": 1}
        ),
    )
    monkeypatch.setattr(
        native_module, "async_get_intercom", AsyncMock(return_value=manager)
    )
    authorized_targets = AsyncMock(return_value=["assist_satellite.kitchen"])
    monkeypatch.setattr(
        native_module, "async_authorized_broadcast_targets", authorized_targets
    )

    result = await NativeFunction().send_broadcast(
        hass,
        {},
        {"destination": "Kitchen", "message": "Dinner is ready"},
        SimpleNamespace(device_id="origin-device"),
        [],
    )

    assert result == {
        "success": True,
        "message_id": "message-1",
        "targets": ["kitchen"],
        "deliveries": 1,
    }
    authorized_targets.assert_awaited_once_with(
        hass,
        manager,
        context=ANY,
        origin_device_id="origin-device",
        whole_home=False,
        entity_ids=["assist_satellite.kitchen"],
    )
    manager.async_send.assert_awaited_once_with(
        "Dinner is ready",
        entity_ids=["assist_satellite.kitchen"],
        origin_device_id="origin-device",
        source="llm_tool",
    )


async def test_broadcast_validates_destination_selection(hass, monkeypatch) -> None:
    manager = SimpleNamespace(resolve_named_target=Mock(return_value=None))
    monkeypatch.setattr(
        native_module, "async_get_intercom", AsyncMock(return_value=manager)
    )
    function = NativeFunction()

    with pytest.raises(HomeAssistantError, match="Unknown Broadcast destination"):
        await function.send_broadcast(
            hass, {}, {"destination": "Missing", "message": "Hi"}, None, []
        )
    with pytest.raises(HomeAssistantError, match="Choose a Broadcast destination"):
        await function.send_broadcast(hass, {}, {"message": "Hi"}, None, [])


def test_indirect_service_target_must_resolve_to_entities(
    hass, exposed_entities, monkeypatch
) -> None:
    referenced = SimpleNamespace(referenced=set(), indirectly_referenced=set())
    monkeypatch.setattr(
        native_module.target_helpers,
        "async_extract_referenced_entity_ids",
        Mock(return_value=referenced),
    )

    with pytest.raises(HomeAssistantError, match="does not resolve"):
        NativeFunction().validate_service_targets(
            hass, {"area_id": "nowhere"}, exposed_entities
        )

    referenced.indirectly_referenced.add("light.living_room")
    NativeFunction().validate_service_targets(
        hass, {"area_id": "living-room"}, exposed_entities
    )


async def test_single_service_parses_comma_separated_entities(
    hass, exposed_entities, monkeypatch
) -> None:
    execute = AsyncMock(return_value=None)
    monkeypatch.setattr(native_module, "async_call_ha_action", execute)

    result = await NativeFunction().execute_service_single(
        hass,
        {},
        {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"entity_id": "light.living_room, sensor.temperature, "},
        },
        None,
        exposed_entities,
    )

    assert result == {"success": True}
    assert execute.await_args.kwargs["data"]["entity_id"] == [
        "light.living_room",
        "sensor.temperature",
    ]


async def test_single_service_requires_target_and_returns_action_error(
    hass, exposed_entities, monkeypatch
) -> None:
    function = NativeFunction()
    with pytest.raises(CallServiceError):
        await function.execute_service_single(
            hass,
            {},
            {"domain": "light", "service": "turn_on", "service_data": {}},
            None,
            exposed_entities,
        )

    monkeypatch.setattr(
        native_module,
        "async_call_ha_action",
        AsyncMock(side_effect=HomeAssistantError("action failed")),
    )
    result = await function.execute_service_single(
        hass,
        {},
        {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"entity_id": ["light.living_room"]},
        },
        None,
        exposed_entities,
    )
    assert result == {"error": "action failed"}


async def test_failed_reload_logs_secondary_reload_failure(hass, monkeypatch) -> None:
    monkeypatch.setattr(
        native_module.automation.config,
        "_async_validate_config_item",
        AsyncMock(),
    )
    hass.services.async_call = AsyncMock(
        side_effect=[RuntimeError("initial reload"), RuntimeError("rollback reload")]
    )

    with pytest.raises(RuntimeError, match="initial reload"):
        await NativeFunction().add_automation(
            hass,
            {},
            {"automation_config": "alias: New\ntriggers: []\nactions: []\n"},
            None,
            [],
        )
    assert hass.services.async_call.await_count == 2


async def test_energy_without_configuration_returns_empty(hass, monkeypatch) -> None:
    monkeypatch.setattr(
        native_module.energy,
        "async_get_manager",
        AsyncMock(return_value=SimpleNamespace(data=None)),
    )
    assert await NativeFunction().get_energy(hass, {}, {}, None, []) == {}


@pytest.mark.parametrize(
    "llm_context",
    [None, SimpleNamespace(context=None), SimpleNamespace(context=Context())],
)
async def test_user_lookup_without_authenticated_user_is_unknown(
    hass, llm_context
) -> None:
    assert await NativeFunction().get_user_from_user_id(
        hass, {}, {}, llm_context, []
    ) == {"name": "Unknown"}


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        (SimpleNamespace(name="Alice"), "Alice"),
        (SimpleNamespace(name=None), "Unknown"),
        (None, "Unknown"),
    ],
)
async def test_user_lookup_returns_safe_name(hass, user, expected: str) -> None:
    hass.auth.async_get_user = AsyncMock(return_value=user)
    context = SimpleNamespace(context=Context(user_id="user-1"))
    assert await NativeFunction().get_user_from_user_id(hass, {}, {}, context, []) == {
        "name": expected
    }


def test_datetime_and_state_conversion_edges() -> None:
    function = NativeFunction()
    sentinel = object()
    assert function.as_utc(None, sentinel, "bad") is sentinel
    with pytest.raises(HomeAssistantError, match="bad time"):
        function.as_utc("not-a-time", sentinel, "bad time")

    state = State("sensor.temperature", "21")
    assert function.as_dict(state)["entity_id"] == "sensor.temperature"
    mapping = {"state": "21"}
    assert function.as_dict(mapping) is mapping


@pytest.fixture(autouse=True)
def authenticated_automation_admin(hass):
    from types import SimpleNamespace
    from homeassistant.core import Context
    from custom_components.extended_openai_conversation_responses.ha_permissions import (
        bind_active_ha_context,
    )

    hass.auth.async_get_user = AsyncMock(
        return_value=SimpleNamespace(is_active=True, is_admin=True)
    )
    with bind_active_ha_context(Context(user_id="admin")):
        yield
