"""Native tool for Home Assistant operations."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import voluptuous as vol
import yaml

from homeassistant.components import automation, energy, recorder
from homeassistant.components.recorder import history as recorder_history
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import (
    ATTR_AREA_ID,
    ATTR_DEVICE_ID,
    ATTR_FLOOR_ID,
    ATTR_LABEL_ID,
    SERVICE_RELOAD,
)
from homeassistant.core import HomeAssistant, State, valid_entity_id
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers import llm, target as target_helpers
import homeassistant.util.dt as dt_util

from ..const import DOMAIN, EVENT_AUTOMATION_REGISTERED
from ..exceptions import CallServiceError, EntityNotExposed, NativeNotFound
from ..ha_actions import async_call_ha_action
from ..intercom import async_get_intercom
from ..safety_hardening import (
    _async_require_admin,
    _normalized_statistics_arguments,
    _validate_execute_service_request,
    _validate_history_request,
)
from .base import Function

_LOGGER = logging.getLogger(__name__)

_INDIRECT_TARGET_KEYS = (
    ATTR_DEVICE_ID,
    ATTR_AREA_ID,
    ATTR_FLOOR_ID,
    ATTR_LABEL_ID,
)
_AUTOMATION_WRITE_LOCK_KEY = f"{DOMAIN}.automation_write_lock"
_UNCONDITIONAL_WRITE = object()
_MAX_STATISTIC_IDS = 100


def _exposed_entity_ids(exposed_entities: list[dict[str, Any]]) -> set[str]:
    """Return the entity IDs available to the current Assist/HA user scope."""
    return {
        str(entity["entity_id"])
        for entity in exposed_entities
        if isinstance(entity.get("entity_id"), str)
    }


def _energy_entity_ids(value: Any) -> set[str]:
    """Collect entity IDs only from Energy fields that reference entities/stats."""
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                isinstance(item, str)
                and (
                    key.startswith("entity_")
                    or key.startswith("stat_")
                    or key == "included_in_stat"
                )
                and valid_entity_id(item)
            ):
                result.add(item)
            elif isinstance(item, (dict, list, tuple)):
                result.update(_energy_entity_ids(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(_energy_entity_ids(item))
    return result


def _parse_automation_config(raw_config: str) -> dict[str, Any]:
    """Parse exactly one automation and assign an integration-owned unique ID."""
    try:
        parsed = yaml.safe_load(raw_config)
    except yaml.YAMLError as err:
        raise HomeAssistantError(f"Automation YAML is invalid: {err}") from err

    if isinstance(parsed, list):
        if len(parsed) != 1 or not isinstance(parsed[0], dict):
            raise HomeAssistantError(
                "automation_config must contain exactly one automation"
            )
        config = dict(parsed[0])
    elif isinstance(parsed, dict):
        config = dict(parsed)
    else:
        raise HomeAssistantError("automation_config must contain one YAML mapping")

    # IDs are integration-owned so concurrent or model-supplied values cannot collide
    # with another automation created by this tool.
    config["id"] = uuid4().hex
    return config


def _read_optional_text(path: Path) -> str | None:
    """Read a text file, distinguishing a missing file from an empty one."""
    return path.read_text(encoding="utf-8") if path.exists() else None


def _assert_automation_file_unchanged(path: Path, expected: str | None) -> None:
    """Abort instead of overwriting an automation file changed by another writer."""
    if _read_optional_text(path) != expected:
        raise HomeAssistantError(
            "automations.yaml changed while it was being updated; please retry"
        )


def _atomic_write_text(
    path: Path,
    content: str,
    *,
    expected_previous: object = _UNCONDITIONAL_WRITE,
) -> None:
    """Atomically replace text, optionally requiring an unchanged source file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    temp_path = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_path, mode)
        if expected_previous is not _UNCONDITIONAL_WRITE:
            _assert_automation_file_unchanged(
                path, expected_previous if isinstance(expected_previous, str) else None
            )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _append_automation_atomic(
    path: Path, config: dict[str, Any]
) -> tuple[str | None, str, str]:
    """Append one automation without overwriting a concurrent external edit."""
    previous = _read_optional_text(path)
    existing_text = previous or ""
    if existing_text.strip():
        try:
            current = yaml.safe_load(existing_text)
        except yaml.YAMLError as err:
            raise HomeAssistantError(
                f"Existing automations YAML is invalid: {err}"
            ) from err
        if not isinstance(current, list):
            raise HomeAssistantError("Existing automations YAML must contain a list")
    else:
        current = []

    raw_config = yaml.safe_dump([config], allow_unicode=True, sort_keys=False)
    if not current:
        updated = raw_config
    else:
        prefix = existing_text
        if not prefix.endswith("\n"):
            prefix += "\n"
        updated = prefix + raw_config
        try:
            combined = yaml.safe_load(updated)
        except yaml.YAMLError:
            combined = None
        if not isinstance(combined, list) or len(combined) != len(current) + 1:
            # Non-standard flow-style list: correctness is more important than
            # preserving formatting, so fall back to serializing the complete list.
            updated = yaml.safe_dump(
                [*current, config], allow_unicode=True, sort_keys=False
            )

    _atomic_write_text(path, updated, expected_previous=previous)
    return previous, raw_config, updated


def _restore_automation_file(
    path: Path, previous: str | None, expected_current: str
) -> None:
    """Rollback only if no external writer changed our just-written file."""
    if previous is None:
        _assert_automation_file_unchanged(path, expected_current)
        path.unlink(missing_ok=True)
        return
    _atomic_write_text(path, previous, expected_previous=expected_current)


class NativeFunction(Function):
    def __init__(self) -> None:
        """Initialize native tool."""
        super().__init__(vol.Schema({vol.Required("name"): str}))

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        name = function_config["name"]
        if name == "execute_service":
            return await self.execute_service(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "execute_service_single":
            return await self.execute_service_single(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "send_broadcast":
            return await self.send_broadcast(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "add_automation":
            return await self.add_automation(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_history":
            return await self.get_history(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_energy":
            return await self.get_energy(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_statistics":
            return await self.get_statistics(
                hass, function_config, arguments, llm_context, exposed_entities
            )
        if name == "get_user_from_user_id":
            return await self.get_user_from_user_id(
                hass, function_config, arguments, llm_context, exposed_entities
            )

        raise NativeNotFound(name)

    async def send_broadcast(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a targeted or whole-home Assist Satellite announcement."""
        manager = await async_get_intercom(hass)
        destination = arguments.get("destination")
        whole_home = bool(arguments.get("whole_home", False))
        target: dict[str, Any] = {"whole_home": whole_home}
        if destination:
            resolved = manager.resolve_named_target(str(destination))
            if resolved is None:
                raise HomeAssistantError(
                    f"Unknown Broadcast destination: {destination}"
                )
            resolved.pop("name", None)
            target.update(resolved)
        if not whole_home and not destination:
            raise HomeAssistantError("Choose a Broadcast destination or whole_home")
        origin_device_id = (
            getattr(llm_context, "device_id", None) if llm_context is not None else None
        )
        result = await manager.async_send(
            str(arguments.get("message", "")),
            **target,
            origin_device_id=origin_device_id,
            source="llm_tool",
        )
        return {
            "success": True,
            "message_id": result["id"],
            "targets": result["targets"],
            "deliveries": result["deliveries"],
        }

    def validate_service_targets(
        self,
        hass: HomeAssistant,
        service_data: dict[str, Any],
        exposed_entities: list[dict[str, Any]],
    ) -> None:
        """Resolve indirect HA targets and enforce the exposed-entity boundary."""
        selection = {
            key: value if isinstance(value, list) else [value]
            for key in _INDIRECT_TARGET_KEYS
            if (value := service_data.get(key)) is not None
        }
        if not selection:
            return

        referenced = target_helpers.async_extract_referenced_entity_ids(
            hass, target_helpers.TargetSelection(selection)
        )
        entity_ids = sorted(referenced.referenced | referenced.indirectly_referenced)
        if not entity_ids:
            raise HomeAssistantError(
                "Service target does not resolve to any Home Assistant entities"
            )
        self.validate_entity_ids(hass, entity_ids, exposed_entities)

    async def execute_service_single(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        service_argument: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        domain = service_argument["domain"]
        service = service_argument["service"]
        raw_service_data = service_argument.get(
            "service_data", service_argument.get("data", {})
        )
        service_data = dict(raw_service_data)
        entity_id = service_data.get("entity_id", service_argument.get("entity_id"))
        area_id = service_data.get("area_id")
        device_id = service_data.get("device_id")
        floor_id = service_data.get("floor_id")
        label_id = service_data.get("label_id")

        if isinstance(entity_id, str):
            entity_id = [e.strip() for e in entity_id.split(",") if e.strip()]
        if entity_id is not None:
            service_data["entity_id"] = entity_id

        if (
            entity_id is None
            and area_id is None
            and device_id is None
            and floor_id is None
            and label_id is None
        ):
            raise CallServiceError(domain, service, service_data)
        if not hass.services.has_service(domain, service):
            raise ServiceNotFound(domain, service)

        # Explicit entity IDs use the existing policy check. Resolve only indirect
        # area/device/floor/label targets so those selectors cannot bypass it.
        self.validate_entity_ids(hass, entity_id or [], exposed_entities)
        self.validate_service_targets(hass, service_data, exposed_entities)

        try:
            previous_state = await async_call_ha_action(
                hass,
                domain,
                service,
                data=service_data,
                blocking=True,
            )
            result: dict[str, Any] = {"success": True}
            if previous_state:
                result["previous_state"] = previous_state
            return result
        except HomeAssistantError as e:
            _LOGGER.error(e)
            return {"error": str(e)}

    async def execute_service(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        _validate_execute_service_request(arguments)
        result = []
        for service_argument in arguments.get("list", []):
            result.append(
                await self.execute_service_single(
                    hass,
                    function_config,
                    service_argument,
                    llm_context,
                    exposed_entities,
                )
            )
        return result

    async def add_automation(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> str:
        await _async_require_admin(hass, llm_context)
        config = await hass.async_add_executor_job(
            _parse_automation_config, arguments["automation_config"]
        )
        await automation.config._async_validate_config_item(hass, config, True, False)

        automation_path = Path(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH)
        )
        lock = hass.data.setdefault(_AUTOMATION_WRITE_LOCK_KEY, asyncio.Lock())
        async with lock:
            previous, raw_config, written = await hass.async_add_executor_job(
                _append_automation_atomic, automation_path, config
            )
            try:
                await hass.services.async_call(
                    automation.config.DOMAIN, SERVICE_RELOAD, blocking=True
                )
            except Exception:
                try:
                    await hass.async_add_executor_job(
                        _restore_automation_file, automation_path, previous, written
                    )
                except HomeAssistantError as rollback_err:
                    raise HomeAssistantError(
                        "Automation reload failed and automations.yaml changed before "
                        "rollback; external changes were preserved"
                    ) from rollback_err
                try:
                    await hass.services.async_call(
                        automation.config.DOMAIN, SERVICE_RELOAD, blocking=True
                    )
                except Exception:
                    _LOGGER.exception(
                        "Unable to reload automations after rolling back add_automation"
                    )
                raise

        hass.bus.async_fire(
            EVENT_AUTOMATION_REGISTERED,
            {"automation_config": config, "raw_config": raw_config},
        )
        return "Success"

    async def get_history(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        _validate_history_request(arguments)
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        entity_ids = arguments.get("entity_ids", [])
        include_start_time_state = arguments.get("include_start_time_state", True)
        significant_changes_only = arguments.get("significant_changes_only", True)
        minimal_response = arguments.get("minimal_response", True)
        no_attributes = arguments.get("no_attributes", True)

        now = dt_util.utcnow()
        one_day = timedelta(days=1)
        start_time = self.as_utc(start_time, now - one_day, "start_time not valid")
        end_time = self.as_utc(end_time, start_time + one_day, "end_time not valid")

        self.validate_entity_ids(hass, entity_ids, exposed_entities)

        try:
            with recorder.util.session_scope(hass=hass, read_only=True) as session:
                result = await recorder.get_instance(hass).async_add_executor_job(
                    recorder_history.get_significant_states_with_session,
                    hass,
                    session,
                    start_time,
                    end_time,
                    entity_ids,
                    None,
                    include_start_time_state,
                    significant_changes_only,
                    minimal_response,
                    no_attributes,
                )
        except (RuntimeError, OSError) as err:
            raise HomeAssistantError("History is temporarily unavailable") from err

        return [[self.as_dict(item) for item in sublist] for sublist in result.values()]

    async def get_energy(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        energy_manager: energy.data.EnergyManager = await energy.async_get_manager(hass)
        if energy_manager.data is None:
            return {}
        data = dict(energy_manager.data)
        hidden_entity_ids = _energy_entity_ids(data) - _exposed_entity_ids(
            exposed_entities
        )
        if hidden_entity_ids:
            # Do not include the identifiers in the error: the entire purpose of
            # this boundary is to avoid disclosing hidden entities to the model.
            raise HomeAssistantError(
                "Energy configuration references Home Assistant entities that are "
                "not exposed to Assist"
            )
        return data

    async def get_user_from_user_id(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if (
            llm_context is None
            or llm_context.context is None
            or llm_context.context.user_id is None
        ):
            return {"name": "Unknown"}
        user = await hass.auth.async_get_user(llm_context.context.user_id)
        user_name = (
            user.name
            if user and hasattr(user, "name") and user.name is not None
            else "Unknown"
        )
        return {"name": user_name}

    async def get_statistics(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        arguments = _normalized_statistics_arguments(arguments)
        raw_statistic_ids = arguments.get("statistic_ids")
        if (
            not isinstance(raw_statistic_ids, list)
            or not raw_statistic_ids
            or any(not isinstance(item, str) or not item for item in raw_statistic_ids)
        ):
            raise HomeAssistantError("statistic_ids must be a non-empty list of IDs")
        if len(raw_statistic_ids) > _MAX_STATISTIC_IDS:
            raise HomeAssistantError(
                f"statistic_ids may contain at most {_MAX_STATISTIC_IDS} IDs"
            )

        # Recorder uses '<domain>:<statistic>' for integration-owned/external
        # statistic IDs. All other IDs are entity-backed and must remain inside
        # the same Assist/HA READ exposure boundary as current state/history.
        exposed_entity_ids = _exposed_entity_ids(exposed_entities)
        unexposed = sorted(
            {
                statistic_id
                for statistic_id in raw_statistic_ids
                if not recorder.statistics.valid_statistic_id(statistic_id)
                and statistic_id not in exposed_entity_ids
            }
        )
        if unexposed:
            raise EntityNotExposed(", ".join(unexposed))

        statistic_ids = set(raw_statistic_ids)
        start_time_parsed = dt_util.parse_datetime(arguments["start_time"])
        end_time_parsed = dt_util.parse_datetime(arguments["end_time"])
        if start_time_parsed is None or end_time_parsed is None:
            raise HomeAssistantError("Invalid datetime format")
        start_time = dt_util.as_utc(start_time_parsed)
        end_time = dt_util.as_utc(end_time_parsed)

        return await recorder.get_instance(hass).async_add_executor_job(
            recorder.statistics.statistics_during_period,
            hass,
            start_time,
            end_time,
            statistic_ids,
            arguments.get("period", "day"),
            arguments.get("units"),
            arguments.get("types", {"change"}),
        )

    def as_utc(
        self, value: str | None, default_value: Any, parse_error_message: str
    ) -> Any:
        if value is None:
            return default_value

        parsed_datetime = dt_util.parse_datetime(value)
        if parsed_datetime is None:
            raise HomeAssistantError(parse_error_message)

        return dt_util.as_utc(parsed_datetime)

    def as_dict(self, state: State | dict[str, Any]) -> dict[str, Any]:
        if isinstance(state, State):
            return state.as_dict()
        return state
