"""Native tool for Home Assistant operations."""

from __future__ import annotations

from datetime import timedelta
import logging
import os
import time
from typing import Any

import voluptuous as vol
import yaml

from homeassistant.components import automation, energy, recorder
from homeassistant.components.recorder import history as recorder_history
from homeassistant.config import AUTOMATION_CONFIG_PATH
from homeassistant.const import SERVICE_RELOAD
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers import llm
import homeassistant.util.dt as dt_util

from ..const import EVENT_AUTOMATION_REGISTERED
from ..exceptions import CallServiceError, NativeNotFound
from .base import Function

_LOGGER = logging.getLogger(__name__)


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
        service_data = service_argument.get(
            "service_data", service_argument.get("data", {})
        )
        entity_id = service_data.get("entity_id", service_argument.get("entity_id"))
        area_id = service_data.get("area_id")
        device_id = service_data.get("device_id")

        if isinstance(entity_id, str):
            entity_id = [e.strip() for e in entity_id.split(",")]
        service_data["entity_id"] = entity_id

        if entity_id is None and area_id is None and device_id is None:
            raise CallServiceError(domain, service, service_data)
        if not hass.services.has_service(domain, service):
            raise ServiceNotFound(domain, service)
        self.validate_entity_ids(hass, entity_id or [], exposed_entities)

        try:
            await hass.services.async_call(
                domain=domain,
                service=service,
                service_data=service_data,
            )
            return {"success": True}
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
        automation_config = yaml.safe_load(arguments["automation_config"])
        config = {"id": str(round(time.time() * 1000))}
        if isinstance(automation_config, list):
            config.update(automation_config[0])
        if isinstance(automation_config, dict):
            config.update(automation_config)

        await automation.config._async_validate_config_item(hass, config, True, False)

        automations = [config]
        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            encoding="utf-8",
        ) as f:
            current_automations = yaml.safe_load(f.read())

        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "a" if current_automations else "w",
            encoding="utf-8",
        ) as f:
            raw_config = yaml.dump(automations, allow_unicode=True, sort_keys=False)
            f.write("\n" + raw_config)

        await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)
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
        # energy_manager.data is EnergyPreferences which is a TypedDict (already a dict)
        return dict(energy_manager.data)

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
        statistic_ids = arguments.get("statistic_ids", [])
        start_time_parsed = dt_util.parse_datetime(arguments["start_time"])
        end_time_parsed = dt_util.parse_datetime(arguments["end_time"])
        if start_time_parsed is None or end_time_parsed is None:
            raise HomeAssistantError("Invalid datetime format")
        start_time = dt_util.as_utc(start_time_parsed)
        end_time = dt_util.as_utc(end_time_parsed)

        instance = recorder.get_instance(hass)

        statistics = await instance.async_add_executor_job(
            recorder.statistics.statistics_during_period,
            hass,
            start_time,
            end_time,
            statistic_ids,
            arguments.get("period", "day"),
            arguments.get("units"),
            arguments.get("types", {"change"}),
        )

        metadata = await instance.async_add_executor_job(
            recorder.statistics.get_metadata,
            hass,
            statistic_ids,
        )

        # Inject unit_of_measurement into each entry so the LLM knows the
        # actual unit (Wh, kWh, etc.) instead of assuming kWh by default.
        for statistic_id, entries in statistics.items():
            unit = None
            if statistic_id in metadata:
                unit = metadata[statistic_id][1].unit_of_measurement
            for entry in entries:
                entry["unit_of_measurement"] = unit

        return statistics

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
