"""Tests for NativeFunction using yaml definitions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Import Tools and test helpers
from custom_components.extended_openai_conversation.functions import (
    NativeFunction,
    get_function,
)
from tests.helpers import prepare_function_tool_from_yaml


class TestNativeFunctionYaml:
    """Test NativeFunction using yaml definitions."""

    @pytest.fixture
    def function(self):
        """Create NativeFunction instance."""
        return NativeFunction()

    async def test_execute_service_from_yaml(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test service execution from yaml definition."""
        # Load function from yaml
        function_tool = prepare_function_tool_from_yaml(
            "native_execute_service_example.yaml"
        )
        function_config = function_tool["function"]

        # Arguments in the format that LLM would send (matching spec.parameters)
        arguments = {
            "list": [
                {
                    "domain": "light",
                    "service": "turn_on",
                    "service_data": {
                        "entity_id": ["light.living_room"],
                        "brightness_pct": 75,
                        "color_temp": 400,
                    },
                }
            ]
        }

        result = await function.execute(
            hass, function_config, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()
        call_args = hass.services.async_call.call_args
        assert call_args[1]["domain"] == "light"
        assert call_args[1]["service"] == "turn_on"

    async def test_execute_service_with_defaults(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test service execution with minimal parameters."""
        function_tool = prepare_function_tool_from_yaml(
            "native_execute_service_example.yaml"
        )
        function_config = function_tool["function"]

        # Provide all required parameters
        arguments = {
            "list": [
                {
                    "domain": "light",
                    "service": "turn_on",
                    "service_data": {
                        "entity_id": ["light.living_room"],
                        "brightness_pct": 50,
                    },
                }
            ]
        }

        result = await function.execute(
            hass, function_config, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()

    async def test_execute_service_with_delay(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test service execution with delay parameter."""
        function_tool = prepare_function_tool_from_yaml(
            "native_execute_service_example.yaml"
        )
        function_config = function_tool["function"]

        arguments = {
            "delay": {"hours": 0, "minutes": 1, "seconds": 30},
            "list": [
                {
                    "domain": "light",
                    "service": "turn_on",
                    "service_data": {
                        "entity_id": ["light.living_room"],
                    },
                }
            ],
        }

        result = await function.execute(
            hass, function_config, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()

    async def test_execute_service_multiple_services(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test executing multiple services in one call."""
        function_tool = prepare_function_tool_from_yaml(
            "native_execute_service_example.yaml"
        )
        function_config = function_tool["function"]

        arguments = {
            "list": [
                {
                    "domain": "light",
                    "service": "turn_on",
                    "service_data": {"entity_id": ["light.living_room"]},
                },
                {
                    "domain": "switch",
                    "service": "turn_off",
                    "service_data": {"entity_id": ["switch.kitchen"]},
                },
            ]
        }

        result = await function.execute(
            hass, function_config, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}, {"success": True}]
        assert hass.services.async_call.call_count == 2

    async def test_execute_service_with_data_instead_of_service_data(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test service execution using 'data' instead of 'service_data'."""
        function_tool = prepare_function_tool_from_yaml(
            "native_execute_service_example.yaml"
        )
        function_config = function_tool["function"]

        arguments = {
            "list": [
                {
                    "domain": "light",
                    "service": "turn_on",
                    "data": {
                        "entity_id": ["light.living_room"],
                        "brightness_pct": 75,
                    },
                }
            ]
        }

        result = await function.execute(
            hass, function_config, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()

    async def test_execute_service_not_found_error(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test service execution fails when service does not exist."""
        from homeassistant.exceptions import ServiceNotFound

        function_tool = prepare_function_tool_from_yaml(
            "native_execute_service_example.yaml"
        )
        function_config = function_tool["function"]

        # Mock has_service to return False
        hass.services.has_service = MagicMock(return_value=False)

        arguments = {
            "list": [
                {
                    "domain": "light",
                    "service": "invalid_service",
                    "service_data": {"entity_id": ["light.living_room"]},
                }
            ]
        }

        with pytest.raises(ServiceNotFound):
            await function.execute(
                hass, function_config, arguments, llm_context, exposed_entities
            )


class TestNativeGetHistory:
    """Test composite get_history function."""

    async def test_get_history(self, hass, exposed_entities, llm_context):
        """Test getting entity history with composite function."""
        from unittest.mock import MagicMock, patch

        function_tool = prepare_function_tool_from_yaml(
            "native_get_history_example.yaml"
        )
        function_config = function_tool["function"]
        function = get_function(function_config["type"])

        arguments = {
            "entity_ids": ["light.living_room"],
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T23:59:59Z",
        }

        # Mock recorder
        mock_recorder_instance = MagicMock()
        mock_recorder_instance.async_add_executor_job = AsyncMock(
            return_value={"light.living_room": []}
        )

        with (
            patch(
                "custom_components.extended_openai_conversation.functions.native.recorder.get_instance",
                return_value=mock_recorder_instance,
            ),
            patch(
                "custom_components.extended_openai_conversation.functions.native.recorder.util.session_scope"
            ),
        ):
            result = await function.execute(
                hass, function_config, arguments, llm_context, exposed_entities
            )

        # Result is processed through template so it can be string or list
        assert result is not None


class TestNativeGetStatistics:
    """Test NativeFunction get_statistics."""

    @pytest.fixture
    def function(self):
        """Create NativeFunction instance."""
        return NativeFunction()

    async def test_get_statistics(self, hass, function, exposed_entities, llm_context):
        """Test getting statistics."""
        from unittest.mock import MagicMock, patch

        function_tool = prepare_function_tool_from_yaml(
            "native_get_statistics_example.yaml", index=1
        )
        function_config = function_tool["function"]
        function = get_function(function_config["type"])

        arguments = {
            "statistic_ids": ["sensor.temperature"],
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T23:59:59Z",
            "period": "day",
        }

        mock_metadata = MagicMock()
        mock_metadata.unit_of_measurement = "°C"

        mock_recorder_instance = MagicMock()
        mock_recorder_instance.async_add_executor_job = AsyncMock(
            side_effect=[
                {"sensor.temperature": [{"start": "2024-01-01", "change": 5.0}]},
                {"sensor.temperature": (None, mock_metadata)},
            ]
        )

        with patch(
            "custom_components.extended_openai_conversation.functions.native.recorder.get_instance",
            return_value=mock_recorder_instance,
        ):
            result = await function.execute(
                hass, function_config, arguments, llm_context, exposed_entities
            )

        assert isinstance(result, dict)
        assert result["sensor.temperature"][0]["unit_of_measurement"] == "°C"

    async def test_get_statistics_with_options(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test getting statistics with custom options."""
        from unittest.mock import MagicMock, patch

        function_tool = prepare_function_tool_from_yaml(
            "native_get_statistics_example.yaml", index=1
        )
        function_config = function_tool["function"]

        arguments = {
            "statistic_ids": ["sensor.temperature", "sensor.humidity"],
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T23:59:59Z",
            "period": "hour",
            "types": ["mean", "min", "max"],
        }

        mock_recorder_instance = MagicMock()
        mock_recorder_instance.async_add_executor_job = AsyncMock(
            side_effect=[
                {"sensor.temperature": [], "sensor.humidity": []},
                {},
            ]
        )

        with patch(
            "custom_components.extended_openai_conversation.functions.native.recorder.get_instance",
            return_value=mock_recorder_instance,
        ):
            result = await function.execute(
                hass, function_config, arguments, llm_context, exposed_entities
            )

        assert isinstance(result, dict)

    async def test_get_statistics_unit_of_measurement_injected(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test that unit_of_measurement from metadata is injected into each entry."""
        from unittest.mock import MagicMock, patch

        function_tool = prepare_function_tool_from_yaml(
            "native_get_statistics_example.yaml", index=1
        )
        function_config = function_tool["function"]
        function = get_function(function_config["type"])

        arguments = {
            "statistic_ids": ["sensor.energy_meter"],
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "period": "day",
        }

        mock_metadata = MagicMock()
        mock_metadata.unit_of_measurement = "Wh"

        mock_recorder_instance = MagicMock()
        mock_recorder_instance.async_add_executor_job = AsyncMock(
            side_effect=[
                {
                    "sensor.energy_meter": [
                        {"start": "2024-01-01", "change": 1500.0},
                        {"start": "2024-01-02", "change": 2000.0},
                    ]
                },
                {"sensor.energy_meter": (None, mock_metadata)},
            ]
        )

        with patch(
            "custom_components.extended_openai_conversation.functions.native.recorder.get_instance",
            return_value=mock_recorder_instance,
        ):
            result = await function.execute(
                hass, function_config, arguments, llm_context, exposed_entities
            )

        entries = result["sensor.energy_meter"]
        assert all(entry["unit_of_measurement"] == "Wh" for entry in entries)

    async def test_get_statistics_missing_metadata(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test that unit_of_measurement is None when metadata is unavailable."""
        from unittest.mock import MagicMock, patch

        function_tool = prepare_function_tool_from_yaml(
            "native_get_statistics_example.yaml", index=1
        )
        function_config = function_tool["function"]
        function = get_function(function_config["type"])

        arguments = {
            "statistic_ids": ["sensor.unknown"],
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "period": "day",
        }

        mock_recorder_instance = MagicMock()
        mock_recorder_instance.async_add_executor_job = AsyncMock(
            side_effect=[
                {"sensor.unknown": [{"start": "2024-01-01", "change": 42.0}]},
                {},  # no metadata for this id
            ]
        )

        with patch(
            "custom_components.extended_openai_conversation.functions.native.recorder.get_instance",
            return_value=mock_recorder_instance,
        ):
            result = await function.execute(
                hass, function_config, arguments, llm_context, exposed_entities
            )

        assert result["sensor.unknown"][0]["unit_of_measurement"] is None

    async def test_get_statistics_invalid_datetime(
        self, hass, function, exposed_entities, llm_context
    ):
        """Test getting statistics with invalid datetime."""
        from homeassistant.exceptions import HomeAssistantError

        function_tool = prepare_function_tool_from_yaml(
            "native_get_statistics_example.yaml", index=1
        )
        function_config = function_tool["function"]

        arguments = {
            "start_time": "invalid-date",
            "end_time": "2024-01-01T23:59:59Z",
            "statistic_ids": ["sensor.temperature"],
            "period": "day",
        }

        with pytest.raises(HomeAssistantError):
            await function.execute(
                hass, function_config, arguments, llm_context, exposed_entities
            )
