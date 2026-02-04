"""Tests for NativeFunctionExecutor using yaml definitions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Import FunctionExecutors and test helpers
from custom_components.extended_openai_conversation.helpers import (
    NativeFunctionExecutor,
    get_function_executor,
)
from tests.helpers import get_function_from_yaml


class TestNativeFunctionExecutorYaml:
    """Test NativeFunctionExecutor using yaml definitions."""

    @pytest.fixture
    def executor(self):
        """Create NativeFunctionExecutor instance."""
        return NativeFunctionExecutor()

    async def test_execute_service_from_yaml(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test service execution from yaml definition."""
        # Load function from yaml
        func_def = get_function_from_yaml("native_execute_service_example.yaml")

        # Process function through executor's to_arguments
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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

        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()
        call_args = hass.services.async_call.call_args
        assert call_args[1]["domain"] == "light"
        assert call_args[1]["service"] == "turn_on"

    async def test_execute_service_with_defaults(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test service execution with minimal parameters."""
        func_def = get_function_from_yaml("native_execute_service_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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

        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()

    async def test_execute_service_with_delay(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test service execution with delay parameter."""
        func_def = get_function_from_yaml("native_execute_service_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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

        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()

    async def test_execute_service_multiple_services(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test executing multiple services in one call."""
        func_def = get_function_from_yaml("native_execute_service_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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

        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}, {"success": True}]
        assert hass.services.async_call.call_count == 2

    async def test_execute_service_with_data_instead_of_service_data(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test service execution using 'data' instead of 'service_data'."""
        func_def = get_function_from_yaml("native_execute_service_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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

        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        assert result == [{"success": True}]
        hass.services.async_call.assert_called_once()

    async def test_execute_service_not_found_error(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test service execution fails when service does not exist."""
        from custom_components.extended_openai_conversation.helpers import (
            ServiceNotFound,
        )

        func_def = get_function_from_yaml("native_execute_service_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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
            await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )


class TestNativeGetHistory:
    """Test composite get_history function."""

    async def test_get_history(self, hass, exposed_entities, llm_context):
        """Test getting entity history with composite function."""
        from unittest.mock import MagicMock, patch

        from custom_components.extended_openai_conversation.helpers import (
            CompositeFunctionExecutor,
        )

        func_def = get_function_from_yaml("native_get_history_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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

        executor = CompositeFunctionExecutor()

        with (
            patch(
                "custom_components.extended_openai_conversation.helpers.recorder.get_instance",
                return_value=mock_recorder_instance,
            ),
            patch(
                "custom_components.extended_openai_conversation.helpers.recorder.util.session_scope"
            ),
        ):
            result = await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )

        # Result is processed through template so it can be string or list
        assert result is not None


class TestNativeGetStatistics:
    """Test NativeFunctionExecutor get_statistics."""

    @pytest.fixture
    def executor(self):
        """Create NativeFunctionExecutor instance."""
        return NativeFunctionExecutor()

    async def test_get_statistics(self, hass, executor, exposed_entities, llm_context):
        """Test getting statistics."""
        from unittest.mock import MagicMock, patch

        func_def = get_function_from_yaml("native_get_statistics_example.yaml", index=1)
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        arguments = {
            "statistic_ids": ["sensor.temperature"],
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T23:59:59Z",
            "period": "day",
        }

        mock_recorder_instance = MagicMock()
        mock_recorder_instance.async_add_executor_job = AsyncMock(
            return_value={"sensor.temperature": []}
        )

        with patch(
            "custom_components.extended_openai_conversation.helpers.recorder.get_instance",
            return_value=mock_recorder_instance,
        ):
            result = await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )

        assert isinstance(result, dict)

    async def test_get_statistics_with_options(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test getting statistics with custom options."""
        from unittest.mock import MagicMock, patch

        func_def = get_function_from_yaml("native_get_statistics_example.yaml", index=1)
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        arguments = {
            "statistic_ids": ["sensor.temperature", "sensor.humidity"],
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T23:59:59Z",
            "period": "hour",
            "types": ["mean", "min", "max"],
        }

        mock_recorder_instance = MagicMock()
        mock_recorder_instance.async_add_executor_job = AsyncMock(
            return_value={"sensor.temperature": [], "sensor.humidity": []}
        )

        with patch(
            "custom_components.extended_openai_conversation.helpers.recorder.get_instance",
            return_value=mock_recorder_instance,
        ):
            result = await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )

        assert isinstance(result, dict)

    async def test_get_statistics_invalid_datetime(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test getting statistics with invalid datetime."""
        from homeassistant.exceptions import HomeAssistantError

        func_def = get_function_from_yaml("native_get_statistics_example.yaml", index=1)
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        arguments = {
            "start_time": "invalid-date",
            "end_time": "2024-01-01T23:59:59Z",
            "statistic_ids": ["sensor.temperature"],
            "period": "day",
        }

        with pytest.raises(HomeAssistantError):
            await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )
