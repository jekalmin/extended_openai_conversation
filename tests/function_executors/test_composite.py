"""Tests for CompositeFunctionExecutor using yaml definitions."""

from homeassistant.core import State
import pytest

# Import FunctionExecutors and test helpers
from custom_components.extended_openai_conversation.helpers import (
    CompositeFunctionExecutor,
    get_function_executor,
)
from tests.helpers import get_function_from_yaml


class TestCompositeFunctionExecutorYaml:
    """Test CompositeFunctionExecutor using yaml definitions."""

    @pytest.fixture
    def executor(self):
        """Create CompositeFunctionExecutor instance."""
        return CompositeFunctionExecutor()

    async def test_composite_from_yaml(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test composite function execution from yaml definition."""
        # Load function from yaml
        func_def = get_function_from_yaml("composite_example.yaml")

        # Process function through executor's to_arguments
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        # Mock sensor states for the living_room
        def mock_states_get(entity_id):
            states_map = {
                "sensor.living_room_temperature": State("sensor.living_room_temperature", "22.5"),
                "sensor.living_room_humidity": State("sensor.living_room_humidity", "45"),
                "light.living_room": State("light.living_room", "on"),
            }
            return states_map.get(entity_id)

        hass.states.get = mock_states_get

        # Arguments based on yaml spec parameters
        arguments = {"room_name": "living_room"}

        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        # Should return formatted room status
        assert "Room: Living Room" in result
        assert "Temperature: 22.5°C" in result
        assert "Humidity: 45%" in result
        assert "Light: on" in result

    async def test_composite_with_different_rooms(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test composite function with various rooms."""
        func_def = get_function_from_yaml("composite_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        # Mock sensor states for bedroom
        def mock_states_get(entity_id):
            states_map = {
                "sensor.bedroom_temperature": State("sensor.bedroom_temperature", "20.0"),
                "sensor.bedroom_humidity": State("sensor.bedroom_humidity", "50"),
                "light.bedroom": State("light.bedroom", "off"),
            }
            return states_map.get(entity_id)

        hass.states.get = mock_states_get

        arguments = {"room_name": "bedroom"}
        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        assert "Room: Bedroom" in result
        assert "Temperature: 20.0°C" in result
        assert "Humidity: 50%" in result
        assert "Light: off" in result
