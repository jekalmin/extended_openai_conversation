"""Tests for TemplateFunctionExecutor using yaml definitions."""

from homeassistant.core import State
import pytest

# Import FunctionExecutors and test helpers
from custom_components.extended_openai_conversation.helpers import (
    TemplateFunctionExecutor,
    get_function_executor,
)
from tests.helpers import get_function_from_yaml


class TestTemplateFunctionExecutorYaml:
    """Test TemplateFunctionExecutor using yaml definitions."""

    @pytest.fixture
    def executor(self):
        """Create TemplateFunctionExecutor instance."""
        return TemplateFunctionExecutor()

    async def test_execute_template_from_yaml(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test template execution from yaml definition."""
        # Load function from yaml
        func_def = get_function_from_yaml("template_example.yaml")

        # Process function through executor's to_arguments
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        # Mock sensor states for energy monitoring
        def mock_states_get(entity_id):
            states_map = {
                "sensor.house_power": State("sensor.house_power", "1500"),
                "sensor.daily_energy": State("sensor.daily_energy", "12.5"),
            }
            return states_map.get(entity_id)

        hass.states.get = mock_states_get

        # Arguments based on yaml spec parameters
        arguments = {"show_cost": True}

        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        # Should return formatted energy summary
        assert "1500" in result or "1500.0" in result  # Power consumption
        assert "12.5" in result  # Daily energy
        assert "$" in result or "1.5" in result  # Cost (12.5 * 0.12 = 1.5)

    async def test_template_without_cost(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test template without cost calculation."""
        func_def = get_function_from_yaml("template_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        # Mock sensor states
        def mock_states_get(entity_id):
            states_map = {
                "sensor.house_power": State("sensor.house_power", "800"),
                "sensor.daily_energy": State("sensor.daily_energy", "5.0"),
            }
            return states_map.get(entity_id)

        hass.states.get = mock_states_get

        arguments = {"show_cost": False}
        result = await executor.execute(
            hass, processed_function, arguments, llm_context, exposed_entities
        )

        # Should not include cost when show_cost is False
        assert "800" in result or "800.0" in result
        assert "5.0" in result or "5" in result
