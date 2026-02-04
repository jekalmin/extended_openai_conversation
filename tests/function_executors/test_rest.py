"""Tests for RestFunctionExecutor using yaml definitions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import FunctionExecutors and test helpers
from custom_components.extended_openai_conversation.helpers import (
    RestFunctionExecutor,
    get_function_executor,
)
from tests.helpers import get_function_from_yaml


class TestRestFunctionExecutorYaml:
    """Test RestFunctionExecutor using yaml definitions."""

    @pytest.fixture
    def executor(self):
        """Create RestFunctionExecutor instance."""
        return RestFunctionExecutor()

    async def test_execute_rest_from_yaml(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test REST API call from yaml definition."""
        # Load function from yaml
        func_def = get_function_from_yaml("rest_example.yaml")

        # Process function through executor's to_arguments
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        with patch(
            "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
        ) as mock_create_rest:
            mock_rest_data = AsyncMock()
            mock_rest_data.async_update = AsyncMock()
            # Mock weather API response
            weather_response = """{
                "name": "Seoul",
                "weather": [{"description": "clear sky"}],
                "main": {
                    "temp": 15.5,
                    "feels_like": 14.2,
                    "humidity": 65
                },
                "wind": {"speed": 3.5}
            }"""
            mock_rest_data.data_without_xml = MagicMock(return_value=weather_response)
            mock_create_rest.return_value = mock_rest_data

            # Arguments based on yaml spec parameters
            arguments = {"city": "Seoul", "country_code": "KR"}

            result = await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )

            # value_template should format weather information
            assert result is not None
            assert "Seoul" in result
            mock_rest_data.async_update.assert_called_once()
