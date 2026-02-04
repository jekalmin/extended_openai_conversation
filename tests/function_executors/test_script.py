"""Tests for ScriptFunctionExecutor using yaml definitions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import FunctionExecutors and test helpers
from custom_components.extended_openai_conversation.helpers import (
    ScriptFunctionExecutor,
    get_function_executor,
)
from tests.helpers import get_function_from_yaml


class TestScriptFunctionExecutorYaml:
    """Test ScriptFunctionExecutor using yaml definitions."""

    @pytest.fixture
    def executor(self):
        """Create ScriptFunctionExecutor instance."""
        return ScriptFunctionExecutor()

    async def test_execute_script_from_yaml(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test script execution from yaml definition."""
        # Load function from yaml
        func_def = get_function_from_yaml("script_example.yaml")

        # Process function through executor's to_arguments (simulates conversation.py behavior)
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        with patch(
            "custom_components.extended_openai_conversation.helpers.Script"
        ) as mock_script_class:
            # Setup mock
            mock_script = AsyncMock()
            mock_result = MagicMock()
            mock_result.variables = {"_function_result": "Movie mode activated"}
            mock_script.async_run = AsyncMock(return_value=mock_result)
            mock_script_class.return_value = mock_script

            # Arguments based on yaml spec parameters (brightness_pct is optional with default 10)
            arguments = {"brightness_pct": 15}

            result = await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )

            assert result == "Movie mode activated"
            mock_script.async_run.assert_called_once()

    async def test_execute_script_with_defaults(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test script execution with default brightness."""
        func_def = get_function_from_yaml("script_example.yaml")
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

        with patch(
            "custom_components.extended_openai_conversation.helpers.Script"
        ) as mock_script_class:
            mock_script = AsyncMock()
            mock_result = MagicMock()
            mock_result.variables = {"_function_result": "Movie mode activated"}
            mock_script.async_run = AsyncMock(return_value=mock_result)
            mock_script_class.return_value = mock_script

            # No arguments, should use default brightness_pct
            arguments = {}

            result = await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )

            assert result == "Movie mode activated"
            mock_script.async_run.assert_called_once()
