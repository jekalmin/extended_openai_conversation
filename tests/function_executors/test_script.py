"""Tests for ScriptFunctionExecutor."""

from pathlib import Path
import sys

# Add config directory to path for custom_components imports
config_dir = Path(__file__).parent.parent.parent.parent.parent
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))

from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.exceptions import ServiceNotFound
from homeassistant.helpers.template import Template
import pytest
import voluptuous as vol

# Import FunctionExecutors
from custom_components.extended_openai_conversation.exceptions import (
    CallServiceError,
    EntityNotExposed,
    EntityNotFound,
    FunctionNotFound,
    InvalidFunction,
    NativeNotFound,
)
from custom_components.extended_openai_conversation.helpers import (
    CompositeFunctionExecutor,
    NativeFunctionExecutor,
    RestFunctionExecutor,
    ScrapeFunctionExecutor,
    ScriptFunctionExecutor,
    SkillExecFunctionExecutor,
    SkillReadFunctionExecutor,
    SqliteFunctionExecutor,
    TemplateFunctionExecutor,
    get_function_executor,
)



class TestScriptFunctionExecutor:
    """Test ScriptFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create ScriptFunctionExecutor instance."""
        return ScriptFunctionExecutor()

    async def test_execute_script(self, hass, executor, exposed_entities, llm_context):
        """Test script execution."""
        with patch(
            "custom_components.extended_openai_conversation.helpers.Script"
        ) as mock_script_class:
            # Setup mock
            mock_script = AsyncMock()
            mock_result = MagicMock()
            mock_result.variables = {"_function_result": "Script completed"}
            mock_script.async_run = AsyncMock(return_value=mock_result)
            mock_script_class.return_value = mock_script

            function = {
                "sequence": [
                    {"service": "light.turn_on", "target": {"entity_id": "light.test"}}
                ]
            }
            arguments = {"brightness": 255}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == "Script completed"
            mock_script.async_run.assert_called_once()

    async def test_function_result_variable(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test _function_result return value."""
        with patch(
            "custom_components.extended_openai_conversation.helpers.Script"
        ) as mock_script_class:
            mock_script = AsyncMock()
            mock_result = MagicMock()
            mock_result.variables = {"_function_result": {"status": "ok", "value": 42}}
            mock_script.async_run = AsyncMock(return_value=mock_result)
            mock_script_class.return_value = mock_script

            function = {"sequence": []}
            arguments = {}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == {"status": "ok", "value": 42}

    async def test_default_success_return(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test default 'Success' return when no _function_result."""
        with patch(
            "custom_components.extended_openai_conversation.helpers.Script"
        ) as mock_script_class:
            mock_script = AsyncMock()
            mock_result = MagicMock()
            mock_result.variables = {}  # No _function_result
            mock_script.async_run = AsyncMock(return_value=mock_result)
            mock_script_class.return_value = mock_script

            function = {"sequence": []}
            arguments = {}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == "Success"


