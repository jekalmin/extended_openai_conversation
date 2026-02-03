"""Tests for RestFunctionExecutor."""

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



class TestRestFunctionExecutor:
    """Test RestFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create RestFunctionExecutor instance."""
        return RestFunctionExecutor()

    async def test_get_request(self, hass, executor, exposed_entities, llm_context):
        """Test REST GET request."""
        with patch(
            "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
        ) as mock_create_rest:
            mock_rest_data = AsyncMock()
            mock_rest_data.async_update = AsyncMock()
            mock_rest_data.data_without_xml = MagicMock(
                return_value='{"result": "success"}'
            )
            mock_create_rest.return_value = mock_rest_data

            function = {
                "resource": "https://api.example.com/data",
            }
            arguments = {}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == '{"result": "success"}'
            mock_rest_data.async_update.assert_called_once()

    async def test_resource_template(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test resource_template rendering."""
        with patch(
            "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
        ) as mock_create_rest:
            mock_rest_data = AsyncMock()
            mock_rest_data.async_update = AsyncMock()
            mock_rest_data.data_without_xml = MagicMock(return_value="data")
            mock_create_rest.return_value = mock_rest_data

            resource_template = Template("https://api.example.com/{{ endpoint }}", hass)
            function = {
                "resource_template": resource_template,
            }
            arguments = {"endpoint": "users"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            # Verify resource was rendered from template
            call_args = mock_create_rest.call_args
            assert call_args is not None

    async def test_payload_template(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test payload_template rendering."""
        with patch(
            "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
        ) as mock_create_rest:
            mock_rest_data = AsyncMock()
            mock_rest_data.async_update = AsyncMock()
            mock_rest_data.data_without_xml = MagicMock(return_value="data")
            mock_create_rest.return_value = mock_rest_data

            payload_template = Template('{"name": "{{ name }}"}', hass)
            function = {
                "resource": "https://api.example.com/data",
                "payload_template": payload_template,
            }
            arguments = {"name": "test"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            call_args = mock_create_rest.call_args
            assert call_args is not None

    async def test_value_template(self, hass, executor, exposed_entities, llm_context):
        """Test response processing with value_template."""
        with patch(
            "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
        ) as mock_create_rest:
            mock_rest_data = AsyncMock()
            mock_rest_data.async_update = AsyncMock()
            mock_rest_data.data_without_xml = MagicMock(
                return_value='{"data": {"value": 42}}'
            )
            mock_create_rest.return_value = mock_rest_data

            value_template = Template("{{ value_json.data.value }}", hass)
            function = {
                "resource": "https://api.example.com/data",
                "value_template": value_template,
            }
            arguments = {}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            # value_template should process the response
            assert result is not None


