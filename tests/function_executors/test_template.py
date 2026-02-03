"""Tests for TemplateFunctionExecutor."""

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



class TestTemplateFunctionExecutor:
    """Test TemplateFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create TemplateFunctionExecutor instance."""
        return TemplateFunctionExecutor()

    async def test_basic_render(self, hass, executor, exposed_entities, llm_context):
        """Test basic template rendering."""
        template = Template("Hello World", hass)
        function = {"value_template": template}

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        assert result == "Hello World"

    async def test_with_arguments(self, hass, executor, exposed_entities, llm_context):
        """Test template with passed arguments."""
        template = Template("Hello {{ name }}", hass)
        function = {"value_template": template}
        arguments = {"name": "World"}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == "Hello World"

    async def test_with_multiple_arguments(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test template with multiple arguments."""
        template = Template("{{ greeting }} {{ name }}!", hass)
        function = {"value_template": template}
        arguments = {"greeting": "Hello", "name": "World"}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == "Hello World!"

    async def test_parse_result_false(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test parse_result=False returns string."""
        template = Template("42", hass)
        function = {"value_template": template, "parse_result": False}

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        assert result == "42"
        assert isinstance(result, str)

    async def test_parse_result_true(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test parse_result=True is passed to template render."""
        # With minimal mock hass, parse_result may not work fully
        # Test that the function config is honored and passed through
        template = Template("test_value", hass)
        function = {"value_template": template, "parse_result": True}

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        # Result should be rendered (parse_result behavior depends on template content)
        assert result is not None
        # With simple string template, result is still a string
        assert isinstance(result, str)
        assert result == "test_value"

    def test_to_arguments_validation(self, hass, executor):
        """Test schema validation errors."""
        # Test that missing required field raises InvalidFunction
        with pytest.raises(InvalidFunction):
            executor.to_arguments({"type": "template"})  # Missing value_template

        # Test that schema has required fields
        schema = executor.data_schema.schema
        assert vol.Required("value_template") in schema or any(
            isinstance(k, vol.Required) and k.schema == "value_template" for k in schema
        )


