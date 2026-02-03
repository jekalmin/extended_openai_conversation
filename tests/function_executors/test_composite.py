"""Tests for CompositeFunctionExecutor."""

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



class TestCompositeFunctionExecutor:
    """Test CompositeFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create CompositeFunctionExecutor instance."""
        return CompositeFunctionExecutor()

    async def test_sequence_execution(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test executing sequence of functions."""
        template1 = Template("Step 1: {{ value }}", hass)
        template2 = Template("Step 2: {{ value }}", hass)

        function = {
            "sequence": [
                {"type": "template", "value_template": template1},
                {"type": "template", "value_template": template2},
            ]
        }
        arguments = {"value": "test"}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        # Should return result of last function
        assert result == "Step 2: test"

    async def test_response_variable(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test passing results via response_variable."""
        template1 = Template("first_result", hass)
        template2 = Template("Combined: {{ previous }}", hass)

        function = {
            "sequence": [
                {
                    "type": "template",
                    "value_template": template1,
                    "response_variable": "previous",
                },
                {"type": "template", "value_template": template2},
            ]
        }
        arguments = {}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == "Combined: first_result"

    async def test_mixed_executors(
        self, hass, executor, exposed_entities, llm_context, temp_skills_dir
    ):
        """Test combining different executor types."""
        from custom_components.extended_openai_conversation.skills import Skill

        template = Template("Template result", hass)

        mock_manager = MagicMock()
        mock_manager.get_skill = MagicMock(
            return_value=Skill(
                name="test",
                description="Test skill",
                directory=temp_skills_dir / "test_skill",
            )
        )

        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_manager,
        ):
            function = {
                "sequence": [
                    {"type": "template", "value_template": template},
                    {
                        "type": "skill_read",
                        "skills_dir": str(temp_skills_dir),
                    },
                ]
            }
            arguments = {"skill_name": "test"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            # Should return result of skill_read (last in sequence)
            assert result == "Skill body content"

    async def test_arguments_preserved(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test that original arguments are preserved through sequence."""
        template1 = Template("{{ original }}", hass)
        template2 = Template("{{ original }}-{{ added }}", hass)

        function = {
            "sequence": [
                {
                    "type": "template",
                    "value_template": template1,
                    "response_variable": "added",
                },
                {"type": "template", "value_template": template2},
            ]
        }
        arguments = {"original": "value"}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == "value-value"

    def test_function_schema_validation(self, hass, executor):
        """Test composite function schema validation errors."""
        # Test invalid schema - sequence not a list
        with pytest.raises(InvalidFunction):
            executor.to_arguments({"type": "composite", "sequence": "not a list"})

        # Test invalid schema - missing sequence
        with pytest.raises(InvalidFunction):
            executor.to_arguments({"type": "composite"})

        # Verify schema has required sequence field
        schema = executor.data_schema.schema
        has_sequence = vol.Required("sequence") in schema or any(
            isinstance(k, vol.Required) and k.schema == "sequence" for k in schema
        )
        assert has_sequence

    def test_function_schema_nested_validation(self, executor):
        """Test nested function type validation."""
        # Invalid nested type
        with pytest.raises((InvalidFunction, vol.error.Error, FunctionNotFound)):
            executor.to_arguments(
                {
                    "type": "composite",
                    "sequence": [{"type": "nonexistent"}],
                }
            )

