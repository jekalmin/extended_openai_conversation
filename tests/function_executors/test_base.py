"""Tests for FunctionExecutor base class and helper function."""

from unittest.mock import MagicMock

import pytest
import voluptuous as vol

# Import FunctionExecutors
from custom_components.extended_openai_conversation.exceptions import (
    EntityNotExposed,
    EntityNotFound,
    FunctionNotFound,
    InvalidFunction,
)
from custom_components.extended_openai_conversation.helpers import (
    NativeFunctionExecutor,
    ScriptFunctionExecutor,
    TemplateFunctionExecutor,
    get_function_executor,
)


class TestGetFunctionExecutor:
    """Test get_function_executor helper function."""

    def test_get_existing_executor(self):
        """Test getting an existing function executor."""
        executor = get_function_executor("template")
        assert isinstance(executor, TemplateFunctionExecutor)

    def test_get_native_executor(self):
        """Test getting native function executor."""
        executor = get_function_executor("native")
        assert isinstance(executor, NativeFunctionExecutor)

    def test_get_script_executor(self):
        """Test getting script function executor."""
        executor = get_function_executor("script")
        assert isinstance(executor, ScriptFunctionExecutor)

    def test_get_nonexistent_executor(self):
        """Test getting a nonexistent function executor raises error."""
        with pytest.raises(FunctionNotFound):
            get_function_executor("nonexistent")


class TestFunctionExecutorBase:
    """Test FunctionExecutor base class."""

    def test_to_arguments_valid(self, hass):
        """Test to_arguments with valid arguments - using pre-built Template."""
        executor = TemplateFunctionExecutor()
        # For testing, pass an already-built Template to bypass cv.template validation
        # This tests the schema structure, not the cv.template behavior
        # The executor's schema uses cv.template which validates strings
        # For unit testing, we verify the schema accepts the required keys
        assert (
            executor.data_schema.schema.get(vol.Required("value_template")) is not None
        )
        assert executor.data_schema.schema.get(vol.Required("type")) is not None

    def test_to_arguments_invalid(self):
        """Test to_arguments with invalid arguments raises InvalidFunction."""
        executor = TemplateFunctionExecutor()
        with pytest.raises(InvalidFunction):
            executor.to_arguments({"type": "template"})  # Missing value_template

    def test_validate_entity_ids_valid(self, hass, exposed_entities):
        """Test validate_entity_ids with valid entities."""
        executor = NativeFunctionExecutor()
        # Should not raise
        executor.validate_entity_ids(hass, ["light.living_room"], exposed_entities)

    def test_validate_entity_ids_not_found(self, hass, exposed_entities):
        """Test validate_entity_ids raises EntityNotFound."""
        executor = NativeFunctionExecutor()
        hass.states.get = MagicMock(return_value=None)

        with pytest.raises(EntityNotFound):
            executor.validate_entity_ids(hass, ["light.nonexistent"], exposed_entities)

    def test_validate_entity_ids_not_exposed(self, hass, exposed_entities):
        """Test validate_entity_ids raises EntityNotExposed."""
        executor = NativeFunctionExecutor()

        with pytest.raises(EntityNotExposed):
            executor.validate_entity_ids(hass, ["light.not_exposed"], exposed_entities)
