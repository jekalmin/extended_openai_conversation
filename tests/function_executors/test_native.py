"""Tests for NativeFunctionExecutor."""

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



class TestNativeFunctionExecutor:
    """Test NativeFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create NativeFunctionExecutor instance."""
        return NativeFunctionExecutor()

    async def test_execute_service_single(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test executing a single service."""
        function = {"name": "execute_service_single"}
        arguments = {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"entity_id": ["light.living_room"]},
        }

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == {"success": True}
        hass.services.async_call.assert_called_once_with(
            domain="light",
            service="turn_on",
            service_data={"entity_id": ["light.living_room"]},
        )

    async def test_execute_service_single_with_string_entity_id(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test executing a single service with string entity_id."""
        function = {"name": "execute_service_single"}
        arguments = {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room",
        }

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == {"success": True}

    async def test_execute_service_batch(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test executing multiple services."""
        function = {"name": "execute_service"}
        arguments = {
            "list": [
                {
                    "domain": "light",
                    "service": "turn_on",
                    "service_data": {"entity_id": ["light.living_room"]},
                },
                {
                    "domain": "switch",
                    "service": "turn_on",
                    "service_data": {"entity_id": ["switch.kitchen"]},
                },
            ]
        }

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"success": True}
        assert result[1] == {"success": True}
        assert hass.services.async_call.call_count == 2

    async def test_entity_not_found(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test error when entity doesn't exist."""
        hass.states.get = MagicMock(return_value=None)

        function = {"name": "execute_service_single"}
        arguments = {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"entity_id": ["light.nonexistent"]},
        }

        with pytest.raises(EntityNotFound):
            await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

    async def test_entity_not_exposed(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test error when entity not exposed."""
        function = {"name": "execute_service_single"}
        arguments = {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"entity_id": ["light.bedroom"]},  # Not in exposed_entities
        }

        with pytest.raises(EntityNotExposed):
            await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

    async def test_service_not_found(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test error when service doesn't exist."""
        hass.services.has_service = MagicMock(return_value=False)

        function = {"name": "execute_service_single"}
        arguments = {
            "domain": "nonexistent",
            "service": "service",
            "service_data": {"entity_id": ["light.living_room"]},
        }

        with pytest.raises(ServiceNotFound):
            await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

    async def test_native_not_found(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test error for unknown native function."""
        function = {"name": "unknown_function"}
        arguments = {}

        with pytest.raises(NativeNotFound):
            await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

    async def test_missing_entity_id_and_area_id(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test error when entity_id, area_id, and device_id are all missing."""
        function = {"name": "execute_service_single"}
        arguments = {
            "domain": "light",
            "service": "turn_on",
            "service_data": {},
        }

        with pytest.raises(CallServiceError):
            await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

    async def test_service_with_area_id(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test executing service with area_id."""
        function = {"name": "execute_service_single"}
        arguments = {
            "domain": "light",
            "service": "turn_on",
            "service_data": {"area_id": "living_room"},
        }

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == {"success": True}

    async def test_get_user_from_user_id(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test get_user_from_user_id native function."""
        # Create a proper mock user object with name attribute
        mock_user = MagicMock()
        mock_user.name = "Test User"
        hass.auth.async_get_user = AsyncMock(return_value=mock_user)

        function = {"name": "get_user_from_user_id"}
        arguments = {}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert result == {"name": "Test User"}


