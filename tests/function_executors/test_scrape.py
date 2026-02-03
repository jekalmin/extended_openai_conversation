"""Tests for ScrapeFunctionExecutor."""

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



class TestScrapeFunctionExecutor:
    """Test ScrapeFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create ScrapeFunctionExecutor instance."""
        return ScrapeFunctionExecutor()

    async def test_scrape_basic(self, hass, executor, exposed_entities, llm_context):
        """Test basic web scraping."""
        with (
            patch(
                "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
            ) as mock_rest,
            patch(
                "custom_components.extended_openai_conversation.helpers.scrape.coordinator.ScrapeCoordinator"
            ) as mock_coordinator_class,
        ):
            from bs4 import BeautifulSoup

            mock_rest_data = AsyncMock()
            mock_rest.return_value = mock_rest_data

            mock_coordinator = AsyncMock()
            mock_coordinator.data = BeautifulSoup(
                '<html><div class="content">Test Content</div></html>',
                "html.parser",
            )
            mock_coordinator.async_config_entry_first_refresh = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            function = {
                "resource": "https://example.com",
                "sensor": [{"select": "div.content"}],
            }
            arguments = {}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == "Test Content"

    async def test_scrape_with_attribute(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test extracting element attribute."""
        with (
            patch(
                "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
            ) as mock_rest,
            patch(
                "custom_components.extended_openai_conversation.helpers.scrape.coordinator.ScrapeCoordinator"
            ) as mock_coordinator_class,
        ):
            from bs4 import BeautifulSoup

            mock_rest_data = AsyncMock()
            mock_rest.return_value = mock_rest_data

            mock_coordinator = AsyncMock()
            mock_coordinator.data = BeautifulSoup(
                '<html><a href="https://example.com" class="link">Link</a></html>',
                "html.parser",
            )
            mock_coordinator.async_config_entry_first_refresh = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            function = {
                "resource": "https://example.com",
                "sensor": [{"select": "a.link", "attribute": "href"}],
            }
            arguments = {}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == "https://example.com"

    async def test_scrape_with_index(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test scraping with index selection."""
        with (
            patch(
                "custom_components.extended_openai_conversation.helpers.rest.create_rest_data_from_config"
            ) as mock_rest,
            patch(
                "custom_components.extended_openai_conversation.helpers.scrape.coordinator.ScrapeCoordinator"
            ) as mock_coordinator_class,
        ):
            from bs4 import BeautifulSoup

            mock_rest_data = AsyncMock()
            mock_rest.return_value = mock_rest_data

            mock_coordinator = AsyncMock()
            mock_coordinator.data = BeautifulSoup(
                "<html><li>First</li><li>Second</li><li>Third</li></html>",
                "html.parser",
            )
            mock_coordinator.async_config_entry_first_refresh = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            function = {
                "resource": "https://example.com",
                "sensor": [{"select": "li", "index": 1}],
            }
            arguments = {}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == "Second"

    def test_extract_value_index_not_found(self, executor):
        """Test _extract_value when index is out of range."""
        from bs4 import BeautifulSoup

        data = BeautifulSoup("<html><div>Only one</div></html>", "html.parser")
        sensor_config = {"select": "div", "index": 5}

        result = executor._extract_value(data, sensor_config)

        assert result is None

    def test_extract_value_attribute_not_found(self, executor):
        """Test _extract_value when attribute doesn't exist."""
        from bs4 import BeautifulSoup

        data = BeautifulSoup(
            '<html><div class="test">Content</div></html>', "html.parser"
        )
        sensor_config = {"select": "div", "attribute": "nonexistent"}

        result = executor._extract_value(data, sensor_config)

        assert result is None


