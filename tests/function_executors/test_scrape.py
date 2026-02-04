"""Tests for ScrapeFunctionExecutor using yaml definitions."""

from unittest.mock import AsyncMock, patch

import pytest

# Import FunctionExecutors and test helpers
from custom_components.extended_openai_conversation.helpers import (
    ScrapeFunctionExecutor,
    get_function_executor,
)
from tests.helpers import get_function_from_yaml


class TestScrapeFunctionExecutorYaml:
    """Test ScrapeFunctionExecutor using yaml definitions."""

    @pytest.fixture
    def executor(self):
        """Create ScrapeFunctionExecutor instance."""
        return ScrapeFunctionExecutor()

    async def test_execute_scrape_from_yaml(
        self, hass, executor, exposed_entities, llm_context
    ):
        """Test web scraping from yaml definition."""
        # Load function from yaml
        func_def = get_function_from_yaml("scrape_example.yaml")

        # Process function through executor's to_arguments
        function_executor = get_function_executor(func_def["function"]["type"])
        processed_function = function_executor.to_arguments(func_def["function"])

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
            # Mock Hacker News HTML structure
            html = """
            <html>
                <tr class="athing">
                    <td class="title">
                        <span class="titleline">
                            <a href="https://example.com/article">Test Article Title</a>
                        </span>
                    </td>
                </tr>
                <tr>
                    <td colspan="2">
                        <span class="score">123 points</span>
                    </td>
                </tr>
            </html>
            """
            mock_coordinator.data = BeautifulSoup(html, "html.parser")
            mock_coordinator.async_config_entry_first_refresh = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            # Arguments based on yaml spec parameters (category is optional)
            arguments = {}

            result = await executor.execute(
                hass, processed_function, arguments, llm_context, exposed_entities
            )

            # Should return scraped data
            assert result is not None
