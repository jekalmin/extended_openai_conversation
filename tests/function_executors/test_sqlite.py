"""Tests for SqliteFunctionExecutor."""

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



class TestSqliteFunctionExecutor:
    """Test SqliteFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create SqliteFunctionExecutor instance."""
        return SqliteFunctionExecutor()

    async def test_query_execution(
        self, hass, executor, temp_db_path, exposed_entities, llm_context
    ):
        """Test basic SQL query."""
        function = {
            "db_url": f"file:{temp_db_path}",
            "query": "SELECT * FROM states",
        }

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["entity_id"] == "light.living_room"

    async def test_single_row(
        self, hass, executor, temp_db_path, exposed_entities, llm_context
    ):
        """Test single=True returns dict."""
        function = {
            "db_url": f"file:{temp_db_path}",
            "query": "SELECT * FROM states WHERE entity_id = 'light.living_room'",
            "single": True,
        }

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        assert isinstance(result, dict)
        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"

    async def test_multiple_rows(
        self, hass, executor, temp_db_path, exposed_entities, llm_context
    ):
        """Test multiple rows return list of dicts."""
        function = {
            "db_url": f"file:{temp_db_path}",
            "query": "SELECT entity_id, state FROM states ORDER BY entity_id",
        }

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        assert isinstance(result, list)
        assert len(result) == 3
        # Check each row is a dict with expected keys
        for row in result:
            assert "entity_id" in row
            assert "state" in row

    async def test_template_query(
        self, hass, executor, temp_db_path, exposed_entities, llm_context
    ):
        """Test query with template variables."""
        function = {
            "db_url": f"file:{temp_db_path}",
            "query": "SELECT * FROM states WHERE entity_id = '{{ entity_id }}'",
        }
        arguments = {"entity_id": "light.living_room"}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["entity_id"] == "light.living_room"

    async def test_read_only_mode(
        self, hass, executor, temp_db_path, exposed_entities, llm_context
    ):
        """Test that connection is read-only."""
        import sqlite3

        function = {
            "db_url": f"file:{temp_db_path}",
            "query": "INSERT INTO states VALUES ('test', 'test', 'test')",
        }

        with pytest.raises(sqlite3.OperationalError):
            await executor.execute(hass, function, {}, llm_context, exposed_entities)

    async def test_is_exposed_helper(
        self, hass, executor, temp_db_path, exposed_entities, llm_context
    ):
        """Test is_exposed template helper."""
        function = {
            "db_url": f"file:{temp_db_path}",
            "query": "SELECT '{{ is_exposed(\"light.living_room\") }}' as result",
            "single": True,
        }

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        assert result["result"] == "True"

    async def test_is_exposed_helper_false(
        self, hass, executor, temp_db_path, exposed_entities, llm_context
    ):
        """Test is_exposed template helper returns False."""
        function = {
            "db_url": f"file:{temp_db_path}",
            "query": "SELECT '{{ is_exposed(\"light.nonexistent\") }}' as result",
            "single": True,
        }

        result = await executor.execute(
            hass, function, {}, llm_context, exposed_entities
        )

        assert result["result"] == "False"

    def test_set_url_read_only(self, executor):
        """Test set_url_read_only adds mode=ro."""
        url = "file:/path/to/db"
        result = executor.set_url_read_only(url)
        assert "mode=ro" in result

    def test_set_url_read_only_with_existing_params(self, executor):
        """Test set_url_read_only with existing query params."""
        url = "file:/path/to/db?param=value"
        result = executor.set_url_read_only(url)
        assert "mode=ro" in result
        assert "param=value" in result


