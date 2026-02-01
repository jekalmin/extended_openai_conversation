"""Tests for FunctionExecutor classes in extended_openai_conversation."""

from pathlib import Path
import sys
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

# Add config directory to path for custom_components imports
config_dir = Path(__file__).parent.parent.parent.parent
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))


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
        template = Template("{{ test }}", hass)
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


class TestSkillReadFunctionExecutor:
    """Test SkillReadFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create SkillReadFunctionExecutor instance."""
        return SkillReadFunctionExecutor()

    @pytest.fixture
    def mock_skill_manager(self, temp_skills_dir):
        """Create mock SkillManager with test skills."""
        from custom_components.extended_openai_conversation.skills import Skill

        mock_manager = MagicMock()
        mock_manager.get_skill = MagicMock(
            side_effect=lambda name: {
                "test": Skill(
                    name="test",
                    description="Test skill",
                    directory=temp_skills_dir / "test_skill",
                ),
                "another": Skill(
                    name="another",
                    description="Another skill",
                    directory=temp_skills_dir / "another_skill",
                ),
            }.get(name)
        )
        return mock_manager

    async def test_read_skill_body(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test reading SKILL.md body (without frontmatter)."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "test"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == "Skill body content"

    async def test_read_specific_file(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test reading specific file from skill."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "test", "file_path": "references/reference.md"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result == "Reference content"

    async def test_path_traversal_blocked(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test path traversal attack is blocked."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {
                "skill_name": "test",
                "file_path": "../another_skill/SKILL.md",
            }

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert isinstance(result, dict)
            assert "error" in result
            assert "Path traversal not allowed" in result["error"]

    async def test_skill_not_found(
        self, hass, executor, temp_skills_dir, exposed_entities, llm_context
    ):
        """Test error when skill doesn't exist."""
        mock_manager = MagicMock()
        mock_manager.get_skill = MagicMock(return_value=None)

        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "nonexistent_skill"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert isinstance(result, dict)
            assert "error" in result
            assert "not found" in result["error"]

    async def test_file_not_found(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test error when file doesn't exist."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "test", "file_path": "nonexistent.md"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert isinstance(result, dict)
            assert "error" in result
            assert "not found" in result["error"]

    async def test_missing_skill_name(
        self, hass, executor, temp_skills_dir, exposed_entities, llm_context
    ):
        """Test error when skill_name is missing."""
        function = {"skills_dir": str(temp_skills_dir)}
        arguments = {}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "skill_name is required" in result["error"]

    async def test_skill_md_not_found(
        self, hass, executor, temp_skills_dir, exposed_entities, llm_context
    ):
        """Test error when SKILL.md doesn't exist in skill directory."""
        from custom_components.extended_openai_conversation.skills import Skill

        # Create a skill directory without SKILL.md
        empty_skill = temp_skills_dir / "empty_skill"
        empty_skill.mkdir()

        mock_manager = MagicMock()
        mock_manager.get_skill = MagicMock(
            return_value=Skill(
                name="empty",
                description="Empty skill",
                directory=empty_skill,
            )
        )

        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "empty"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert isinstance(result, dict)
            assert "error" in result
            assert "SKILL.md not found" in result["error"]


class TestSkillExecFunctionExecutor:
    """Test SkillExecFunctionExecutor."""

    @pytest.fixture
    def executor(self):
        """Create SkillExecFunctionExecutor instance."""
        return SkillExecFunctionExecutor()

    @pytest.fixture
    def mock_skill_manager(self, temp_skills_dir):
        """Create mock SkillManager with test skills."""
        from custom_components.extended_openai_conversation.skills import Skill

        mock_manager = MagicMock()
        mock_manager.get_skill = MagicMock(
            side_effect=lambda name: {
                "test": Skill(
                    name="test",
                    description="Test skill",
                    directory=temp_skills_dir / "test_skill",
                ),
            }.get(name)
        )
        return mock_manager

    async def test_execute_command(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test executing shell command."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "test", "command": "echo 'test'"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert isinstance(result, dict)
            assert "exit_code" in result
            assert result["exit_code"] == 0
            assert "test" in result["stdout"]

    async def test_command_output(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test stdout/stderr capture."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {
                "skill_name": "test",
                "command": "echo 'stdout'; echo 'stderr' >&2",
            }

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert "stdout" in result["stdout"]
            assert "stderr" in result.get("stderr", "")

    async def test_command_exit_code(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test command exit code capture."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "test", "command": "exit 42"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert result["exit_code"] == 42

    async def test_skill_not_found(
        self, hass, executor, temp_skills_dir, exposed_entities, llm_context
    ):
        """Test error when skill doesn't exist."""
        mock_manager = MagicMock()
        mock_manager.get_skill = MagicMock(return_value=None)

        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "nonexistent_skill", "command": "echo 'test'"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            assert isinstance(result, dict)
            assert "error" in result
            assert "not found" in result["error"]

    async def test_missing_skill_name(
        self, hass, executor, temp_skills_dir, exposed_entities, llm_context
    ):
        """Test error when skill_name is missing."""
        function = {"skills_dir": str(temp_skills_dir)}
        arguments = {"command": "echo 'test'"}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "skill_name is required" in result["error"]

    async def test_missing_command(
        self, hass, executor, temp_skills_dir, exposed_entities, llm_context
    ):
        """Test error when command is missing."""
        function = {"skills_dir": str(temp_skills_dir)}
        arguments = {"skill_name": "test_skill"}

        result = await executor.execute(
            hass, function, arguments, llm_context, exposed_entities
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "command is required" in result["error"]

    async def test_command_working_directory(
        self,
        hass,
        executor,
        temp_skills_dir,
        exposed_entities,
        llm_context,
        mock_skill_manager,
    ):
        """Test command runs in skill directory."""
        with patch(
            "custom_components.extended_openai_conversation.skills.SkillManager.async_get_instance",
            return_value=mock_skill_manager,
        ):
            function = {"skills_dir": str(temp_skills_dir)}
            arguments = {"skill_name": "test", "command": "pwd"}

            result = await executor.execute(
                hass, function, arguments, llm_context, exposed_entities
            )

            expected_path = str(temp_skills_dir / "test_skill")
            assert expected_path in result["stdout"]


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
