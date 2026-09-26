"""Fixtures for extended_openai_conversation_responses tests."""

from functools import wraps
from inspect import signature
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

# Add config directory to path for custom_components imports
config_dir = Path(__file__).parent.parent
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))

import pytest  # noqa: E402

from homeassistant.components import conversation  # noqa: E402
from homeassistant.helpers import config_validation as cv, llm  # noqa: E402
from homeassistant.helpers.template import TemplateEnvironment  # noqa: E402


@pytest.fixture(autouse=True)
def mock_ha_action_owners(monkeypatch):
    """Unit HA doubles do not include registries or service ownership records."""
    from custom_components.extended_openai_conversation_responses import ha_actions

    monkeypatch.setattr(ha_actions, "_target_identity", lambda *_: ())
    monkeypatch.setattr(ha_actions, "_service_identity", lambda *_: None)

# Home Assistant dev changed ToolResultContent(tool_result=...) to
# ToolResultContent(result=llm.ToolResult(...)). Keep legacy unit-test fixtures
# valid on both APIs without replacing the class or changing isinstance checks.
if "result" in signature(conversation.ToolResultContent).parameters:
    _tool_result_content_init = conversation.ToolResultContent.__init__

    @wraps(_tool_result_content_init)
    def _compat_tool_result_content_init(self, *args, **kwargs):
        if "tool_result" in kwargs and "result" not in kwargs:
            kwargs["result"] = llm.ToolResult(data=kwargs.pop("tool_result"))
        _tool_result_content_init(self, *args, **kwargs)

    conversation.ToolResultContent.__init__ = _compat_tool_result_content_init


@pytest.fixture
def hass(tmp_path: Path) -> MagicMock:
    """Mock Home Assistant instance."""
    hass = MagicMock()
    # Create extended_openai_conversation_responses directory in tmp_path
    workdir = tmp_path / "extended_openai_conversation_responses"
    workdir.mkdir(parents=True, exist_ok=True)
    hass.config.config_dir = str(tmp_path)
    hass.config.path.side_effect = lambda *parts: str(tmp_path.joinpath(*parts))
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=MagicMock(state="on"))
    hass.states.async_all = MagicMock(return_value=[])
    hass.services = MagicMock()
    hass.services.has_service = MagicMock(return_value=True)
    hass.services.async_call = AsyncMock()
    hass.auth = MagicMock()
    hass.auth.async_get_user = AsyncMock(return_value=MagicMock(name="Test User"))
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()

    # Create a minimal template environment for rendering
    template_env = TemplateEnvironment(hass, limited=False, strict=False)
    hass.data = {
        "template.environment": template_env,
        "template.environment_limited": template_env,
        "template.environment_strict": template_env,
    }

    # For async_add_executor_job - run the function directly
    async def run_executor_job(func, *args):
        return func(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=run_executor_job)

    # Set hass in thread-local storage for cv.template validation
    cv._hass.hass = hass

    return hass


@pytest.fixture
def llm_context() -> MagicMock:
    """Mock LLM context."""
    context = MagicMock()
    context.context = MagicMock()
    context.context.user_id = "test_user_id"
    return context


@pytest.fixture
def exposed_entities() -> list:
    """Sample exposed entities."""
    return [
        {
            "entity_id": "light.living_room",
            "name": "Living Room",
            "state": "on",
            "aliases": [],
        },
        {
            "entity_id": "switch.kitchen",
            "name": "Kitchen",
            "state": "off",
            "aliases": [],
        },
        {
            "entity_id": "sensor.temperature",
            "name": "Temperature",
            "state": "22.5",
            "aliases": ["temp"],
        },
        {
            "entity_id": "sensor.humidity",
            "name": "Humidity",
            "state": "45",
            "aliases": [],
        },
    ]


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary SQLite database for testing."""
    import sqlite3

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create test table
    cursor.execute("""
        CREATE TABLE states (
            entity_id TEXT,
            state TEXT,
            last_updated TEXT
        )
    """)

    # Insert test data
    cursor.executemany(
        "INSERT INTO states VALUES (?, ?, ?)",
        [
            ("light.living_room", "on", "2024-01-01 12:00:00"),
            ("switch.kitchen", "off", "2024-01-01 12:00:00"),
            ("sensor.temperature", "22.5", "2024-01-01 12:00:00"),
        ],
    )
    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def management_agent(hass, monkeypatch):
    """Select one mutable agent for Management command contract tests."""
    from types import SimpleNamespace

    from custom_components.extended_openai_conversation_responses import management_ui
    from custom_components.extended_openai_conversation_responses.agent_config import (
        agent_config_defaults,
    )
    from custom_components.extended_openai_conversation_responses.const import DOMAIN

    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Jarvis",
        data=agent_config_defaults(),
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain=DOMAIN,
        title="Provider",
        data={},
        subentries={subentry.subentry_id: subentry},
    )
    monkeypatch.setattr(management_ui, "entry_and_agent", lambda *_: (entry, subentry))

    def update(selected_entry, selected_subentry, *, data, title=None):
        assert selected_entry is entry and selected_subentry is subentry
        subentry.data = data
        if title is not None:
            subentry.title = title

    hass.config_entries.async_update_subentry.side_effect = update
    hass.config_entries.async_get_entry.return_value = entry
    return entry, subentry


@pytest.fixture
def management_message(management_agent):
    """Build a request selecting the fixture's existing conversation agent."""
    entry, subentry = management_agent

    def message(section, action, **fields):
        return {
            "entry_id": entry.entry_id,
            "subentry_id": subentry.subentry_id,
            "section": section,
            "action": action,
            **fields,
        }

    return message


@pytest.fixture
def entry_agent(hass, monkeypatch):
    """Exercise the real entry owner with only its inner continuity work stubbed."""
    from types import SimpleNamespace
    from custom_components.extended_openai_conversation_responses import conversation

    agent = object.__new__(conversation.ExtendedOpenAIAgentEntity)
    agent.hass = hass
    agent.entry = SimpleNamespace(entry_id="entry", data={})
    agent.subentry = SimpleNamespace(subentry_id="agent", data={})
    agent._async_process_with_continuity = AsyncMock(return_value="processed")
    monkeypatch.setattr(conversation, "async_reconcile_runtime_configuration", AsyncMock())
    return agent


@pytest.fixture
def entry_input():
    """A caller-owned request with both satellite and device-registry metadata."""
    from types import SimpleNamespace
    from homeassistant.core import Context

    request = SimpleNamespace(
        text="hello", language="en", context=Context(), conversation_id=None,
        device_id="device-registry-id", satellite_id="assist_satellite.kitchen",
    )
    request.as_llm_context = lambda _domain: SimpleNamespace(context=request.context)
    return request
