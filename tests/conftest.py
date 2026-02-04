"""Fixtures for extended_openai_conversation tests."""
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

# Add config directory to path for custom_components imports
config_dir = Path(__file__).parent.parent
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))

from homeassistant.helpers import config_validation as cv  # noqa: E402
from homeassistant.helpers.template import TemplateEnvironment  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def hass(tmp_path: Path) -> MagicMock:
    """Mock Home Assistant instance."""
    hass = MagicMock()
    hass.config.config_dir = str(tmp_path)
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
