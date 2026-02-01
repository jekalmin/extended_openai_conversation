"""Fixtures for extended_openai_conversation tests."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from homeassistant.helpers.template import TemplateEnvironment
import pytest


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
def temp_skills_dir(tmp_path: Path) -> Path:
    """Create temporary skills directory with test skill."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create test_skill
    test_skill = skills_dir / "test_skill"
    test_skill.mkdir()
    (test_skill / "SKILL.md").write_text(
        "---\nname: test\ndescription: Test skill\n---\nSkill body content"
    )
    # Create references directory
    references_dir = test_skill / "references"
    references_dir.mkdir()
    (references_dir / "reference.md").write_text("Reference content")
    # Create scripts directory
    scripts_dir = test_skill / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "script.sh").write_text("#!/bin/bash\necho 'Hello World'")

    # Create another skill for testing
    another_skill = skills_dir / "another_skill"
    another_skill.mkdir()
    (another_skill / "SKILL.md").write_text(
        "---\nname: another\ndescription: Another skill\n---\nAnother skill body"
    )

    return skills_dir


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
