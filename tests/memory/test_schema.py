"""Tests for memory database schema."""

import sqlite3
from pathlib import Path

import pytest

from custom_components.extended_openai_conversation.memory.schema import (
    SCHEMA_VERSION,
    has_fts,
    has_vec,
    init_database,
)


class TestSchema:
    """Test database schema initialization."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return str(tmp_path / "test_memory.sqlite")

    def test_init_creates_tables(self, db_path: str):
        """Test that init_database creates all required tables."""
        conn = init_database(db_path)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}

        assert "meta" in table_names
        assert "memories" in table_names
        # Old tables must NOT exist
        assert "files" not in table_names
        assert "chunks" not in table_names
        assert "embedding_cache" not in table_names

        conn.close()

    def test_schema_version_set(self, db_path: str):
        """Test that schema version is recorded."""
        conn = init_database(db_path)

        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row[0] == str(SCHEMA_VERSION)

        conn.close()

    def test_memories_table_structure(self, db_path: str):
        """Test memories table can store and retrieve data."""
        conn = init_database(db_path)

        conn.execute(
            "INSERT INTO memories (content) VALUES (?)",
            ("test content",),
        )
        conn.commit()

        row = conn.execute("SELECT id, content, embedding, model FROM memories").fetchone()
        assert row is not None
        assert row[1] == "test content"
        assert row[2] is None  # embedding nullable
        assert row[3] is None  # model nullable

        conn.close()

    def test_memories_table_with_embedding(self, db_path: str):
        """Test storing a memory with an embedding."""
        import json

        conn = init_database(db_path)

        embedding = [0.1, 0.2, 0.3]
        conn.execute(
            "INSERT INTO memories (content, embedding, model) VALUES (?, ?, ?)",
            ("test", json.dumps(embedding), "text-embedding-3-small"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT content, embedding, model FROM memories"
        ).fetchone()
        assert row[0] == "test"
        assert json.loads(row[1]) == embedding
        assert row[2] == "text-embedding-3-small"

        conn.close()

    def test_has_fts_detection(self, db_path: str):
        """Test FTS5 table detection."""
        conn = init_database(db_path)
        result = has_fts(conn)
        assert isinstance(result, bool)
        conn.close()

    def test_has_vec_detection(self, db_path: str):
        """Test vec table detection."""
        conn = init_database(db_path)
        result = has_vec(conn)
        assert isinstance(result, bool)
        conn.close()

    def test_idempotent_init(self, db_path: str):
        """Test that init_database can be called multiple times without losing data."""
        conn1 = init_database(db_path)
        conn1.execute(
            "INSERT INTO memories (content) VALUES (?)",
            ("persist",),
        )
        conn1.commit()
        conn1.close()

        # Re-init should not drop data
        conn2 = init_database(db_path)
        row = conn2.execute(
            "SELECT content FROM memories WHERE content = 'persist'"
        ).fetchone()
        assert row is not None
        assert row[0] == "persist"
        conn2.close()

    def test_created_at_default(self, db_path: str):
        """Test that created_at is automatically set."""
        conn = init_database(db_path)
        conn.execute("INSERT INTO memories (content) VALUES (?)", ("auto-time",))
        conn.commit()

        row = conn.execute("SELECT created_at FROM memories").fetchone()
        assert row is not None
        assert row[0] > 0  # unixepoch() > 0

        conn.close()
