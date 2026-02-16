"""SQLite schema for memory storage."""

from __future__ import annotations

import logging
import sqlite3

_LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 3

SCHEMA_SQL = """
-- Metadata (schema version, embedding fingerprint, etc.)
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Memory unit storage (no chunking, no files)
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    embedding  BLOB,          -- JSON float array (nullable)
    model      TEXT,          -- Embedding model name
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    id UNINDEXED
);
"""


def init_database(db_path: str, vector_dims: int | None = None) -> sqlite3.Connection:
    """Initialize the memory database with V3 schema."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(SCHEMA_SQL)

    # Try to create FTS5 table (may not be available on all platforms)
    try:
        conn.executescript(FTS_SQL)
    except sqlite3.OperationalError:
        _LOGGER.debug("FTS5 not available, keyword search disabled")

    # Try to load sqlite-vec for vector search
    _init_vec_table(conn, vector_dims)

    # Set schema version
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()
    return conn


def _init_vec_table(conn: sqlite3.Connection, dims: int | None = None) -> None:
    """Try to initialize sqlite-vec virtual table."""
    if dims is None:
        _LOGGER.debug("Vector dims unknown, skipping sqlite-vec table creation")
        return
    try:
        import sqlite_vec

        sqlite_vec.load(conn)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                id        INTEGER PRIMARY KEY,
                embedding FLOAT[{dims}]
            )
        """)
        conn.commit()
        _LOGGER.debug("sqlite-vec loaded successfully (dims=%d)", dims)
    except (ImportError, sqlite3.OperationalError):
        _LOGGER.debug(
            "sqlite-vec not available, falling back to Python cosine similarity"
        )


def has_fts(conn: sqlite3.Connection) -> bool:
    """Check if FTS5 table exists."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    )
    return cursor.fetchone() is not None


def has_vec(conn: sqlite3.Connection) -> bool:
    """Check if sqlite-vec table exists."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_vec'"
    )
    return cursor.fetchone() is not None
