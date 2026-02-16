"""Memory manager for long-term memory storage and retrieval."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import logging
import math
import os
import sqlite3

from openai import AsyncClient

from .embeddings import generate_embedding
from .schema import has_fts, has_vec, init_database

_LOGGER = logging.getLogger(__name__)

# Hybrid search weights
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Default vector dims for text-embedding-3-small
DEFAULT_VECTOR_DIMS = 1536


@dataclass
class MemorySearchResult:
    """A single memory search result."""

    id: int
    text: str
    score: float
    created_at: int


class MemoryManager:
    """Manages long-term memory for a conversation entity.

    SQLite is the single source of truth — no file watching, no chunking.
    Owned by the conversation subentry entity; looked up via hass.data.
    """

    def __init__(
        self,
        db_path: str,
        client: AsyncClient,
        embedding_model: str,
    ) -> None:
        """Initialize MemoryManager."""
        self._db_path = db_path
        self._client = client
        self._embedding_model = embedding_model
        self._conn: sqlite3.Connection | None = None

        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = init_database(self._db_path, DEFAULT_VECTOR_DIMS)
        return self._conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def async_store(self, content: str) -> dict:
        """Insert a memory entry and sync to FTS5/vec."""
        if not content.strip():
            return {"stored": False, "message": "Empty content"}

        conn = self._get_conn()

        # Generate embedding (async API call)
        embedding: list[float] | None = None
        try:
            embedding = await generate_embedding(
                self._client, content, self._embedding_model
            )
        except Exception:
            _LOGGER.warning("Failed to generate embedding for memory", exc_info=True)

        embedding_blob = json.dumps(embedding) if embedding else None
        cursor = conn.execute(
            "INSERT INTO memories (content, embedding, model) VALUES (?, ?, ?)",
            (content, embedding_blob, self._embedding_model if embedding else None),
        )
        memory_id = cursor.lastrowid

        if has_fts(conn):
            with contextlib.suppress(sqlite3.Error):
                conn.execute(
                    "INSERT INTO memories_fts (rowid, content, id) VALUES (?, ?, ?)",
                    (memory_id, content, memory_id),
                )

        if has_vec(conn) and embedding:
            with contextlib.suppress(sqlite3.Error):
                conn.execute(
                    "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                    (memory_id, json.dumps(embedding)),
                )

        conn.commit()
        return {"id": memory_id, "stored": True}

    async def async_search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.35,
    ) -> list[MemorySearchResult]:
        """Search memory for relevant content."""
        conn = self._get_conn()

        # Generate query embedding (async API call)
        query_embedding: list[float] | None = None
        try:
            query_embedding = await generate_embedding(
                self._client, query, self._embedding_model
            )
        except Exception:
            _LOGGER.warning("Failed to generate query embedding", exc_info=True)

        vector_results: dict[int, float] = {}
        if query_embedding:
            vector_results = _vector_search(conn, query_embedding, max_results * 2)

        keyword_results: dict[int, float] = {}
        if has_fts(conn):
            keyword_results = _keyword_search(conn, query, max_results * 2)

        merged = _merge_results(vector_results, keyword_results)

        filtered = [
            (mem_id, score)
            for mem_id, score in merged.items()
            if score >= min_score
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:max_results]

        results: list[MemorySearchResult] = []
        for mem_id, score in filtered:
            row = conn.execute(
                "SELECT content, created_at FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
            if row:
                results.append(
                    MemorySearchResult(
                        id=mem_id,
                        text=row[0],
                        score=score,
                        created_at=row[1],
                    )
                )

        return results

    async def async_delete(self, memory_id: int) -> dict:
        """Delete a memory entry."""
        conn = self._get_conn()

        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

        if has_fts(conn):
            with contextlib.suppress(sqlite3.Error):
                conn.execute(
                    "DELETE FROM memories_fts WHERE rowid = ?", (memory_id,)
                )

        if has_vec(conn):
            with contextlib.suppress(sqlite3.Error):
                conn.execute(
                    "DELETE FROM memories_vec WHERE id = ?", (memory_id,)
                )

        conn.commit()
        return {"deleted": True}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def async_close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _vector_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int,
) -> dict[int, float]:
    """Perform vector similarity search."""
    results: dict[int, float] = {}

    if has_vec(conn):
        try:
            rows = conn.execute(
                "SELECT id, distance FROM memories_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (json.dumps(query_embedding), limit),
            ).fetchall()
            for row in rows:
                results[row[0]] = 1.0 - row[1]
            return results
        except sqlite3.Error:
            _LOGGER.debug("sqlite-vec search failed, falling back", exc_info=True)

    # Fallback: Python cosine similarity
    rows = conn.execute(
        "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
    ).fetchall()
    for row in rows:
        mem_id = row[0]
        try:
            mem_embedding = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue
        similarity = _cosine_similarity(query_embedding, mem_embedding)
        results[mem_id] = similarity

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1], reverse=True)[:limit]
    )
    return sorted_results


def _keyword_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
) -> dict[int, float]:
    """Perform FTS5 keyword search."""
    results: dict[int, float] = {}
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
        if not rows:
            return results
        max_rank = max(abs(row[1]) for row in rows) if rows else 1.0
        for row in rows:
            results[row[0]] = abs(row[1]) / max_rank if max_rank > 0 else 0.0
    except sqlite3.Error:
        _LOGGER.debug("FTS search failed", exc_info=True)
    return results


def _merge_results(
    vector_results: dict[int, float],
    keyword_results: dict[int, float],
) -> dict[int, float]:
    """Merge vector and keyword search results with weighted scoring."""
    merged: dict[int, float] = {}
    all_ids = set(vector_results.keys()) | set(keyword_results.keys())
    for mem_id in all_ids:
        v_score = vector_results.get(mem_id, 0.0)
        k_score = keyword_results.get(mem_id, 0.0)
        merged[mem_id] = (VECTOR_WEIGHT * v_score) + (KEYWORD_WEIGHT * k_score)
    return merged


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot_product = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
