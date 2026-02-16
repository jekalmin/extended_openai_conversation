"""Tests for MemoryManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.extended_openai_conversation.memory.manager import (
    MemoryManager,
    MemorySearchResult,
    _cosine_similarity,
)


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()

    async def mock_create(input, model):
        result = MagicMock()
        item = MagicMock()
        # Create a simple deterministic embedding based on text content
        text = input[0] if input else ""
        embedding = [0.1] * 1536
        for i, ch in enumerate(text[:10]):
            if i < 1536:
                embedding[i] = ord(ch) / 256.0
        item.embedding = embedding
        result.data = [item]
        return result

    client.embeddings.create = AsyncMock(side_effect=mock_create)
    return client


@pytest.fixture
def manager(tmp_path: Path, mock_client) -> MemoryManager:
    """Create a MemoryManager instance."""
    db_path = str(tmp_path / "memory" / "test.sqlite")
    return MemoryManager(
        db_path=db_path,
        client=mock_client,
        embedding_model="text-embedding-3-small",
    )


class TestMemoryManager:
    """Test MemoryManager."""

    def test_creates_db_dir_on_init(self, tmp_path: Path, mock_client):
        """Test that MemoryManager creates the db directory on init."""
        db_path = str(tmp_path / "deep" / "nested" / "memory.sqlite")
        MemoryManager(db_path=db_path, client=mock_client, embedding_model="model")
        assert (tmp_path / "deep" / "nested").exists()

    async def test_store_returns_id(self, manager: MemoryManager):
        """Test that async_store returns an id and stored=True."""
        result = await manager.async_store("The user prefers warm lighting.")
        assert result["stored"] is True
        assert "id" in result
        assert result["id"] is not None

    async def test_store_and_search(self, manager: MemoryManager):
        """Test storing and searching memory."""
        await manager.async_store("The user prefers warm lighting in the evening.")

        results = await manager.async_search(
            query="lighting preference",
            min_score=0.0,  # Low threshold for test
        )
        assert len(results) >= 1
        assert isinstance(results[0], MemorySearchResult)
        assert "warm lighting" in results[0].text

    async def test_store_empty_content(self, manager: MemoryManager):
        """Test storing empty content returns stored=False."""
        result = await manager.async_store("")
        assert result["stored"] is False
        assert "message" in result

    async def test_store_whitespace_only(self, manager: MemoryManager):
        """Test storing whitespace-only content returns stored=False."""
        result = await manager.async_store("   ")
        assert result["stored"] is False

    async def test_search_no_results(self, manager: MemoryManager):
        """Test search when no memories stored."""
        results = await manager.async_search(query="anything")
        assert results == []

    async def test_search_returns_sorted_by_score(self, manager: MemoryManager):
        """Test that search results are sorted by score descending."""
        await manager.async_store("Warm lighting preference")
        await manager.async_store("Temperature set to 22C")

        results = await manager.async_search(query="lighting", min_score=0.0)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    async def test_delete(self, manager: MemoryManager):
        """Test deleting a memory entry."""
        store_result = await manager.async_store("Content to delete")
        memory_id = store_result["id"]

        delete_result = await manager.async_delete(memory_id)
        assert delete_result["deleted"] is True

        # Verify it's gone
        conn = manager._get_conn()
        row = conn.execute(
            "SELECT id FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        assert row is None

    async def test_delete_nonexistent(self, manager: MemoryManager):
        """Test deleting a non-existent memory returns deleted=True (no error)."""
        result = await manager.async_delete(999999)
        assert result["deleted"] is True

    async def test_close_releases_connection(self, manager: MemoryManager):
        """Test that async_close closes the database connection."""
        manager._get_conn()
        assert manager._conn is not None
        await manager.async_close()
        assert manager._conn is None

    async def test_search_result_has_correct_fields(self, manager: MemoryManager):
        """Test that search results contain expected fields."""
        await manager.async_store("User likes jazz music")

        results = await manager.async_search(query="jazz music", min_score=0.0)
        if results:
            r = results[0]
            assert isinstance(r.id, int)
            assert isinstance(r.text, str)
            assert isinstance(r.score, float)
            assert isinstance(r.created_at, int)

    async def test_multiple_stores_and_search(self, manager: MemoryManager):
        """Test storing multiple memories and searching."""
        await manager.async_store("User's favorite color is blue")
        await manager.async_store("User prefers morning workouts")
        await manager.async_store("User lives in Seoul")

        results = await manager.async_search(query="color preference", min_score=0.0)
        assert len(results) >= 1


class TestCosineSimilarity:
    """Test cosine similarity function."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        a = [1.0, 2.0]
        b = [0.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_different_length_vectors(self):
        """Test similarity with different length vectors."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0
