"""Tests for embedding generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.extended_openai_conversation.memory.embeddings import (
    generate_embedding,
)


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()

    async def mock_create(input, model):
        result = MagicMock()
        item = MagicMock()
        item.embedding = [0.1, 0.2, 0.3]
        result.data = [item]
        return result

    client.embeddings.create = AsyncMock(side_effect=mock_create)
    return client


class TestGenerateEmbedding:
    """Test single embedding generation."""

    async def test_returns_embedding(self, mock_client):
        """Test that generate_embedding returns a list of floats."""
        result = await generate_embedding(mock_client, "hello world", "model")
        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    async def test_calls_api_with_correct_args(self, mock_client):
        """Test that the API is called with correct input."""
        await generate_embedding(mock_client, "test text", "text-embedding-3-small")
        mock_client.embeddings.create.assert_called_once_with(
            input=["test text"], model="text-embedding-3-small"
        )

    async def test_each_call_hits_api(self, mock_client):
        """Test that each generate_embedding call hits the API (no caching)."""
        await generate_embedding(mock_client, "text", "model")
        await generate_embedding(mock_client, "text", "model")
        assert mock_client.embeddings.create.call_count == 2

    async def test_retry_on_failure(self, mock_client):
        """Test retry logic on transient API failure."""
        call_count = 0

        async def failing_then_succeeding(input, model):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            result = MagicMock()
            item = MagicMock()
            item.embedding = [0.5, 0.6]
            result.data = [item]
            return result

        mock_client.embeddings.create = AsyncMock(side_effect=failing_then_succeeding)

        # Patch asyncio.sleep to avoid delays in tests
        import asyncio
        from unittest.mock import patch

        with patch.object(asyncio, "sleep", new=AsyncMock()):
            result = await generate_embedding(mock_client, "text", "model")

        assert result == [0.5, 0.6]
        assert call_count == 3
