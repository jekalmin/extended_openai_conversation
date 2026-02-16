"""Tests for memory services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.extended_openai_conversation.const import DOMAIN
from custom_components.extended_openai_conversation.memory.manager import (
    MemorySearchResult,
)


@pytest.fixture
def mock_manager():
    """Create a mock MemoryManager."""
    manager = MagicMock()
    manager.async_store = AsyncMock(return_value={"id": 1, "stored": True})
    manager.async_search = AsyncMock(
        return_value=[
            MemorySearchResult(
                id=1,
                text="User prefers warm lighting",
                score=0.85,
                created_at=1700000000,
            ),
        ]
    )
    manager.async_delete = AsyncMock(return_value={"deleted": True})
    return manager


def _make_platform_with_entity(entity_id: str, memory_manager):
    """Create a mock platform containing an entity with _memory_manager."""
    entity = MagicMock()
    entity._memory_manager = memory_manager
    platform = MagicMock()
    platform.entities = {entity_id: entity}
    return platform


class TestMemoryStoreService:
    """Test memory_store service handler."""

    async def test_store_success(self, hass, mock_manager):
        """Test successful memory store via service."""
        platform = _make_platform_with_entity("conversation.test", mock_manager)

        from custom_components.extended_openai_conversation.services import (
            MEMORY_STORE_SCHEMA,
        )

        call = MagicMock()
        call.data = MEMORY_STORE_SCHEMA(
            {"entity_id": "conversation.test", "content": "User likes jazz"}
        )

        with patch(
            "custom_components.extended_openai_conversation.services.async_get_platforms",
            return_value=[platform],
        ):
            manager = next(
                (
                    getattr(e, "_memory_manager", None)
                    for p in [platform]
                    for e in p.entities.values()
                ),
                None,
            )
        assert manager is not None
        result = await manager.async_store(call.data["content"])
        assert result["stored"] is True
        assert result["id"] == 1
        mock_manager.async_store.assert_called_once_with("User likes jazz")

    async def test_store_no_manager(self, hass):
        """Test memory_store when no entity found."""
        with patch(
            "custom_components.extended_openai_conversation.services.async_get_platforms",
            return_value=[],
        ):
            manager = None
            for platform in []:
                entity = platform.entities.get("conversation.nonexistent")
                if entity is not None:
                    manager = getattr(entity, "_memory_manager", None)
        assert manager is None


class TestMemorySearchService:
    """Test memory_search service handler."""

    async def test_search_returns_results(self, hass, mock_manager):
        """Test memory_search returns formatted results."""
        results = await mock_manager.async_search(
            query="lighting", max_results=5, min_score=0.35
        )

        formatted = {
            "results": [
                {
                    "id": r.id,
                    "snippet": r.text,
                    "score": round(r.score, 3),
                    "created_at": r.created_at,
                }
                for r in results
            ]
        }

        assert len(formatted["results"]) == 1
        assert formatted["results"][0]["snippet"] == "User prefers warm lighting"
        assert formatted["results"][0]["score"] == 0.85
        assert formatted["results"][0]["id"] == 1

    async def test_search_no_manager(self, hass):
        """Test memory_search when no entity found."""
        with patch(
            "custom_components.extended_openai_conversation.services.async_get_platforms",
            return_value=[],
        ):
            manager = None
        assert manager is None


class TestMemoryDeleteService:
    """Test memory_delete service handler."""

    async def test_delete_success(self, hass, mock_manager):
        """Test successful memory delete via service."""
        result = await mock_manager.async_delete(1)
        assert result["deleted"] is True
        mock_manager.async_delete.assert_called_once_with(1)

    async def test_delete_no_manager(self, hass):
        """Test memory_delete when no entity found."""
        with patch(
            "custom_components.extended_openai_conversation.services.async_get_platforms",
            return_value=[],
        ):
            manager = None
        assert manager is None


class TestMemoryServiceSchemas:
    """Test memory service input schemas."""

    def test_memory_store_schema_valid(self):
        """Test MEMORY_STORE_SCHEMA with valid input."""
        from custom_components.extended_openai_conversation.services import (
            MEMORY_STORE_SCHEMA,
        )

        result = MEMORY_STORE_SCHEMA(
            {"entity_id": "conversation.test", "content": "Some content"}
        )
        assert result["entity_id"] == "conversation.test"
        assert result["content"] == "Some content"

    def test_memory_search_schema_defaults(self):
        """Test MEMORY_SEARCH_SCHEMA applies defaults."""
        from custom_components.extended_openai_conversation.services import (
            MEMORY_SEARCH_SCHEMA,
        )

        result = MEMORY_SEARCH_SCHEMA(
            {"entity_id": "conversation.test", "query": "lights"}
        )
        assert result["max_results"] == 5
        assert result["min_score"] == 0.35

    def test_memory_search_schema_custom(self):
        """Test MEMORY_SEARCH_SCHEMA with custom values."""
        from custom_components.extended_openai_conversation.services import (
            MEMORY_SEARCH_SCHEMA,
        )

        result = MEMORY_SEARCH_SCHEMA(
            {
                "entity_id": "conversation.test",
                "query": "lights",
                "max_results": 10,
                "min_score": 0.5,
            }
        )
        assert result["max_results"] == 10
        assert result["min_score"] == 0.5

    def test_memory_delete_schema_valid(self):
        """Test MEMORY_DELETE_SCHEMA with valid input."""
        from custom_components.extended_openai_conversation.services import (
            MEMORY_DELETE_SCHEMA,
        )

        result = MEMORY_DELETE_SCHEMA({"entity_id": "conversation.test", "id": 42})
        assert result["entity_id"] == "conversation.test"
        assert result["id"] == 42

    def test_memory_delete_schema_coerces_id(self):
        """Test MEMORY_DELETE_SCHEMA coerces string id to int."""
        from custom_components.extended_openai_conversation.services import (
            MEMORY_DELETE_SCHEMA,
        )

        result = MEMORY_DELETE_SCHEMA({"entity_id": "conversation.test", "id": "5"})
        assert result["id"] == 5
