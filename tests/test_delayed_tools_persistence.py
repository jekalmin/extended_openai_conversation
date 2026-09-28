"""Additional persistence-state tests for durable delayed Function Tools."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.extended_openai_conversation_responses.delayed_tools import (
    DelayedToolManager,
)
from tests.test_delayed_tools import _record


async def test_setup_can_retry_after_store_load_failure(hass) -> None:
    """A failed initial load must not publish a half-initialized scheduler."""
    manager = DelayedToolManager(hass)
    manager._store = SimpleNamespace(
        async_load=AsyncMock(
            side_effect=[OSError("storage unavailable"), {"calls": []}]
        ),
        async_save=AsyncMock(),
    )

    with pytest.raises(OSError, match="storage unavailable"):
        await manager.async_setup()

    assert manager._setup_complete is False
    assert manager._records == {}

    await manager.async_setup()

    assert manager._setup_complete is True
    assert manager._store.async_load.await_count == 2


async def test_failed_cancellation_keeps_due_call_retryable(hass, monkeypatch) -> None:
    """A Store failure cannot orphan a forbidden call without a live retry path."""
    manager = DelayedToolManager(hass)
    record = _record()
    manager._records = {record.call_id: record}
    manager._store = SimpleNamespace(
        async_save=AsyncMock(side_effect=OSError("storage unavailable")),
        async_load=AsyncMock(return_value={"calls": [record.as_dict()]}),
    )
    hass.config_entries.async_get_entry = MagicMock(
        return_value=SimpleNamespace(
            disabled_by=None,
            subentries={
                "agent": SimpleNamespace(subentry_type="conversation", data={})
            },
        )
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [
            {
                "enabled": False,
                "spec": {"name": "control_light"},
                "function": {"type": "native", "name": "execute_service_single"},
            }
        ],
    )

    assert await manager._async_execute_due(record.call_id) is True
    assert manager._records[record.call_id] == record
