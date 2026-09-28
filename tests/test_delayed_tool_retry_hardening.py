"""Regression tests for delayed Function Tool retry safety."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from custom_components.extended_openai_conversation_responses.delayed_tools import (
    _MAX_AGENT_RETRIES,
    DelayedToolCall,
    DelayedToolManager,
)
from homeassistant.util import dt as dt_util


def _record(*, retry_count: int = 0) -> DelayedToolCall:
    now = dt_util.utcnow()
    return DelayedToolCall(
        call_id="call-1",
        entry_id="entry",
        subentry_id="agent",
        tool_name="control_light",
        arguments={"delay": {"seconds": 5}},
        due_at=(now - timedelta(seconds=1)).isoformat(),
        created_at=(now - timedelta(seconds=6)).isoformat(),
        retry_count=retry_count,
    )


async def test_retry_budget_advances_when_retry_state_persistence_fails(hass) -> None:
    """A Store outage must not make agent-resolution retries unbounded."""
    manager = DelayedToolManager(hass)
    record = _record()
    manager._records = {record.call_id: record}
    manager._store = SimpleNamespace(
        async_save=AsyncMock(side_effect=OSError("storage unavailable")),
        async_load=AsyncMock(return_value={"calls": [record.as_dict()]}),
    )
    manager._async_discard = AsyncMock(return_value=True)

    for retry_count in range(_MAX_AGENT_RETRIES):
        current = manager._records[record.call_id]
        assert current.retry_count == retry_count
        assert await manager._async_retry_agent(current) is True

    assert manager._store.async_save.await_count == _MAX_AGENT_RETRIES
    current = manager._records[record.call_id]
    assert current.retry_count == _MAX_AGENT_RETRIES

    # Once the in-memory safety budget is exhausted, no further retry-state write is
    # attempted; the normal terminal discard path takes over.
    assert await manager._async_retry_agent(current) is False
    assert manager._store.async_save.await_count == _MAX_AGENT_RETRIES
    manager._async_discard.assert_awaited_once_with(
        record.call_id,
        "conversation agent did not become available",
    )


async def test_successful_retry_write_catches_up_after_previous_failure(hass) -> None:
    """A later healthy Store write persists the advanced in-memory retry count."""
    manager = DelayedToolManager(hass)
    record = _record()
    manager._records = {record.call_id: record}
    save = AsyncMock(side_effect=[OSError("storage unavailable"), None])
    manager._store = SimpleNamespace(
        async_save=save,
        async_load=AsyncMock(return_value={"calls": [record.as_dict()]}),
    )

    assert await manager._async_retry_agent(record) is True
    assert manager._records[record.call_id].retry_count == 1

    assert await manager._async_retry_agent(manager._records[record.call_id]) is True
    assert manager._records[record.call_id].retry_count == 2

    persisted = save.await_args_list[1].args[0]
    assert persisted["calls"][0]["retry_count"] == 2
