"""Regression tests for durable delayed Function Tools."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.extended_openai_conversation_responses.const import DOMAIN
from custom_components.extended_openai_conversation_responses.conversation import (
    _ACTIVE_GUEST_POLICY,
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation_responses.delayed_tools import (
    _AGENT_RETRY_SECONDS,
    _DELAYED_EXECUTION_MARKER,
    _EXECUTING,
    _MAX_AGENT_RETRIES,
    DATA_DELAYED_TOOL_MANAGER,
    DelayedToolCall,
    DelayedToolManager,
    _delay_as_timedelta,
    async_setup_delayed_tools,
)
from custom_components.extended_openai_conversation_responses.entity import (
    ExtendedOpenAIBaseLLMEntity,
)
from custom_components.extended_openai_conversation_responses.guest_mode import (
    GuestCapabilityPolicy,
)
from custom_components.extended_openai_conversation_responses.ha_tool_result_compat import (
    tool_result_data,
)
from homeassistant.components import conversation
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util


def _entity(hass):
    return SimpleNamespace(
        hass=hass,
        entry=SimpleNamespace(entry_id="entry"),
        subentry=SimpleNamespace(subentry_id="agent"),
    )


def _record(*, status: str = "pending", retry_count: int = 0) -> DelayedToolCall:
    now = dt_util.utcnow()
    return DelayedToolCall(
        call_id="call-1",
        entry_id="entry",
        subentry_id="agent",
        tool_name="control_light",
        arguments={"delay": {"seconds": 5}, "value": 1},
        due_at=(now - timedelta(seconds=1)).isoformat(),
        created_at=(now - timedelta(seconds=6)).isoformat(),
        user_id="user-1",
        device_id="device-1",
        status=status,
        retry_count=retry_count,
    )


def test_delay_normalization_accepts_ha_time_period_shapes() -> None:
    """Delay persistence uses the same shapes accepted by HA script delays."""
    assert _delay_as_timedelta({"minutes": 1, "seconds": 5}) == timedelta(seconds=65)
    assert _delay_as_timedelta("00:00:03") == timedelta(seconds=3)

    with pytest.raises(HomeAssistantError, match="Invalid Function Tool delay"):
        _delay_as_timedelta("not-a-duration")


async def test_schedule_is_persisted_before_becoming_live(hass) -> None:
    """A failed Store write must not leave a volatile scheduled action behind."""
    manager = DelayedToolManager(hass)
    manager._setup_complete = True
    manager._store = SimpleNamespace(async_save=AsyncMock())
    context = SimpleNamespace(
        context=Context(user_id="user-1"), device_id="device-1"
    )

    record = await manager.async_schedule(
        _entity(hass),
        "control_light",
        {"delay": {"seconds": 30}, "value": 1},
        context,
    )

    manager._store.async_save.assert_awaited_once()
    assert manager._records[record.call_id] == record
    assert record.user_id == "user-1"
    assert record.device_id == "device-1"

    failing = DelayedToolManager(hass)
    failing._setup_complete = True
    failing._store = SimpleNamespace(
        async_save=AsyncMock(side_effect=OSError("storage unavailable"))
    )
    with pytest.raises(OSError, match="storage unavailable"):
        await failing.async_schedule(
            _entity(hass),
            "control_light",
            {"delay": {"seconds": 30}},
            context,
        )
    assert failing._records == {}


async def test_schedule_reconciles_after_lost_store_acknowledgement(hass) -> None:
    """A committed schedule remains operational after Store reports failure."""
    persisted: dict[str, Any] = {}

    async def save_then_fail(payload: dict[str, Any]) -> None:
        persisted.clear()
        persisted.update(payload)
        raise OSError("directory fsync acknowledgement failed")

    async def load() -> dict[str, Any]:
        return persisted

    manager = DelayedToolManager(hass)
    manager._setup_complete = True
    manager._store = SimpleNamespace(async_save=save_then_fail, async_load=load)
    context = SimpleNamespace(
        context=Context(user_id="user-1"), device_id="device-1"
    )

    with pytest.raises(OSError, match="directory fsync acknowledgement failed"):
        await manager.async_schedule(
            _entity(hass),
            "control_light",
            {"delay": {"seconds": 30}, "value": 1},
            context,
        )

    assert len(manager._records) == 1
    scheduled = next(iter(manager._records.values()))
    assert persisted["calls"] == [scheduled.as_dict()]


async def test_delayed_state_transitions_reconcile_after_lost_acknowledgement(
    hass,
) -> None:
    """Execution tombstones and removals adopt the Store's replaced generation."""
    pending = _record()
    persisted: dict[str, Any] = {"calls": [pending.as_dict()]}

    async def save_then_fail(payload: dict[str, Any]) -> None:
        persisted.clear()
        persisted.update(payload)
        raise OSError("directory fsync acknowledgement failed")

    async def load() -> dict[str, Any]:
        return persisted

    manager = DelayedToolManager(hass)
    manager._records = {pending.call_id: pending}
    manager._setup_complete = True
    manager._store = SimpleNamespace(async_save=save_then_fail, async_load=load)

    with pytest.raises(OSError, match="directory fsync acknowledgement failed"):
        await manager._async_replace_record(replace(pending, status=_EXECUTING))
    # Startup semantics discard an executing tombstone rather than replay it.
    assert manager._records == {}
    assert persisted["calls"][0]["status"] == _EXECUTING

    persisted["calls"] = [pending.as_dict()]
    manager._records = {pending.call_id: pending}
    assert await manager._async_discard(pending.call_id, "test cancellation")
    assert manager._records == {}
    assert persisted["calls"] == []


async def test_due_call_uses_current_tool_and_current_exposure(hass, monkeypatch) -> None:
    """Execution re-resolves the tool and exposure instead of stale snapshots."""
    manager = DelayedToolManager(hass)
    manager._setup_complete = True
    record = _record()
    manager._records = {record.call_id: record}
    manager._store = SimpleNamespace(async_save=AsyncMock())

    latest_subentry = SimpleNamespace(subentry_type="conversation", data={})
    latest_entry = SimpleNamespace(
        disabled_by=None, subentries={"agent": latest_subentry}
    )
    hass.config_entries.async_get_entry = MagicMock(return_value=latest_entry)
    hass.auth.async_get_user = AsyncMock(return_value=SimpleNamespace(is_active=True))

    current_tool = {
        "enabled": True,
        "spec": {"name": "control_light"},
        "function": {"type": "native", "name": "execute_service_single"},
    }
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [current_tool],
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.get_exposed_entities",
        lambda _hass: [{"entity_id": "light.current"}],
    )

    agent = SimpleNamespace(_execute_function_tool=AsyncMock(return_value=object()))
    monkeypatch.setattr(manager, "_resolve_agent", lambda *_args: agent)

    retry = await manager._async_execute_due(record.call_id)

    assert retry is False
    assert record.call_id not in manager._records
    agent._execute_function_tool.assert_awaited_once()
    called_tool, tool_input, context, exposed = agent._execute_function_tool.await_args.args
    assert called_tool is current_tool
    assert isinstance(tool_input, llm.ToolInput)
    assert tool_input.tool_args == record.arguments
    assert context.context.user_id == "user-1"
    assert context.device_id == "device-1"
    assert getattr(context, _DELAYED_EXECUTION_MARKER) is True
    assert exposed == [{"entity_id": "light.current"}]
    assert manager._store.async_save.await_count == 2


@pytest.mark.parametrize(
    ("inherited_policy", "live_policy"),
    [
        (
            GuestCapabilityPolicy(
                True, configured_tool_names=frozenset({"control_light"})
            ),
            GuestCapabilityPolicy(True, configured_tool_names=frozenset()),
        ),
        (
            GuestCapabilityPolicy(
                True, configured_tool_names=frozenset({"control_light"})
            ),
            GuestCapabilityPolicy.unrestricted(),
        ),
        (
            None,
            GuestCapabilityPolicy(True, configured_tool_names=frozenset()),
        ),
    ],
)
async def test_due_call_uses_live_guest_policy_not_inherited_request_policy(
    hass,
    monkeypatch,
    inherited_policy: GuestCapabilityPolicy | None,
    live_policy: GuestCapabilityPolicy,
) -> None:
    """In-process and recovered delayed calls authorize from the same live policy."""
    manager = DelayedToolManager(hass)
    record = _record()
    manager._records = {record.call_id: record}
    manager._store = SimpleNamespace(async_save=AsyncMock())
    hass.config_entries.async_get_entry = MagicMock(
        return_value=SimpleNamespace(
            disabled_by=None,
            subentries={
                "agent": SimpleNamespace(subentry_type="conversation", data={})
            },
        )
    )
    hass.auth.async_get_user = AsyncMock(return_value=SimpleNamespace(is_active=True))

    current_tool = {
        "enabled": True,
        "spec": {"name": "control_light"},
        "function": {"type": "native", "name": "execute_service_single"},
    }
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [current_tool],
    )

    observed: list[GuestCapabilityPolicy] = []

    class GuestAwareAgent:
        def _resolve_live_guest_policy(self) -> GuestCapabilityPolicy:
            return live_policy

        def _effective_guest_policy(self) -> GuestCapabilityPolicy:
            return ExtendedOpenAIAgentEntity._effective_guest_policy(self)  # type: ignore[arg-type]

        async def _execute_function_tool(self, *_args) -> object:
            observed.append(self._effective_guest_policy())
            return object()

    agent = GuestAwareAgent()
    monkeypatch.setattr(manager, "_resolve_agent", lambda *_args: agent)

    token = _ACTIVE_GUEST_POLICY.set(inherited_policy)
    try:
        assert await manager._async_execute_due(record.call_id) is False
        assert _ACTIVE_GUEST_POLICY.get() is inherited_policy
    finally:
        _ACTIVE_GUEST_POLICY.reset(token)

    assert observed == [live_policy]


async def test_due_call_is_cancelled_when_tool_is_disabled(hass, monkeypatch) -> None:
    """Disabling a Function Tool after scheduling prevents later execution."""
    manager = DelayedToolManager(hass)
    record = _record()
    manager._records = {record.call_id: record}
    manager._store = SimpleNamespace(async_save=AsyncMock())
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
    resolve_agent = MagicMock()
    monkeypatch.setattr(manager, "_resolve_agent", resolve_agent)

    assert await manager._async_execute_due(record.call_id) is False
    assert record.call_id not in manager._records
    resolve_agent.assert_not_called()


async def test_execution_boundary_failure_never_runs_tool(hass, monkeypatch) -> None:
    """No action runs unless the executing tombstone is durably persisted first."""
    manager = DelayedToolManager(hass)
    record = _record()
    manager._records = {record.call_id: record}
    manager._store = SimpleNamespace(
        async_save=AsyncMock(side_effect=OSError("storage unavailable"))
    )
    hass.config_entries.async_get_entry = MagicMock(
        return_value=SimpleNamespace(
            disabled_by=None,
            subentries={
                "agent": SimpleNamespace(subentry_type="conversation", data={})
            },
        )
    )
    hass.auth.async_get_user = AsyncMock(return_value=SimpleNamespace(is_active=True))
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [
            {
                "enabled": True,
                "spec": {"name": "control_light"},
                "function": {"type": "native", "name": "execute_service_single"},
            }
        ],
    )
    agent = SimpleNamespace(_execute_function_tool=AsyncMock())
    monkeypatch.setattr(manager, "_resolve_agent", lambda *_args: agent)

    assert await manager._async_execute_due(record.call_id) is True
    agent._execute_function_tool.assert_not_awaited()
    assert record.call_id not in manager._records
    assert not manager._setup_complete


async def test_interrupted_executing_calls_are_not_replayed(hass) -> None:
    """Startup discards an indeterminate executing record to avoid duplicate effects."""
    executing = _record(status=_EXECUTING)
    pending = DelayedToolCall(
        **{
            **_record().as_dict(),
            "call_id": "call-2",
            "status": "pending",
        }
    )
    manager = DelayedToolManager(hass)
    manager._store = SimpleNamespace(
        async_load=AsyncMock(
            return_value={"calls": [executing.as_dict(), pending.as_dict()]}
        ),
        async_save=AsyncMock(),
    )

    await manager.async_setup()

    assert executing.call_id not in manager._records
    assert manager._records[pending.call_id] == pending
    manager._store.async_save.assert_awaited_once()


def _coverage_record(
    *,
    call_id: str = "call-1",
    status: str = "pending",
    retry_count: int = 0,
    user_id: str | None = "user-1",
    due_at: str | None = None,
) -> DelayedToolCall:
    now = dt_util.utcnow()
    return DelayedToolCall(
        call_id=call_id,
        entry_id="entry",
        subentry_id="agent",
        tool_name="control_light",
        arguments={"delay": {"seconds": 5}, "value": 1},
        due_at=due_at or (now - timedelta(seconds=1)).isoformat(),
        created_at=(now - timedelta(seconds=6)).isoformat(),
        user_id=user_id,
        device_id="device-1",
        status=status,
        retry_count=retry_count,
    )


def _valid_tool(*, function_type: str = "native") -> dict[str, Any]:
    return {
        "enabled": True,
        "spec": {"name": "control_light"},
        "function": {"type": function_type, "name": "execute_service_single"},
    }


def _live_entry() -> SimpleNamespace:
    return SimpleNamespace(
        disabled_by=None,
        subentries={"agent": SimpleNamespace(subentry_type="conversation", data={})},
    )


def _stored_call() -> dict[str, Any]:
    return _coverage_record().as_dict()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda raw: "not-an-object", "not an object"),
        (lambda raw: {**raw, "call_id": ""}, "invalid call_id"),
        (lambda raw: {**raw, "arguments": []}, "invalid arguments"),
        (lambda raw: {**raw, "status": "done"}, "invalid status"),
        (lambda raw: {**raw, "retry_count": -1}, "invalid retry_count"),
        (lambda raw: {**raw, "due_at": "not-a-date"}, "invalid due_at timestamp"),
        (
            lambda raw: {**raw, "created_at": "not-a-date"},
            "invalid created_at timestamp",
        ),
        (lambda raw: {**raw, "user_id": 123}, "invalid user_id"),
        (lambda raw: {**raw, "device_id": 123}, "invalid device_id"),
    ],
)
def test_persisted_call_validation_rejects_corrupt_shapes(mutate, match: str) -> None:
    """Every persisted field used for later authorization must be validated."""
    with pytest.raises(ValueError, match=match):
        DelayedToolCall.from_dict(mutate(_stored_call()))


def test_negative_delay_is_rejected() -> None:
    """A delayed tool can never be scheduled backwards in time."""
    with pytest.raises(HomeAssistantError, match="cannot be negative"):
        _delay_as_timedelta({"seconds": -1})


async def test_setup_cleans_invalid_persisted_records_and_is_idempotent(hass) -> None:
    """Corrupt records are removed durably and a second setup is a no-op."""
    valid = _coverage_record(call_id="valid")
    manager = DelayedToolManager(hass)
    manager._store = SimpleNamespace(
        async_load=AsyncMock(return_value={"calls": [valid.as_dict(), {"bad": True}]}),
        async_save=AsyncMock(),
    )

    await manager.async_setup()
    await manager.async_setup()

    assert manager._records == {"valid": valid}
    manager._store.async_load.assert_awaited_once()
    manager._store.async_save.assert_awaited_once_with({"calls": [valid.as_dict()]})


async def test_schedule_requires_setup_and_arms_when_already_started(
    hass, monkeypatch
) -> None:
    """Scheduling is unavailable before recovery and arms immediately after start."""
    manager = DelayedToolManager(hass)
    entity = SimpleNamespace(
        entry=SimpleNamespace(entry_id="entry"),
        subentry=SimpleNamespace(subentry_id="agent"),
    )

    with pytest.raises(HomeAssistantError, match="scheduler is unavailable"):
        await manager.async_schedule(
            entity,
            "control_light",
            {"delay": {"seconds": 1}},
            None,
        )

    manager._setup_complete = True
    manager._started = True
    manager._store = SimpleNamespace(async_save=AsyncMock())
    arm = MagicMock()
    monkeypatch.setattr(manager, "_arm", arm)

    record = await manager.async_schedule(
        entity,
        "control_light",
        {"delay": {"seconds": 1}},
        None,
    )

    arm.assert_called_once_with(record.call_id)
    assert record.user_id is None
    assert record.device_id is None


def test_start_stop_lifecycle_arms_once_and_preserves_executing_task(
    hass, monkeypatch
) -> None:
    """Startup arms records once; shutdown cancels only replay-safe pending work."""
    manager = DelayedToolManager(hass)
    manager._records = {
        "pending": _coverage_record(call_id="pending"),
        "executing": _coverage_record(call_id="executing", status=_EXECUTING),
    }
    arm = MagicMock()
    monkeypatch.setattr(manager, "_arm", arm)

    manager._handle_started()
    manager._handle_started()

    assert manager._started is True
    assert [item.args[0] for item in arm.call_args_list] == ["pending", "executing"]

    pending_task = MagicMock()
    executing_task = MagicMock()
    orphan_task = MagicMock()
    manager._tasks = {
        "pending": pending_task,
        "executing": executing_task,
        "missing": orphan_task,
    }

    manager._handle_stop()

    assert manager._started is False
    pending_task.cancel.assert_called_once_with()
    orphan_task.cancel.assert_called_once_with()
    executing_task.cancel.assert_not_called()


def test_arm_ignores_live_waiter_missing_record_and_nonpending_record(hass) -> None:
    """Arming cannot duplicate waiters or resurrect absent/completed records."""
    manager = DelayedToolManager(hass)
    live_task = MagicMock()
    live_task.done.return_value = False
    manager._tasks["call-1"] = live_task
    manager._records["call-1"] = _coverage_record()

    manager._arm("call-1")
    assert manager._tasks["call-1"] is live_task

    manager._tasks.clear()
    manager._arm("missing")
    assert manager._tasks == {}

    manager._records["executing"] = _coverage_record(call_id="executing", status=_EXECUTING)
    manager._arm("executing")
    assert manager._tasks == {}


async def test_waiter_discards_invalid_due_timestamp_and_cleans_task(
    hass, monkeypatch
) -> None:
    """A corrupt in-memory due timestamp is discarded and cannot strand a waiter."""
    manager = DelayedToolManager(hass)
    manager._started = True
    record = _coverage_record(due_at="not-a-date")
    manager._records = {record.call_id: record}
    manager._tasks = {record.call_id: MagicMock()}
    discard = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_async_discard", discard)

    await manager._async_wait_and_execute(record.call_id)

    discard.assert_awaited_once_with(record.call_id, "invalid due timestamp")
    assert record.call_id not in manager._tasks


async def test_waiter_uses_maintenance_gate_and_stops_after_execution(
    hass, monkeypatch
) -> None:
    """Due execution is serialized through the ordinary-agent maintenance gate."""
    manager = DelayedToolManager(hass)
    manager._started = True
    record = _coverage_record()
    manager._records = {record.call_id: record}
    manager._tasks = {record.call_id: MagicMock()}

    class Gate:
        def shared(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    gate = Gate()
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.get_agent_maintenance_gate",
        lambda *_args: gate,
    )
    sleep = AsyncMock()
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.asyncio.sleep",
        sleep,
    )
    execute_due = AsyncMock(return_value=False)
    monkeypatch.setattr(manager, "_async_execute_due", execute_due)

    await manager._async_wait_and_execute(record.call_id)

    execute_due.assert_awaited_once_with(record.call_id)
    assert record.call_id not in manager._tasks


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (None, "config entry is unavailable"),
        (
            SimpleNamespace(disabled_by="user", subentries={}),
            "config entry is unavailable",
        ),
        (
            SimpleNamespace(disabled_by=None, subentries={}),
            "conversation agent is unavailable",
        ),
        (
            SimpleNamespace(
                disabled_by=None,
                subentries={"agent": SimpleNamespace(subentry_type="other", data={})},
            ),
            "conversation agent is unavailable",
        ),
    ],
)
async def test_due_call_discards_when_live_agent_configuration_disappears(
    hass, monkeypatch, entry, reason: str
) -> None:
    """Persisted work is re-authorized against the live config-entry topology."""
    manager = DelayedToolManager(hass)
    record = _coverage_record()
    manager._records = {record.call_id: record}
    hass.config_entries.async_get_entry = MagicMock(return_value=entry)
    discard = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_async_discard", discard)

    assert await manager._async_execute_due(record.call_id) is False
    discard.assert_awaited_once_with(record.call_id, reason)


async def test_due_call_discards_invalid_live_tool_configuration(
    hass, monkeypatch
) -> None:
    """A now-invalid Function Tool config cannot execute stale persisted arguments."""
    manager = DelayedToolManager(hass)
    record = _coverage_record()
    manager._records = {record.call_id: record}
    hass.config_entries.async_get_entry = MagicMock(return_value=_live_entry())
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        MagicMock(side_effect=ValueError("bad config")),
    )
    discard = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_async_discard", discard)

    assert await manager._async_execute_due(record.call_id) is False
    discard.assert_awaited_once_with(
        record.call_id, "live Function Tool configuration is invalid"
    )


async def test_due_call_discards_tool_that_is_no_longer_delay_eligible(
    hass, monkeypatch
) -> None:
    """A delayed call cannot become an HA-owned LLM tool after scheduling."""
    manager = DelayedToolManager(hass)
    record = _coverage_record()
    manager._records = {record.call_id: record}
    hass.config_entries.async_get_entry = MagicMock(return_value=_live_entry())
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [_valid_tool(function_type="ha_llm")],
    )
    discard = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_async_discard", discard)

    assert await manager._async_execute_due(record.call_id) is False
    discard.assert_awaited_once_with(
        record.call_id, "Function Tool was removed or disabled"
    )


async def test_due_call_discards_inactive_originating_user(hass, monkeypatch) -> None:
    """A revoked user identity invalidates a delayed action before agent resolution."""
    manager = DelayedToolManager(hass)
    record = _coverage_record()
    manager._records = {record.call_id: record}
    hass.config_entries.async_get_entry = MagicMock(return_value=_live_entry())
    hass.auth.async_get_user = AsyncMock(return_value=SimpleNamespace(is_active=False))
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [_valid_tool()],
    )
    resolve_agent = MagicMock()
    monkeypatch.setattr(manager, "_resolve_agent", resolve_agent)
    discard = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_async_discard", discard)

    assert await manager._async_execute_due(record.call_id) is False
    discard.assert_awaited_once_with(
        record.call_id, "originating user is no longer active"
    )
    resolve_agent.assert_not_called()


async def test_due_call_retries_when_agent_is_temporarily_missing(
    hass, monkeypatch
) -> None:
    """A transient entity reload persists a retry count instead of losing the call."""
    manager = DelayedToolManager(hass)
    record = _coverage_record(user_id=None)
    manager._records = {record.call_id: record}
    manager._store = SimpleNamespace(async_save=AsyncMock())
    hass.config_entries.async_get_entry = MagicMock(return_value=_live_entry())
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [_valid_tool()],
    )
    monkeypatch.setattr(manager, "_resolve_agent", lambda *_args: None)

    assert await manager._async_execute_due(record.call_id) is True
    assert manager._records[record.call_id].retry_count == 1
    manager._store.async_save.assert_awaited_once()


async def test_retry_limit_discards_and_retry_save_failure_still_retries(
    hass, monkeypatch
) -> None:
    """Retry exhaustion cancels; a transient save failure remains retryable."""
    manager = DelayedToolManager(hass)
    exhausted = _coverage_record(retry_count=_MAX_AGENT_RETRIES)
    discard = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_async_discard", discard)

    assert await manager._async_retry_agent(exhausted) is False
    discard.assert_awaited_once_with(
        exhausted.call_id, "conversation agent did not become available"
    )

    replace_record = AsyncMock(side_effect=OSError("storage unavailable"))
    monkeypatch.setattr(manager, "_async_replace_record", replace_record)
    assert await manager._async_retry_agent(_coverage_record()) is True
    replace_record.assert_awaited_once()


async def test_due_call_discards_when_live_tool_resolution_fails(
    hass, monkeypatch
) -> None:
    """Runtime tool-resolution errors cancel stale work instead of executing it."""
    manager = DelayedToolManager(hass)
    record = _coverage_record(user_id=None)
    manager._records = {record.call_id: record}
    hass.config_entries.async_get_entry = MagicMock(return_value=_live_entry())
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.configured_function_tools_from_data",
        lambda _data: [_valid_tool()],
    )
    monkeypatch.setattr(manager, "_resolve_agent", lambda *_args: object())
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.latest_function_tool_for_execution",
        MagicMock(side_effect=HomeAssistantError("gone")),
    )
    discard = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, "_async_discard", discard)

    assert await manager._async_execute_due(record.call_id) is False
    discard.assert_awaited_once_with(
        record.call_id, "Function Tool is unavailable: gone"
    )


def test_resolve_agent_filters_registry_and_returns_live_agent(
    hass, monkeypatch
) -> None:
    """Agent lookup ignores unrelated registry entries and returns only the live match."""
    manager = DelayedToolManager(hass)
    unrelated = SimpleNamespace(
        platform="other",
        domain=conversation.DOMAIN,
        config_entry_id="entry",
        config_subentry_id="agent",
        entity_id="conversation.other",
    )
    matching = SimpleNamespace(
        platform=DOMAIN,
        domain=conversation.DOMAIN,
        config_entry_id="entry",
        config_subentry_id="agent",
        entity_id="conversation.agent",
    )
    registry = SimpleNamespace(entities={"one": unrelated, "two": matching})
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.er.async_get",
        lambda _hass: registry,
    )
    agent = object()
    get_agent = MagicMock(return_value=agent)
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.conversation.async_get_agent",
        get_agent,
    )

    assert manager._resolve_agent("entry", "agent") is agent
    get_agent.assert_called_once_with(hass, "conversation.agent")

    get_agent.return_value = None
    assert manager._resolve_agent("entry", "agent") is None


async def test_record_storage_helpers_preserve_durability_on_failures(hass) -> None:
    """Replace/discard/finalize handle absent records and failed persistence safely."""
    manager = DelayedToolManager(hass)
    manager._store = SimpleNamespace(async_save=AsyncMock())

    await manager._async_replace_record(_coverage_record(call_id="missing"))
    manager._store.async_save.assert_not_awaited()
    assert await manager._async_discard("missing", "gone") is True
    await manager._async_finalize("missing")

    record = _coverage_record()
    manager._records = {record.call_id: record}
    manager._store.async_save.side_effect = OSError("storage unavailable")

    assert await manager._async_discard(record.call_id, "cancel") is False
    assert record.call_id not in manager._records
    assert not manager._setup_complete

    executing = DelayedToolCall.from_dict({**record.as_dict(), "status": _EXECUTING})
    manager._records = {record.call_id: executing}
    await manager._async_finalize(record.call_id)
    assert record.call_id not in manager._records


async def test_shared_setup_reuses_manager_without_replacing_executor(
    hass, monkeypatch
) -> None:
    """Integration setup owns one manager and preserves the source-owned executor."""
    setup = AsyncMock()
    monkeypatch.setattr(DelayedToolManager, "async_setup", setup)
    original_method = ExtendedOpenAIBaseLLMEntity._execute_function_tool
    hass.data.setdefault(DOMAIN, {}).pop(DATA_DELAYED_TOOL_MANAGER, None)
    created = await async_setup_delayed_tools(hass)
    assert hass.data[DOMAIN][DATA_DELAYED_TOOL_MANAGER] is created
    setup.assert_awaited_once_with()
    assert ExtendedOpenAIBaseLLMEntity._execute_function_tool is original_method

    setup.reset_mock()
    reused = await async_setup_delayed_tools(hass)
    assert reused is created
    setup.assert_awaited_once_with()
    assert ExtendedOpenAIBaseLLMEntity._execute_function_tool is original_method


async def test_owned_executor_blocks_ha_llm_replay_and_delegates_live_ha_llm(
    hass, monkeypatch
) -> None:
    """HA-owned tools delegate normally but are forbidden inside delayed replay."""
    executor = ExtendedOpenAIBaseLLMEntity._execute_function_tool
    assert ExtendedOpenAIBaseLLMEntity._execute_function_tool is executor

    entity = SimpleNamespace(hass=hass)
    function_tool = {"function": {"type": "ha_llm"}, "spec": {"name": "ha"}}
    tool_input = llm.ToolInput(id="call", tool_name="ha", tool_args={}, external=True)

    delayed_context = SimpleNamespace(**{_DELAYED_EXECUTION_MARKER: True})
    with pytest.raises(
        HomeAssistantError, match="cannot execute in the delayed scheduler"
    ):
        await executor(entity, function_tool, tool_input, delayed_context, [])


async def test_owned_executor_executes_native_replay_directly(
    hass, monkeypatch
) -> None:
    """A replay-marked native call executes once without scheduling itself again."""
    validate = AsyncMock(return_value={"delay": {"seconds": 10}, "value": 7})
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.entity.async_execution_arguments",
        validate,
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.entity.split_legacy_execution_delay",
        lambda _spec, _args: ({"value": 7}, timedelta(seconds=10)),
    )
    function = SimpleNamespace(execute=AsyncMock(return_value="done"))
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.entity.get_function",
        lambda _type: function,
    )
    executor = ExtendedOpenAIBaseLLMEntity._execute_function_tool
    entity = SimpleNamespace(hass=hass, entity_id="conversation.agent")
    tool = {
        "spec": {"name": "control_light"},
        "function": {"type": "native", "name": "execute_service_single"},
    }
    tool_input = llm.ToolInput(
        id="call", tool_name="control_light", tool_args={"value": 7}, external=True
    )
    context = SimpleNamespace(**{_DELAYED_EXECUTION_MARKER: True})

    result = await executor(entity, tool, tool_input, context, [])

    function.execute.assert_awaited_once_with(
        hass,
        tool["function"],
        {"value": 7},
        context,
        [],
    )
    assert tool_result_data(result) == {"result": "done"}


async def test_owned_executor_requires_scheduler_for_background_call(
    hass, monkeypatch
) -> None:
    """A background-eligible call fails closed if the durable manager is absent."""
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.entity.async_execution_arguments",
        AsyncMock(return_value={"delay": {"seconds": 10}}),
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.entity.split_legacy_execution_delay",
        lambda _spec, _args: ({}, timedelta(seconds=10)),
    )
    executor = ExtendedOpenAIBaseLLMEntity._execute_function_tool
    hass.data.setdefault(DOMAIN, {}).pop(DATA_DELAYED_TOOL_MANAGER, None)
    entity = SimpleNamespace(
        hass=hass,
        entity_id="conversation.agent",
        should_run_in_background=lambda _delay: True,
    )
    tool = {
        "spec": {"name": "control_light"},
        "function": {"type": "native", "name": "execute_service_single"},
    }
    tool_input = llm.ToolInput(
        id="call", tool_name="control_light", tool_args={}, external=True
    )

    with pytest.raises(HomeAssistantError, match="scheduler is unavailable"):
        await executor(entity, tool, tool_input, None, [])


async def test_owned_executor_schedules_background_call_and_returns_receipt(
    hass, monkeypatch
) -> None:
    """Background routing persists the validated call and returns a scheduled receipt."""
    arguments = {"delay": {"seconds": 10}, "value": 7}
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.entity.async_execution_arguments",
        AsyncMock(return_value=arguments),
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.entity.split_legacy_execution_delay",
        lambda _spec, _args: ({"value": 7}, timedelta(seconds=10)),
    )
    executor = ExtendedOpenAIBaseLLMEntity._execute_function_tool
    manager = DelayedToolManager(hass)
    manager.async_schedule = AsyncMock(return_value=_coverage_record())
    hass.data.setdefault(DOMAIN, {})[DATA_DELAYED_TOOL_MANAGER] = manager
    entity = SimpleNamespace(
        hass=hass,
        entity_id="conversation.agent",
        should_run_in_background=lambda _delay: True,
    )
    tool = {
        "spec": {"name": "control_light"},
        "function": {"type": "native", "name": "execute_service_single"},
    }
    tool_input = llm.ToolInput(
        id="call", tool_name="control_light", tool_args={"value": 7}, external=True
    )
    context = SimpleNamespace(context=Context(user_id="user-1"), device_id="device-1")

    result = await executor(entity, tool, tool_input, context, [])

    manager.async_schedule.assert_awaited_once_with(
        entity, "control_light", arguments, context
    )
    assert tool_result_data(result) == {"result": "Scheduled"}


def _race_record(*, due_at: str | None = None) -> DelayedToolCall:
    now = dt_util.utcnow()
    return DelayedToolCall(
        call_id="call-1",
        entry_id="entry",
        subentry_id="agent",
        tool_name="control_light",
        arguments={"delay": {"seconds": 5}, "value": 1},
        due_at=due_at or (now - timedelta(seconds=1)).isoformat(),
        created_at=(now - timedelta(seconds=6)).isoformat(),
        user_id="user-1",
        device_id="device-1",
    )


async def test_concurrent_setup_loads_and_registers_once(hass) -> None:
    """A second setup waiter must observe completion instead of loading twice."""
    manager = DelayedToolManager(hass)
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    load_count = 0

    async def async_load():
        nonlocal load_count
        load_count += 1
        load_started.set()
        await release_load.wait()
        return {"calls": []}

    manager._store = SimpleNamespace(
        async_load=async_load,
        async_save=AsyncMock(),
    )

    first = asyncio.create_task(manager.async_setup())
    await load_started.wait()
    second = asyncio.create_task(manager.async_setup())
    await asyncio.sleep(0)

    release_load.set()
    await asyncio.gather(first, second)

    assert load_count == 1
    assert manager._setup_complete is True


async def test_arm_does_not_duplicate_an_active_waiter(hass, monkeypatch) -> None:
    """Repeated arming of one call must not create duplicate execution waiters."""
    manager = DelayedToolManager(hass)
    record = _race_record()
    manager._records = {record.call_id: record}
    started = asyncio.Event()
    release = asyncio.Event()
    invocations = 0

    async def waiter(call_id: str) -> None:
        nonlocal invocations
        assert call_id == record.call_id
        invocations += 1
        started.set()
        await release.wait()

    monkeypatch.setattr(manager, "_async_wait_and_execute", waiter)

    manager._arm(record.call_id)
    first_task = manager._tasks[record.call_id]
    await started.wait()

    manager._arm(record.call_id)

    assert manager._tasks[record.call_id] is first_task
    assert invocations == 1

    release.set()
    await first_task
    manager._tasks.pop(record.call_id, None)


async def test_waiter_does_not_execute_record_removed_while_waiting(
    hass, monkeypatch
) -> None:
    """A call removed during its due-time wait must never execute afterward."""
    manager = DelayedToolManager(hass)
    record = _race_record()
    manager._records = {record.call_id: record}
    manager._started = True
    execute_due = AsyncMock()
    monkeypatch.setattr(manager, "_async_execute_due", execute_due)

    async def remove_during_sleep(_delay: float) -> None:
        manager._records.pop(record.call_id, None)

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.asyncio.sleep",
        remove_during_sleep,
    )

    await manager._async_wait_and_execute(record.call_id)

    execute_due.assert_not_awaited()
    assert record.call_id not in manager._tasks


async def test_invalid_due_timestamp_retries_when_discard_cannot_persist(
    hass, monkeypatch
) -> None:
    """A failed discard is retried rather than executing a malformed persisted call."""
    manager = DelayedToolManager(hass)
    record = _race_record(due_at="not-a-timestamp")
    manager._records = {record.call_id: record}
    manager._started = True
    discard = AsyncMock(side_effect=[False, True])
    execute_due = AsyncMock()
    monkeypatch.setattr(manager, "_async_discard", discard)
    monkeypatch.setattr(manager, "_async_execute_due", execute_due)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.asyncio.sleep",
        fake_sleep,
    )

    await manager._async_wait_and_execute(record.call_id)

    assert discard.await_count == 2
    assert sleeps == [_AGENT_RETRY_SECONDS]
    execute_due.assert_not_awaited()


async def test_transient_execution_failure_retries_same_pending_call(
    hass, monkeypatch
) -> None:
    """A retryable due-call failure must loop instead of abandoning the call."""
    manager = DelayedToolManager(hass)
    record = _race_record()
    manager._records = {record.call_id: record}
    manager._started = True
    execute_due = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr(manager, "_async_execute_due", execute_due)
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    class Gate:
        def shared(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.asyncio.sleep",
        fake_sleep,
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.get_agent_maintenance_gate",
        lambda *_args: Gate(),
    )

    await manager._async_wait_and_execute(record.call_id)

    assert execute_due.await_count == 2
    assert _AGENT_RETRY_SECONDS in sleeps


async def test_waiter_cancellation_propagates_and_removes_task(hass, monkeypatch) -> None:
    """Stopping a pending waiter must not leave a stale task registration behind."""
    manager = DelayedToolManager(hass)
    record = _race_record()
    manager._records = {record.call_id: record}
    manager._started = True
    sleeping = asyncio.Event()
    never_release = asyncio.Event()

    async def blocking_sleep(_delay: float) -> None:
        sleeping.set()
        await never_release.wait()

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.delayed_tools.asyncio.sleep",
        blocking_sleep,
    )

    task = asyncio.create_task(manager._async_wait_and_execute(record.call_id))
    manager._tasks[record.call_id] = task
    await sleeping.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert record.call_id not in manager._tasks
