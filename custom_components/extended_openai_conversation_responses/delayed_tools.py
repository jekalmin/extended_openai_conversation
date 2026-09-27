"""Durable scheduling for delayed configured Function Tools."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import logging
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Context, CoreState, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_registry as er, llm
from homeassistant.util import dt as dt_util

from .agent_config import configured_function_tools_from_data, function_tool_enabled
from .agent_maintenance import get_agent_maintenance_gate
from .const import DOMAIN
from .function_tool_resolution import latest_function_tool_for_execution
from .ha_permissions import bind_active_ha_context
from .helpers import get_exposed_entities
from .persistence_hardening import _async_repair_private_store_mode
from .strict_store import PropagatingWriteStore

_LOGGER = logging.getLogger(__name__)

DATA_DELAYED_TOOL_MANAGER = "delayed_tool_manager"
DELAYED_TOOL_STORAGE_VERSION = 1
DELAYED_TOOL_STORAGE_KEY = f"{DOMAIN}.delayed_tools"
_AGENT_RETRY_SECONDS = 30
_MAX_AGENT_RETRIES = 10
_PENDING = "pending"
_EXECUTING = "executing"
_DELAYED_EXECUTION_MARKER = "_extended_openai_delayed_execution"


@dataclass(slots=True, frozen=True)
class DelayedToolCall:
    """Persisted delayed Function Tool invocation."""

    call_id: str
    entry_id: str
    subentry_id: str
    tool_name: str
    arguments: dict[str, Any]
    due_at: str
    created_at: str
    user_id: str | None = None
    device_id: str | None = None
    status: str = _PENDING
    retry_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialize the call for Home Assistant storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> DelayedToolCall:
        """Validate and deserialize one stored call."""
        if not isinstance(raw, dict):
            raise ValueError("stored delayed call is not an object")
        required_strings = (
            "call_id",
            "entry_id",
            "subentry_id",
            "tool_name",
            "due_at",
            "created_at",
        )
        for key in required_strings:
            if not isinstance(raw.get(key), str) or not raw[key]:
                raise ValueError(f"stored delayed call has invalid {key}")
        if not isinstance(raw.get("arguments"), dict):
            raise ValueError("stored delayed call has invalid arguments")
        status = raw.get("status", _PENDING)
        if status not in {_PENDING, _EXECUTING}:
            raise ValueError("stored delayed call has invalid status")
        retry_count = raw.get("retry_count", 0)
        if not isinstance(retry_count, int) or retry_count < 0:
            raise ValueError("stored delayed call has invalid retry_count")
        for timestamp_key in ("due_at", "created_at"):
            parsed = dt_util.parse_datetime(raw[timestamp_key])
            if parsed is None:
                raise ValueError(
                    f"stored delayed call has invalid {timestamp_key} timestamp"
                )
        user_id = raw.get("user_id")
        device_id = raw.get("device_id")
        if user_id is not None and not isinstance(user_id, str):
            raise ValueError("stored delayed call has invalid user_id")
        if device_id is not None and not isinstance(device_id, str):
            raise ValueError("stored delayed call has invalid device_id")
        return cls(
            call_id=raw["call_id"],
            entry_id=raw["entry_id"],
            subentry_id=raw["subentry_id"],
            tool_name=raw["tool_name"],
            arguments=deepcopy(raw["arguments"]),
            due_at=raw["due_at"],
            created_at=raw["created_at"],
            user_id=user_id,
            device_id=device_id,
            status=status,
            retry_count=retry_count,
        )


def _delay_as_timedelta(value: Any) -> timedelta:
    """Normalize the same time-period shapes accepted by Home Assistant scripts."""
    try:
        delay = cast(timedelta, cv.time_period(value))
    except (TypeError, ValueError, vol.Invalid) as err:
        raise HomeAssistantError(f"Invalid Function Tool delay: {err}") from err
    if delay.total_seconds() < 0:
        raise HomeAssistantError("Function Tool delay cannot be negative")
    return delay


class DelayedToolManager:
    """Persist, recover, and execute delayed configured Function Tools."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the scheduler."""
        self.hass = hass
        self._store = PropagatingWriteStore(
            hass,
            DELAYED_TOOL_STORAGE_VERSION,
            DELAYED_TOOL_STORAGE_KEY,
            private=True,
            atomic_writes=True,
        )
        self._records: dict[str, DelayedToolCall] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._setup_lock = asyncio.Lock()
        self._started = False
        self._setup_complete = False

    async def async_setup(self) -> None:
        """Load persisted calls and arm recovery once Home Assistant is running."""
        await _async_repair_private_store_mode(self._store)
        if self._setup_complete:
            return
        async with self._setup_lock:
            if self._setup_complete:
                return

            raw_data = await self._store.async_load() or {}
            raw_calls = raw_data.get("calls", []) if isinstance(raw_data, dict) else []
            dirty = not isinstance(raw_calls, list)
            recovered: dict[str, DelayedToolCall] = {}
            for raw in raw_calls if isinstance(raw_calls, list) else []:
                try:
                    record = DelayedToolCall.from_dict(raw)
                except ValueError as err:
                    dirty = True
                    _LOGGER.warning(
                        "Ignoring invalid persisted delayed Function Tool: %s", err
                    )
                    continue
                if record.status == _EXECUTING:
                    # A previous process persisted the execution boundary before
                    # invoking the tool. Replaying it could duplicate a side effect.
                    dirty = True
                    _LOGGER.warning(
                        "Not replaying interrupted delayed Function Tool `%s`; its "
                        "prior execution outcome is indeterminate",
                        record.tool_name,
                    )
                    continue
                recovered[record.call_id] = record

            if dirty:
                await self._store.async_save(self._storage_payload(recovered))
            self._records = recovered

            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, self._handle_stop)
            running = self.hass.state == CoreState.running
            if not running:
                self.hass.bus.async_listen_once(
                    EVENT_HOMEASSISTANT_STARTED, self._handle_started
                )
            self._setup_complete = True
            if running:
                self._handle_started()

    async def async_schedule(
        self,
        entity: Any,
        tool_name: str,
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
    ) -> DelayedToolCall:
        """Persist a delayed call before exposing it as scheduled."""
        if not self._setup_complete:
            raise HomeAssistantError("Delayed Function Tool scheduler is unavailable")
        delay = _delay_as_timedelta(arguments.get("delay"))
        now = dt_util.utcnow()
        context = getattr(llm_context, "context", None)
        record = DelayedToolCall(
            call_id=uuid4().hex,
            entry_id=entity.entry.entry_id,
            subentry_id=entity.subentry.subentry_id,
            tool_name=tool_name,
            arguments=deepcopy(arguments),
            due_at=(now + delay).isoformat(),
            created_at=now.isoformat(),
            user_id=getattr(context, "user_id", None),
            device_id=getattr(llm_context, "device_id", None),
        )
        async with self._lock:
            updated = dict(self._records)
            updated[record.call_id] = record
            await self._store.async_save(self._storage_payload(updated))
            self._records = updated
        if self._started:
            self._arm(record.call_id)
        return record

    @callback
    def _handle_started(self, _event: Any = None) -> None:
        """Arm all pending calls after startup or an in-process setup."""
        if self._started:
            return
        self._started = True
        for call_id in tuple(self._records):
            self._arm(call_id)

    @callback
    def _handle_stop(self, _event: Any = None) -> None:
        """Cancel only calls that have not crossed the persisted execution boundary."""
        self._started = False
        for call_id, task in tuple(self._tasks.items()):
            record = self._records.get(call_id)
            if record is None or record.status == _PENDING:
                task.cancel()

    def _arm(self, call_id: str) -> None:
        """Create one waiter for a pending persisted call."""
        existing = self._tasks.get(call_id)
        if existing is not None and not existing.done():
            return
        record = self._records.get(call_id)
        if record is None or record.status != _PENDING:
            return
        self._tasks[call_id] = asyncio.create_task(
            self._async_wait_and_execute(call_id),
            name=f"{DOMAIN}-delayed-{call_id[:8]}",
        )

    async def _async_wait_and_execute(self, call_id: str) -> None:
        """Wait until due, retry transient state failures, then execute once."""
        try:
            while self._started:
                record = self._records.get(call_id)
                if record is None or record.status != _PENDING:
                    return
                due_at = dt_util.parse_datetime(record.due_at)
                if due_at is None:
                    if await self._async_discard(call_id, "invalid due timestamp"):
                        return
                    await asyncio.sleep(_AGENT_RETRY_SECONDS)
                    continue
                await asyncio.sleep(
                    max(
                        0.0, (dt_util.as_utc(due_at) - dt_util.utcnow()).total_seconds()
                    )
                )
                record = self._records.get(call_id)
                if record is None or record.status != _PENDING:
                    return
                gate = get_agent_maintenance_gate(
                    self.hass, record.entry_id, record.subentry_id
                )
                # Live resolution, the durable execution boundary, the external tool
                # action and finalization form one ordinary-agent lease. Restore must
                # not overlap any of them.
                async with gate.shared():
                    retry = await self._async_execute_due(call_id)
                if not retry:
                    return
                await asyncio.sleep(_AGENT_RETRY_SECONDS)
        except asyncio.CancelledError:
            raise
        finally:
            self._tasks.pop(call_id, None)

    async def _async_execute_due(self, call_id: str) -> bool:
        """Re-authorize a due call against live configuration and execute it once."""
        record = self._records.get(call_id)
        if record is None or record.status != _PENDING:
            return False

        with bind_active_ha_context(Context(user_id=record.user_id)):
            entry = self.hass.config_entries.async_get_entry(record.entry_id)
            if entry is None or entry.disabled_by is not None:
                return not await self._async_discard(
                    call_id, "config entry is unavailable"
                )
            subentry = entry.subentries.get(record.subentry_id)
            if subentry is None or subentry.subentry_type != "conversation":
                return not await self._async_discard(
                    call_id, "conversation agent is unavailable"
                )

            try:
                current_tools = configured_function_tools_from_data(subentry.data)
            except Exception:
                _LOGGER.exception(
                    "Unable to validate live Function Tools for delayed `%s`",
                    record.tool_name,
                )
                return not await self._async_discard(
                    call_id, "live Function Tool configuration is invalid"
                )
            current_tool = next(
                (
                    tool
                    for tool in current_tools
                    if tool.get("spec", {}).get("name") == record.tool_name
                ),
                None,
            )
            if (
                current_tool is None
                or not function_tool_enabled(current_tool)
                or current_tool.get("function", {}).get("type") == "ha_llm"
            ):
                return not await self._async_discard(
                    call_id, "Function Tool was removed or disabled"
                )

            if record.user_id is not None:
                user = await self.hass.auth.async_get_user(record.user_id)
                if user is None or getattr(user, "is_active", True) is not True:
                    return not await self._async_discard(
                        call_id, "originating user is no longer active"
                    )

            agent = self._resolve_agent(record.entry_id, record.subentry_id)
            if agent is None:
                return await self._async_retry_agent(record)

            try:
                current_tool = latest_function_tool_for_execution(agent, current_tool)
            except HomeAssistantError as err:
                return not await self._async_discard(
                    call_id, f"Function Tool is unavailable: {err}"
                )

            # A waiter armed from an Assist request inherits that request's ContextVars.
            # Delayed Function Tools deliberately authorize at execution time, so discard
            # only the pinned request Guest policy before entering the normal tool seam.
            # The live agent then resolves the Guest policy that exists when the call is
            # actually due, matching calls recovered after a Home Assistant restart.
            from .conversation import _ACTIVE_GUEST_POLICY

            executing = replace(record, status=_EXECUTING)
            try:
                await self._async_replace_record(executing)
            except Exception:
                _LOGGER.exception(
                    "Unable to persist execution boundary for delayed Function Tool `%s`",
                    record.tool_name,
                )
                return True

            execution_context = Context(user_id=record.user_id)
            delayed_context = SimpleNamespace(
                context=execution_context,
                device_id=record.device_id,
                **{_DELAYED_EXECUTION_MARKER: True},
            )
            guest_policy_token = _ACTIVE_GUEST_POLICY.set(None)
            try:
                with bind_active_ha_context(execution_context):
                    await agent._execute_function_tool(
                        current_tool,
                        llm.ToolInput(
                            id=record.call_id,
                            tool_name=record.tool_name,
                            tool_args=deepcopy(record.arguments),
                            external=True,
                        ),
                        delayed_context,
                        get_exposed_entities(self.hass),
                    )
            except Exception:
                _LOGGER.exception(
                    "Delayed Function Tool `%s` failed during execution",
                    record.tool_name,
                )
            else:
                _LOGGER.info("Executed delayed Function Tool `%s`", record.tool_name)
            finally:
                _ACTIVE_GUEST_POLICY.reset(guest_policy_token)
                await self._async_finalize(call_id)
            return False

    async def _async_retry_agent(self, record: DelayedToolCall) -> bool:
        """Retry a transient agent reload without losing the persisted call."""
        if record.retry_count >= _MAX_AGENT_RETRIES:
            return not await self._async_discard(
                record.call_id, "conversation agent did not become available"
            )
        updated = replace(record, retry_count=record.retry_count + 1)
        try:
            await self._async_replace_record(updated)
        except Exception:
            _LOGGER.exception(
                "Unable to persist retry state for delayed Function Tool `%s`",
                record.tool_name,
            )
        # Advance the live safety budget even if its durable update failed.
        current_record = self._records.get(record.call_id)
        if (
            current_record is not None
            and current_record.status == record.status
            and current_record.retry_count == record.retry_count
        ):
            self._records[record.call_id] = updated
        return True

    def _resolve_agent(self, entry_id: str, subentry_id: str) -> Any | None:
        """Resolve the live conversation entity without retaining stale objects."""
        registry = er.async_get(self.hass)
        for registry_entry in registry.entities.values():
            if (
                registry_entry.platform == DOMAIN
                and registry_entry.domain == conversation.DOMAIN
                and registry_entry.config_entry_id == entry_id
                and registry_entry.config_subentry_id == subentry_id
            ):
                agent = conversation.async_get_agent(
                    self.hass, registry_entry.entity_id
                )
                if agent is not None:
                    return agent
        return None

    async def _async_replace_record(self, record: DelayedToolCall) -> None:
        """Persist one state transition before updating live state."""
        async with self._lock:
            if record.call_id not in self._records:
                return
            updated = dict(self._records)
            updated[record.call_id] = record
            await self._store.async_save(self._storage_payload(updated))
            self._records = updated

    async def _async_discard(self, call_id: str, reason: str) -> bool:
        """Persist cancellation, returning whether the pending call was removed."""
        record = self._records.get(call_id)
        if record is None:
            return True
        try:
            async with self._lock:
                updated = dict(self._records)
                updated.pop(call_id, None)
                await self._store.async_save(self._storage_payload(updated))
                self._records = updated
        except Exception:
            _LOGGER.exception(
                "Unable to persist cancellation for delayed Function Tool `%s`",
                record.tool_name,
            )
            return False
        _LOGGER.info(
            "Cancelled delayed Function Tool `%s`: %s", record.tool_name, reason
        )
        return True

    async def _async_finalize(self, call_id: str) -> None:
        """Remove an executed call without ever making it eligible for replay again."""
        record = self._records.get(call_id)
        if record is None:
            return
        try:
            async with self._lock:
                updated = dict(self._records)
                updated.pop(call_id, None)
                await self._store.async_save(self._storage_payload(updated))
                self._records = updated
        except Exception:
            # The persisted record is already marked executing. Drop the live copy;
            # startup recovery also discards executing records, so this cannot replay.
            self._records.pop(call_id, None)
            _LOGGER.exception(
                "Unable to remove completed delayed Function Tool `%s` from storage; "
                "its executing tombstone will be discarded on next startup",
                record.tool_name,
            )

    @staticmethod
    def _storage_payload(
        records: dict[str, DelayedToolCall],
    ) -> dict[str, list[dict[str, Any]]]:
        return {"calls": [record.as_dict() for record in records.values()]}


async def async_setup_delayed_tools(hass: HomeAssistant) -> DelayedToolManager:
    """Set up the shared durable scheduler used by the base tool executor."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    existing = domain_data.get(DATA_DELAYED_TOOL_MANAGER)
    if isinstance(existing, DelayedToolManager):
        manager = existing
    else:
        manager = DelayedToolManager(hass)
        domain_data[DATA_DELAYED_TOOL_MANAGER] = manager
    await manager.async_setup()
    return manager
