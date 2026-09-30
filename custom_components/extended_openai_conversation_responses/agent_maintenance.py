"""Per-agent quiescence barrier for multi-store maintenance operations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, cast

from homeassistant.core import HomeAssistant

from .const import DOMAIN

_AGENT_MAINTENANCE_GATES = f"{DOMAIN}.agent_maintenance_gates"


class AgentMaintenanceGate:
    """Writer-preferring reader/writer gate for one conversation agent."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_readers = 0
        self._reader_owner: ContextVar[object | None] = ContextVar(
            f"extended_openai_maintenance_reader_{id(self)}", default=None
        )
        self._reader_depth: dict[object, int] = {}
        self._waiting_writers = 0
        self._writer_active = False

    @asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        """Enter ordinary agent work, preserving a logical lease across child tasks."""
        owner = self._reader_owner.get()
        owner_token = None

        async with self._condition:
            depth = self._reader_depth.get(owner, 0) if owner is not None else 0
            if depth:
                # Context variables propagate into child tasks created by Home
                # Assistant Script. Treat those descendants as part of the same
                # logical request so a waiting writer cannot deadlock on its parent.
                self._reader_depth[owner] = depth + 1
            else:
                await self._condition.wait_for(
                    lambda: not self._writer_active and self._waiting_writers == 0
                )
                owner = object()
                owner_token = self._reader_owner.set(owner)
                self._reader_depth[owner] = 1
                self._active_readers += 1
        try:
            yield
        finally:
            if owner is None:  # pragma: no cover - every admitted reader owns a token
                raise RuntimeError("Agent maintenance reader lease was not established")
            async with self._condition:
                depth = self._reader_depth[owner]
                if depth > 1:
                    self._reader_depth[owner] = depth - 1
                else:
                    del self._reader_depth[owner]
                    self._active_readers -= 1
                    if self._active_readers == 0:
                        self._condition.notify_all()
            if owner_token is not None:
                self._reader_owner.reset(owner_token)

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Drain readers and exclude new work for one maintenance operation."""
        acquired = False
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writer_active and self._active_readers == 0
                )
                self._writer_active = True
                acquired = True
            finally:
                self._waiting_writers -= 1
                if not acquired:
                    # A cancelled writer must not strand readers behind its waiter bit.
                    self._condition.notify_all()
        try:
            yield
        finally:
            async with self._condition:
                self._writer_active = False
                self._condition.notify_all()


def get_agent_maintenance_gate(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> AgentMaintenanceGate:
    """Return the process-local gate for one exact agent."""
    gates = cast(
        dict[tuple[str, str], AgentMaintenanceGate],
        hass.data.setdefault(_AGENT_MAINTENANCE_GATES, {}),
    )
    return gates.setdefault((entry_id, subentry_id), AgentMaintenanceGate())


async def _async_run_exclusive_operation[T](
    gate: AgentMaintenanceGate,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Finish one exclusive operation before propagating caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    async with gate.exclusive():
        task: asyncio.Future[T] = asyncio.ensure_future(operation())
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as err:
                if task.cancelled():
                    raise
                if cancellation is None:
                    cancellation = err
            except Exception:
                # The operation has reached an outcome. Inspect it below so its
                # deterministic restore/rollback error takes precedence over a
                # concurrent caller cancellation.
                break

        result = task.result()

    # Release exclusivity before surfacing deferred cancellation so the next
    # ordinary operation cannot observe a permanently wedged maintenance gate.
    if cancellation is not None:
        raise cancellation
    return result


@asynccontextmanager
async def conversation_request_lease(entity: Any) -> AsyncIterator[None]:
    """Hold one agent lease across preparation, processing and result capture."""
    # Retain support for focused tests of an unregistered/partial entity. Every
    # registered conversation entity supplies these three identity fields.
    entry = getattr(entity, "entry", None)
    subentry = getattr(entity, "subentry", None)
    hass = getattr(entity, "hass", None)
    entry_id = getattr(entry, "entry_id", None)
    subentry_id = getattr(subentry, "subentry_id", None)
    if (
        hass is None
        or not isinstance(entry_id, str)
        or not isinstance(subentry_id, str)
    ):
        yield
        return
    async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
        yield


def _management_operation_owns_its_gate(message: dict[str, Any]) -> bool:
    """Avoid nesting around paths that enter a guarded operation in another task."""
    section = message.get("section", "overview")
    action = message.get("action")
    return section == "backup" or (
        (section == "request_rules" and action == "test")
        or (section == "diagnostics" and action == "test_agent")
    )


@asynccontextmanager
async def management_command_lease(
    hass: HomeAssistant,
    message: dict[str, Any],
) -> AsyncIterator[None]:
    """Hold the original Management lease across validation, dispatch and projection.

    Backup, provider diagnostics and live Request Rules tests own their gates
    inside the normal execution pipeline.
    """
    entry_id = message.get("entry_id")
    subentry_id = message.get("subentry_id")
    if (
        message.get("action") == "agents"
        or _management_operation_owns_its_gate(message)
        or not isinstance(entry_id, str)
        or not isinstance(subentry_id, str)
    ):
        yield
        return
    async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
        yield
