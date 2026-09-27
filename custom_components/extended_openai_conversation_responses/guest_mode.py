"""Backend-enforced Guest Mode state and capability policy."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import logging
from typing import Any, cast

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    target as target_helpers,
)
from homeassistant.util import dt as dt_util

from .agent_config import CONF_GUEST_WEB_SEARCH, DEFAULT_GUEST_WEB_SEARCH
from .const import (
    CONF_FUNCTION_GROUPS,
    CONF_GUEST_ALLOWED_FUNCTION_NAMES,
    CONF_GUEST_ALLOWED_GROUP_IDS,
    CONF_GUEST_CONTROL_EXCLUDED_AREAS,
    CONF_GUEST_CONTROL_EXCLUDED_DOMAINS,
    CONF_GUEST_CONTROL_EXCLUDED_ENTITIES,
    CONF_GUEST_CONTROL_EXCLUDED_LABELS,
    CONF_GUEST_CONTROLLABLE_AREAS,
    CONF_GUEST_CONTROLLABLE_DOMAINS,
    CONF_GUEST_CONTROLLABLE_ENTITIES,
    CONF_GUEST_CONTROLLABLE_LABELS,
    CONF_GUEST_EXCLUDED_AREAS,
    CONF_GUEST_EXCLUDED_DOMAINS,
    CONF_GUEST_EXCLUDED_ENTITIES,
    CONF_GUEST_EXCLUDED_LABELS,
    CONF_GUEST_FUNCTION_POLICY,
    CONF_GUEST_KNOWLEDGE_ENABLED,
    CONF_GUEST_KNOWLEDGE_POLICY,
    CONF_GUEST_KNOWLEDGE_SOURCE_IDS,
    CONF_GUEST_MODE_ENABLED,
    CONF_GUEST_POLICY_VERSION,
    CONF_GUEST_READABLE_AREAS,
    CONF_GUEST_READABLE_DOMAINS,
    CONF_GUEST_READABLE_ENTITIES,
    CONF_GUEST_READABLE_LABELS,
    CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS,
    CONF_GUEST_SHARED_MEMORY_POLICY,
    CONF_GUEST_SHARED_MEMORY_READ,
    CONF_GUEST_SHARED_MEMORY_WRITE,
    DOMAIN,
    GUEST_POLICY_VERSION,
)
from .functions.security import FunctionSecurity, classify_tool
from .helpers import get_exposed_entities
from .strict_store import PropagatingWriteStore

_LOGGER = logging.getLogger(__name__)

GUEST_MODE_STORAGE_VERSION = 1
GUEST_MODE_STORAGE_PREFIX = f"{DOMAIN}.guest_mode"
GUEST_MODE_UNAVAILABLE = "This capability is unavailable in Guest Mode."
EXECUTION_FAILED = "The requested action could not be completed."
GUEST_MODE_PROMPT = """
## Guest Mode
Guest Mode is active. You have access only to capabilities explicitly permitted
for guests. Do not infer or request hidden owner or private information. If a
requested capability is unavailable, say briefly that it is unavailable in Guest
Mode. Do not reveal what the owner normally has access to.
"""

_MANAGERS = f"{DOMAIN}.guest_mode_managers"


class GuestModeDenied(HomeAssistantError):
    """Raised when Guest Mode policy denies an otherwise valid action."""


def guest_mode_denial_result() -> dict[str, str]:
    """Return the stable, model-visible Guest Mode denial result."""
    return {
        "status": "denied",
        "reason": "guest_mode",
        "error": GUEST_MODE_UNAVAILABLE,
    }


def execution_failure_result() -> dict[str, str]:
    """Return a generic result without leaking private exception details."""
    return {
        "status": "error",
        "reason": "execution_failed",
        "error": EXECUTION_FAILED,
    }


@dataclass(slots=True, frozen=True)
class GuestModeSchedule:
    """Persisted interval during which Guest Mode is active."""

    active_from: str
    active_until: str | None = None
    source: str = "home_assistant"
    updated_at: str | None = None


@dataclass(slots=True, frozen=True)
class GuestCapabilityPolicy:
    """Resolved capabilities held stable for one user request."""

    guest_active: bool
    readable_entity_ids: frozenset[str] | None = None
    controllable_entity_ids: frozenset[str] | None = None
    configured_tool_names: frozenset[str] | None = None
    knowledge_source_ids: frozenset[str] | None = None
    legacy_function_flags: bool = False
    personal_memory_read: bool = True
    personal_memory_write: bool = True
    shared_memory_read: bool = True
    shared_memory_write: bool = True
    archive_access: bool = True
    archive_retention: bool = True
    knowledge_access: bool = True
    temporary_memory: bool = True
    skills: bool = True
    web_search: bool = True
    private_capabilities: bool = True

    @classmethod
    def unrestricted(cls) -> GuestCapabilityPolicy:
        """Return the normal baseline policy."""
        return cls(False)

    def allows_entity_read(self, entity_id: str) -> bool:
        return self.readable_entity_ids is None or entity_id in self.readable_entity_ids

    def allows_entity_control(self, entity_id: str) -> bool:
        return (
            self.controllable_entity_ids is None
            or entity_id in self.controllable_entity_ids
        )

    def allows_configured_tool(self, name: str) -> bool:
        return self.configured_tool_names is None or name in self.configured_tool_names

    def as_diagnostics(self) -> dict[str, Any]:
        """Return counts and booleans without private identifiers."""
        return {
            "active": self.guest_active,
            "readable_entity_count": (
                None
                if self.readable_entity_ids is None
                else len(self.readable_entity_ids)
            ),
            "controllable_entity_count": (
                None
                if self.controllable_entity_ids is None
                else len(self.controllable_entity_ids)
            ),
            "configured_tool_count": (
                None
                if self.configured_tool_names is None
                else len(self.configured_tool_names)
            ),
            "memory": {
                "personal_read": self.personal_memory_read,
                "personal_write": self.personal_memory_write,
                "shared_read": self.shared_memory_read,
                "shared_write": self.shared_memory_write,
            },
            "archive_access": self.archive_access,
            "archive_retention": self.archive_retention,
            "knowledge_access": self.knowledge_access,
            "knowledge_source_count": (
                None
                if self.knowledge_source_ids is None
                else len(self.knowledge_source_ids)
            ),
            "temporary_memory": self.temporary_memory,
            "web_search": self.web_search,
        }


def can_reuse_request_policy(request_policy: Any, guest_mode: Any) -> bool:
    """Reuse a request policy only while a normal-owner request stays unrestricted."""
    return bool(
        request_policy is not None
        and not request_policy.guest_active
        and (guest_mode is None or not guest_mode.is_active())
    )


class GuestModeManager:
    """Integration-owned, per-agent Guest Mode schedule."""

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str) -> None:
        self.hass = hass
        self._store = PropagatingWriteStore(
            hass,
            GUEST_MODE_STORAGE_VERSION,
            f"{GUEST_MODE_STORAGE_PREFIX}.{entry_id}.{subentry_id}",
            private=True,
            atomic_writes=True,
            serialize_in_event_loop=False,
        )
        self._schedule: GuestModeSchedule | None = None
        self._listeners: set[Callable[[], None]] = set()
        self._mutation_lock = asyncio.Lock()
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def async_initialize(self) -> None:
        """Publish validated state once, leaving Store failures retryable."""
        async with self._initialization_lock:
            if self._initialized:
                return
            # Storage I/O failures are not malformed state. Propagate them so a
            # later getter/setup can retry instead of silently disabling Guest Mode.
            data = await self._store.async_load()
            raw = data.get("schedule") if isinstance(data, Mapping) else None
            schedule = None
            if isinstance(raw, Mapping):
                try:
                    candidate = GuestModeSchedule(**dict(raw))
                    _parse_timestamp(self.hass, candidate.active_from, "active_from")
                    if candidate.active_until is not None:
                        _parse_timestamp(
                            self.hass, candidate.active_until, "active_until"
                        )
                    schedule = candidate
                except TypeError, ValueError:
                    _LOGGER.warning(
                        "Ignoring malformed Guest Mode state", exc_info=True
                    )
            self._schedule = schedule
            self._initialized = True

    @property
    def schedule(self) -> GuestModeSchedule | None:
        return self._schedule

    def status(self, now: Any | None = None) -> dict[str, Any]:
        """Return the current non-sensitive state and schedule."""
        current = _as_utc(now or dt_util.utcnow())
        schedule = self._schedule
        if schedule is None:
            state = "inactive"
            active = scheduled = indefinite = False
        else:
            start = _parse_timestamp(self.hass, schedule.active_from, "active_from")
            end = (
                _parse_timestamp(self.hass, schedule.active_until, "active_until")
                if schedule.active_until is not None
                else None
            )
            active = start <= current and (end is None or current < end)
            scheduled = current < start
            indefinite = end is None
            state = (
                "active_indefinitely"
                if active and indefinite
                else "active"
                if active
                else "scheduled"
                if scheduled
                else "inactive"
            )
        return {
            "state": state,
            "active_from": schedule.active_from if schedule else None,
            "active_until": schedule.active_until if schedule else None,
            "indefinite": indefinite,
            "currently_active": active,
            "scheduled": scheduled,
            "source": schedule.source if schedule else None,
            "updated_at": schedule.updated_at if schedule else None,
        }

    def is_active(self, now: Any | None = None) -> bool:
        return bool(self.status(now)["currently_active"])

    async def async_restrict(
        self,
        *,
        active_from: str | None = None,
        active_until: str | None = None,
        make_indefinite: bool = False,
        now: Any | None = None,
    ) -> dict[str, Any]:
        """Apply a monotonic LLM restriction that can only widen the interval."""
        current = _as_utc(now or dt_util.utcnow())
        requested_start = (
            _parse_timestamp(self.hass, active_from, "active_from")
            if active_from is not None
            else current
        )
        requested_end = (
            _parse_timestamp(self.hass, active_until, "active_until")
            if active_until is not None
            else None
        )
        if requested_end is not None and requested_end <= requested_start:
            raise ValueError("active_until must be later than active_from")

        async with self._mutation_lock:
            existing = self._live_or_future_schedule(current)
            if existing is None:
                start = requested_start
                end = None if make_indefinite or active_until is None else requested_end
            else:
                existing_start = _parse_timestamp(
                    self.hass, existing.active_from, "active_from"
                )
                existing_end = (
                    _parse_timestamp(self.hass, existing.active_until, "active_until")
                    if existing.active_until is not None
                    else None
                )
                start = min(existing_start, requested_start)
                if existing_end is None or make_indefinite:
                    end = None
                elif requested_end is None:
                    end = existing_end
                else:
                    end = max(existing_end, requested_end)
            schedule = self._schedule_value(start, end, "llm")
            await self._async_commit_schedule(schedule)
            return self.status(current)

    async def async_update_trusted(
        self,
        *,
        active_from: str | None = None,
        active_until: str | None = None,
        indefinite: bool = False,
        now: Any | None = None,
    ) -> dict[str, Any]:
        """Replace the interval from an authenticated HA control surface."""
        current = _as_utc(now or dt_util.utcnow())
        start = (
            _parse_timestamp(self.hass, active_from, "active_from")
            if active_from is not None
            else current
        )
        end = (
            None
            if indefinite
            else _parse_timestamp(self.hass, active_until, "active_until")
            if active_until is not None
            else None
        )
        if end is not None and end <= start:
            raise ValueError("active_until must be later than active_from")
        schedule = self._schedule_value(start, end, "home_assistant")
        async with self._mutation_lock:
            await self._async_commit_schedule(schedule)
            return self.status(current)

    async def async_disable_trusted(self) -> dict[str, Any]:
        """End or cancel Guest Mode from a trusted HA control surface."""
        async with self._mutation_lock:
            await self._async_commit_schedule(None)
            return self.status()

    async def async_backup_data(self) -> dict[str, Any]:
        """Return JSON-compatible Guest Mode state for a private agent backup."""
        return {"schedule": asdict(self._schedule) if self._schedule else None}

    @staticmethod
    def validate_backup_data(value: Any) -> GuestModeSchedule | None:
        """Validate schedule data without mutating runtime state."""
        if not isinstance(value, Mapping) or set(value) != {"schedule"}:
            raise ValueError("Guest Mode backup is invalid")
        raw = value["schedule"]
        if raw is None:
            return None
        if not isinstance(raw, Mapping) or set(raw) != {
            "active_from",
            "active_until",
            "source",
            "updated_at",
        }:
            raise ValueError("Guest Mode schedule backup is invalid")
        schedule = GuestModeSchedule(**dict(raw))
        for field, timestamp in (
            ("active_from", schedule.active_from),
            ("active_until", schedule.active_until),
            ("updated_at", schedule.updated_at),
        ):
            if timestamp is None and field != "active_from":
                continue
            if not isinstance(timestamp, str):
                raise ValueError(f"Guest Mode {field} is invalid")
            parsed = dt_util.parse_datetime(timestamp)
            if parsed is None or parsed.tzinfo is None:
                raise ValueError(f"Guest Mode {field} is invalid")
        if not isinstance(schedule.source, str) or not schedule.source:
            raise ValueError("Guest Mode source is invalid")
        return schedule

    async def async_replace_backup(self, schedule: GuestModeSchedule | None) -> None:
        """Replace durable Guest Mode state during an atomic agent restore."""
        async with self._mutation_lock:
            await self._async_commit_schedule(schedule)

    def async_add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        self._listeners.add(listener)

        def remove() -> None:
            self._listeners.discard(listener)

        return remove

    @staticmethod
    def _schedule_value(start: Any, end: Any | None, source: str) -> GuestModeSchedule:
        return GuestModeSchedule(
            active_from=_as_utc(start).isoformat(),
            active_until=_as_utc(end).isoformat() if end is not None else None,
            source=source,
            updated_at=dt_util.utcnow().isoformat(),
        )

    async def _async_commit_schedule(self, schedule: GuestModeSchedule | None) -> None:
        """Persist one schedule before publishing it to live Guest Mode readers."""
        payload = {"schedule": asdict(schedule) if schedule is not None else None}
        save_task = asyncio.ensure_future(self._store.async_save(payload))
        cancellation: asyncio.CancelledError | None = None

        while not save_task.done():
            try:
                await asyncio.shield(save_task)
            except asyncio.CancelledError as err:
                if save_task.cancelled():
                    raise
                if cancellation is None:
                    cancellation = err
            except Exception:
                break

        try:
            save_task.result()
        except asyncio.CancelledError:
            raise
        except Exception as err:
            if cancellation is not None:
                raise cancellation from err
            raise

        self._schedule = schedule
        self._notify()
        if cancellation is not None:
            raise cancellation

    def _live_or_future_schedule(self, now: Any) -> GuestModeSchedule | None:
        schedule = self._schedule
        if schedule is None or schedule.active_until is None:
            return schedule
        end = _parse_timestamp(self.hass, schedule.active_until, "active_until")
        return schedule if now < end else None

    def _notify(self) -> None:
        for listener in tuple(self._listeners):
            listener()


async def async_get_guest_mode(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> GuestModeManager:
    managers: dict[tuple[str, str], GuestModeManager] = hass.data.setdefault(
        _MANAGERS, {}
    )
    key = (entry_id, subentry_id)
    if key not in managers:
        managers[key] = GuestModeManager(hass, entry_id, subentry_id)
    manager = managers[key]
    await manager.async_initialize()
    return manager


def get_loaded_guest_mode(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> GuestModeManager | None:
    return cast(
        GuestModeManager | None,
        hass.data.get(_MANAGERS, {}).get((entry_id, subentry_id)),
    )


def resolve_guest_policy(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    manager: GuestModeManager | None,
    configured_tools: Sequence[Mapping[str, Any]] = (),
    *,
    exposed_entities: Sequence[Mapping[str, Any]] | None = None,
) -> GuestCapabilityPolicy:
    """Resolve one restrictive policy from the current Guest state and config."""
    if manager is None or not manager.is_active():
        return GuestCapabilityPolicy.unrestricted()
    if options.get(CONF_GUEST_POLICY_VERSION) == GUEST_POLICY_VERSION:
        if exposed_entities is None:
            return _resolve_exclusion_policy(hass, options, configured_tools)
        return _resolve_exclusion_policy(
            hass,
            options,
            configured_tools,
            exposed_entities=exposed_entities,
        )
    return _resolve_legacy_policy(hass, options, configured_tools)


def _resolve_legacy_policy(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    configured_tools: Sequence[Mapping[str, Any]],
) -> GuestCapabilityPolicy:
    """Preserve the pre-v2 allow-list policy without broadening access."""
    readable = _resolve_legacy_entity_ids(hass, options, control=False)
    controllable = _resolve_legacy_entity_ids(hass, options, control=True) & readable
    groups = options.get(CONF_FUNCTION_GROUPS, ())
    membership = (
        {
            name: group
            for group in groups
            if isinstance(group, Mapping)
            for name in group.get("functions", ())
            if isinstance(name, str)
        }
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes))
        else {}
    )
    guest_tools = frozenset(
        str(tool.get("spec", {}).get("name"))
        for tool in configured_tools
        if tool.get("guest_allowed", False) is True
        and isinstance(tool.get("spec"), Mapping)
        and isinstance(tool.get("spec", {}).get("name"), str)
        and (
            tool.get("spec", {}).get("name") not in membership
            or membership[str(tool.get("spec", {}).get("name"))].get("guest_allowed")
            is True
        )
        and classify_tool(tool) <= FunctionSecurity.CONTROL
    )
    return GuestCapabilityPolicy(
        True,
        readable_entity_ids=frozenset(readable),
        controllable_entity_ids=frozenset(controllable),
        configured_tool_names=guest_tools,
        legacy_function_flags=True,
        personal_memory_read=False,
        personal_memory_write=False,
        shared_memory_read=bool(options.get(CONF_GUEST_SHARED_MEMORY_READ, False)),
        shared_memory_write=bool(options.get(CONF_GUEST_SHARED_MEMORY_WRITE, False)),
        archive_access=False,
        archive_retention=False,
        knowledge_access=bool(options.get(CONF_GUEST_KNOWLEDGE_ENABLED, False)),
        temporary_memory=False,
        skills=False,
        web_search=False,
        private_capabilities=False,
    )


def _resolve_exclusion_policy(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    configured_tools: Sequence[Mapping[str, Any]],
    *,
    exposed_entities: Sequence[Mapping[str, Any]] | None = None,
) -> GuestCapabilityPolicy:
    """Resolve v2 against HA's normal assistant exposure, then subtract denies."""
    exposed = (
        exposed_entities if exposed_entities is not None else get_exposed_entities(hass)
    )
    baseline = {
        item["entity_id"]
        for item in exposed
        if isinstance(item, Mapping) and isinstance(item.get("entity_id"), str)
    }
    readable = baseline - resolve_guest_selector_entity_ids(
        hass,
        baseline,
        entities=options.get(CONF_GUEST_EXCLUDED_ENTITIES, ()),
        domains=options.get(CONF_GUEST_EXCLUDED_DOMAINS, ()),
        areas=options.get(CONF_GUEST_EXCLUDED_AREAS, ()),
        labels=options.get(CONF_GUEST_EXCLUDED_LABELS, ()),
    )
    controllable = set(readable)
    if options.get(CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS) is True:
        controllable -= resolve_guest_selector_entity_ids(
            hass,
            baseline,
            entities=options.get(CONF_GUEST_CONTROL_EXCLUDED_ENTITIES, ()),
            domains=options.get(CONF_GUEST_CONTROL_EXCLUDED_DOMAINS, ()),
            areas=options.get(CONF_GUEST_CONTROL_EXCLUDED_AREAS, ()),
            labels=options.get(CONF_GUEST_CONTROL_EXCLUDED_LABELS, ()),
        )

    configured_by_name = {
        str(tool["spec"]["name"]): tool
        for tool in configured_tools
        if isinstance(tool.get("spec"), Mapping)
        and isinstance(tool["spec"].get("name"), str)
    }
    function_policy = options.get(CONF_GUEST_FUNCTION_POLICY, "off")
    if function_policy == "on":
        guest_tools = set(configured_by_name)
    elif function_policy == "custom":
        guest_tools = _string_set(options.get(CONF_GUEST_ALLOWED_FUNCTION_NAMES, ()))
        selected_groups = _string_set(options.get(CONF_GUEST_ALLOWED_GROUP_IDS, ()))
        groups = options.get(CONF_FUNCTION_GROUPS, ())
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
            guest_tools.update(
                name
                for group in groups
                if isinstance(group, Mapping) and group.get("id") in selected_groups
                for name in group.get("functions", ())
                if isinstance(name, str)
            )
        guest_tools &= set(configured_by_name)
    else:
        guest_tools = set()
    guest_tools = {
        name
        for name in guest_tools
        if classify_tool(configured_by_name[name]) <= FunctionSecurity.CONTROL
    }

    knowledge_policy = options.get(CONF_GUEST_KNOWLEDGE_POLICY, "off")
    knowledge_sources = (
        frozenset(_string_set(options.get(CONF_GUEST_KNOWLEDGE_SOURCE_IDS, ())))
        if knowledge_policy == "custom"
        else None
    )
    memory_policy = options.get(CONF_GUEST_SHARED_MEMORY_POLICY, "off")
    return GuestCapabilityPolicy(
        True,
        readable_entity_ids=frozenset(readable),
        controllable_entity_ids=frozenset(controllable),
        configured_tool_names=frozenset(guest_tools),
        knowledge_source_ids=knowledge_sources,
        personal_memory_read=False,
        personal_memory_write=False,
        shared_memory_read=memory_policy in ("read_only", "read_write"),
        shared_memory_write=memory_policy == "read_write",
        archive_access=False,
        archive_retention=False,
        knowledge_access=knowledge_policy == "on" or bool(knowledge_sources),
        temporary_memory=False,
        skills=False,
        web_search=bool(options.get(CONF_GUEST_WEB_SEARCH, DEFAULT_GUEST_WEB_SEARCH)),
        private_capabilities=False,
    )


def guest_mode_restrict_tool() -> dict[str, Any]:
    """Return the structurally one-way model tool."""
    return {
        "spec": {
            "name": "guest_mode_restrict",
            "description": (
                "Enable, start sooner, or extend Guest Mode. This operation cannot "
                "disable, cancel, delay, or shorten Guest Mode. Omit active_until "
                "for indefinite activation when enabling it for the first time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "active_from": {
                        "type": "string",
                        "description": "ISO 8601 start time; omit to start now.",
                    },
                    "active_until": {
                        "type": "string",
                        "description": "ISO 8601 expiry used only to extend restrictions.",
                    },
                    "make_indefinite": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "function": {"type": "guest_mode", "operation": "restrict"},
    }


def guest_policy_editor_snapshot(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    configured_tools: Sequence[Mapping[str, Any]],
    *,
    exposed_entities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a v2 editor draft, conservatively translating legacy settings."""
    keys = (
        CONF_GUEST_EXCLUDED_ENTITIES,
        CONF_GUEST_EXCLUDED_DOMAINS,
        CONF_GUEST_EXCLUDED_AREAS,
        CONF_GUEST_EXCLUDED_LABELS,
        CONF_GUEST_CONTROL_EXCLUDED_ENTITIES,
        CONF_GUEST_CONTROL_EXCLUDED_DOMAINS,
        CONF_GUEST_CONTROL_EXCLUDED_AREAS,
        CONF_GUEST_CONTROL_EXCLUDED_LABELS,
        CONF_GUEST_KNOWLEDGE_SOURCE_IDS,
        CONF_GUEST_ALLOWED_FUNCTION_NAMES,
        CONF_GUEST_ALLOWED_GROUP_IDS,
    )
    if options.get(CONF_GUEST_POLICY_VERSION) == GUEST_POLICY_VERSION:
        return {
            CONF_GUEST_POLICY_VERSION: GUEST_POLICY_VERSION,
            CONF_GUEST_MODE_ENABLED: options.get(CONF_GUEST_MODE_ENABLED, True),
            CONF_GUEST_WEB_SEARCH: bool(
                options.get(CONF_GUEST_WEB_SEARCH, DEFAULT_GUEST_WEB_SEARCH)
            ),
            **{key: sorted(_string_set(options.get(key, ()))) for key in keys},
            CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS: options.get(
                CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS, False
            ),
            CONF_GUEST_KNOWLEDGE_POLICY: options.get(
                CONF_GUEST_KNOWLEDGE_POLICY, "off"
            ),
            CONF_GUEST_FUNCTION_POLICY: options.get(CONF_GUEST_FUNCTION_POLICY, "off"),
            CONF_GUEST_SHARED_MEMORY_POLICY: options.get(
                CONF_GUEST_SHARED_MEMORY_POLICY, "off"
            ),
        }

    exposed = (
        exposed_entities if exposed_entities is not None else get_exposed_entities(hass)
    )
    baseline = {
        item["entity_id"]
        for item in exposed
        if isinstance(item, Mapping) and isinstance(item.get("entity_id"), str)
    }
    readable = _resolve_legacy_entity_ids(hass, options, control=False)
    controllable = _resolve_legacy_entity_ids(hass, options, control=True) & readable
    legacy_tools = _resolve_legacy_policy(hass, options, configured_tools)
    shared_read = bool(options.get(CONF_GUEST_SHARED_MEMORY_READ, False))
    shared_write = bool(options.get(CONF_GUEST_SHARED_MEMORY_WRITE, False))
    return {
        CONF_GUEST_POLICY_VERSION: GUEST_POLICY_VERSION,
        CONF_GUEST_MODE_ENABLED: options.get(CONF_GUEST_MODE_ENABLED, True),
        CONF_GUEST_WEB_SEARCH: False,
        CONF_GUEST_EXCLUDED_ENTITIES: sorted(baseline - readable),
        CONF_GUEST_EXCLUDED_DOMAINS: [],
        CONF_GUEST_EXCLUDED_AREAS: [],
        CONF_GUEST_EXCLUDED_LABELS: [],
        CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS: readable != controllable,
        CONF_GUEST_CONTROL_EXCLUDED_ENTITIES: sorted(readable - controllable),
        CONF_GUEST_CONTROL_EXCLUDED_DOMAINS: [],
        CONF_GUEST_CONTROL_EXCLUDED_AREAS: [],
        CONF_GUEST_CONTROL_EXCLUDED_LABELS: [],
        CONF_GUEST_KNOWLEDGE_POLICY: (
            "on" if options.get(CONF_GUEST_KNOWLEDGE_ENABLED, False) else "off"
        ),
        CONF_GUEST_KNOWLEDGE_SOURCE_IDS: [],
        CONF_GUEST_FUNCTION_POLICY: (
            "custom" if legacy_tools.configured_tool_names else "off"
        ),
        CONF_GUEST_ALLOWED_FUNCTION_NAMES: sorted(
            legacy_tools.configured_tool_names or ()
        ),
        CONF_GUEST_ALLOWED_GROUP_IDS: [],
        # A legacy write-only combination has no v2 equivalent and maps to Off,
        # avoiding a silent grant of read access.
        CONF_GUEST_SHARED_MEMORY_POLICY: (
            "read_write"
            if shared_read and shared_write
            else "read_only"
            if shared_read
            else "off"
        ),
    }


def _resolve_legacy_entity_ids(
    hass: HomeAssistant, options: Mapping[str, Any], *, control: bool
) -> set[str]:
    prefix = "controllable" if control else "readable"
    entity_key = (
        CONF_GUEST_CONTROLLABLE_ENTITIES if control else CONF_GUEST_READABLE_ENTITIES
    )
    domain_key = (
        CONF_GUEST_CONTROLLABLE_DOMAINS if control else CONF_GUEST_READABLE_DOMAINS
    )
    area_key = CONF_GUEST_CONTROLLABLE_AREAS if control else CONF_GUEST_READABLE_AREAS
    label_key = (
        CONF_GUEST_CONTROLLABLE_LABELS if control else CONF_GUEST_READABLE_LABELS
    )
    explicit = _string_set(options.get(entity_key, ()))
    domains = _string_set(options.get(domain_key, ()))
    areas = _string_set(options.get(area_key, ()))
    labels = _string_set(options.get(label_key, ()))
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    allowed: set[str] = set()
    for state in hass.states.async_all():
        entity_id = state.entity_id
        registry_entry = entity_registry.async_get(entity_id)
        device = (
            device_registry.async_get(registry_entry.device_id)
            if registry_entry is not None and registry_entry.device_id
            else None
        )
        entity_areas = {
            value
            for value in (
                getattr(registry_entry, "area_id", None),
                getattr(device, "area_id", None),
            )
            if value
        }
        entity_labels = set(getattr(registry_entry, "labels", ()) or ()) | set(
            getattr(device, "labels", ()) or ()
        )
        if (
            entity_id in explicit
            or entity_id.partition(".")[0] in domains
            or bool(entity_areas & areas)
            or bool(entity_labels & labels)
        ):
            allowed.add(entity_id)
    _LOGGER.debug("Resolved %d Guest %s entities", len(allowed), prefix)
    return allowed


def resolve_guest_selector_entity_ids(
    hass: HomeAssistant,
    candidates: set[str],
    *,
    entities: Any = (),
    domains: Any = (),
    areas: Any = (),
    labels: Any = (),
    devices: Any = (),
) -> set[str]:
    """Resolve a union of selector matches, limited to candidate entities."""
    explicit = _string_set(entities)
    selected_domains = _string_set(domains)
    selected_areas = _string_set(areas)
    selected_labels = _string_set(labels)
    selected_devices = _string_set(devices)
    if not candidates or not (
        explicit
        or selected_domains
        or selected_areas
        or selected_labels
        or selected_devices
    ):
        return set()
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    matched: set[str] = set()
    for entity_id in candidates:
        registry_entry = entity_registry.async_get(entity_id)
        device = (
            device_registry.async_get(registry_entry.device_id)
            if registry_entry is not None and registry_entry.device_id
            else None
        )
        entity_areas = {
            value
            for value in (
                getattr(registry_entry, "area_id", None),
                getattr(device, "area_id", None),
            )
            if value
        }
        entity_labels = set(getattr(registry_entry, "labels", ()) or ()) | set(
            getattr(device, "labels", ()) or ()
        )
        if (
            entity_id in explicit
            or entity_id.partition(".")[0] in selected_domains
            or bool(entity_areas & selected_areas)
            or bool(entity_labels & selected_labels)
            or bool(
                registry_entry is not None
                and registry_entry.device_id in selected_devices
            )
        ):
            matched.add(entity_id)
    return matched


def guest_arguments_allowed_runtime(
    hass: HomeAssistant,
    value: Any,
    policy: GuestCapabilityPolicy,
    *,
    control: bool,
    require_entity_selector: bool = False,
) -> bool:
    """Resolve HA selectors and require every selected entity to be guest-safe."""
    if not policy.guest_active:
        return True

    selected: dict[str, set[str]] = {
        "areas": set(),
        "devices": set(),
        "floors": set(),
        "labels": set(),
    }
    explicit_entity_selector = False

    def string_values(item: Any) -> set[str]:
        values = item if isinstance(item, list) else [item]
        return {
            part.strip()
            for entry in values
            if isinstance(entry, str)
            for part in entry.split(",")
            if part.strip()
        }

    def collect(item: Any) -> None:
        nonlocal explicit_entity_selector
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).lower()
                if normalized in {
                    "entity_id",
                    "entity_ids",
                    "statistic_id",
                    "statistic_ids",
                } and string_values(child):
                    explicit_entity_selector = True
                bucket = (
                    "areas"
                    if normalized in {"area_id", "area_ids"}
                    else "devices"
                    if normalized in {"device_id", "device_ids"}
                    else "floors"
                    if normalized in {"floor_id", "floor_ids"}
                    else "labels"
                    if normalized in {"label_id", "label_ids"}
                    else None
                )
                if bucket is not None:
                    selected[bucket].update(string_values(child))
                else:
                    collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    if (
        require_entity_selector
        and not explicit_entity_selector
        and not any(selected.values())
    ):
        return False
    allows = policy.allows_entity_control if control else policy.allows_entity_read
    if any(selected.values()):
        # Resolve with HA's service-target rules. Assist exposure is intentionally
        # not a candidate filter: an area, device, floor, or label can also target
        # unexposed entities, which must fail the Guest policy check below.
        target_selection = target_helpers.TargetSelection(
            {
                "area_id": list(selected["areas"]),
                "device_id": list(selected["devices"]),
                "floor_id": list(selected["floors"]),
                "label_id": list(selected["labels"]),
            }
        )
        referenced = target_helpers.async_extract_referenced_entity_ids(
            hass,
            target_selection,
        )
        matched = referenced.referenced | referenced.indirectly_referenced
        if not matched or not all(allows(entity_id) for entity_id in matched):
            return False

    broad_keys = {
        "area_id",
        "area_ids",
        "device_id",
        "device_ids",
        "floor_id",
        "floor_ids",
        "label_id",
        "label_ids",
    }

    def inspect(item: Any, key: str | None = None) -> bool:
        if isinstance(item, Mapping):
            return all(
                inspect(child, str(child_key).lower())
                for child_key, child in item.items()
                if str(child_key).lower() not in broad_keys
            )
        if isinstance(item, list):
            return all(inspect(child, key) for child in item)
        if key in {"entity_id", "entity_ids", "statistic_id", "statistic_ids"}:
            values = string_values(item)
            return bool(values) and all(allows(entity_id) for entity_id in values)
        return True

    return inspect(value)


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _parse_timestamp(hass: HomeAssistant, value: str, field: str) -> Any:
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        raise ValueError(f"{field} must be an ISO 8601 timestamp")
    if parsed.tzinfo is None:
        zone = dt_util.get_time_zone(hass.config.time_zone)
        if zone is None:
            raise ValueError("Home Assistant timezone is unavailable")
        parsed = parsed.replace(tzinfo=zone)
    return dt_util.as_utc(parsed)


def _as_utc(value: Any) -> Any:
    if value.tzinfo is None:
        raise ValueError("timestamp must include timezone information")
    return dt_util.as_utc(value)
