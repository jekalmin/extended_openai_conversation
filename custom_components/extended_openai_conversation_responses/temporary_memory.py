"""Automatic, expiring temporary context for conversation agents."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import logging
from typing import Any, cast
from uuid import uuid4

from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .memory import validate_memory_privacy
from .persistence_hardening import _async_settle_transactional_save
from .scope import SHARED_HOUSEHOLD_SCOPE_ID
from .strict_store import PropagatingWriteStore

_LOGGER = logging.getLogger(__name__)
_ACTIVE_OWNER_SCOPE_ID: ContextVar[str | None] = ContextVar(
    "extended_openai_temporary_memory_owner_scope_id", default=None
)

STORAGE_VERSION = 1
STORAGE_KEY_PREFIX = f"{DOMAIN}.temporary_memory"
MAX_ACTIVE_RECORDS = 100
MAX_CONTENT_LENGTH = 500
MAX_CATEGORY_LENGTH = 64
MAX_DELETE_RECORDS = 50
MAX_INJECT_RECORDS = 30
MAX_INJECT_CHARACTERS = 6_000

TEMPORARY_MEMORY_TOOL_NAMES = {
    "temporary_memory_add",
    "temporary_memory_update",
    "temporary_memory_delete",
}


@dataclass(slots=True, frozen=True)
class TemporaryMemoryRecord:
    """One concise fact that expires automatically."""

    memory_id: str
    scope_id: str
    content: str
    category: str
    source: str
    expires_at: str
    created_at: str
    updated_at: str
    owner_scope_id: str | None = None


class TemporaryMemoryStore(PropagatingWriteStore):
    """Versioned private Home Assistant storage."""


class TemporaryMemory:
    """Concurrency-safe short-lived context store."""

    def __init__(self, store: TemporaryMemoryStore) -> None:
        self._store = store
        self._records: dict[str, TemporaryMemoryRecord] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self.expired_pruned = 0
        self.invalid_owners_pruned = 0
        self.overflow_pruned = 0
        self._normalization_pending = False
        self._prune_save_task: asyncio.Task[None] | None = None
        self._committed_state: tuple[dict[str, TemporaryMemoryRecord], int] | None = (
            None
        )

    async def async_initialize(self) -> None:
        """Load retryably, then persist canonical ownership and the startup ceiling."""
        async with self._lock:
            try:
                if not self._initialized:
                    data = await self._store.async_load()
                    raw_records = (
                        data.get("records", []) if isinstance(data, Mapping) else []
                    )
                    for raw in raw_records:
                        try:
                            record = _record_from_storage(raw)
                            if _parse_expiry(record.expires_at) > dt_util.utcnow():
                                self._records[record.memory_id] = record
                            else:
                                self.expired_pruned += 1
                        except TypeError, ValueError:
                            continue
                    self._initialized = True
                    if self.expired_pruned:
                        await self._async_save_locked(reconcile_failure=False)
            except Exception:
                self._records.clear()
                self.expired_pruned = 0
                self._initialized = False
                self._committed_state = None
                raise
            self._remember_committed_state()
            await self._async_normalize_loaded_records_locked()

    async def async_active(
        self, scope_id: str, owner_scope_id: str | None = None
    ) -> list[TemporaryMemoryRecord]:
        """Return owner-only context; persist expiry pruning off the read path."""
        owner = _resolve_owner_scope_id(owner_scope_id)
        if owner is None:
            return []
        async with self._lock:
            expired = self._prune_expired_locked()
            result = self._active_snapshot_locked(scope_id, owner)
        if expired:
            self._schedule_pruned_state_save()
        return result

    async def async_active_snapshot(
        self, scope_id: str, owner_scope_id: str | None = None
    ) -> list[TemporaryMemoryRecord]:
        """Return owner-only context without pruning, saving, or changing counters."""
        owner = _resolve_owner_scope_id(owner_scope_id)
        if owner is None:
            return []
        async with self._lock:
            return self._active_snapshot_locked(scope_id, owner)

    def _active_snapshot_locked(
        self, scope_id: str, owner_scope_id: str | None = None
    ) -> list[TemporaryMemoryRecord]:
        """Select bounded, currently active records while the lock is held."""
        return self.select_active_snapshot(
            self._records.values(), scope_id, owner_scope_id
        )

    @staticmethod
    def select_active_snapshot(
        records: Iterable[TemporaryMemoryRecord],
        scope_id: str,
        owner_scope_id: str | None = None,
    ) -> list[TemporaryMemoryRecord]:
        """Select bounded active records for one scope and optional owner."""
        now = dt_util.utcnow()
        records = [
            record
            for record in records
            if _matches_owner(record, scope_id, owner_scope_id)
            and _parse_expiry(record.expires_at) > now
        ]
        records.sort(key=lambda item: (item.updated_at, item.expires_at), reverse=True)
        selected: list[TemporaryMemoryRecord] = []
        characters = 0
        for record in records:
            if len(selected) >= MAX_INJECT_RECORDS:
                break
            size = len(record.content)
            if selected and characters + size > MAX_INJECT_CHARACTERS:
                continue
            selected.append(record)
            characters += size
        return selected

    async def async_add(
        self,
        scope_id: str,
        content: str,
        expires_at: str,
        category: str = "general",
        *,
        owner_scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Add an automatic fact, coalescing an exact owned active duplicate."""
        owner_scope_id = _require_owner_scope_id(owner_scope_id)
        content = _clean(content, MAX_CONTENT_LENGTH, "content")
        category = _clean(category, MAX_CATEGORY_LENGTH, "category")
        validate_memory_privacy(content, automatic=True)
        expiry = _parse_future_expiry(expires_at)
        async with self._lock:
            await self._async_prune_locked()
            now = dt_util.utcnow().isoformat()
            for current in self._records.values():
                if (
                    _matches_owner(current, scope_id, owner_scope_id)
                    and current.content.casefold() == content.casefold()
                ):
                    updated = TemporaryMemoryRecord(
                        current.memory_id,
                        current.scope_id,
                        content,
                        category,
                        "automatic",
                        expiry.isoformat(),
                        current.created_at,
                        now,
                        current.owner_scope_id,
                    )
                    self._records[current.memory_id] = updated
                    await self._async_save_locked()
                    return {
                        "status": "updated",
                        "memory": temporary_memory_as_dict(updated),
                    }
            if len(self._records) >= MAX_ACTIVE_RECORDS:
                raise ValueError("temporary memory limit reached")
            record = TemporaryMemoryRecord(
                uuid4().hex,
                scope_id,
                content,
                category,
                "automatic",
                expiry.isoformat(),
                now,
                now,
                owner_scope_id,
            )
            self._records[record.memory_id] = record
            await self._async_save_locked()
            return {"status": "created", "memory": temporary_memory_as_dict(record)}

    async def async_update(
        self,
        scope_id: str,
        memory_id: str,
        content: str | None,
        expires_at: str | None,
        category: str | None,
        *,
        owner_scope_id: str | None = None,
    ) -> TemporaryMemoryRecord:
        """Update/supersede a temporary fact owned by the current request."""
        owner_scope_id = _require_owner_scope_id(owner_scope_id)
        async with self._lock:
            await self._async_prune_locked()
            current = self._owned(scope_id, memory_id, owner_scope_id)
            new_content = (
                _clean(content, MAX_CONTENT_LENGTH, "content")
                if content is not None
                else current.content
            )
            validate_memory_privacy(new_content, automatic=True)
            new_expiry = (
                _parse_future_expiry(expires_at).isoformat()
                if expires_at is not None
                else current.expires_at
            )
            updated = TemporaryMemoryRecord(
                current.memory_id,
                current.scope_id,
                new_content,
                _clean(category, MAX_CATEGORY_LENGTH, "category")
                if category is not None
                else current.category,
                current.source,
                new_expiry,
                current.created_at,
                dt_util.utcnow().isoformat(),
                current.owner_scope_id,
            )
            self._records[memory_id] = updated
            await self._async_save_locked()
            return updated

    async def async_delete(
        self,
        scope_id: str,
        memory_ids: list[str],
        *,
        owner_scope_id: str | None = None,
    ) -> int:
        """Delete selected records only from the current scope and optional owner."""
        owner_scope_id = _require_owner_scope_id(owner_scope_id)
        if not memory_ids or len(memory_ids) > MAX_DELETE_RECORDS:
            raise ValueError(f"memory_ids must contain 1 to {MAX_DELETE_RECORDS} IDs")
        async with self._lock:
            deleted = 0
            for memory_id in set(memory_ids):
                record = self._records.get(memory_id)
                if record is not None and _matches_owner(
                    record, scope_id, owner_scope_id
                ):
                    del self._records[memory_id]
                    deleted += 1
            if deleted:
                await self._async_save_locked()
            return deleted

    async def async_list(
        self,
        _scope_id: str | None = None,
        owner_scope_id: str | None = None,
    ) -> list[TemporaryMemoryRecord]:
        owner = _require_owner_scope_id(owner_scope_id)
        return await self.async_list_owned(owner)

    async def async_list_all(
        self,
        owner_scope_id: str | None = None,
    ) -> list[TemporaryMemoryRecord]:
        owner = _require_owner_scope_id(owner_scope_id)
        return await self.async_list_owned(owner)

    def stats(self) -> dict[str, int]:
        """Return non-sensitive diagnostics."""
        return {
            "active_temporary_memory_count": len(self._records),
            "expired_temporary_memories_pruned": self.expired_pruned,
            "invalid_owner_records_pruned": self.invalid_owners_pruned,
            "startup_overflow_records_pruned": self.overflow_pruned,
        }

    def scope_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.scope_id] = counts.get(record.scope_id, 0) + 1
        return counts

    async def async_backup_data(self) -> dict[str, Any]:
        """Return active records with their original absolute expiry."""
        async with self._lock:
            await self._async_prune_locked()
            return {"records": [asdict(record) for record in self._records.values()]}

    @staticmethod
    def validate_backup_data(data: Any) -> list[TemporaryMemoryRecord]:
        """Validate and drop records that have expired since backup creation."""
        if not isinstance(data, Mapping) or set(data) != {"records"}:
            raise ValueError("temporary memories are incomplete or corrupted")
        raw_records = data["records"]
        if not isinstance(raw_records, list) or len(raw_records) > MAX_ACTIVE_RECORDS:
            raise ValueError("temporary memory count is invalid")
        records: list[TemporaryMemoryRecord] = []
        seen: set[str] = set()
        now = dt_util.utcnow()
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise ValueError("temporary memory record must be an object")
            try:
                record = _record_from_storage(raw)
            except TypeError as err:
                raise ValueError("temporary memory record is invalid") from err
            if not all(
                isinstance(value, str)
                for value in (
                    record.memory_id,
                    record.scope_id,
                    record.content,
                    record.category,
                    record.source,
                    record.expires_at,
                    record.created_at,
                    record.updated_at,
                )
            ):
                raise ValueError("temporary memory fields must be strings")
            if record.owner_scope_id is not None and not isinstance(
                record.owner_scope_id, str
            ):
                raise ValueError("temporary memory owner must be a string")
            if (
                not record.memory_id
                or len(record.memory_id) > 128
                or record.memory_id in seen
                or not record.scope_id
                or len(record.scope_id) > 128
                or (
                    record.owner_scope_id is not None
                    and (not record.owner_scope_id or len(record.owner_scope_id) > 128)
                )
                or record.source != "automatic"
            ):
                raise ValueError("temporary memory metadata is invalid")
            _clean(record.content, MAX_CONTENT_LENGTH, "content")
            _clean(record.category, MAX_CATEGORY_LENGTH, "category")
            expiry = _parse_expiry(record.expires_at)
            if (
                dt_util.parse_datetime(record.created_at) is None
                or dt_util.parse_datetime(record.updated_at) is None
            ):
                raise ValueError("temporary memory timestamp is invalid")
            seen.add(record.memory_id)
            if (
                expiry > now
                and (normalized := _normalize_record_owner(record)) is not None
            ):
                records.append(normalized)
        return records

    async def async_replace_backup(self, records: list[TemporaryMemoryRecord]) -> None:
        """Replace active temporary memories without changing their expiry."""
        normalized = [
            migrated
            for record in records
            if (migrated := _normalize_record_owner(record)) is not None
        ]
        if len(normalized) > MAX_ACTIVE_RECORDS:
            normalized = sorted(normalized, key=_owner_record_sort_key, reverse=True)[
                :MAX_ACTIVE_RECORDS
            ]
        async with self._lock:
            self._records = {record.memory_id: record for record in normalized}
            await self._async_save_locked()

    def _owned(
        self,
        scope_id: str,
        memory_id: str,
        owner_scope_id: str | None = None,
    ) -> TemporaryMemoryRecord:
        record = self._records.get(memory_id)
        if record is None or not _matches_owner(record, scope_id, owner_scope_id):
            raise ValueError("temporary memory not found")
        return record

    async def _async_prune_locked(self) -> None:
        if self._prune_expired_locked():
            await self._async_save_locked()

    async def _async_save_locked(self, *, reconcile_failure: bool = True) -> None:
        """Settle writes and reload validated records after ambiguous failures."""
        await _async_settle_transactional_save(
            self._store.async_save(
                {"records": [asdict(record) for record in self._records.values()]}
            ),
            self._restore_committed_state,
            self._remember_committed_state,
            self._async_reconcile_failed_save if reconcile_failure else None,
            self._invalidate_after_unreadable_store,
        )

    async def _async_reconcile_failed_save(self) -> None:
        """Rebuild validated records without starting a second cleanup write."""
        data = await self._store.async_load()
        if data is not None and not isinstance(data, Mapping):
            raise ValueError("Temporary Memory store has invalid structure")
        raw_records = data.get("records", []) if isinstance(data, Mapping) else []
        if not isinstance(raw_records, list):
            raise ValueError("Temporary Memory records have invalid structure")
        records: list[TemporaryMemoryRecord] = []
        invalid_owners = 0
        normalized_owners = 0
        for raw in raw_records:
            try:
                record = _record_from_storage(raw)
                _parse_expiry(record.expires_at)
                normalized = _normalize_record_owner(record)
            except TypeError, ValueError:
                continue
            if normalized is None:
                invalid_owners += 1
                continue
            normalized_owners += normalized != record
            records.append(normalized)
        overflow = max(0, len(records) - MAX_ACTIVE_RECORDS)
        if overflow:
            records = sorted(records, key=_owner_record_sort_key, reverse=True)[
                :MAX_ACTIVE_RECORDS
            ]
        self._records = {record.memory_id: record for record in records}
        self.expired_pruned = 0
        self.invalid_owners_pruned = invalid_owners
        self.overflow_pruned = overflow
        self._normalization_pending = bool(
            invalid_owners or normalized_owners or overflow
        )
        self._initialized = True
        self._remember_committed_state()

    def _invalidate_after_unreadable_store(self) -> None:
        """Block context reads if persisted temporary memory cannot be validated."""
        self._records.clear()
        self._initialized = False
        self._committed_state = None
        self._normalization_pending = False

    async def _async_normalize_loaded_records_locked(self) -> None:
        """Persist safe owner migration while retaining the pre-save retry state."""
        original = self._records
        normalized: list[TemporaryMemoryRecord] = []
        invalid = 0
        for record in original.values():
            migrated = _normalize_record_owner(record)
            if migrated is None:
                invalid += 1
                continue
            normalized.append(migrated)

        overflow = max(0, len(normalized) - MAX_ACTIVE_RECORDS)
        if overflow:
            normalized = sorted(normalized, key=_owner_record_sort_key, reverse=True)[
                :MAX_ACTIVE_RECORDS
            ]

        replacement = {record.memory_id: record for record in normalized}
        if replacement == original and not self._normalization_pending:
            return

        already_counted = self._normalization_pending
        self._records = replacement
        await self._async_save_locked()
        self._normalization_pending = False

        if not already_counted:
            self.invalid_owners_pruned += invalid
            self.overflow_pruned += overflow
        if invalid:
            _LOGGER.warning(
                "Removed %s Temporary Memory record(s) without a valid retained owner",
                invalid,
            )
        if overflow:
            _LOGGER.warning(
                "Temporary Memory exceeded the %s-record ceiling at startup; "
                "kept the newest %s records and removed %s",
                MAX_ACTIVE_RECORDS,
                MAX_ACTIVE_RECORDS,
                overflow,
            )

    async def async_list_owned(
        self, owner_scope_id: str
    ) -> list[TemporaryMemoryRecord]:
        await self.async_initialize()
        async with self._lock:
            await self._async_prune_locked()
            return self._records_for_owner(owner_scope_id)

    async def async_update_owned(
        self,
        owner_scope_id: str,
        memory_id: str,
        content: str | None,
        expires_at: str | None,
        category: str | None,
    ) -> TemporaryMemoryRecord:
        owner = _require_owner_scope_id(owner_scope_id)
        return await self.async_update(
            owner,
            memory_id,
            content,
            expires_at,
            category,
            owner_scope_id=owner,
        )

    async def async_delete_owned(
        self,
        owner_scope_id: str,
        memory_ids: list[str],
    ) -> int:
        owner = _require_owner_scope_id(owner_scope_id)
        return await self.async_delete(owner, memory_ids, owner_scope_id=owner)

    def owner_counts(self) -> dict[str, int]:
        now = dt_util.utcnow()
        counts: Counter[str] = Counter()
        for record in self._records.values():
            owner = _valid_owner_scope_id(record.owner_scope_id)
            expiry = dt_util.parse_datetime(record.expires_at)
            if owner is not None and expiry is not None and expiry > now:
                counts[owner] += 1
        return dict(counts)

    def _records_for_owner(self, owner_scope_id: str) -> list[TemporaryMemoryRecord]:
        """Return every active record belonging to an owner without injection limits."""
        owner = _require_owner_scope_id(owner_scope_id)
        now = dt_util.utcnow()
        records = [
            record
            for record in self._records.values()
            if record.owner_scope_id == owner
            and (dt_util.parse_datetime(record.expires_at) or now) > now
        ]
        records.sort(key=lambda record: (record.expires_at, record.memory_id))
        return records

    def _prune_expired_locked(self) -> bool:
        """Remove expired records in RAM; the caller chooses the save boundary."""
        now = dt_util.utcnow()
        expired = [
            memory_id
            for memory_id, record in self._records.items()
            if _parse_expiry(record.expires_at) <= now
        ]
        for memory_id in expired:
            del self._records[memory_id]
        if expired:
            self.expired_pruned += len(expired)
        return bool(expired)

    def _schedule_pruned_state_save(self) -> None:
        """Persist an expiry-only mutation later through the transactional save seam."""
        current = self._prune_save_task
        if current is not None and not current.done():
            return

        async def persist() -> None:
            try:
                async with self._lock:
                    await self._async_save_locked()
            except Exception:
                # Expired records remain invisible by timestamp even if persistence fails;
                # the owner's transactional save restores the last committed state.
                _LOGGER.exception("Unable to persist pruned temporary memories")

        task = asyncio.create_task(
            persist(), name="extended_openai_temporary_memory_expiry_persistence"
        )
        self._prune_save_task = task

        def done(completed: asyncio.Task[None]) -> None:
            if self._prune_save_task is completed:
                self._prune_save_task = None

        task.add_done_callback(done)

    def _remember_committed_state(self) -> None:
        self._committed_state = (dict(self._records), self.expired_pruned)

    def _restore_committed_state(self) -> None:
        if self._committed_state is not None:
            records, expired_pruned = self._committed_state
            self._records = dict(records)
            self.expired_pruned = expired_pruned


def _record_from_storage(raw: Mapping[str, Any]) -> TemporaryMemoryRecord:
    """Load a record while preserving safe compatibility with pre-owner storage."""
    values = dict(raw)
    if "owner_scope_id" not in values:
        scope_id = values.get("scope_id")
        values["owner_scope_id"] = (
            scope_id
            if isinstance(scope_id, str)
            and (scope_id.startswith("user:") or scope_id.startswith("shared:"))
            else None
        )
    return TemporaryMemoryRecord(**values)


def _matches_owner(
    record: TemporaryMemoryRecord, _scope_id: str, owner_scope_id: str | None
) -> bool:
    """Continuity is metadata, never authorization for retained data."""
    owner = _valid_owner_scope_id(owner_scope_id)
    return owner is not None and record.owner_scope_id == owner


def _parse_expiry(value: str):
    parsed = dt_util.parse_datetime(value)
    if parsed is None:
        raise ValueError("expires_at must be an ISO date-time with a timezone")
    if parsed.tzinfo is None:
        raise ValueError("expires_at must include a timezone")
    return dt_util.as_utc(parsed)


def _parse_future_expiry(value: str):
    parsed = _parse_expiry(value)
    if parsed <= dt_util.utcnow():
        raise ValueError("expires_at must be in the future")
    if parsed > dt_util.utcnow() + timedelta(days=366):
        raise ValueError("temporary memory cannot last longer than one year")
    return parsed


def _clean(value: str, limit: int, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = " ".join(value.split()).strip()
    if not value or len(value) > limit:
        raise ValueError(f"{field} must contain 1 to {limit} characters")
    return value


def temporary_memory_as_dict(
    record: TemporaryMemoryRecord, *, include_scope: bool = False
) -> dict[str, str]:
    result = {
        "memory_id": record.memory_id,
        "content": record.content,
        "category": record.category,
        "source": record.source,
        "expires_at": record.expires_at,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    if include_scope:
        result["scope_id"] = record.scope_id
    return result


def temporary_memory_tools() -> list[dict[str, Any]]:
    """Return the small model-facing maintenance surface."""
    common = {"type": "temporary_memory"}
    return [
        {
            "spec": {
                "name": "temporary_memory_add",
                "description": "Silently store one useful short-lived fact for the current context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "expires_at": {
                            "type": "string",
                            "description": "ISO date-time with timezone, inferred reasonably from ordinary language.",
                        },
                        "category": {"type": "string"},
                    },
                    "required": ["content", "expires_at"],
                    "additionalProperties": False,
                },
            },
            "function": {**common, "operation": "add"},
        },
        {
            "spec": {
                "name": "temporary_memory_update",
                "description": "Silently supersede an active temporary fact when circumstances change.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "content": {"type": "string"},
                        "expires_at": {"type": "string"},
                        "category": {"type": "string"},
                    },
                    "required": ["memory_id"],
                    "additionalProperties": False,
                },
            },
            "function": {**common, "operation": "update"},
        },
        {
            "spec": {
                "name": "temporary_memory_delete",
                "description": "Silently forget one or more active temporary facts for the current context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": MAX_DELETE_RECORDS,
                        },
                    },
                    "required": ["memory_ids"],
                    "additionalProperties": False,
                },
            },
            "function": {**common, "operation": "delete"},
        },
    ]


_MANAGERS = f"{DOMAIN}.temporary_memory_managers"


def _temporary_memory_store(
    hass: Any, entry_id: str, subentry_id: str
) -> TemporaryMemoryStore:
    """Build the versioned store adapter without loading or changing manager state."""
    return TemporaryMemoryStore(
        hass,
        STORAGE_VERSION,
        f"{STORAGE_KEY_PREFIX}.{entry_id}.{subentry_id}",
        private=True,
        atomic_writes=True,
        serialize_in_event_loop=False,
    )


def get_loaded_temporary_memory(
    hass: Any, entry_id: str, subentry_id: str
) -> TemporaryMemory | None:
    """Return an already initialized manager without creating runtime state."""
    return cast(
        TemporaryMemory | None,
        hass.data.get(_MANAGERS, {}).get((entry_id, subentry_id)),
    )


async def async_read_temporary_memory_snapshot(
    hass: Any,
    entry_id: str,
    subentry_id: str,
    scope_id: str,
    owner_scope_id: str | None = None,
) -> list[TemporaryMemoryRecord]:
    """Read active stored context without creating, pruning, or saving a manager."""
    owner_scope_id = _resolve_owner_scope_id(owner_scope_id)
    if owner_scope_id is None:
        return []
    data = await _temporary_memory_store(hass, entry_id, subentry_id).async_load()
    raw_records = data.get("records", []) if isinstance(data, Mapping) else []
    records: list[TemporaryMemoryRecord] = []
    for raw in raw_records:
        try:
            record = _record_from_storage(raw)
            _parse_expiry(record.expires_at)
            records.append(record)
        except TypeError, ValueError:
            continue
    return TemporaryMemory.select_active_snapshot(records, scope_id, owner_scope_id)


async def async_get_temporary_memory(
    hass: Any, entry_id: str, subentry_id: str
) -> TemporaryMemory:
    managers = hass.data.setdefault(_MANAGERS, {})
    key = (entry_id, subentry_id)
    if key not in managers:
        managers[key] = TemporaryMemory(
            _temporary_memory_store(hass, entry_id, subentry_id)
        )
    manager = managers[key]
    await manager.async_initialize()
    return cast(TemporaryMemory, manager)


def _valid_owner_scope_id(value: object) -> str | None:
    """Return one canonical retained owner, or None when it is not valid."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if candidate == SHARED_HOUSEHOLD_SCOPE_ID:
        return candidate
    if candidate.startswith("user:"):
        user_id = candidate[5:]
        if user_id and len(candidate) <= 128:
            return candidate
    return None


def _require_owner_scope_id(value: object = None) -> str:
    """Resolve and validate the retained owner for one operation."""
    valid = _resolve_owner_scope_id(value)
    if valid is None:
        raise ValueError(
            "Temporary Memory requires a resolved Personal or Shared owner"
        )
    return valid


def _owner_from_resolved_scope(scope: object) -> str | None:
    """Translate only the established data-scope contract into a retained owner."""
    if scope is None:
        return None
    scope_type = getattr(scope, "scope_type", None)
    if scope_type == "user":
        user_id = getattr(scope, "user_id", None)
        return _valid_owner_scope_id(f"user:{user_id}") if user_id else None
    if scope_type == "shared":
        return SHARED_HOUSEHOLD_SCOPE_ID
    return None


def _normalize_record_owner(
    record: TemporaryMemoryRecord,
) -> TemporaryMemoryRecord | None:
    """Preserve proven ownership and conservatively migrate canonical legacy data."""
    if record.owner_scope_id is not None:
        owner = _valid_owner_scope_id(record.owner_scope_id)
        if owner is None:
            return None
        return (
            record
            if owner == record.owner_scope_id
            else replace(record, owner_scope_id=owner)
        )

    # Pre-owner records whose scope itself was a canonical retained scope are safe
    # to migrate. Device/conversation continuity keys cannot establish ownership.
    inferred = _valid_owner_scope_id(record.scope_id)
    if inferred is None:
        return None
    return replace(record, owner_scope_id=inferred)


def _owner_record_sort_key(record: TemporaryMemoryRecord) -> tuple[Any, ...]:
    """Keep the newest bounded records deterministically during startup recovery."""

    def parsed(value: str) -> float:
        parsed_value = dt_util.parse_datetime(value)
        return parsed_value.timestamp() if parsed_value is not None else 0.0

    return (
        parsed(record.updated_at),
        parsed(record.expires_at),
        parsed(record.created_at),
        record.memory_id,
    )


def _resolve_owner_scope_id(value: object = None) -> str | None:
    """Resolve an explicit owner or the request-bound owner, failing closed."""
    return _valid_owner_scope_id(
        value if value is not None else _ACTIVE_OWNER_SCOPE_ID.get()
    )
