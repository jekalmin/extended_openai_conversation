"""Local, scoped and searchable conversation transcript archive."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import logging
import re
from typing import Any, Protocol
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .scope import ResolvedDataScope
from .strict_store import PropagatingWriteStore

_LOGGER = logging.getLogger(__name__)
STORAGE_VERSION = 1
STORAGE_KEY_PREFIX = f"{DOMAIN}.archive"
MAX_SEARCH_LIMIT = 50
MAX_GET_LIMIT = 50
MAX_TEXT_LENGTH = 40_000
MAX_TITLE_LENGTH = 80
MAX_EXCERPT_LENGTH = 500
_ARCHIVE_MANAGERS = f"{DOMAIN}.archive_managers"
_TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
_SPACE_PATTERN = re.compile(r"\s+")


def _scope_retention_allowed(
    scope: ResolvedDataScope, shared_archive_enabled: bool
) -> bool:
    """Return whether the current scope policy permits transcript retention."""
    return bool(
        scope.allows_retention
        and (scope.scope_type != "shared" or shared_archive_enabled)
    )


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
}


@dataclass(slots=True, frozen=True)
class ArchiveSession:
    """Content-free session metadata."""

    session_id: str
    home_assistant_conversation_id: str | None
    agent_subentry_id: str
    scope_id: str
    scope_type: str
    scope_source: str
    source_device_id: str | None
    started_at: str
    last_message_at: str
    title: str
    turn_count: int
    retention_state: str


@dataclass(slots=True, frozen=True)
class ArchiveTurn:
    """Exactly one user message and final assistant response."""

    turn_id: str
    session_id: str
    run_id: str | None
    timestamp: str
    user_text: str
    assistant_text: str
    successful: bool


class ArchiveStorage(Protocol):
    """Partitioned persistence boundary."""

    async def async_load_metadata(self) -> dict[str, Any] | None: ...
    async def async_save_metadata(self, data: dict[str, Any]) -> None: ...
    async def async_load_partition(self, partition: str) -> dict[str, Any] | None: ...
    async def async_save_partition(
        self, partition: str, data: dict[str, Any]
    ) -> None: ...


class HomeAssistantArchiveStorage:
    """Small metadata store plus independent monthly transcript stores."""

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str) -> None:
        prefix = f"{STORAGE_KEY_PREFIX}.{entry_id}.{subentry_id}"
        self._hass = hass
        self._prefix = prefix
        self._metadata = PropagatingWriteStore(
            hass,
            STORAGE_VERSION,
            f"{prefix}.metadata",
            private=True,
            atomic_writes=True,
        )

    def _partition_store(self, partition: str) -> Store[dict[str, Any]]:
        return PropagatingWriteStore(
            self._hass,
            STORAGE_VERSION,
            f"{self._prefix}.turns.{partition}",
            private=True,
            atomic_writes=True,
            serialize_in_event_loop=False,
        )

    async def async_load_metadata(self) -> dict[str, Any] | None:
        return await self._metadata.async_load()

    async def async_save_metadata(self, data: dict[str, Any]) -> None:
        await self._metadata.async_save(data)

    async def async_load_partition(self, partition: str) -> dict[str, Any] | None:
        return await self._partition_store(partition).async_load()

    async def async_save_partition(self, partition: str, data: dict[str, Any]) -> None:
        await self._partition_store(partition).async_save(data)

    async def async_remove_partition(self, partition: str) -> None:
        """Remove an obsolete monthly transcript store."""
        await self._partition_store(partition).async_remove()


class ConversationArchive:
    """Concurrency-safe scoped archive with deterministic session privacy."""

    def __init__(self, storage: ArchiveStorage, agent_subentry_id: str) -> None:
        self._storage = storage
        self._agent_subentry_id = agent_subentry_id
        self._sessions: dict[str, ArchiveSession] = {}
        self._turns: dict[str, list[ArchiveTurn]] = defaultdict(list)
        self._active: dict[str, str] = {}
        self._partitions: set[str] = set()
        self._pending_partitions: set[str] = set()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def async_initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            data = await self._storage.async_load_metadata() or {}
            if not isinstance(data, dict):
                data = {}
            pending = data.get("pending_partitions")
            if pending is not None:
                # Metadata is the transaction marker. Complete any interrupted
                # cross-store write before exposing archive state after restart.
                if not isinstance(pending, dict) or any(
                    not isinstance(partition, str)
                    or re.fullmatch(r"\d{4}-\d{2}", partition) is None
                    or not isinstance(payload, dict)
                    or not isinstance(payload.get("turns"), list)
                    for partition, payload in pending.items()
                ):
                    raise ValueError("conversation archive transaction is corrupted")
                for partition, payload in sorted(pending.items()):
                    await self._storage.async_save_partition(partition, payload)
                data = {
                    key: value
                    for key, value in data.items()
                    if key != "pending_partitions"
                }
                await self._storage.async_save_metadata(data)
            sessions = data.get("sessions")
            for raw in sessions if isinstance(sessions, list) else []:
                try:
                    session = ArchiveSession(**raw)
                except TypeError, ValueError:
                    _LOGGER.warning("Ignoring malformed conversation archive session")
                    continue
                self._sessions[session.session_id] = session
            active = data.get("active")
            self._active = {
                str(key): str(value)
                for key, value in (active if isinstance(active, dict) else {}).items()
                if isinstance(value, str) and value in self._sessions
            }
            partitions = data.get("partitions")
            self._partitions = {
                value
                for value in (partitions if isinstance(partitions, list) else [])
                if isinstance(value, str)
                and re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])", value)
            }
            for partition in sorted(self._partitions):
                payload = await self._storage.async_load_partition(partition) or {}
                turns = payload.get("turns") if isinstance(payload, dict) else None
                for raw in turns if isinstance(turns, list) else []:
                    try:
                        turn = ArchiveTurn(**raw)
                    except TypeError, ValueError:
                        _LOGGER.warning("Ignoring malformed conversation archive turn")
                        continue
                    if turn.session_id in self._sessions:
                        self._turns[turn.session_id].append(turn)
            self._initialized = True

    async def async_begin_session(
        self,
        session_key: str,
        scope: ResolvedDataScope,
        home_assistant_conversation_id: str | None,
        *,
        archive_enabled: bool,
        shared_archive_enabled: bool,
        inactivity_minutes: int,
    ) -> ArchiveSession | None:
        """Resolve the stable scope once and return the exact active session."""
        if not archive_enabled:
            return None
        async with self._lock:
            self._ensure_initialized()
            current = self._sessions.get(self._active.get(session_key, ""))
            now = dt_util.utcnow()
            may_retain = _scope_retention_allowed(scope, shared_archive_enabled)
            if current is not None:
                expired = (
                    _parse_time(current.last_message_at)
                    + timedelta(minutes=max(1, inactivity_minutes))
                    < now
                )
                if (
                    not expired
                    and current.scope_id == scope.scope_id
                    and current.home_assistant_conversation_id
                    == home_assistant_conversation_id
                    and current.retention_state != "closed"
                    and (may_retain or current.retention_state != "retained")
                ):
                    return current

            timestamp = now.isoformat()
            session = ArchiveSession(
                session_id=uuid4().hex,
                home_assistant_conversation_id=home_assistant_conversation_id,
                agent_subentry_id=self._agent_subentry_id,
                scope_id=scope.scope_id,
                scope_type=scope.scope_type,
                scope_source=scope.source,
                source_device_id=scope.device_id,
                started_at=timestamp,
                last_message_at=timestamp,
                title="",
                turn_count=0,
                retention_state="retained" if may_retain else "unretained",
            )
            await self._async_publish_session_locked(session_key, session)
            return session

    async def async_record_turn(
        self,
        session_id: str,
        *,
        run_id: str | None,
        user_text: str,
        assistant_text: str,
        successful: bool,
    ) -> ArchiveTurn | None:
        """Retain one turn using the Archive journal before publishing live state."""
        user_text = _clean_text(user_text)
        assistant_text = _clean_text(assistant_text)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.retention_state != "retained":
                return None
            timestamp = dt_util.utcnow().isoformat()
            turn = ArchiveTurn(
                turn_id=uuid4().hex,
                session_id=session_id,
                run_id=run_id,
                timestamp=timestamp,
                user_text=user_text,
                assistant_text=assistant_text,
                successful=successful,
            )
            sessions = dict(self._sessions)
            sessions[session_id] = ArchiveSession(
                **{
                    **asdict(session),
                    "last_message_at": timestamp,
                    "title": session.title or _title(user_text),
                    "turn_count": session.turn_count + 1,
                }
            )
            turns = dict(self._turns)
            turns[session_id] = [*self._turns.get(session_id, ()), turn]
            await self._async_commit_state_locked(
                sessions=sessions,
                turns=turns,
                active=dict(self._active),
                changed_partitions={timestamp[:7]},
            )
            return turn

    async def async_make_private(self, session_id: str) -> dict[str, Any]:
        """Delete retained turns and journal the privacy boundary before publishing."""
        async with self._lock:
            session = self._require_session(session_id)
            turns = dict(self._turns)
            removed = list(turns.pop(session_id, ()))
            sessions = dict(self._sessions)
            sessions[session_id] = ArchiveSession(
                **{**asdict(session), "turn_count": 0, "retention_state": "private"}
            )
            await self._async_commit_state_locked(
                sessions=sessions,
                turns=turns,
                active=dict(self._active),
                changed_partitions=self._changed_partitions(removed),
            )
            return {
                "private_mode_enabled": True,
                "session_id": session_id,
                "deleted_turns": len(removed),
                "future_turns_retained": False,
            }

    async def async_resume_saving(
        self,
        session_key: str,
        previous_session_id: str,
        scope: ResolvedDataScope,
        *,
        shared_archive_enabled: bool,
    ) -> ArchiveSession:
        """End a private boundary and create a fresh retained session."""
        async with self._lock:
            previous = self._require_session(previous_session_id)
            timestamp = dt_util.utcnow().isoformat()
            session = ArchiveSession(
                session_id=uuid4().hex,
                home_assistant_conversation_id=previous.home_assistant_conversation_id,
                agent_subentry_id=self._agent_subentry_id,
                scope_id=scope.scope_id,
                scope_type=scope.scope_type,
                scope_source=scope.source,
                source_device_id=scope.device_id,
                started_at=timestamp,
                last_message_at=timestamp,
                title="",
                turn_count=0,
                retention_state=(
                    "retained"
                    if _scope_retention_allowed(scope, shared_archive_enabled)
                    else "unretained"
                ),
            )
            await self._async_publish_session_locked(session_key, session)
            return session

    async def async_search(
        self,
        scope_id: str,
        query: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 5,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Perform bounded phrase/token lexical search within one exact scope."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        offset = max(0, offset)
        async with self._lock:
            self._ensure_initialized()
            snapshot = tuple(
                (session, tuple(self._turns.get(session.session_id, ())))
                for session in self._sessions.values()
                if session.scope_id == scope_id
                and session.retention_state == "retained"
            )
        return await asyncio.to_thread(
            _search_archive_snapshot,
            snapshot,
            query,
            start_date,
            end_date,
            limit,
            offset,
        )

    async def async_get(
        self, scope_id: str, session_id: str, start_turn: int = 0, limit: int = 6
    ) -> dict[str, Any]:
        """Return bounded turns only after exact ownership validation."""
        session = self._require_owned_session(scope_id, session_id)
        start_turn = max(0, start_turn)
        limit = max(1, min(limit, MAX_GET_LIMIT))
        turns = self._turns.get(session_id, [])
        page = turns[start_turn : start_turn + limit]
        return {
            "session": asdict(session),
            "turns": [asdict(turn) for turn in page],
            "start_turn": start_turn,
            "limit": limit,
            "has_more": len(turns) > start_turn + limit,
        }

    async def async_list_sessions(
        self, scope_id: str, *, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """List bounded content-free metadata for one scope."""
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        sessions = [
            session
            for session in self._sessions.values()
            if session.scope_id == scope_id and session.retention_state == "retained"
        ]
        sessions.sort(key=lambda item: item.last_message_at, reverse=True)
        page = sessions[max(0, offset) : max(0, offset) + limit]
        return {
            "sessions": [asdict(session) for session in page],
            "offset": max(0, offset),
            "limit": limit,
            "has_more": len(sessions) > max(0, offset) + limit,
        }

    async def async_delete_session(
        self, scope_id: str, session_id: str
    ) -> dict[str, int]:
        """Delete one exact owned transcript using the restart-safe journal."""
        async with self._lock:
            self._require_owned_session(scope_id, session_id)
            sessions = dict(self._sessions)
            del sessions[session_id]
            turns = dict(self._turns)
            removed = list(turns.pop(session_id, ()))
            active = {
                key: value for key, value in self._active.items() if value != session_id
            }
            await self._async_commit_state_locked(
                sessions=sessions,
                turns=turns,
                active=active,
                changed_partitions=self._changed_partitions(removed),
            )
            return {"deleted_sessions": 1, "deleted_turns": len(removed)}

    async def async_clear_scope(
        self, scope_id: str, *, confirm: bool
    ) -> dict[str, int]:
        """Clear one scope transactionally after explicit confirmation."""
        if not confirm:
            raise ValueError("Explicit confirmation is required")
        async with self._lock:
            targets = {
                session.session_id
                for session in self._sessions.values()
                if session.scope_id == scope_id
            }
            if not targets:
                return {"deleted_sessions": 0, "deleted_turns": 0}
            sessions = {
                session_id: session
                for session_id, session in self._sessions.items()
                if session_id not in targets
            }
            turns = dict(self._turns)
            removed = [
                turn for session_id in targets for turn in turns.pop(session_id, ())
            ]
            active = {
                key: value
                for key, value in self._active.items()
                if value not in targets
            }
            await self._async_commit_state_locked(
                sessions=sessions,
                turns=turns,
                active=active,
                changed_partitions=self._changed_partitions(removed),
            )
            return {
                "deleted_sessions": len(targets),
                "deleted_turns": len(removed),
            }

    async def async_delete_selected(
        self, scope_id: str, session_ids: list[str], *, confirm: bool
    ) -> dict[str, int]:
        """Delete selected owned sessions transactionally after confirmation."""
        if not confirm:
            raise ValueError("Explicit confirmation is required")
        if not session_ids or len(session_ids) > MAX_SEARCH_LIMIT:
            raise ValueError(f"session_ids must contain 1 to {MAX_SEARCH_LIMIT} IDs")
        async with self._lock:
            targets = set(session_ids)
            # Validate every supplied ID while holding the same lock used for commit.
            for session_id in targets:
                self._require_owned_session(scope_id, session_id)
            sessions = {
                session_id: session
                for session_id, session in self._sessions.items()
                if session_id not in targets
            }
            turns = dict(self._turns)
            removed = [
                turn for session_id in targets for turn in turns.pop(session_id, ())
            ]
            active = {
                key: value
                for key, value in self._active.items()
                if value not in targets
            }
            await self._async_commit_state_locked(
                sessions=sessions,
                turns=turns,
                active=active,
                changed_partitions=self._changed_partitions(removed),
            )
            return {
                "deleted_sessions": len(targets),
                "deleted_turns": len(removed),
            }

    async def async_delete_date_range(
        self,
        scope_id: str,
        start_date: str,
        end_date: str,
        *,
        confirm: bool,
    ) -> dict[str, int]:
        """Delete sessions whose last retained turn is inside an exact date range."""
        if not confirm:
            raise ValueError("Explicit confirmation is required")
        if (
            not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start_date)
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date)
            or start_date > end_date
        ):
            raise ValueError("a valid start_date and end_date are required")
        targets = [
            session.session_id
            for session in self._sessions.values()
            if session.scope_id == scope_id
            and session.retention_state == "retained"
            and start_date <= session.last_message_at[:10] <= end_date
        ]
        if not targets:
            return {"deleted_sessions": 0, "deleted_turns": 0}
        return await self.async_delete_selected(scope_id, targets, confirm=True)

    async def async_prune(self, retention_days: int) -> dict[str, int]:
        """Delete expired transcript sessions using the restart-safe journal."""
        cutoff = dt_util.utcnow() - timedelta(days=max(1, retention_days))
        async with self._lock:
            targets = {
                session.session_id
                for session in self._sessions.values()
                if _parse_time(session.last_message_at) < cutoff
            }
            if not targets:
                return {"deleted_sessions": 0, "deleted_turns": 0}
            sessions = {
                session_id: session
                for session_id, session in self._sessions.items()
                if session_id not in targets
            }
            turns = dict(self._turns)
            removed = [
                turn for session_id in targets for turn in turns.pop(session_id, ())
            ]
            active = {
                key: value
                for key, value in self._active.items()
                if value not in targets
            }
            await self._async_commit_state_locked(
                sessions=sessions,
                turns=turns,
                active=active,
                changed_partitions=self._changed_partitions(removed),
            )
            return {
                "deleted_sessions": len(targets),
                "deleted_turns": len(removed),
            }

    def active_session(self, session_key: str) -> ArchiveSession | None:
        return self._sessions.get(self._active.get(session_key, ""))

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "home_assistant_monthly_store",
            "storage_version": STORAGE_VERSION,
            "session_count": len(self._sessions),
            "turn_count": sum(len(turns) for turns in self._turns.values()),
            "partition_count": len(self._partitions),
        }

    def scope_counts(self) -> dict[str, int]:
        """Return retained session totals grouped by data scope."""
        counts: dict[str, int] = {}
        for session in self._sessions.values():
            if session.retention_state != "retained":
                continue
            counts[session.scope_id] = counts.get(session.scope_id, 0) + 1
        return counts

    async def async_backup_data(self) -> dict[str, Any]:
        """Return retained archive content without active-session runtime state."""
        async with self._lock:
            self._ensure_initialized()
            sessions = [
                session
                for session in self._sessions.values()
                if session.retention_state != "unretained"
            ]
            retained_ids = {session.session_id for session in sessions}
            return {
                "sessions": [asdict(session) for session in sessions],
                "turns": [
                    asdict(turn)
                    for session_id, turns in self._turns.items()
                    if session_id in retained_ids
                    for turn in turns
                ],
            }

    @staticmethod
    def validate_backup_data(
        data: Any, target_agent_id: str
    ) -> tuple[list[ArchiveSession], list[ArchiveTurn]]:
        """Validate archive source data and bind it to the selected agent."""
        if not isinstance(data, dict) or set(data) != {"sessions", "turns"}:
            raise ValueError("archive data is incomplete or corrupted")
        if not isinstance(data["sessions"], list) or not isinstance(
            data["turns"], list
        ):
            raise ValueError("archive sessions and turns must be lists")
        sessions: list[ArchiveSession] = []
        session_ids: set[str] = set()
        for raw in data["sessions"]:
            if not isinstance(raw, dict):
                raise ValueError("archive session must be an object")
            raw = {**raw, "agent_subentry_id": target_agent_id}
            try:
                session = ArchiveSession(**raw)
            except TypeError as err:
                raise ValueError("archive session is invalid") from err
            string_values = (
                session.session_id,
                session.agent_subentry_id,
                session.scope_id,
                session.scope_type,
                session.scope_source,
                session.started_at,
                session.last_message_at,
                session.title,
                session.retention_state,
            )
            optional_values = (
                session.home_assistant_conversation_id,
                session.source_device_id,
            )
            if not all(isinstance(value, str) for value in string_values) or not all(
                value is None or isinstance(value, str) for value in optional_values
            ):
                raise ValueError("archive session fields have invalid types")
            if (
                not session.session_id
                or len(session.session_id) > 128
                or session.session_id in session_ids
                or session.retention_state not in {"retained", "private"}
                or not isinstance(session.turn_count, int)
                or isinstance(session.turn_count, bool)
                or session.turn_count < 0
                or len(session.title) > MAX_TITLE_LENGTH
                or _parse_time(session.started_at)
                == datetime.min.replace(tzinfo=dt_util.UTC)
                or _parse_time(session.last_message_at)
                == datetime.min.replace(tzinfo=dt_util.UTC)
            ):
                raise ValueError("archive session metadata is invalid")
            session_ids.add(session.session_id)
            sessions.append(session)

        turns: list[ArchiveTurn] = []
        turn_ids: set[str] = set()
        per_session: dict[str, int] = defaultdict(int)
        for raw in data["turns"]:
            if not isinstance(raw, dict):
                raise ValueError("archive turn must be an object")
            try:
                turn = ArchiveTurn(**raw)
            except TypeError as err:
                raise ValueError("archive turn is invalid") from err
            if (
                not all(
                    isinstance(value, str)
                    for value in (
                        turn.turn_id,
                        turn.session_id,
                        turn.timestamp,
                        turn.user_text,
                        turn.assistant_text,
                    )
                )
                or (turn.run_id is not None and not isinstance(turn.run_id, str))
                or not isinstance(turn.successful, bool)
                or not turn.turn_id
                or len(turn.turn_id) > 128
                or turn.turn_id in turn_ids
                or turn.session_id not in session_ids
                or len(turn.user_text) > MAX_TEXT_LENGTH
                or len(turn.assistant_text) > MAX_TEXT_LENGTH
                or _parse_time(turn.timestamp)
                == datetime.min.replace(tzinfo=dt_util.UTC)
            ):
                raise ValueError("archive turn metadata is invalid")
            turn_ids.add(turn.turn_id)
            per_session[turn.session_id] += 1
            turns.append(turn)
        if any(
            session.turn_count != per_session[session.session_id]
            for session in sessions
        ):
            raise ValueError("archive turn counts do not match session metadata")
        return sessions, turns

    async def async_replace_backup(
        self, sessions: list[ArchiveSession], turns: list[ArchiveTurn]
    ) -> None:
        """Replace durable archive data through the same restart-safe journal."""
        async with self._lock:
            self._ensure_initialized()
            session_map = {session.session_id: session for session in sessions}
            turn_map: dict[str, list[ArchiveTurn]] = defaultdict(list)
            for turn in turns:
                turn_map[turn.session_id].append(turn)
            new_partitions = {turn.timestamp[:7] for turn in turns}
            await self._async_commit_state_locked(
                sessions=session_map,
                turns=dict(turn_map),
                active={},
                changed_partitions=set(self._partitions) | new_partitions,
            )

    def _require_session(self, session_id: str) -> ArchiveSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("conversation session not found")
        return session

    def _require_owned_session(self, scope_id: str, session_id: str) -> ArchiveSession:
        session = self._require_session(session_id)
        if session.scope_id != scope_id or session.retention_state != "retained":
            raise ValueError("conversation session not found")
        return session

    async def _async_publish_session_locked(
        self, session_key: str, session: ArchiveSession
    ) -> None:
        """Publish a new active session, persisting only durable archive state."""
        if session.retention_state == "unretained" and not self._pending_partitions:
            self._sessions[session.session_id] = session
            self._active[session_key] = session.session_id
            return
        sessions = {**self._sessions, session.session_id: session}
        active = {**self._active, session_key: session.session_id}
        pending = (
            {
                partition: self._partition_payload_locked(partition)
                for partition in sorted(self._pending_partitions)
            }
            if self._pending_partitions
            else None
        )
        await self._storage.async_save_metadata(
            self._metadata_payload_locked(
                pending,
                sessions=sessions,
                active=active,
            )
        )
        self._sessions[session.session_id] = session
        self._active[session_key] = session.session_id

    @staticmethod
    def _metadata_payload_for_state(
        sessions: dict[str, ArchiveSession],
        active: dict[str, str],
        partitions: set[str],
        pending_partitions: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Serialize one candidate Archive metadata state."""
        persisted_sessions = {
            session.session_id: session
            for session in sessions.values()
            if session.retention_state != "unretained"
        }
        payload: dict[str, Any] = {
            "sessions": [asdict(session) for session in persisted_sessions.values()],
            "active": {
                key: value
                for key, value in active.items()
                if value in persisted_sessions
            },
            "partitions": sorted(partitions),
        }
        if pending_partitions is not None:
            payload["pending_partitions"] = pending_partitions
        return payload

    @staticmethod
    def _partition_payload_for_state(
        partition: str, turns: dict[str, list[ArchiveTurn]]
    ) -> dict[str, Any]:
        return {
            "turns": [
                asdict(turn)
                for session_turns in turns.values()
                for turn in session_turns
                if turn.timestamp.startswith(partition)
            ]
        }

    @staticmethod
    def _partitions_for_turns(turns: dict[str, list[ArchiveTurn]]) -> set[str]:
        return {
            turn.timestamp[:7]
            for session_turns in turns.values()
            for turn in session_turns
        }

    @staticmethod
    def _changed_partitions(turns: list[ArchiveTurn]) -> set[str]:
        return {turn.timestamp[:7] for turn in turns}

    async def _async_remove_partition_locked(self, partition: str) -> None:
        """Remove one obsolete monthly store when the storage boundary supports it."""
        remover = getattr(self._storage, "async_remove_partition", None)
        if callable(remover):
            await remover(partition)
            return

        # Compatibility for custom/test storage boundaries that predate the explicit
        # removal method. Archive callers do not need to know about Store internals.
        store_factory = getattr(self._storage, "_partition_store", None)
        if not callable(store_factory):
            return
        store = store_factory(partition)
        remove = getattr(store, "async_remove", None)
        if callable(remove):
            await remove()

    async def _async_commit_state_locked(
        self,
        *,
        sessions: dict[str, ArchiveSession],
        turns: dict[str, list[ArchiveTurn]],
        active: dict[str, str],
        changed_partitions: set[str],
    ) -> None:
        """Journal candidate Archive state before publishing it in memory.

        The first metadata write is the durable commit point. If it fails, live
        state remains unchanged. Once it succeeds, restart recovery can complete
        every pending partition write, so RAM is moved to the target state even if
        a later partition or final-metadata write fails.
        """
        partitions = self._partitions_for_turns(turns)
        removed_partitions = set(self._partitions) - partitions
        pending_names = (
            set(changed_partitions) | set(self._pending_partitions) | removed_partitions
        )
        pending = {
            partition: self._partition_payload_for_state(partition, turns)
            for partition in sorted(pending_names)
        }

        await self._storage.async_save_metadata(
            self._metadata_payload_for_state(sessions, active, partitions, pending)
        )

        # Durable intent exists now; publish exactly what restart recovery will finish.
        self._sessions = sessions
        self._turns = defaultdict(list, turns)
        self._active = active
        self._partitions = partitions
        self._pending_partitions = set(pending_names)

        for partition, payload in pending.items():
            await self._storage.async_save_partition(partition, payload)

        # Empty months are already durably overwritten above. Physical deletion is
        # housekeeping only; metadata remains the authoritative partition index.
        for partition in sorted(pending_names - partitions):
            try:
                await self._async_remove_partition_locked(partition)
            except Exception:
                _LOGGER.warning(
                    "Unable to remove obsolete conversation archive partition %s",
                    partition,
                    exc_info=True,
                )

        await self._storage.async_save_metadata(
            self._metadata_payload_for_state(sessions, active, partitions)
        )
        self._pending_partitions.clear()

    def _metadata_payload_locked(
        self,
        pending_partitions: dict[str, dict[str, Any]] | None = None,
        *,
        sessions: dict[str, ArchiveSession] | None = None,
        active: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._metadata_payload_for_state(
            self._sessions if sessions is None else sessions,
            self._active if active is None else active,
            set(self._partitions),
            pending_partitions,
        )

    def _partition_payload_locked(self, partition: str) -> dict[str, Any]:
        return self._partition_payload_for_state(partition, dict(self._turns))

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("conversation archive has not been initialized")


async def async_get_archive(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> ConversationArchive:
    """Return one shared archive manager per conversation agent."""
    managers: dict[tuple[str, str], ConversationArchive] = hass.data.setdefault(
        _ARCHIVE_MANAGERS, {}
    )
    key = (entry_id, subentry_id)
    if key not in managers:
        managers[key] = ConversationArchive(
            HomeAssistantArchiveStorage(hass, entry_id, subentry_id), subentry_id
        )
    await managers[key].async_initialize()
    return managers[key]


def archive_tools() -> list[dict[str, Any]]:
    """Return bounded model-facing archive and deterministic privacy tools."""
    return [
        _tool(
            "conversation_search",
            "Search prior retained discussions only when the user refers to them.",
            {
                "query": {"type": "string", "minLength": 1},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            ["query"],
            "search",
        ),
        _tool(
            "conversation_get",
            "Read a bounded page from one search result.",
            {
                "session_id": {"type": "string"},
                "start_turn": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            ["session_id"],
            "get",
        ),
        _tool(
            "conversation_private",
            "Make the exact active conversation private and delete its retained turns.",
            {},
            [],
            "private",
        ),
        _tool(
            "conversation_resume_saving",
            "Start a new retained session after private mode.",
            {},
            [],
            "resume",
        ),
        _tool(
            "conversation_delete_current",
            "Delete the exact active retained conversation.",
            {},
            [],
            "delete_current",
        ),
        _tool(
            "conversation_delete_selected",
            "Delete only explicitly selected retained sessions after user confirmation.",
            {
                "session_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 50,
                },
                "confirm": {"type": "boolean"},
            },
            ["session_ids", "confirm"],
            "delete_selected",
        ),
        _tool(
            "conversation_delete_date_range",
            "Bulk-delete retained sessions in an exact date range after user confirmation.",
            {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "confirm": {"type": "boolean"},
            },
            ["start_date", "end_date", "confirm"],
            "delete_range",
        ),
    ]


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    operation: str,
) -> dict[str, Any]:
    return {
        "spec": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
        "function": {"type": "archive", "operation": operation},
    }


def _search_archive_snapshot(
    snapshot: tuple[tuple[ArchiveSession, tuple[ArchiveTurn, ...]], ...],
    query: str,
    start_date: str | None,
    end_date: str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Rank one immutable Archive snapshot outside the event loop."""
    query_tokens = _tokens(query)
    normalized_query = _normalize(query)
    ranked: list[tuple[float, str, ArchiveSession, ArchiveTurn]] = []
    for session, turns in snapshot:
        for turn in turns:
            date = turn.timestamp[:10]
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            combined = f"{turn.user_text} {turn.assistant_text}"
            normalized_combined = _normalize(combined)
            tokens = _tokens(combined)
            overlap = len(query_tokens & tokens)
            if (
                query_tokens
                and not overlap
                and normalized_query not in normalized_combined
            ):
                continue
            score = overlap / max(1, len(query_tokens))
            if normalized_query and normalized_query in normalized_combined:
                score += 2
            ranked.append((score, turn.timestamp, session, turn))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    page = ranked[offset : offset + limit]
    return {
        "results": [
            {
                "session_id": session.session_id,
                "turn_id": turn.turn_id,
                "date": turn.timestamp[:10],
                "timestamp": turn.timestamp,
                "title": session.title,
                "excerpt": _excerpt(f"{turn.user_text}\n{turn.assistant_text}", query),
            }
            for _, _, session, turn in page
        ],
        "offset": offset,
        "limit": limit,
        "has_more": len(ranked) > offset + limit,
    }


def _clean_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("archive text must be a string")
    value = value.strip()
    if len(value) > MAX_TEXT_LENGTH:
        value = value[:MAX_TEXT_LENGTH]
    return value


def _title(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", value).strip()[:MAX_TITLE_LENGTH]


def _normalize(value: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(value.casefold()))


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {
        _stem(token)
        for token in _TOKEN_PATTERN.findall(value.casefold())
        if len(token) > 1 and token not in _STOP_WORDS
    }


def _excerpt(value: str, query: str) -> str:
    normalized = _SPACE_PATTERN.sub(" ", value).strip()
    lower = normalized.casefold()
    needle = _normalize(query)
    index = lower.find(needle) if needle else 0
    start = max(0, index - MAX_EXCERPT_LENGTH // 3) if index >= 0 else 0
    excerpt = normalized[start : start + MAX_EXCERPT_LENGTH]
    return (
        ("…" if start else "")
        + excerpt
        + ("…" if start + MAX_EXCERPT_LENGTH < len(normalized) else "")
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=dt_util.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_util.UTC)
    return parsed
