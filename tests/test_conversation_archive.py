"""Tests for scoped archive search, privacy, deletion, and session stability."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from custom_components.extended_openai_conversation_responses import (
    conversation_archive as archive_module,
)
from custom_components.extended_openai_conversation_responses.conversation_archive import (
    MAX_SEARCH_LIMIT,
    MAX_TEXT_LENGTH,
    ArchiveSession,
    ArchiveTurn,
    ConversationArchive,
    HomeAssistantArchiveStorage,
    _clean_text,
    _excerpt,
    _parse_time,
    _search_archive_snapshot,
    _stem,
    _title,
    _tokens,
    archive_tools,
    async_get_archive,
)
from custom_components.extended_openai_conversation_responses.scope import (
    shared_scope,
    unretained_scope,
    user_scope,
)


class FakeArchiveStorage:
    def __init__(self):
        self.metadata = None
        self.partitions = {}
        self.metadata_save_count = 0
        self.fail_metadata_on = None
        self.partition_save_count = 0
        self.fail_partition_on = None

    async def async_load_metadata(self):
        return deepcopy(self.metadata)

    async def async_save_metadata(self, data):
        self.metadata_save_count += 1
        if self.metadata_save_count == self.fail_metadata_on:
            raise OSError("metadata write failed")
        self.metadata = deepcopy(data)

    async def async_load_partition(self, partition):
        return deepcopy(self.partitions.get(partition))

    async def async_save_partition(self, partition, data):
        self.partition_save_count += 1
        if self.partition_save_count == self.fail_partition_on:
            raise OSError("partition write failed")
        self.partitions[partition] = deepcopy(data)


async def _archive():
    archive = ConversationArchive(FakeArchiveStorage(), "agent-1")
    await archive.async_initialize()
    return archive


async def test_archive_is_unretained_when_disabled_or_scope_is_unresolved() -> None:
    archive = await _archive()
    disabled = await archive.async_begin_session(
        "one",
        user_scope("alice", source="authenticated_user"),
        "conversation-1",
        archive_enabled=False,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    unresolved = await archive.async_begin_session(
        "two",
        unretained_scope(device_id="satellite"),
        "conversation-2",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    assert disabled is None
    assert unresolved.retention_state == "unretained"
    assert (
        await archive.async_record_turn(
            unresolved.session_id,
            run_id="run",
            user_text="secret",
            assistant_text="reply",
            successful=True,
        )
        is None
    )
    assert archive.stats()["turn_count"] == 0


async def test_scope_is_stable_and_different_sessions_are_not_joined() -> None:
    archive = await _archive()
    alice = user_scope("alice", source="device_mapping", device_id="kitchen")
    first = await archive.async_begin_session(
        "key",
        alice,
        "conversation-1",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    same = await archive.async_begin_session(
        "key",
        alice,
        "conversation-1",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    changed = await archive.async_begin_session(
        "key",
        user_scope("bob", source="device_mapping", device_id="kitchen"),
        "conversation-1",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    assert same.session_id == first.session_id
    assert changed.session_id != first.session_id
    assert changed.scope_id == "user:bob"


async def test_archive_scope_counts_include_only_retained_sessions() -> None:
    archive = await _archive()
    alice = user_scope("alice", source="authenticated_user")
    await archive.async_begin_session(
        "a",
        alice,
        "one",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_begin_session(
        "b",
        alice,
        "two",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_begin_session(
        "private",
        unretained_scope(),
        "three",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )

    assert archive.scope_counts() == {"user:alice": 2}


async def test_private_mode_deletes_only_active_session_and_resume_is_new_boundary() -> (
    None
):
    archive = await _archive()
    scope = user_scope("alice", source="authenticated_user")
    active = await archive.async_begin_session(
        "browser",
        scope,
        "one",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    other = await archive.async_begin_session(
        "satellite",
        scope,
        "two",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        active.session_id,
        run_id="run-1",
        user_text="medical appointment",
        assistant_text="call on Tuesday",
        successful=True,
    )
    await archive.async_record_turn(
        other.session_id,
        run_id="run-2",
        user_text="weather",
        assistant_text="sunny",
        successful=True,
    )
    result = await archive.async_make_private(active.session_id)
    assert result["deleted_turns"] == 1
    assert archive.stats()["turn_count"] == 1
    assert (
        await archive.async_record_turn(
            active.session_id,
            run_id="run-3",
            user_text="private",
            assistant_text="private",
            successful=True,
        )
        is None
    )
    resumed = await archive.async_resume_saving(
        "browser", active.session_id, scope, shared_archive_enabled=False
    )
    assert resumed.session_id != active.session_id
    assert resumed.retention_state == "retained"
    assert archive.stats()["turn_count"] == 1


async def test_search_and_get_cannot_cross_scope_or_return_whole_archive() -> None:
    archive = await _archive()
    alice = await archive.async_begin_session(
        "a",
        user_scope("alice", source="authenticated_user"),
        "one",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        alice.session_id,
        run_id="run-1",
        user_text="Which restaurant was it?",
        assistant_text="The locally owned Cedar House.",
        successful=True,
    )
    found = await archive.async_search("user:alice", "restaurant cedar", limit=1)
    assert len(found["results"]) == 1
    assert len(found["results"][0]["excerpt"]) <= 502
    assert (await archive.async_search("user:bob", "restaurant cedar"))["results"] == []
    with pytest.raises(ValueError, match="not found"):
        await archive.async_get("user:bob", alice.session_id)


async def test_shared_archive_requires_explicit_agent_permission() -> None:
    archive = await _archive()
    blocked = await archive.async_begin_session(
        "shared-1",
        shared_scope(source="shared_voice_policy"),
        "one",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    allowed = await archive.async_begin_session(
        "shared-2",
        shared_scope(source="shared_voice_policy"),
        "two",
        archive_enabled=True,
        shared_archive_enabled=True,
        inactivity_minutes=30,
    )
    assert blocked.retention_state == "unretained"
    assert allowed.retention_state == "retained"


async def test_interrupted_turn_commit_is_completed_on_restart() -> None:
    storage = FakeArchiveStorage()
    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()
    session = await archive.async_begin_session(
        "key",
        user_scope("alice", source="test"),
        "conversation",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    # Staging metadata succeeds, then the final metadata cleanup fails.
    storage.fail_metadata_on = storage.metadata_save_count + 2
    with pytest.raises(OSError, match="metadata write failed"):
        await archive.async_record_turn(
            session.session_id,
            run_id="run",
            user_text="hello",
            assistant_text="hi",
            successful=True,
        )

    storage.fail_metadata_on = None
    restarted = ConversationArchive(storage, "agent-1")
    await restarted.async_initialize()
    result = await restarted.async_get("user:alice", session.session_id)
    assert [turn["user_text"] for turn in result["turns"]] == ["hello"]
    assert "pending_partitions" not in storage.metadata


async def test_interrupted_private_commit_cannot_resurrect_content() -> None:
    storage = FakeArchiveStorage()
    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()
    session = await archive.async_begin_session(
        "key",
        user_scope("alice", source="test"),
        "conversation",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        session.session_id,
        run_id="run",
        user_text="secret",
        assistant_text="reply",
        successful=True,
    )
    storage.fail_metadata_on = storage.metadata_save_count + 2
    with pytest.raises(OSError, match="metadata write failed"):
        await archive.async_make_private(session.session_id)

    storage.fail_metadata_on = None
    restarted = ConversationArchive(storage, "agent-1")
    await restarted.async_initialize()
    assert restarted.stats()["turn_count"] == 0
    with pytest.raises(ValueError, match="not found"):
        await restarted.async_get("user:alice", session.session_id)


async def test_partition_write_failure_recovers_private_state_on_restart() -> None:
    storage = FakeArchiveStorage()
    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()
    session = await archive.async_begin_session(
        "key",
        user_scope("alice", source="test"),
        "conversation",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        session.session_id,
        run_id="run",
        user_text="secret",
        assistant_text="reply",
        successful=True,
    )
    storage.fail_partition_on = storage.partition_save_count + 1

    with pytest.raises(OSError, match="partition write failed"):
        await archive.async_make_private(session.session_id)

    assert "pending_partitions" in storage.metadata
    restarted = ConversationArchive(storage, "agent-1")
    await restarted.async_initialize()

    assert restarted.stats()["turn_count"] == 0
    with pytest.raises(ValueError, match="not found"):
        await restarted.async_get("user:alice", session.session_id)
    assert "pending_partitions" not in storage.metadata


@pytest.mark.parametrize(
    "operation", ["delete", "clear", "selected", "date_range", "prune"]
)
async def test_interrupted_deletion_commits_recover_deterministically(
    operation: str,
) -> None:
    storage = FakeArchiveStorage()
    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()
    session = await archive.async_begin_session(
        "key",
        user_scope("alice", source="test"),
        "conversation",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        session.session_id,
        run_id="run",
        user_text="remove me",
        assistant_text="removed",
        successful=True,
    )
    if operation == "prune":
        archive._sessions[session.session_id] = replace(
            archive._sessions[session.session_id], last_message_at="2000-01-01T00:00:00+00:00"
        )
    storage.fail_metadata_on = storage.metadata_save_count + 2
    with pytest.raises(OSError, match="metadata write failed"):
        if operation == "delete":
            await archive.async_delete_session("user:alice", session.session_id)
        elif operation == "clear":
            await archive.async_clear_scope("user:alice", confirm=True)
        elif operation == "selected":
            await archive.async_delete_selected(
                "user:alice", [session.session_id], confirm=True
            )
        elif operation == "date_range":
            await archive.async_delete_date_range(
                "user:alice", "2000-01-01", "2100-01-01", confirm=True
            )
        else:
            await archive.async_prune(1)

    storage.fail_metadata_on = None
    restarted = ConversationArchive(storage, "agent-1")
    await restarted.async_initialize()
    assert restarted.stats()["session_count"] == 0
    assert restarted.stats()["turn_count"] == 0


async def test_interrupted_backup_replacement_recovers_exact_target() -> None:
    storage = FakeArchiveStorage()
    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()
    session = await archive.async_begin_session(
        "key",
        user_scope("alice", source="test"),
        "conversation",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        session.session_id,
        run_id="run",
        user_text="old",
        assistant_text="content",
        successful=True,
    )
    storage.fail_metadata_on = storage.metadata_save_count + 2
    with pytest.raises(OSError, match="metadata write failed"):
        await archive.async_replace_backup([], [])

    storage.fail_metadata_on = None
    restarted = ConversationArchive(storage, "agent-1")
    await restarted.async_initialize()
    assert restarted.stats()["session_count"] == 0
    assert restarted.stats()["turn_count"] == 0


class _RecoveringArchiveStorage:
    """In-memory archive storage with a one-shot partition failure."""

    def __init__(self) -> None:
        self.metadata = None
        self.partitions: dict[str, dict] = {}
        self.fail_next_partition = False

    async def async_load_metadata(self):
        return deepcopy(self.metadata)

    async def async_save_metadata(self, data):
        self.metadata = deepcopy(data)

    async def async_load_partition(self, partition):
        return deepcopy(self.partitions.get(partition))

    async def async_save_partition(self, partition, data):
        if self.fail_next_partition:
            self.fail_next_partition = False
            raise OSError("partition write failed")
        self.partitions[partition] = deepcopy(data)


async def _archive_with_failed_turn():
    storage = _RecoveringArchiveStorage()
    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()
    session = await archive.async_begin_session(
        "browser",
        user_scope("alice", source="test"),
        "conversation",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )

    storage.fail_next_partition = True
    with pytest.raises(OSError, match="partition write failed"):
        await archive.async_record_turn(
            session.session_id,
            run_id="run-1",
            user_text="first turn",
            assistant_text="first reply",
            successful=True,
        )

    assert archive._pending_partitions
    assert "pending_partitions" in storage.metadata
    return archive, storage, session


async def test_loaded_archive_converges_after_partition_failure_without_restart() -> None:
    """A later successful commit must absorb and clear an earlier pending journal."""
    archive, storage, session = await _archive_with_failed_turn()

    second = await archive.async_record_turn(
        session.session_id,
        run_id="run-2",
        user_text="second turn",
        assistant_text="second reply",
        successful=True,
    )
    assert second is not None
    assert archive._pending_partitions == set()
    assert "pending_partitions" not in storage.metadata

    restarted = ConversationArchive(storage, "agent-1")
    await restarted.async_initialize()
    result = await restarted.async_get("user:alice", session.session_id)
    assert [turn["user_text"] for turn in result["turns"]] == [
        "first turn",
        "second turn",
    ]
    assert result["session"]["turn_count"] == 2


async def test_later_session_publish_preserves_pending_transaction_for_restart() -> None:
    """Unrelated metadata writes cannot erase an unfinished partition journal."""
    archive, storage, failed_session = await _archive_with_failed_turn()

    later_session = await archive.async_begin_session(
        "kitchen",
        user_scope("bob", source="test"),
        "conversation-2",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )

    assert later_session is not None
    assert "pending_partitions" in storage.metadata
    persisted_ids = {item["session_id"] for item in storage.metadata["sessions"]}
    assert failed_session.session_id in persisted_ids
    assert later_session.session_id in persisted_ids

    restarted = ConversationArchive(storage, "agent-1")
    await restarted.async_initialize()

    failed = await restarted.async_get("user:alice", failed_session.session_id)
    assert [turn["user_text"] for turn in failed["turns"]] == ["first turn"]
    listed = await restarted.async_list_sessions("user:bob")
    assert [item["session_id"] for item in listed["sessions"]] == [
        later_session.session_id
    ]
    assert "pending_partitions" not in storage.metadata


class CoverageArchiveStorage:
    """Small in-memory persistence boundary for archive tests."""

    def __init__(self) -> None:
        self.metadata = None
        self.partitions: dict[str, dict] = {}
        self.metadata_writes: list[dict] = []
        self.partition_writes: list[tuple[str, dict]] = []

    async def async_load_metadata(self):
        return deepcopy(self.metadata)

    async def async_save_metadata(self, data):
        self.metadata = deepcopy(data)
        self.metadata_writes.append(deepcopy(data))

    async def async_load_partition(self, partition):
        return deepcopy(self.partitions.get(partition))

    async def async_save_partition(self, partition, data):
        self.partitions[partition] = deepcopy(data)
        self.partition_writes.append((partition, deepcopy(data)))


async def _coverage_archive(storage: CoverageArchiveStorage | None = None) -> ConversationArchive:
    archive = ConversationArchive(storage or CoverageArchiveStorage(), "agent-1")
    await archive.async_initialize()
    return archive


def _session(
    session_id: str = "session-1",
    *,
    scope_id: str = "user:alice",
    turn_count: int = 1,
    retention_state: str = "retained",
) -> ArchiveSession:
    return ArchiveSession(
        session_id=session_id,
        home_assistant_conversation_id="conversation-1",
        agent_subentry_id="source-agent",
        scope_id=scope_id,
        scope_type="user",
        scope_source="authenticated_user",
        source_device_id=None,
        started_at="2026-09-01T10:00:00+00:00",
        last_message_at="2026-09-01T10:01:00+00:00",
        title="Archive title",
        turn_count=turn_count,
        retention_state=retention_state,
    )


def _turn(
    session_id: str = "session-1",
    *,
    turn_id: str = "turn-1",
    timestamp: str = "2026-09-01T10:01:00+00:00",
    user_text: str = "Remember the running shoes",
    assistant_text: str = "You chose the blue pair",
) -> ArchiveTurn:
    return ArchiveTurn(
        turn_id=turn_id,
        session_id=session_id,
        run_id="run-1",
        timestamp=timestamp,
        user_text=user_text,
        assistant_text=assistant_text,
        successful=True,
    )


async def test_initialize_salvages_valid_state_around_malformed_records() -> None:
    storage = CoverageArchiveStorage()
    valid = _session()
    storage.metadata = {
        "sessions": [asdict(valid), {"session_id": "broken"}, "not-an-object"],
        "active": {"browser": valid.session_id, "stale": "missing", "bad": 4},
        "partitions": ["2026-09", "2026-13", 12],
    }
    storage.partitions["2026-09"] = {
        "turns": [asdict(_turn()), {"turn_id": "broken"}, "not-an-object"]
    }

    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()
    await archive.async_initialize()

    assert archive.active_session("browser") == valid
    assert archive.active_session("stale") is None
    assert archive.stats()["session_count"] == 1
    assert archive.stats()["turn_count"] == 1
    assert archive.stats()["partition_count"] == 1
    assert (await archive.async_get("user:alice", valid.session_id))["turns"][0][
        "turn_id"
    ] == "turn-1"


@pytest.mark.parametrize(
    "pending",
    [
        [],
        {"bad": {"turns": []}},
        {"2026-09": []},
        {"2026-09": {}},
    ],
)
async def test_initialize_rejects_corrupted_pending_transaction(pending) -> None:
    storage = CoverageArchiveStorage()
    storage.metadata = {"pending_partitions": pending}

    with pytest.raises(ValueError, match="transaction is corrupted"):
        await ConversationArchive(storage, "agent-1").async_initialize()


async def test_initialize_completes_pending_transaction_before_loading() -> None:
    storage = CoverageArchiveStorage()
    session = _session()
    payload = {"turns": [asdict(_turn())]}
    storage.metadata = {
        "sessions": [asdict(session)],
        "active": {"browser": session.session_id},
        "partitions": ["2026-09"],
        "pending_partitions": {"2026-09": payload},
    }

    archive = ConversationArchive(storage, "agent-1")
    await archive.async_initialize()

    assert storage.partitions["2026-09"] == payload
    assert "pending_partitions" not in storage.metadata
    assert archive.stats()["turn_count"] == 1


async def test_begin_session_expires_old_active_session() -> None:
    archive = await _coverage_archive()
    scope = user_scope("alice", source="authenticated_user")
    first = await archive.async_begin_session(
        "browser",
        scope,
        "conversation-1",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    archive._sessions[first.session_id] = replace(
        first, last_message_at="2000-01-01T00:00:00+00:00"
    )

    second = await archive.async_begin_session(
        "browser",
        scope,
        "conversation-1",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=0,
    )

    assert second.session_id != first.session_id
    assert archive.active_session("browser") == second


async def test_record_turn_bounds_text_and_get_paginates() -> None:
    archive = await _coverage_archive()
    session = await archive.async_begin_session(
        "browser",
        user_scope("alice", source="authenticated_user"),
        "conversation-1",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )

    with pytest.raises(ValueError, match="archive text must be a string"):
        await archive.async_record_turn(
            session.session_id,
            run_id=None,
            user_text=None,
            assistant_text="reply",
            successful=True,
        )

    first = await archive.async_record_turn(
        session.session_id,
        run_id=None,
        user_text="x" * (MAX_TEXT_LENGTH + 10),
        assistant_text=" first reply ",
        successful=True,
    )
    await archive.async_record_turn(
        session.session_id,
        run_id="run-2",
        user_text="second",
        assistant_text="reply",
        successful=False,
    )

    assert len(first.user_text) == MAX_TEXT_LENGTH
    assert first.assistant_text == "first reply"
    page = await archive.async_get("user:alice", session.session_id, -4, 1)
    assert page["start_turn"] == 0
    assert page["limit"] == 1
    assert page["has_more"] is True
    assert page["session"]["title"] == "x" * 80


async def test_list_sessions_clamps_orders_and_reports_more() -> None:
    archive = await _coverage_archive()
    scope = user_scope("alice", source="authenticated_user")
    first = await archive.async_begin_session(
        "first",
        scope,
        "one",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    second = await archive.async_begin_session(
        "second",
        scope,
        "two",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    archive._sessions[first.session_id] = replace(
        first, last_message_at="2026-01-01T00:00:00+00:00"
    )
    archive._sessions[second.session_id] = replace(
        second, last_message_at="2026-02-01T00:00:00+00:00"
    )

    page = await archive.async_list_sessions("user:alice", limit=1, offset=-3)

    assert page["offset"] == 0
    assert page["limit"] == 1
    assert page["has_more"] is True
    assert page["sessions"][0]["session_id"] == second.session_id


async def test_bulk_delete_validation_is_atomic_and_date_range_can_noop() -> None:
    archive = await _coverage_archive()
    alice = await archive.async_begin_session(
        "alice",
        user_scope("alice", source="authenticated_user"),
        "one",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    bob = await archive.async_begin_session(
        "bob",
        user_scope("bob", source="authenticated_user"),
        "two",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )

    with pytest.raises(ValueError, match="Explicit confirmation"):
        await archive.async_clear_scope("user:alice", confirm=False)
    with pytest.raises(ValueError, match="Explicit confirmation"):
        await archive.async_delete_selected("user:alice", [alice.session_id], confirm=False)
    with pytest.raises(ValueError, match="session_ids must contain"):
        await archive.async_delete_selected("user:alice", [], confirm=True)
    with pytest.raises(ValueError, match="session_ids must contain"):
        await archive.async_delete_selected(
            "user:alice",
            [str(index) for index in range(MAX_SEARCH_LIMIT + 1)],
            confirm=True,
        )
    with pytest.raises(ValueError, match="not found"):
        await archive.async_delete_selected(
            "user:alice", [alice.session_id, bob.session_id], confirm=True
        )
    assert archive.stats()["session_count"] == 2

    with pytest.raises(ValueError, match="Explicit confirmation"):
        await archive.async_delete_date_range(
            "user:alice", "2026-01-01", "2026-12-31", confirm=False
        )
    for start, end in (
        ("bad", "2026-12-31"),
        ("2026-01-01", "bad"),
        ("2026-12-31", "2026-01-01"),
    ):
        with pytest.raises(ValueError, match="valid start_date"):
            await archive.async_delete_date_range("user:alice", start, end, confirm=True)
    assert await archive.async_delete_date_range(
        "user:alice", "1990-01-01", "1990-01-02", confirm=True
    ) == {"deleted_sessions": 0, "deleted_turns": 0}


async def test_delete_selected_deduplicates_and_clears_active_mapping() -> None:
    archive = await _coverage_archive()
    session = await archive.async_begin_session(
        "browser",
        user_scope("alice", source="authenticated_user"),
        "conversation",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        session.session_id,
        run_id="run",
        user_text="hello",
        assistant_text="reply",
        successful=True,
    )

    result = await archive.async_delete_selected(
        "user:alice", [session.session_id, session.session_id], confirm=True
    )

    assert result == {"deleted_sessions": 1, "deleted_turns": 1}
    assert archive.active_session("browser") is None


async def test_backup_data_omits_unretained_runtime_sessions() -> None:
    archive = await _coverage_archive()
    retained = await archive.async_begin_session(
        "retained",
        user_scope("alice", source="authenticated_user"),
        "one",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    bob = await archive.async_begin_session(
        "unretained",
        user_scope("bob", source="authenticated_user"),
        "two",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    archive._sessions[bob.session_id] = replace(bob, retention_state="unretained")
    await archive.async_record_turn(
        retained.session_id,
        run_id="run",
        user_text="hello",
        assistant_text="reply",
        successful=True,
    )

    backup = await archive.async_backup_data()

    assert [item["session_id"] for item in backup["sessions"]] == [retained.session_id]
    assert [item["session_id"] for item in backup["turns"]] == [retained.session_id]
    assert "active" not in backup


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "incomplete or corrupted"),
        ({"sessions": []}, "incomplete or corrupted"),
        ({"sessions": {}, "turns": []}, "sessions and turns must be lists"),
        ({"sessions": ["bad"], "turns": []}, "session must be an object"),
        ({"sessions": [{"session_id": "only"}], "turns": []}, "session is invalid"),
    ],
)
def test_validate_backup_rejects_structural_corruption(data, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ConversationArchive.validate_backup_data(data, "target-agent")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: {**raw, "session_id": ""},
        lambda raw: {**raw, "session_id": "x" * 129},
        lambda raw: {**raw, "retention_state": "unretained"},
        lambda raw: {**raw, "turn_count": True},
        lambda raw: {**raw, "turn_count": -1},
        lambda raw: {**raw, "title": "x" * 81},
        lambda raw: {**raw, "started_at": "not-a-time"},
        lambda raw: {**raw, "last_message_at": "not-a-time"},
        lambda raw: {**raw, "scope_id": 7},
        lambda raw: {**raw, "source_device_id": 7},
    ],
)
def test_validate_backup_rejects_invalid_session_metadata(mutator) -> None:
    data = {"sessions": [mutator(asdict(_session(turn_count=0)))], "turns": []}

    with pytest.raises(
        ValueError, match="session metadata is invalid|fields have invalid types"
    ):
        ConversationArchive.validate_backup_data(data, "target-agent")


def test_validate_backup_rebinds_agent_and_rejects_duplicate_sessions() -> None:
    session = _session(turn_count=0)
    sessions, turns = ConversationArchive.validate_backup_data(
        {"sessions": [asdict(session)], "turns": []}, "target-agent"
    )
    assert sessions[0].agent_subentry_id == "target-agent"
    assert turns == []

    with pytest.raises(ValueError, match="session metadata is invalid"):
        ConversationArchive.validate_backup_data(
            {"sessions": [asdict(session), asdict(session)], "turns": []},
            "target-agent",
        )


@pytest.mark.parametrize(
    "turn_data",
    [
        "bad",
        {"turn_id": "only"},
        {**asdict(_turn()), "turn_id": ""},
        {**asdict(_turn()), "turn_id": "x" * 129},
        {**asdict(_turn()), "session_id": "missing"},
        {**asdict(_turn()), "run_id": 7},
        {**asdict(_turn()), "successful": 1},
        {**asdict(_turn()), "timestamp": "bad"},
        {**asdict(_turn()), "user_text": "x" * (MAX_TEXT_LENGTH + 1)},
        {**asdict(_turn()), "assistant_text": "x" * (MAX_TEXT_LENGTH + 1)},
    ],
)
def test_validate_backup_rejects_invalid_turn_metadata(turn_data) -> None:
    data = {"sessions": [asdict(_session())], "turns": [turn_data]}

    with pytest.raises(
        ValueError, match="turn must be an object|turn is invalid|turn metadata is invalid"
    ):
        ConversationArchive.validate_backup_data(data, "target-agent")


def test_validate_backup_rejects_duplicate_turns_and_count_mismatch() -> None:
    session = _session(turn_count=2)
    turn = _turn()
    with pytest.raises(ValueError, match="turn metadata is invalid"):
        ConversationArchive.validate_backup_data(
            {"sessions": [asdict(session)], "turns": [asdict(turn), asdict(turn)]},
            "target-agent",
        )

    with pytest.raises(ValueError, match="turn counts do not match"):
        ConversationArchive.validate_backup_data(
            {"sessions": [asdict(session)], "turns": [asdict(turn)]},
            "target-agent",
        )


async def test_replace_backup_rebuilds_partitions_and_clears_active() -> None:
    storage = CoverageArchiveStorage()
    archive = await _coverage_archive(storage)
    original = await archive.async_begin_session(
        "browser",
        user_scope("alice", source="authenticated_user"),
        "old",
        archive_enabled=True,
        shared_archive_enabled=False,
        inactivity_minutes=30,
    )
    await archive.async_record_turn(
        original.session_id,
        run_id="old",
        user_text="old",
        assistant_text="old",
        successful=True,
    )
    replacement = _session("replacement", turn_count=2)
    turns = [
        _turn("replacement", turn_id="jan", timestamp="2026-01-02T00:00:00+00:00"),
        _turn("replacement", turn_id="feb", timestamp="2026-02-02T00:00:00+00:00"),
    ]

    await archive.async_replace_backup([replacement], turns)

    assert archive.active_session("browser") is None
    assert archive.stats()["partition_count"] == 2
    assert set(storage.partitions) >= {"2026-01", "2026-02"}
    result = await archive.async_get("user:alice", "replacement")
    assert [turn["turn_id"] for turn in result["turns"]] == ["jan", "feb"]


async def test_uninitialized_operations_fail_fast() -> None:
    archive = ConversationArchive(CoverageArchiveStorage(), "agent-1")

    with pytest.raises(RuntimeError, match="has not been initialized"):
        await archive.async_begin_session(
            "browser",
            user_scope("alice", source="authenticated_user"),
            "conversation",
            archive_enabled=True,
            shared_archive_enabled=False,
            inactivity_minutes=30,
        )
    with pytest.raises(RuntimeError, match="has not been initialized"):
        await archive.async_backup_data()
    with pytest.raises(RuntimeError, match="has not been initialized"):
        await archive.async_replace_backup([], [])


async def test_async_get_archive_reuses_one_manager_per_agent(monkeypatch) -> None:
    storages = []

    def fake_storage(hass, entry_id: str, subentry_id: str):
        storage = CoverageArchiveStorage()
        storages.append((entry_id, subentry_id, storage))
        return storage

    monkeypatch.setattr(archive_module, "HomeAssistantArchiveStorage", fake_storage)
    hass = SimpleNamespace(data={})

    first = await async_get_archive(hass, "entry-1", "agent-1")
    second = await async_get_archive(hass, "entry-1", "agent-1")
    other = await async_get_archive(hass, "entry-1", "agent-2")

    assert first is second
    assert other is not first
    assert [(entry_id, subentry_id) for entry_id, subentry_id, _ in storages] == [
        ("entry-1", "agent-1"),
        ("entry-1", "agent-2"),
    ]


async def test_home_assistant_storage_delegates_to_private_atomic_stores(monkeypatch) -> None:
    created = []

    class FakeStore:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, hass, version, key, **kwargs):
            created.append((hass, version, key, kwargs))
            self.loaded = {"key": key}
            self.saved = []

        async def async_load(self):
            return self.loaded

        async def async_save(self, data):
            self.saved.append(data)

    monkeypatch.setattr(archive_module, "PropagatingWriteStore", FakeStore)
    hass = object()
    storage = HomeAssistantArchiveStorage(hass, "entry", "agent")

    assert await storage.async_load_metadata() == {
        "key": f"{archive_module.STORAGE_KEY_PREFIX}.entry.agent.metadata"
    }
    await storage.async_save_metadata({"sessions": []})
    assert storage._metadata.saved == [{"sessions": []}]
    assert await storage.async_load_partition("2026-09") == {
        "key": f"{archive_module.STORAGE_KEY_PREFIX}.entry.agent.turns.2026-09"
    }
    await storage.async_save_partition("2026-09", {"turns": []})

    assert len(created) == 3
    assert created[0][3]["private"] is True
    assert created[0][3]["atomic_writes"] is True
    assert created[1][3]["serialize_in_event_loop"] is False


def test_archive_tools_are_bounded_and_map_to_expected_operations() -> None:
    tools = archive_tools()

    assert [tool["function"]["operation"] for tool in tools] == [
        "search",
        "get",
        "private",
        "resume",
        "delete_current",
        "delete_selected",
        "delete_range",
    ]
    assert tools[0]["spec"]["parameters"]["additionalProperties"] is False
    assert tools[0]["spec"]["parameters"]["properties"]["limit"]["maximum"] == 10
    selected = tools[5]["spec"]["parameters"]["properties"]["session_ids"]
    assert selected["maxItems"] == 50


def test_search_snapshot_filters_dates_ranks_phrase_and_paginates() -> None:
    session = _session(turn_count=3)
    turns = (
        _turn(
            turn_id="older",
            timestamp="2026-08-01T10:00:00+00:00",
            user_text="Where are the running shoes?",
            assistant_text="By the door",
        ),
        _turn(
            turn_id="phrase",
            timestamp="2026-09-02T10:00:00+00:00",
            user_text="The blue running shoes",
            assistant_text="were selected",
        ),
        _turn(
            turn_id="unrelated",
            timestamp="2026-09-03T10:00:00+00:00",
            user_text="weather forecast",
            assistant_text="sunny",
        ),
    )

    phrase = _search_archive_snapshot(
        ((session, turns),), "blue running shoes", "2026-09-01", None, 1, 0
    )
    assert [item["turn_id"] for item in phrase["results"]] == ["phrase"]
    assert phrase["has_more"] is False

    first = _search_archive_snapshot(((session, turns),), "shoes", None, None, 1, 0)
    second = _search_archive_snapshot(((session, turns),), "shoes", None, None, 1, 1)
    assert first["has_more"] is True
    assert first["results"][0]["turn_id"] != second["results"][0]["turn_id"]


def test_text_search_and_time_helpers_cover_edge_forms() -> None:
    assert _clean_text("  hello  ") == "hello"
    assert _clean_text("x" * (MAX_TEXT_LENGTH + 4)) == "x" * MAX_TEXT_LENGTH
    with pytest.raises(ValueError, match="must be a string"):
        _clean_text(3)

    assert _title("  many\n  spaces  ") == "many spaces"
    assert _stem("stories") == "story"
    assert _stem("running") == "runn"
    assert _stem("cars") == "cars"
    assert _stem("glass") == "glass"
    assert _stem("cat") == "cat"
    assert _tokens("The stories and running cars") == {"story", "runn", "cars"}

    value = "prefix " * 100 + "needle phrase" + " suffix" * 100
    excerpt = _excerpt(value, "needle phrase")
    assert excerpt.startswith("…")
    assert "needle phrase" in excerpt
    assert excerpt.endswith("…")
    assert _excerpt("short text", "missing") == "short text"

    aware = _parse_time("2026-09-13T10:00:00+01:00")
    naive = _parse_time("2026-09-13T10:00:00")
    invalid = _parse_time("not-a-time")
    assert aware.utcoffset() is not None
    assert naive.tzinfo is not None
    assert invalid == datetime.min.replace(tzinfo=archive_module.dt_util.UTC)
