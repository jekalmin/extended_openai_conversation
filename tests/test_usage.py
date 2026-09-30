"""Tests for provider-reported persistent usage statistics."""

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace

import pytest
from homeassistant.util import dt as dt_util

from custom_components.extended_openai_conversation_responses import usage
from custom_components.extended_openai_conversation_responses.entity import (
    ExtendedOpenAIBaseLLMEntity,
)
from custom_components.extended_openai_conversation_responses.usage import (
    RequestUsage,
    UsageManager,
    extract_usage,
)


class FakeStream:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self) -> AsyncIterator:
        for item in self.items:
            yield item


class FakeStorage:
    """In-memory persistence boundary."""

    data: dict | None = None

    async def async_load(self):
        return deepcopy(self.data)

    async def async_save(self, data):
        self.data = deepcopy(data)


class FailingStorage(FakeStorage):
    async def async_save(self, data):
        raise OSError("disk full")


async def _manager(storage: FakeStorage | None = None) -> UsageManager:
    manager = UsageManager(storage or FakeStorage())
    await manager.async_initialize()
    return manager


async def test_single_request_usage_and_conversation_count() -> None:
    """A conversation and its one API request remain distinct counters."""
    manager = await _manager()
    await manager.async_record_conversation()
    await manager.async_record_request(
        successful=True,
        usage=RequestUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )

    assert manager.as_dict() == {
        "conversation_count": 1,
        "api_request_count": 1,
        "successful_request_count": 1,
        "failed_request_count": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "details": {},
    }


async def test_multi_round_tool_usage_is_aggregated() -> None:
    """Every provider request in one tool-call conversation is accumulated."""
    manager = await _manager()
    await manager.async_record_conversation()
    await manager.async_record_request(
        successful=True,
        usage=RequestUsage(
            input_tokens=20,
            output_tokens=3,
            total_tokens=23,
            cached_input_tokens=4,
        ),
    )
    await manager.async_record_request(
        successful=True,
        usage=RequestUsage(
            input_tokens=30,
            output_tokens=7,
            total_tokens=37,
            reasoning_tokens=2,
        ),
    )

    assert manager.totals.conversation_count == 1
    assert manager.totals.api_request_count == 2
    assert manager.totals.total_tokens == 60
    assert manager.totals.cached_input_tokens == 4
    assert manager.totals.reasoning_tokens == 2


async def test_missing_usage_metadata_degrades_gracefully() -> None:
    """Compatible providers may omit usage while the successful request is counted."""
    manager = await _manager()
    await manager.async_record_request(successful=True, usage=extract_usage(None))

    assert manager.totals.successful_request_count == 1
    assert manager.totals.total_tokens == 0


async def test_failed_api_request_is_counted_without_tokens() -> None:
    """A rejected request increments request and failure counters only."""
    manager = await _manager()
    await manager.async_record_request(successful=False)

    assert manager.totals.api_request_count == 1
    assert manager.totals.failed_request_count == 1
    assert manager.totals.successful_request_count == 0
    assert manager.totals.total_tokens == 0


async def test_usage_persists_across_reload() -> None:
    """Cumulative statistics survive manager recreation."""
    storage = FakeStorage()
    first = await _manager(storage)
    await first.async_record_conversation()
    await first.async_record_request(
        successful=True,
        usage=RequestUsage(input_tokens=8, output_tokens=2, total_tokens=10),
    )

    second = await _manager(storage)
    assert second.totals.conversation_count == 1
    assert second.totals.total_tokens == 10


def test_chat_and_responses_detailed_usage_fields() -> None:
    """Known detailed token fields are normalized without provider assumptions."""
    chat = extract_usage(
        SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=6,
            total_tokens=18,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )
    responses = extract_usage(
        {
            "input_tokens": 7,
            "output_tokens": 4,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        }
    )

    assert (chat.cached_input_tokens, chat.reasoning_tokens) == (5, 3)
    assert (responses.total_tokens, responses.cached_input_tokens) == (11, 2)


async def test_chat_stream_consumes_usage_chunk_after_stop() -> None:
    """Chat Completions does not stop before the final usage-only chunk."""
    stop = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="OK", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    final_usage = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=2, total_tokens=11),
    )
    entity = ExtendedOpenAIBaseLLMEntity.__new__(ExtendedOpenAIBaseLLMEntity)
    request_usage = RequestUsage()
    chat_log = SimpleNamespace(async_trace=lambda _: None)

    deltas = [
        delta
        async for delta in entity._transform_chat_stream(
            chat_log, FakeStream([stop, final_usage]), request_usage
        )
    ]

    assert any(delta.get("content") == "OK" for delta in deltas)
    assert request_usage.total_tokens == 11


async def test_run_lifecycle_groups_multiple_requests_and_finalizes_once() -> None:
    manager = await _manager()
    async with manager.async_run(
        home_assistant_conversation_id="conversation-1",
        source_device_id="satellite-1",
    ) as run:
        await manager.async_record_request(
            successful=True,
            usage=RequestUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            provider="openai",
            model="gpt-5-mini",
            api_mode="responses",
            request_stage="initial",
            tool_calls_requested=1,
        )
        await manager.async_record_request(
            successful=False,
            provider="openai",
            model="gpt-5-mini",
            api_mode="responses",
            request_stage="after_tool",
            error_type="APIError",
        )
    assert manager.totals.conversation_count == 1
    assert run.request_count == 2
    assert run.tool_call_count == 1
    assert run.successful is False
    assert manager.today_summary()["run_count"] == 1
    assert manager.today_summary()["total_tokens"] == 12
    assert manager.breakdowns()["providers"] == {"openai": 12}
    await manager.async_clear_details(confirm=True)
    assert manager.breakdowns()["providers"] == {"openai": 12}
    assert manager.today_summary()["total_tokens"] == 12


async def test_exception_run_is_finalized_without_request_metadata() -> None:
    manager = await _manager()
    try:
        async with manager.async_run() as run:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert run.completed_at is not None
    assert run.error_type == "RuntimeError"
    assert run.request_count == 0
    assert manager.totals.conversation_count == 1


async def test_persistence_failure_keeps_successful_request_and_run_in_memory(
    caplog,
) -> None:
    manager = UsageManager(
        FailingStorage(), FailingStorage(), FailingStorage(), agent_subentry_id="agent"
    )
    await manager.async_initialize()

    async with manager.async_run() as run:
        await manager.async_record_request(
            successful=True,
            usage=RequestUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        )

    assert run.successful is True
    assert manager.totals.api_request_count == 1
    assert manager.totals.conversation_count == 1
    assert manager.totals.total_tokens == 5
    assert "Unable to persist usage" in caplog.text


class CoverageMemoryStorage:
    def __init__(self, data=None, *, fail_load: bool = False, fail_save: bool = False):
        self.data = deepcopy(data)
        self.fail_load = fail_load
        self.fail_save = fail_save
        self.saves = []

    async def async_load(self):
        if self.fail_load:
            raise OSError("load failed")
        return deepcopy(self.data)

    async def async_save(self, data):
        if self.fail_save:
            raise OSError("save failed")
        self.data = deepcopy(data)
        self.saves.append(deepcopy(data))


async def _coverage_manager(
    storage=None,
    daily_storage=None,
    detail_storage=None,
    **kwargs,
):
    manager = usage.UsageManager(
        storage or CoverageMemoryStorage(),
        daily_storage,
        detail_storage,
        agent_subentry_id="agent-target",
        **kwargs,
    )
    await manager.async_initialize()
    return manager


def _coverage_request(*, request_id="request-1", run_id="run-1", timestamp=None, **changes):
    values = {
        "request_id": request_id,
        "run_id": run_id,
        "timestamp": timestamp or dt_util.utcnow().isoformat(),
        "agent_subentry_id": "source-agent",
        "provider": "openai",
        "model": "gpt-test",
        "api_mode": "responses",
        "successful": True,
        "duration_ms": 10,
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "cached_input_tokens": 1,
        "reasoning_tokens": 0,
        "request_stage": "initial",
        "tool_calls_requested": 1,
        "web_search_used": False,
        "error_type": None,
        "details": {"input_cached_tokens": 1},
    }
    values.update(changes)
    return values


def _coverage_run(*, run_id="run-1", started_at=None, **changes):
    values = {
        "run_id": run_id,
        "started_at": started_at or dt_util.utcnow().isoformat(),
        "completed_at": dt_util.utcnow().isoformat(),
        "duration_ms": 20,
        "agent_subentry_id": "source-agent",
        "home_assistant_conversation_id": "conversation-1",
        "source_device_id": "device-1",
        "request_count": 1,
        "successful_request_count": 1,
        "failed_request_count": 0,
        "tool_call_count": 1,
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "cached_input_tokens": 1,
        "reasoning_tokens": 0,
        "successful": True,
        "models": ["gpt-test"],
        "providers": ["openai"],
        "api_modes": ["responses"],
        "web_search_used": False,
        "error_type": None,
    }
    values.update(changes)
    return values


def _coverage_backup():
    day = usage._empty_day("2026-09-13")
    day.update(
        {
            "run_count": 1,
            "successful_run_count": 1,
            "api_request_count": 1,
            "successful_request_count": 1,
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "cached_input_tokens": 1,
            "tool_call_count": 1,
            "total_run_duration_ms": 20,
            "average_tokens_per_completed_run": 5,
            "average_requests_per_completed_run": 1.0,
            "average_duration_ms_per_completed_run": 20.0,
            "provider_breakdown": {"openai": 5},
            "model_breakdown": {"gpt-test": 5},
            "api_mode_breakdown": {"responses": 5},
        }
    )
    return {
        "totals": asdict(
            usage.UsageTotals(
                conversation_count=1,
                api_request_count=1,
                successful_request_count=1,
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
                cached_input_tokens=1,
                details={"input_cached_tokens": 1},
            )
        ),
        "daily": {"2026-09-13": day},
        "requests": [_coverage_request()],
        "runs": [_coverage_run()],
    }


def test_extract_usage_salvages_object_and_model_dump_details() -> None:
    class Details:
        def __init__(self):
            self.cached_tokens = 4
            self.extra = 2

    class DumpDetails:
        def __init__(self):
            self.reasoning_tokens = 3

        def model_dump(self, *, exclude_none):
            assert exclude_none is True
            return {"reasoning_tokens": 3, "accepted_prediction_tokens": 2, "zero": 0}

    result = usage.extract_usage(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=0,
            prompt_tokens_details=Details(),
            completion_tokens_details=DumpDetails(),
        )
    )

    assert result.total_tokens == 15
    assert result.cached_input_tokens == 4
    assert result.reasoning_tokens == 3
    assert result.details == {
        "input_cached_tokens": 4,
        "input_extra": 2,
        "output_reasoning_tokens": 3,
        "output_accepted_prediction_tokens": 2,
    }


def test_totals_storage_salvages_only_positive_integer_counters() -> None:
    totals = usage._totals_from_storage(
        {
            "totals": {
                "conversation_count": 2,
                "api_request_count": True,
                "successful_request_count": -1,
                "failed_request_count": "3",
                "input_tokens": 5,
                "details": {"good": 7, "zero": 0, "bad": "8", "negative": -1},
                "unknown": 99,
            }
        }
    )

    assert totals.conversation_count == 2
    assert totals.api_request_count == 0
    assert totals.successful_request_count == 0
    assert totals.failed_request_count == 0
    assert totals.input_tokens == 5
    assert totals.details == {"good": 7}
    assert usage._totals_from_storage(None) == usage.UsageTotals()
    assert usage._totals_from_storage({"totals": []}) == usage.UsageTotals()


async def test_initialize_salvages_valid_detail_records_and_applies_retention() -> None:
    now = dt_util.utcnow()
    detail_storage = CoverageMemoryStorage(
        {
            "requests": [
                _coverage_request(request_id="recent", timestamp=(now - timedelta(days=1)).isoformat()),
                {"bad": "request"},
                _coverage_request(request_id="old", timestamp=(now - timedelta(days=90)).isoformat()),
            ],
            "runs": [
                _coverage_run(run_id="recent", started_at=(now - timedelta(days=1)).isoformat()),
                {"bad": "run"},
                _coverage_run(run_id="old", started_at=(now - timedelta(days=90)).isoformat()),
            ],
        }
    )
    daily_storage = CoverageMemoryStorage(
        {
            "totals": {"conversation_count": 4, "details": {}},
            "days": {"2026-09-13": {"total_tokens": 9}, "bad": []},
        }
    )
    manager = await _coverage_manager(
        CoverageMemoryStorage({"conversation_count": 99}),
        daily_storage,
        detail_storage,
        request_retention_days=30,
        run_retention_days=30,
    )

    assert manager.totals.conversation_count == 4
    assert list(manager.daily) == ["2026-09-13"]
    assert [item.request_id for item in manager.requests] == ["recent"]
    assert [item.run_id for item in manager.runs] == ["recent"]
    await manager.async_initialize()
    assert [item.request_id for item in manager.requests] == ["recent"]


async def test_request_outside_run_updates_aggregates_without_detail_record() -> None:
    details = CoverageMemoryStorage()
    manager = await _coverage_manager(CoverageMemoryStorage(), CoverageMemoryStorage(), details)
    await manager.async_record_request(
        successful=False,
        usage=usage.RequestUsage(total_tokens=4),
        provider="provider",
        model="model",
        api_mode="chat",
        error_type="ProviderError",
    )

    assert manager.totals.api_request_count == 1
    assert manager.totals.failed_request_count == 1
    assert manager.requests == []
    assert details.saves == []
    assert manager.latest_run is None


async def test_mark_failed_clamps_error_and_finalize_is_idempotent() -> None:
    manager = await _coverage_manager()
    manager.mark_current_run_failed("ignored-without-run")

    async with manager.async_run() as run:
        manager.mark_current_run_failed("x" * 200)
        assert manager.current_run() is run
    assert manager.current_run() is None
    assert run.successful is False
    assert run.error_type == "x" * 128
    assert manager.totals.conversation_count == 1

    await manager._async_finalize_run(run)
    assert manager.totals.conversation_count == 1


async def test_prune_and_clear_details_cover_zero_retention_and_confirmation() -> None:
    details = CoverageMemoryStorage()
    manager = await _coverage_manager(
        CoverageMemoryStorage(),
        CoverageMemoryStorage(),
        details,
        request_retention_days=0,
        run_retention_days=0,
    )
    manager.requests = [usage.UsageRequest(**_coverage_request())]
    manager.runs = [usage.UsageRun(**_coverage_run())]

    result = await manager.async_prune_details(save=False)
    assert result == {"deleted_requests": 1, "deleted_runs": 1}
    assert details.saves == []

    manager.requests = [usage.UsageRequest(**_coverage_request())]
    manager.runs = [usage.UsageRun(**_coverage_run())]
    with pytest.raises(ValueError, match="confirmation"):
        await manager.async_clear_details(confirm=False)
    result = await manager.async_clear_details(confirm=True)
    assert result == {"deleted_requests": 1, "deleted_runs": 1}
    assert details.data == {"requests": [], "runs": []}


async def test_summary_series_pagination_breakdowns_and_listener_lifecycle() -> None:
    manager = await _coverage_manager()
    first_day = usage._empty_day("2026-09-01")
    first_day.update(
        {
            "run_count": 2,
            "api_request_count": 3,
            "total_tokens": 20,
            "total_run_duration_ms": 100,
            "provider_breakdown": {"openai": 20},
            "model_breakdown": {"gpt-a": 20},
            "api_mode_breakdown": {"responses": 20},
        }
    )
    usage._calculate_averages(first_day)
    second_day = usage._empty_day("2026-09-02")
    second_day.update(
        {
            "run_count": 1,
            "api_request_count": 1,
            "total_tokens": 5,
            "total_run_duration_ms": 50,
            "provider_breakdown": {"other": 5},
            "model_breakdown": {"gpt-b": 5},
            "api_mode_breakdown": {"chat": 5},
        }
    )
    usage._calculate_averages(second_day)
    manager.daily = {"2026-09-01": first_day, "2026-09-02": second_day}
    manager.runs = [
        usage.UsageRun(**_coverage_run(run_id="one", successful=True)),
        usage.UsageRun(**_coverage_run(run_id="two", successful=False)),
        usage.UsageRun(**_coverage_run(run_id="three", successful=True)),
    ]
    manager.requests = [
        usage.UsageRequest(**_coverage_request(request_id="a", run_id="one")),
        usage.UsageRequest(**_coverage_request(request_id="b", run_id="one")),
        usage.UsageRequest(**_coverage_request(request_id="c", run_id="two")),
    ]

    assert manager.summary_for_date("missing")["run_count"] == 0
    month = manager.month_summary("2026-09")
    assert month["run_count"] == 3
    assert month["total_tokens"] == 25
    assert month["average_requests_per_completed_run"] == pytest.approx(1.33)
    assert [d["date"] for d in manager.daily_series("2026-09-01", "2026-09-30", limit=1)] == ["2026-09-01"]

    page = manager.recent_runs(limit=1, offset=1, successful=True)
    assert [item["run_id"] for item in page["runs"]] == ["one"]
    assert page["has_more"] is False
    assert manager.recent_runs(limit=999, offset=-5)["limit"] == usage.MAX_RECENT_LIMIT

    requests = manager.requests_for_run("one", limit=1)
    assert requests["requests"][0]["request_id"] == "a"
    assert requests["has_more"] is True
    assert manager.requests_for_run("missing", offset=-2)["offset"] == 0

    assert manager.breakdowns("2026-09-02", "2026-09-02") == {
        "providers": {"other": 5},
        "models": {"gpt-b": 5},
        "api_modes": {"chat": 5},
    }
    assert manager.latest_run.run_id == "three"

    called = []
    remove = manager.async_add_listener(lambda: called.append(True))
    manager._notify()
    remove()
    manager._notify()
    assert called == [True]


async def test_backup_export_requires_initialization_and_returns_deep_copy() -> None:
    manager = usage.UsageManager(CoverageMemoryStorage())
    with pytest.raises(RuntimeError, match="not been initialized"):
        await manager.async_backup_data()

    await manager.async_initialize()
    manager.daily["2026-09-13"] = usage._empty_day("2026-09-13")
    exported = await manager.async_backup_data()
    exported["daily"]["2026-09-13"]["total_tokens"] = 999
    assert manager.daily["2026-09-13"]["total_tokens"] == 0


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda data: data.pop("runs"), "incomplete or corrupted"),
        (lambda data: data.__setitem__("totals", {}), "totals are invalid"),
        (lambda data: data["totals"].__setitem__("input_tokens", True), "counter input_tokens"),
        (lambda data: data["totals"].__setitem__("details", {"x": -1}), "total details"),
        (lambda data: data.__setitem__("daily", []), "daily usage data"),
        (lambda data: data["daily"].__setitem__("bad", usage._empty_day("bad")), "daily usage date"),
        (lambda data: data["daily"]["2026-09-13"].__setitem__("date", "2026-09-12"), "does not match"),
        (lambda data: data.__setitem__("requests", {}), "details must be lists"),
        (lambda data: data["requests"].append(deepcopy(data["requests"][0])), "request IDs must be unique"),
        (lambda data: data["runs"].append(deepcopy(data["runs"][0])), "run IDs must be unique"),
    ],
)
def test_validate_backup_rejects_corrupted_top_level_shapes(mutator, match) -> None:
    data = _coverage_backup()
    mutator(data)
    with pytest.raises(ValueError, match=match):
        usage.UsageManager.validate_backup_data(data, "target")


def test_validate_backup_rebinds_agent_and_accepts_valid_state() -> None:
    totals, daily, requests, runs = usage.UsageManager.validate_backup_data(
        _coverage_backup(), "target-agent"
    )
    assert totals.total_tokens == 5
    assert daily["2026-09-13"]["provider_breakdown"] == {"openai": 5}
    assert requests[0].agent_subentry_id == "target-agent"
    assert runs[0].agent_subentry_id == "target-agent"


@pytest.mark.parametrize(
    ("target", "change", "match"),
    [
        ("request", {"timestamp": "not-a-time"}, "request metadata"),
        ("request", {"successful": 1}, "request metadata"),
        ("request", {"duration_ms": -1}, "counter duration_ms"),
        ("request", {"details": {"bad": True}}, "request details"),
        ("run", {"started_at": "bad"}, "run metadata"),
        ("run", {"models": "gpt-test"}, "run metadata"),
        ("run", {"models": [1]}, "run metadata"),
        ("run", {"request_count": -1}, "counter request_count"),
    ],
)
def test_backup_detail_validators_reject_invalid_metadata(target, change, match) -> None:
    raw = _coverage_request(**change) if target == "request" else _coverage_run(**change)
    validator = usage._usage_request_from_backup if target == "request" else usage._usage_run_from_backup
    with pytest.raises(ValueError, match=match):
        validator(raw, "target")


def test_backup_detail_validators_reject_non_objects_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="request must be an object"):
        usage._usage_request_from_backup([], "target")
    with pytest.raises(ValueError, match="request is invalid"):
        usage._usage_request_from_backup({**_coverage_request(), "unknown": 1}, "target")
    with pytest.raises(ValueError, match="run must be an object"):
        usage._usage_run_from_backup([], "target")
    with pytest.raises(ValueError, match="run is invalid"):
        usage._usage_run_from_backup({**_coverage_run(), "unknown": 1}, "target")


async def test_replace_backup_persists_then_reapplies_retention_and_notifies() -> None:
    details = CoverageMemoryStorage()
    daily = CoverageMemoryStorage()
    manager = await _coverage_manager(
        CoverageMemoryStorage(),
        daily,
        details,
        request_retention_days=0,
        run_retention_days=0,
    )
    totals, days, requests, runs = usage.UsageManager.validate_backup_data(
        _coverage_backup(), "agent-target"
    )
    calls = []
    manager.async_add_listener(lambda: calls.append(True))

    await manager.async_replace_backup(totals, days, requests, runs)

    assert manager.totals.total_tokens == 5
    assert manager.requests == []
    assert manager.runs == []
    assert daily.data["totals"]["total_tokens"] == 5
    assert details.data == {"requests": [], "runs": []}
    assert calls == [True]


async def test_legacy_totals_mirror_failure_is_non_fatal(caplog) -> None:
    primary = CoverageMemoryStorage(fail_save=True)
    daily = CoverageMemoryStorage()
    manager = await _coverage_manager(primary, daily)

    await manager.async_record_conversation()

    assert daily.data["totals"]["conversation_count"] == 1
    assert "legacy usage totals mirror" in caplog.text


async def test_save_guards_and_listener_failures_do_not_escape(caplog) -> None:
    manager = usage.UsageManager(CoverageMemoryStorage())
    with pytest.raises(RuntimeError, match="not been initialized"):
        await manager._async_save_totals()
    with pytest.raises(RuntimeError, match="not been initialized"):
        await manager._async_save_aggregates()

    await manager.async_initialize()

    async def fail():
        raise OSError("disk full")

    await manager._async_save_safely("test data", fail)
    manager.async_add_listener(lambda: (_ for _ in ()).throw(RuntimeError("listener")))
    manager._notify()
    assert "Unable to persist usage test data" in caplog.text
    assert "Usage listener failed" in caplog.text


def test_time_day_and_merge_helpers_cover_edge_cases() -> None:
    naive = usage._parse_time("2026-09-13T12:00:00")
    aware = usage._parse_time("2026-09-13T12:00:00+00:00")
    invalid = usage._parse_time(None)
    assert naive.tzinfo is not None
    assert aware.tzinfo is not None
    assert invalid == usage.datetime.min.replace(tzinfo=dt_util.UTC)

    target = usage._empty_day("2026-09")
    source = usage._empty_day("2026-09-01")
    source.update(
        {
            "run_count": 2,
            "total_tokens": 11,
            "provider_breakdown": {"openai": 11},
            "average_requests_per_completed_run": 99.0,
        }
    )
    usage._merge_day(target, source)
    usage._calculate_averages(target)
    assert target["run_count"] == 2
    assert target["total_tokens"] == 11
    assert target["provider_breakdown"] == {"openai": 11}
    assert target["average_requests_per_completed_run"] == 0


async def test_async_get_durable_usage_publishes_single_manager_before_initialization(monkeypatch) -> None:
    created = []

    class FakeStore(CoverageMemoryStorage):
        def __init__(self, hass, version, key, **kwargs):
            super().__init__()
            created.append((version, key, kwargs))

    real_initialize = usage.UsageManager.async_initialize
    entered = asyncio.Event()
    release = asyncio.Event()
    initialize_calls = 0

    async def blocked_initialize(self):
        nonlocal initialize_calls
        initialize_calls += 1
        entered.set()
        await release.wait()
        await real_initialize(self)

    monkeypatch.setattr(usage, "Store", FakeStore)
    monkeypatch.setattr(usage.UsageManager, "async_initialize", blocked_initialize)
    listeners = []
    hass = SimpleNamespace(
        data={},
        bus=SimpleNamespace(async_listen_once=lambda *args: listeners.append(args)),
    )

    first_task = asyncio.create_task(usage.async_get_durable_usage(hass, "entry", "agent"))
    await entered.wait()
    second_task = asyncio.create_task(usage.async_get_durable_usage(hass, "entry", "agent"))
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first is second
    assert initialize_calls == 2
    assert len(created) == 3
    assert created[2][2]["private"] is True
    assert created[2][2]["serialize_in_event_loop"] is False
    assert await usage.async_get_durable_usage(hass, "entry", "agent") is first
    assert len(listeners) == 1

class ResidualMemoryStorage:
    """Small in-memory UsageStorage implementation for residual tests."""

    def __init__(self, data=None):
        self.data = deepcopy(data)
        self.saves: list[dict] = []

    async def async_load(self):
        return deepcopy(self.data)

    async def async_save(self, data):
        self.data = deepcopy(data)
        self.saves.append(deepcopy(data))


async def _residual_manager(
    *,
    storage: ResidualMemoryStorage | None = None,
    daily_storage: ResidualMemoryStorage | None = None,
    detail_storage: ResidualMemoryStorage | None = None,
    request_retention_days: int = 30,
    run_retention_days: int = 30,
) -> usage.UsageManager:
    manager = usage.UsageManager(
        storage or ResidualMemoryStorage(),
        daily_storage,
        detail_storage,
        agent_subentry_id="agent",
        request_retention_days=request_retention_days,
        run_retention_days=run_retention_days,
    )
    await manager.async_initialize()
    return manager


def _residual_request(
    request_id: str,
    timestamp: str,
    *,
    run_id: str = "run",
) -> usage.UsageRequest:
    return usage.UsageRequest(
        request_id=request_id,
        run_id=run_id,
        timestamp=timestamp,
        agent_subentry_id="agent",
        provider="openai",
        model="gpt-test",
        api_mode="responses",
        successful=True,
        duration_ms=1,
    )


def _residual_run(run_id: str, started_at: str) -> usage.UsageRun:
    return usage.UsageRun(
        run_id=run_id,
        started_at=started_at,
        completed_at=started_at,
        duration_ms=1,
        agent_subentry_id="agent",
        home_assistant_conversation_id=None,
        source_device_id=None,
    )


def test_extract_usage_ignores_unmappable_detail_objects() -> None:
    """Opaque SDK detail objects still contribute named fields without iteration."""

    class OpaqueDetails:
        __slots__ = ("cached_tokens",)

        def __init__(self) -> None:
            self.cached_tokens = 7

    result = usage.extract_usage(
        {
            "input_tokens": 11,
            "output_tokens": 3,
            "input_tokens_details": OpaqueDetails(),
            "output_tokens_details": object(),
        }
    )

    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.total_tokens == 14
    assert result.cached_input_tokens == 7
    assert result.reasoning_tokens == 0
    assert result.details == {}


def _minimal_backup(*, date: str) -> dict:
    return {
        "totals": asdict(usage.UsageTotals()),
        "daily": {date: usage._empty_day(date)},
        "requests": [],
        "runs": [],
    }


def test_backup_rejects_iso_length_date_that_is_not_a_real_calendar_date() -> None:
    """Reach the datetime parser failure after the length check succeeds."""
    with pytest.raises(ValueError, match="daily usage date is invalid"):
        usage.UsageManager.validate_backup_data(
            _minimal_backup(date="2026-99-99"), "target-agent"
        )


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda day: day.pop("run_count"), "daily usage record is invalid"),
        (
            lambda day: day.__setitem__("average_requests_per_completed_run", True),
            "daily usage value average_requests_per_completed_run is invalid",
        ),
        (
            lambda day: day.__setitem__("average_duration_ms_per_completed_run", -0.1),
            "daily usage value average_duration_ms_per_completed_run is invalid",
        ),
    ],
)
def test_usage_day_validation_rejects_residual_invalid_shapes(mutator, match) -> None:
    day = usage._empty_day("2026-09-13")
    mutator(day)

    with pytest.raises(ValueError, match=match):
        usage._validate_usage_day("2026-09-13", day)


async def test_replace_backup_without_detail_store_uses_aggregate_only_path() -> None:
    """Restoring totals/days must work when bounded detail persistence is absent."""
    primary = ResidualMemoryStorage()
    manager = await _residual_manager(storage=primary)
    day = usage._empty_day("2026-09-13")
    day["total_tokens"] = 9
    totals = usage.UsageTotals(total_tokens=9)

    await manager.async_replace_backup(totals, {"2026-09-13": day}, [], [])

    assert manager.totals.total_tokens == 9
    assert manager.daily["2026-09-13"]["total_tokens"] == 9
    assert primary.data["total_tokens"] == 9
    assert manager.requests == []
    assert manager.runs == []


def test_request_day_omits_empty_breakdown_labels() -> None:
    """Blank provider/model/mode values must not create meaningless buckets."""
    day = usage._empty_day("2026-09-13")

    usage._add_request_to_day(
        day,
        successful=True,
        usage=usage.RequestUsage(total_tokens=4),
        provider="",
        model="",
        api_mode="",
    )

    assert day["api_request_count"] == 1
    assert day["total_tokens"] == 4
    assert day["provider_breakdown"] == {}
    assert day["model_breakdown"] == {}
    assert day["api_mode_breakdown"] == {}


def test_merge_day_ignores_date_averages_and_non_integer_scalars() -> None:
    """Monthly aggregation only sums counters and nested breakdowns."""
    target = usage._empty_day("2026-09")
    source = usage._empty_day("2026-09-13")
    source.update(
        {
            "run_count": 2,
            "total_tokens": 10,
            "average_tokens_per_completed_run": 5,
            "average_requests_per_completed_run": 1.5,
            "average_duration_ms_per_completed_run": 20.0,
            "provider_breakdown": {"openai": 10},
        }
    )

    usage._merge_day(target, source)

    assert target["date"] == "2026-09"
    assert target["run_count"] == 2
    assert target["total_tokens"] == 10
    assert target["average_tokens_per_completed_run"] == 0
    assert target["average_requests_per_completed_run"] == 0.0
    assert target["average_duration_ms_per_completed_run"] == 0.0
    assert target["provider_breakdown"] == {"openai": 10}
