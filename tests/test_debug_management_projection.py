"""Tests for bounded management projection of volatile request debug captures."""

from __future__ import annotations

import json

from custom_components.extended_openai_conversation_responses.debug import DebugManager
from custom_components.extended_openai_conversation_responses import (
    debug_management_projection as projection,
)
from custom_components.extended_openai_conversation_responses.debug_management_projection import (
    debug_run_summaries,
    debug_trace_page,
)
from custom_components.extended_openai_conversation_responses.management_result_limits import (
    MANAGEMENT_DEBUG_PAGE_CHARACTERS,
    MANAGEMENT_DEBUG_SUMMARY_VALUE_CHARACTERS,
)


def _captured_manager() -> tuple[DebugManager, str]:
    manager = DebugManager()
    manager.configure(enabled=True, limit=10)
    trace = manager.begin(
        entry_id="entry",
        subentry_id="agent",
        user_input={"text": "hello"},
        incoming_conversation_id="conversation",
    )
    trace.system_prompt = "system " + ("s" * 100_000)
    trace.memory = {"retrieved": "m" * 100_000}
    trace.result = {"speech": "r" * 100_000}
    trace.error = {"message": "e" * 10_000}
    trace.continuity = {
        "resolved_conversation_id": "resolved",
        "resumed": True,
        "mode": "device",
    }
    for index in range(7):
        request = trace.start_provider_request(
            "responses",
            (),
            {
                "model": "gpt-test",
                "input": "i" * 100_000,
                "tools": [{"name": f"tool-{index}", "description": "t" * 5_000}],
            },
        )
        request.response_events = [{"type": "delta", "delta": "o" * 100_000}]
        request._event_bytes = 100_000
        request.successful = True
    manager.finish(trace, successful=True)
    return manager, trace.debug_id


def test_debug_trace_provider_requests_are_paged_and_aggregate_bounded() -> None:
    manager, debug_id = _captured_manager()

    first = debug_trace_page(manager, debug_id, provider_offset=0, provider_limit=2)
    assert first is not None
    provider_meta = first["management_projection"]["provider_requests"]
    assert len(first["provider_requests"]) == 2
    assert provider_meta == {
        "offset": 0,
        "limit": 2,
        "returned": 2,
        "has_more": True,
        "next_offset": 2,
        "total": 7,
    }
    assert first["management_projection"]["truncated"] is True
    assert first["management_projection"]["page"]["limit_characters"] == (
        MANAGEMENT_DEBUG_PAGE_CHARACTERS
    )
    assert first["management_projection"]["page"]["remaining_characters"] >= 0
    assert first["management_projection"]["system_prompt"]["truncated"] is True
    # The hard content budget deliberately leaves only modest JSON-structure overhead.
    assert len(json.dumps(first)) < MANAGEMENT_DEBUG_PAGE_CHARACTERS + 100_000

    second = debug_trace_page(manager, debug_id, provider_offset=2, provider_limit=2)
    assert second is not None
    second_meta = second["management_projection"]["provider_requests"]
    assert second_meta["offset"] == 2
    assert second_meta["next_offset"] == 4
    assert second_meta["total"] == 7
    assert [item["request_id"] for item in first["provider_requests"]] != [
        item["request_id"] for item in second["provider_requests"]
    ]


def test_debug_run_summary_cannot_return_an_unbounded_error_object() -> None:
    manager, _debug_id = _captured_manager()

    summary = debug_run_summaries(manager)[0]

    assert summary["error_meta"]["limit_characters"] == (
        MANAGEMENT_DEBUG_SUMMARY_VALUE_CHARACTERS
    )
    assert summary["error_meta"]["truncated"] is True
    assert len(json.dumps(summary["error"])) < MANAGEMENT_DEBUG_SUMMARY_VALUE_CHARACTERS + 500
    assert summary["provider_request_count"] == 7
    assert summary["continuity_mode"] == "device"


def test_projection_depth_container_and_budget_limits_fail_closed() -> None:
    deep: object = "leaf"
    for _ in range(projection._MAX_DEPTH + 2):
        deep = {"next": deep}

    budget = projection._ProjectionBudget(10_000)
    projected = projection._project_value(deep, budget)
    serialized = json.dumps(projected)
    assert "management debug depth limit reached" in serialized
    assert budget.truncated is True

    many = {f"k{index}": index for index in range(projection._MAX_CONTAINER_ITEMS + 20)}
    budget = projection._ProjectionBudget(100_000)
    projected_many = projection._project_value(many, budget)
    assert projected_many["__management_truncated__"] is True
    assert len([key for key in projected_many if key != "__management_truncated__"]) == (
        projection._MAX_CONTAINER_ITEMS
    )
    assert budget.truncated is True

    budget = projection._ProjectionBudget(4)
    projected_small = projection._project_value({"abcdef": "value"}, budget)
    assert projected_small == {"__management_truncated__": True}
    assert budget.remaining == 0
    assert budget.truncated is True


def test_bounded_value_and_text_share_page_budget_and_handle_non_json_values() -> None:
    page = projection._ProjectionBudget(5)
    text, meta = projection._bounded_text("abcdefgh", 100, page)
    assert text == "abcde\n<management debug text truncated>"
    assert meta["truncated"] is True
    assert page.remaining == 0
    assert page.truncated is True

    class Custom:
        def __str__(self) -> str:
            return "custom-value"

    page = projection._ProjectionBudget(6)
    value, value_meta = projection._bounded_value(
        {"custom": Custom()}, 100, page
    )
    assert value_meta["truncated"] is True
    assert page.remaining == 0
    assert page.truncated is True
    assert "__management_truncated__" in value or "custom" in value


def test_debug_trace_page_clamps_provider_window_and_missing_id() -> None:
    manager, debug_id = _captured_manager()

    assert projection.debug_trace_page(manager, "missing") is None

    page = projection.debug_trace_page(
        manager,
        debug_id,
        provider_offset=-50,
        provider_limit=10_000,
    )
    assert page is not None
    provider = page["management_projection"]["provider_requests"]
    assert provider["offset"] == 0
    assert provider["limit"] == projection.MANAGEMENT_DEBUG_PROVIDER_PAGE_MAX
    assert provider["returned"] == min(
        7, projection.MANAGEMENT_DEBUG_PROVIDER_PAGE_MAX
    )
