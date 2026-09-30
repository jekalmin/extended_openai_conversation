"""Handled failures and shutdown share normal terminal accounting."""

import asyncio

import pytest

from custom_components.extended_openai_conversation_responses.debug import (
    _ACTIVE_DEBUG_TRACE,
    DebugManager,
)
from custom_components.extended_openai_conversation_responses.usage import (
    RequestUsage,
    UsageManager,
)
from tests.test_usage import FakeStorage


@pytest.mark.parametrize(
    "error_type",
    ["HomeAssistantError", "RequestRuleExecutionFailed", "ProviderStreamError"],
)
async def test_handled_failure_signal_reaches_debug_and_usage(error_type):
    usage = UsageManager(FakeStorage(), FakeStorage(), FakeStorage())
    await usage.async_initialize()
    debug = DebugManager()
    trace = debug.begin(
        entry_id="entry",
        subentry_id="agent",
        user_input={},
        incoming_conversation_id=None,
    )
    token = _ACTIVE_DEBUG_TRACE.set(trace)
    try:
        async with usage.async_run():
            usage.mark_current_run_failed(error_type)
        debug.finish(trace, successful=trace.error_type is None)
    finally:
        _ACTIVE_DEBUG_TRACE.reset(token)
    assert not trace.successful
    assert trace.error_type == error_type
    assert not usage.runs[0].successful
    assert usage.runs[0].error_type == error_type


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_shutdown_finalizes_interrupted_continuation_once(cancel_first):
    totals, daily, details = FakeStorage(), FakeStorage(), FakeStorage()
    usage = UsageManager(totals, daily, details)
    await usage.async_initialize()
    entered = asyncio.Event()

    async def request():
        async with usage.async_run():
            await usage.async_record_request(
                successful=True,
                tool_calls_requested=1,
                usage=RequestUsage(total_tokens=17),
            )
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await usage.async_record_request(
                    successful=False,
                    request_stage="after_tool",
                    error_type="CancelledError",
                )
                raise

    task = asyncio.create_task(request())
    await asyncio.wait_for(entered.wait(), 2)
    if cancel_first:
        task.cancel()
    await usage.async_shutdown()
    await usage.async_shutdown()
    assert task.cancelled()
    restored = UsageManager(totals, daily, details)
    await restored.async_initialize()
    assert restored.totals.conversation_count == 1
    assert restored.totals.api_request_count == 2
    assert restored.totals.total_tokens == 17
    assert len(restored.runs) == 1
    run = restored.runs[0]
    assert run.completed_at
    assert not run.successful
    assert run.error_type == "CancelledError"
    assert run.tool_call_count == 1
    assert run.failed_request_count == 1
