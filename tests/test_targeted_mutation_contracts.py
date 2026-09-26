"""Behavioral contracts discovered by targeted mutation testing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from custom_components.extended_openai_conversation_responses.exceptions import (
    FunctionNotFound,
)
from custom_components.extended_openai_conversation_responses.function_tool_recovery import (
    ToolRecoveryState,
    bind_tool_recovery_state,
    correctable_validation_failure,
    recovery_tool_result,
    strict_execution_failures_enabled,
)
from custom_components.extended_openai_conversation_responses.function_tool_resolution import (
    configured_function_tool_for_execution,
    latest_function_tool_for_execution,
)
from custom_components.extended_openai_conversation_responses.ha_tool_result_compat import (
    tool_result_data,
)
from custom_components.extended_openai_conversation_responses.parallel_tool_execution import (
    async_execute_parallel_safe_batch,
    async_execute_parallel_safe_batch_outcomes,
    is_parallel_safe_integration_tool,
)
from homeassistant.components import conversation
from homeassistant.helpers import llm


def _call(name: str, call_id: str) -> llm.ToolInput:
    return llm.ToolInput(
        id=call_id,
        tool_name=name,
        tool_args={},
        external=True,
    )


def _tool(
    name: str,
    function: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "spec": {
            "name": name,
            "parameters": parameters
            or {"type": "object", "properties": {}, "additionalProperties": False},
        },
        "function": function,
    }


def _resolution_agent(
    current_tools: list[dict[str, Any]],
    *,
    groups: list[dict[str, Any]] | None = None,
) -> tuple[Any, Mock, Mock]:
    latest_data = {"function_groups": groups or []}
    latest_subentry = SimpleNamespace(data=latest_data)
    latest_entry = SimpleNamespace(subentries={"agent-1": latest_subentry})
    entry_lookup = Mock(return_value=latest_entry)
    resolver = Mock(return_value=current_tools)
    agent = SimpleNamespace(
        hass=SimpleNamespace(
            config_entries=SimpleNamespace(async_get_entry=entry_lookup)
        ),
        entry=SimpleNamespace(entry_id="entry-1"),
        subentry=SimpleNamespace(subentry_id="agent-1", data={"revision": 1}),
        _configured_function_tools_from_data=resolver,
    )
    return agent, entry_lookup, resolver


def _result(tool_input: llm.ToolInput) -> conversation.ToolResultContent:
    return conversation.ToolResultContent(
        agent_id="conversation.mutation",
        tool_call_id=tool_input.id,
        tool_name=tool_input.tool_name,
        tool_result={"result": tool_input.id},
    )


def test_strict_execution_failures_follow_bound_recovery_policy() -> None:
    """Strict runtime failures are enabled only for an enabled bound recovery state."""
    assert strict_execution_failures_enabled() is False

    with bind_tool_recovery_state(ToolRecoveryState(enabled=False)):
        assert strict_execution_failures_enabled() is False

    with bind_tool_recovery_state(ToolRecoveryState(enabled=True)):
        assert strict_execution_failures_enabled() is True

    assert strict_execution_failures_enabled() is False


def test_recovery_feedback_has_bounded_text_and_stable_protocol_fields() -> None:
    """Recovery feedback preserves the model-visible boundary and result structure."""
    exactly_at_limit = "x" * 320
    exact_failure = correctable_validation_failure(ValueError(exactly_at_limit))
    assert exact_failure.message == exactly_at_limit

    over_limit = correctable_validation_failure(ValueError("x" * 321))
    assert len(over_limit.message) == 320
    assert over_limit.message.endswith("…")

    blank = correctable_validation_failure(ValueError(" \n\t "))
    assert blank.message
    assert "schema" in blank.message.lower()

    call = _call("action", "call-1")
    result = recovery_tool_result("conversation.mutation", call, over_limit)
    assert result.agent_id == "conversation.mutation"
    assert result.tool_call_id == "call-1"
    assert result.tool_name == "action"
    assert tool_result_data(result) == {
        "result": {
            "status": "error",
            "reason": "correctable_tool_error",
            "code": "invalid_arguments",
            "stage": "pre_dispatch_validation",
            "error": over_limit.message,
        }
    }


def test_parallel_safe_classification_rejects_malformed_function_metadata() -> None:
    """Missing or malformed function metadata must never become parallel-safe."""
    assert is_parallel_safe_integration_tool({"spec": {"name": "missing"}}) is False
    assert (
        is_parallel_safe_integration_tool(
            {"spec": {"name": "bad"}, "function": "knowledge"}
        )
        is False
    )


async def test_parallel_batch_passes_each_resolved_tool_to_executor() -> None:
    """Parallel dispatch must not lose or substitute the resolved Function Tool."""
    first = _tool("first", {"type": "knowledge", "operation": "search"})
    second = _tool("second", {"type": "knowledge", "operation": "list"})
    calls = [(first, _call("first", "1")), (second, _call("second", "2"))]
    seen: dict[str, dict[str, Any]] = {}

    async def execute(
        function_tool: dict[str, Any], tool_input: llm.ToolInput
    ) -> conversation.ToolResultContent:
        seen[tool_input.id] = function_tool
        return _result(tool_input)

    results = await async_execute_parallel_safe_batch(calls, execute)

    assert seen["1"] is first
    assert seen["2"] is second
    assert [result.tool_call_id for result in results] == ["1", "2"]


async def test_parallel_batch_cancels_pending_sibling_after_ordered_failure() -> None:
    """An earlier failure cancels and drains a still-running sibling before propagating."""
    first = _tool("first", {"type": "knowledge", "operation": "search"})
    second = _tool("second", {"type": "knowledge", "operation": "list"})
    calls = [(first, _call("first", "1")), (second, _call("second", "2"))]
    second_started = asyncio.Event()
    second_cancelled = asyncio.Event()
    never_release = asyncio.Event()

    async def execute(
        _function_tool: dict[str, Any], tool_input: llm.ToolInput
    ) -> conversation.ToolResultContent:
        if tool_input.id == "1":
            await second_started.wait()
            raise RuntimeError("first failed")
        second_started.set()
        try:
            await never_release.wait()
        except asyncio.CancelledError:
            second_cancelled.set()
            raise
        raise AssertionError("blocked sibling unexpectedly resumed")

    with pytest.raises(RuntimeError, match="first failed"):
        await async_execute_parallel_safe_batch(calls, execute)

    assert second_cancelled.is_set()


async def test_parallel_outcomes_capture_child_exception_without_aborting_batch() -> (
    None
):
    """Outcome mode returns ordinary child exceptions in provider order."""
    first = _tool("first", {"type": "knowledge", "operation": "search"})
    second = _tool("second", {"type": "knowledge", "operation": "list"})
    calls = [(first, _call("first", "1")), (second, _call("second", "2"))]

    async def execute(
        function_tool: dict[str, Any], tool_input: llm.ToolInput
    ) -> conversation.ToolResultContent:
        assert function_tool is (first if tool_input.id == "1" else second)
        if tool_input.id == "2":
            raise RuntimeError("second failed")
        return _result(tool_input)

    outcomes = await async_execute_parallel_safe_batch_outcomes(calls, execute)

    assert isinstance(outcomes[0], conversation.ToolResultContent)
    assert outcomes[0].tool_call_id == "1"
    assert isinstance(outcomes[1], RuntimeError)
    assert str(outcomes[1]) == "second failed"


@pytest.mark.parametrize(
    "runner",
    [async_execute_parallel_safe_batch, async_execute_parallel_safe_batch_outcomes],
    ids=["ordered", "outcomes"],
)
async def test_parallel_helpers_cancel_children_when_parent_is_cancelled(
    runner: Any,
) -> None:
    """Cancelling the parent leaves no still-running child tool tasks."""
    first = _tool("first", {"type": "knowledge", "operation": "search"})
    second = _tool("second", {"type": "knowledge", "operation": "list"})
    calls = [(first, _call("first", "1")), (second, _call("second", "2"))]
    all_started = asyncio.Event()
    never_release = asyncio.Event()
    started: set[str] = set()
    cancelled: set[str] = set()

    async def execute(
        _function_tool: dict[str, Any], tool_input: llm.ToolInput
    ) -> conversation.ToolResultContent:
        started.add(tool_input.id)
        if len(started) == 2:
            all_started.set()
        try:
            await never_release.wait()
        except asyncio.CancelledError:
            cancelled.add(tool_input.id)
            raise
        raise AssertionError("cancelled child unexpectedly resumed")

    parent = asyncio.create_task(runner(calls, execute))
    await asyncio.wait_for(all_started.wait(), timeout=1)
    parent.cancel()

    with pytest.raises(asyncio.CancelledError):
        await parent

    assert cancelled == {"1", "2"}


def test_configured_resolution_uses_live_entry_and_allows_enabled_group() -> None:
    """Live config lookup and an explicitly enabled Function Group permit dispatch."""
    current = _tool("notify", {"type": "service", "service": "notify.current"})
    groups = [
        {
            "id": "notifications",
            "name": "Notifications",
            "description": "Enabled notification tools",
            "loading_mode": "always",
            "functions": ["notify"],
            "enabled": True,
        }
    ]
    agent, entry_lookup, _resolver = _resolution_agent([current], groups=groups)

    assert configured_function_tool_for_execution(agent, "notify") is current
    entry_lookup.assert_called_once_with("entry-1")


def test_missing_resolver_fails_closed_with_requested_name() -> None:
    """A missing configured-tool resolver yields the public not-found contract."""
    agent = SimpleNamespace()

    with pytest.raises(FunctionNotFound) as caught:
        configured_function_tool_for_execution(agent, "notify")

    assert caught.value.function == "notify"


def test_missing_current_tool_preserves_requested_name() -> None:
    """A live deletion reports the Function Tool name that disappeared."""
    agent, _entry_lookup, _resolver = _resolution_agent([])

    with pytest.raises(FunctionNotFound) as caught:
        configured_function_tool_for_execution(agent, "notify")

    assert caught.value.function == "notify"


def test_truthy_non_string_tool_name_keeps_request_round_definition() -> None:
    """Malformed non-string names do not enter configured-tool resolution."""
    agent, _entry_lookup, resolver = _resolution_agent([])
    request_tool = _tool("placeholder", {"type": "service", "service": "test.call"})
    request_tool["spec"]["name"] = 123

    assert latest_function_tool_for_execution(agent, request_tool) is request_tool
    resolver.assert_not_called()


def test_ha_llm_schema_edit_rejects_stale_request_round() -> None:
    """A changed saved schema invalidates provider arguments from the old round."""
    reference = {
        "type": "ha_llm",
        "source_type": "platform",
        "source_id": "test",
        "api_id": "assist",
        "tool_name": "do_thing",
    }
    request_tool = _tool(
        "ha_saved",
        dict(reference),
        parameters={
            "type": "object",
            "properties": {"old_schema": {"type": "string"}},
            "additionalProperties": False,
        },
    )
    current_tool = _tool(
        "ha_saved",
        dict(reference),
        parameters={
            "type": "object",
            "properties": {"new_schema": {"type": "string"}},
            "additionalProperties": False,
        },
    )
    agent, entry_lookup, _resolver = _resolution_agent([current_tool])

    with pytest.raises(FunctionNotFound):
        latest_function_tool_for_execution(agent, request_tool)
    entry_lookup.assert_called_once_with("entry-1")


def test_ha_llm_reference_change_fails_closed_with_tool_name() -> None:
    """Changing the saved HA reference after provider emission invalidates the call."""
    original_reference = {
        "type": "ha_llm",
        "source_type": "platform",
        "source_id": "test",
        "api_id": "assist",
        "tool_name": "do_thing",
    }
    changed_reference = dict(original_reference, tool_name="other_thing")
    request_tool = _tool("ha_saved", original_reference)
    current_tool = _tool("ha_saved", changed_reference)
    agent, _entry_lookup, _resolver = _resolution_agent([current_tool])

    with pytest.raises(FunctionNotFound) as caught:
        latest_function_tool_for_execution(agent, request_tool)

    assert caught.value.function == "ha_saved"
