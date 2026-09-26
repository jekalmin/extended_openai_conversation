"""Protocol-safe retained Function Tool exchange helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from .exceptions import FunctionNotFound
from .function_call_budget import FunctionCallBudget
from .function_execution import (
    async_validate_function_arguments,
    validated_function_call,
)
from .function_tool_recovery import (
    CorrectableToolFailure,
    ToolRecoveryState,
    bind_tool_recovery_state,
    correctable_validation_failure,
    recovery_tool_result,
)
from .function_tool_resolution import latest_function_tool_for_execution
from .ha_llm_tools import is_ha_tool
from .ha_tool_result_compat import is_tool_result_content, make_tool_result_content
from .parallel_tool_execution import (
    async_execute_parallel_safe_batch_outcomes,
    resolve_parallel_safe_batch,
)

_MAX_ERROR_TEXT = 512


def _error_text(error: BaseException | None) -> str:
    """Return a bounded model-visible description for a failed exchange."""
    if error is None:
        return "Tool execution was interrupted before completion"
    name = type(error).__name__
    detail = str(error).strip()
    text = f"{name}: {detail}" if detail else name
    if len(text) <= _MAX_ERROR_TEXT:
        return text
    return f"{text[: _MAX_ERROR_TEXT - 1]}…"


def retained_tool_calls_since(
    chat_log: conversation.ChatLog, existing_content_ids: set[int]
) -> list[llm.ToolInput]:
    """Return external tool calls actually retained during one provider round."""
    calls: list[llm.ToolInput] = []
    for content in chat_log.content:
        if id(content) in existing_content_ids:
            continue
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            calls.extend(content.tool_calls)
    return calls


def _reject_duplicate_tool_call_ids(tool_calls: Iterable[llm.ToolInput]) -> None:
    """Reject one provider round whose call IDs cannot be paired unambiguously."""
    seen: set[str] = set()
    for tool_call in tool_calls:
        if tool_call.id in seen:
            raise HomeAssistantError(
                f"Provider returned duplicate tool call id `{tool_call.id}`"
            )
        seen.add(tool_call.id)


def append_unresolved_tool_results(
    chat_log: conversation.ChatLog,
    agent_id: str,
    tool_calls: Iterable[llm.ToolInput],
    *,
    failed_call_id: str | None = None,
    error: BaseException | None = None,
) -> None:
    """Close every still-retained call exactly once before an error escapes.

    Results already recorded by successful calls are preserved. The call that caused
    the abort is represented as an error; other calls that never reached a completed
    result are explicitly marked skipped. If the failure is at provider/round level
    rather than attributable to one call, the first unresolved call carries the error
    and the remaining calls are skipped.
    """
    calls = list(tool_calls)
    if not calls:
        return

    retained_ids = {
        tool_call.id
        for content in chat_log.content
        if isinstance(content, conversation.AssistantContent) and content.tool_calls
        for tool_call in content.tool_calls
    }
    completed_ids = {
        tool_call_id
        for content in chat_log.content
        if is_tool_result_content(content)
        and isinstance((tool_call_id := getattr(content, "tool_call_id", None)), str)
    }
    unresolved = [
        tool_call
        for tool_call in calls
        if tool_call.id in retained_ids and tool_call.id not in completed_ids
    ]
    if not unresolved:
        return

    actual_failed_id = (
        failed_call_id
        if failed_call_id is not None
        and any(call.id == failed_call_id for call in unresolved)
        else unresolved[0].id
    )
    failure_text = _error_text(error)
    failed_name = next(
        (call.tool_name for call in unresolved if call.id == actual_failed_id),
        "another tool call",
    )

    for tool_call in unresolved:
        result: dict[str, Any]
        if tool_call.id == actual_failed_id:
            result = {"status": "error", "error": failure_text}
        else:
            result = {
                "status": "skipped",
                "error": (
                    f"Skipped because tool call `{failed_name}` failed before this "
                    "exchange completed"
                ),
            }
        chat_log.async_add_assistant_content_without_tools(
            make_tool_result_content(
                agent_id=agent_id,
                tool_call_id=tool_call.id,
                tool_name=tool_call.tool_name,
                tool_result={"result": result},
            )
        )


def _index_tools(function_tools: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index effective Function Tools without changing first-definition precedence."""
    indexed: dict[str, dict[str, Any]] = {}
    for function_tool in function_tools:
        name = function_tool.get("spec", {}).get("name")
        if isinstance(name, str):
            indexed.setdefault(name, function_tool)
    return indexed


def _resolve_current_tool(
    entity: Any,
    tool_input: llm.ToolInput,
    request_tools_by_name: dict[str, dict[str, Any]],
    function_tools_factory: Callable[[], list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Resolve one request-round call against current effective availability."""
    check_aba = getattr(entity, "_assert_no_aba_configuration", None)
    if callable(check_aba):
        check_aba()
    request_tool = request_tools_by_name.get(tool_input.tool_name)
    if request_tool is None:
        raise FunctionNotFound(tool_input.tool_name)

    candidate = request_tool
    if function_tools_factory is not None:
        current_effective = _index_tools(function_tools_factory())
        current_candidate = current_effective.get(tool_input.tool_name)
        if current_candidate is None:
            raise FunctionNotFound(tool_input.tool_name)
        candidate = current_candidate

    return latest_function_tool_for_execution(entity, candidate)


def _append_recovery(
    chat_log: conversation.ChatLog,
    entity: Any,
    tool_input: llm.ToolInput,
    failure: CorrectableToolFailure,
    recovery_state: ToolRecoveryState,
) -> bool:
    """Append one correctable result only while the independent cap permits it."""
    if not recovery_state.consume():
        return False
    chat_log.async_add_assistant_content_without_tools(
        recovery_tool_result(entity.entity_id, tool_input, failure)
    )
    return True


def _validated_tool_input(
    tool_input: llm.ToolInput, arguments: dict[str, Any]
) -> llm.ToolInput:
    """Preserve call identity while dispatching locally validated/coerced arguments."""
    return llm.ToolInput(
        id=tool_input.id,
        tool_name=tool_input.tool_name,
        tool_args=arguments,
        external=tool_input.external,
    )


async def _async_validate_recoverable_call(
    entity: Any,
    function_tool: dict[str, Any],
    tool_input: llm.ToolInput,
    recovery_state: ToolRecoveryState,
) -> llm.ToolInput | CorrectableToolFailure:
    """Validate one already-resolved call without crossing dispatch."""
    malformed = recovery_state.pop_malformed(tool_input.id)
    if malformed is not None:
        return malformed

    # HA LLM Tools own their live schema and permission boundary. Do not
    # reinterpret that external schema with our configured-tool validator.
    if is_ha_tool(function_tool):
        return tool_input

    try:
        arguments = await async_validate_function_arguments(
            entity.hass,
            function_tool.get("spec", {}),
            tool_input.tool_args,
            distinguish_infrastructure=True,
        )
    except HomeAssistantError as err:
        return correctable_validation_failure(err)

    return _validated_tool_input(tool_input, arguments)


async def _execute_bound(
    entity: Any,
    function_tool: dict[str, Any],
    tool_input: llm.ToolInput,
    llm_context: llm.LLMContext | None,
    exposed_entities: list[dict[str, Any]],
    recovery_state: ToolRecoveryState,
) -> conversation.ToolResultContent:
    """Execute one prepared call with strict runtime-failure semantics bound."""
    with (
        bind_tool_recovery_state(recovery_state),
        validated_function_call(tool_input, function_tool.get("spec", {})),
    ):
        return cast(
            conversation.ToolResultContent,
            await entity._execute_function_tool(
                function_tool, tool_input, llm_context, exposed_entities
            ),
        )


async def _async_execute_with_recovery(
    entity: Any,
    chat_log: conversation.ChatLog,
    pending_tool_calls: list[llm.ToolInput],
    request_function_tools: list[dict[str, Any]],
    function_call_budget: FunctionCallBudget,
    llm_context: llm.LLMContext | None,
    exposed_entities: list[dict[str, Any]],
    function_tools_factory: Callable[[], list[dict[str, Any]]] | None,
    recovery_state: ToolRecoveryState,
) -> None:
    """Execute one batch with recovery restricted to pre-dispatch stages."""
    request_tools_by_name = _index_tools(request_function_tools)
    potential_parallel = resolve_parallel_safe_batch(
        pending_tool_calls, request_tools_by_name
    )
    parallel_batch = None

    # Match the legacy budget boundary exactly: current availability is
    # resolved before any budget claim. Availability may encode security or
    # Guest policy, so failures here are always fail-fast and consume no call.
    if potential_parallel is not None:
        current_tools: dict[str, dict[str, Any]] = {}
        resolving_call: llm.ToolInput | None = None
        try:
            for resolving_call in pending_tool_calls:
                current_tools[resolving_call.tool_name] = _resolve_current_tool(
                    entity,
                    resolving_call,
                    request_tools_by_name,
                    function_tools_factory,
                )
        except BaseException as err:
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                failed_call_id=(
                    resolving_call.id if resolving_call is not None else None
                ),
                error=err,
            )
            raise
        parallel_batch = resolve_parallel_safe_batch(pending_tool_calls, current_tools)

    if parallel_batch is not None:
        remaining = function_call_budget.remaining
        try:
            function_call_budget.claim_many(
                tool_input.tool_name for _, tool_input in parallel_batch
            )
        except BaseException as err:
            failed_index = (
                0 if remaining is None else min(remaining, len(parallel_batch) - 1)
            )
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                failed_call_id=parallel_batch[failed_index][1].id,
                error=err,
            )
            raise

        prepared: list[tuple[dict[str, Any], llm.ToolInput]] = []
        recovery_results: dict[str, conversation.ToolResultContent] = {}
        validating_call: llm.ToolInput | None = None
        try:
            for function_tool, tool_input in parallel_batch:
                validating_call = tool_input
                validation_outcome = await _async_validate_recoverable_call(
                    entity, function_tool, tool_input, recovery_state
                )
                if isinstance(validation_outcome, CorrectableToolFailure):
                    if not recovery_state.consume():
                        append_unresolved_tool_results(
                            chat_log,
                            entity.entity_id,
                            pending_tool_calls,
                            failed_call_id=tool_input.id,
                            error=validation_outcome.original,
                        )
                        raise validation_outcome.original
                    recovery_results[tool_input.id] = recovery_tool_result(
                        entity.entity_id, tool_input, validation_outcome
                    )
                else:
                    prepared.append((function_tool, validation_outcome))
        except BaseException as err:
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                failed_call_id=(
                    validating_call.id if validating_call is not None else None
                ),
                error=err,
            )
            raise

        execution_outcomes: dict[
            str, conversation.ToolResultContent | BaseException
        ] = {}
        if prepared:
            try:
                outcomes = await async_execute_parallel_safe_batch_outcomes(
                    prepared,
                    lambda function_tool, tool_input: _execute_bound(
                        entity,
                        function_tool,
                        tool_input,
                        llm_context,
                        exposed_entities,
                        recovery_state,
                    ),
                )
            except BaseException as err:
                append_unresolved_tool_results(
                    chat_log,
                    entity.entity_id,
                    pending_tool_calls,
                    error=err,
                )
                raise
            execution_outcomes = {
                tool_input.id: execution_outcome
                for (_, tool_input), execution_outcome in zip(
                    prepared, outcomes, strict=True
                )
            }

        first_error: BaseException | None = None
        for tool_input in pending_tool_calls:
            recovery_result = recovery_results.get(tool_input.id)
            if recovery_result is not None:
                chat_log.async_add_assistant_content_without_tools(recovery_result)
                continue
            execution_outcome = execution_outcomes.get(tool_input.id)
            if isinstance(execution_outcome, BaseException):
                append_unresolved_tool_results(
                    chat_log,
                    entity.entity_id,
                    [tool_input],
                    failed_call_id=tool_input.id,
                    error=execution_outcome,
                )
                if first_error is None:
                    first_error = execution_outcome
            elif execution_outcome is not None:
                chat_log.async_add_assistant_content_without_tools(execution_outcome)
        if first_error is not None:
            raise first_error
        return

    # Preserve the old serial ordering: resolve, claim, then cross into
    # validation/recovery and finally dispatch.
    for tool_input in pending_tool_calls:
        try:
            function_tool = _resolve_current_tool(
                entity,
                tool_input,
                request_tools_by_name,
                function_tools_factory,
            )
            function_call_budget.claim(tool_input.tool_name)
            outcome = await _async_validate_recoverable_call(
                entity, function_tool, tool_input, recovery_state
            )
            if isinstance(outcome, CorrectableToolFailure):
                if _append_recovery(
                    chat_log, entity, tool_input, outcome, recovery_state
                ):
                    continue
                raise outcome.original
            tool_result_content = await _execute_bound(
                entity,
                function_tool,
                outcome,
                llm_context,
                exposed_entities,
                recovery_state,
            )
        except BaseException as err:
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                failed_call_id=tool_input.id,
                error=err,
            )
            raise
        chat_log.async_add_assistant_content_without_tools(tool_result_content)


async def async_execute_tool_exchange(
    entity: Any,
    chat_log: conversation.ChatLog,
    pending_tool_calls: list[llm.ToolInput],
    request_function_tools: list[dict[str, Any]],
    function_call_budget: FunctionCallBudget,
    llm_context: llm.LLMContext | None,
    exposed_entities: list[dict[str, Any]],
    *,
    function_tools_factory: Callable[[], list[dict[str, Any]]] | None = None,
    recovery_state: ToolRecoveryState | None = None,
) -> None:
    """Execute one provider tool batch while keeping retained history complete."""
    if not pending_tool_calls:
        return
    _reject_duplicate_tool_call_ids(pending_tool_calls)
    if recovery_state is not None and recovery_state.enabled:
        await _async_execute_with_recovery(
            entity,
            chat_log,
            pending_tool_calls,
            request_function_tools,
            function_call_budget,
            llm_context,
            exposed_entities,
            function_tools_factory,
            recovery_state,
        )
        return

    # Disabled mode deliberately retains the previous fail-fast path byte-for-byte in
    # behavior so opting out cannot change execution, budgets, or error history.
    request_tools_by_name = _index_tools(request_function_tools)
    potential_parallel = resolve_parallel_safe_batch(
        pending_tool_calls, request_tools_by_name
    )
    parallel_batch = None

    if potential_parallel is not None:
        current_tools: dict[str, dict[str, Any]] = {}
        resolving_call: llm.ToolInput | None = None
        try:
            for resolving_call in pending_tool_calls:
                current_tools[resolving_call.tool_name] = _resolve_current_tool(
                    entity,
                    resolving_call,
                    request_tools_by_name,
                    function_tools_factory,
                )
        except BaseException as err:
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                failed_call_id=resolving_call.id
                if resolving_call is not None
                else None,
                error=err,
            )
            raise
        parallel_batch = resolve_parallel_safe_batch(pending_tool_calls, current_tools)

    if parallel_batch is not None:
        remaining = function_call_budget.remaining
        try:
            function_call_budget.claim_many(
                tool_input.tool_name for _, tool_input in parallel_batch
            )
        except BaseException as err:
            failed_index = (
                0 if remaining is None else min(remaining, len(parallel_batch) - 1)
            )
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                failed_call_id=parallel_batch[failed_index][1].id,
                error=err,
            )
            raise

        try:
            outcomes = await async_execute_parallel_safe_batch_outcomes(
                parallel_batch,
                lambda function_tool, tool_input: entity._execute_function_tool(
                    function_tool,
                    tool_input,
                    llm_context,
                    exposed_entities,
                ),
            )
        except BaseException as err:
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                error=err,
            )
            raise

        first_error: BaseException | None = None
        for (_, tool_input), outcome in zip(parallel_batch, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                append_unresolved_tool_results(
                    chat_log,
                    entity.entity_id,
                    [tool_input],
                    failed_call_id=tool_input.id,
                    error=outcome,
                )
                if first_error is None:
                    first_error = outcome
            else:
                chat_log.async_add_assistant_content_without_tools(outcome)
        if first_error is not None:
            raise first_error
        return

    for tool_input in pending_tool_calls:
        try:
            function_tool = _resolve_current_tool(
                entity,
                tool_input,
                request_tools_by_name,
                function_tools_factory,
            )
            function_call_budget.claim(tool_input.tool_name)
            tool_result_content = await entity._execute_function_tool(
                function_tool,
                tool_input,
                llm_context,
                exposed_entities,
            )
        except BaseException as err:
            append_unresolved_tool_results(
                chat_log,
                entity.entity_id,
                pending_tool_calls,
                failed_call_id=tool_input.id,
                error=err,
            )
            raise
        chat_log.async_add_assistant_content_without_tools(tool_result_content)
