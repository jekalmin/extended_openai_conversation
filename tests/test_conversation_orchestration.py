"""Conversation pipeline orchestration regressions."""

from __future__ import annotations

from contextlib import asynccontextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.extended_openai_conversation_responses import (
    conversation as conversation_module,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_CONTINUE_CONVERSATION,
    CONF_CONVERSATION_CONTINUITY,
    CONF_CONVERSATION_TIMEOUT_MINUTES,
    CONTINUE_CONVERSATION_ALWAYS,
    CONTINUE_CONVERSATION_CONDITIONAL,
    CONVERSATION_CONTINUITY_DEVICE,
    CONVERSATION_CONTINUITY_HA_DEFAULT,
    CONVERSATION_CONTINUITY_USER,
    DEFAULT_CONTINUE_CONVERSATION,
)
from custom_components.extended_openai_conversation_responses.continuity import (
    ConversationContinuity,
)
from custom_components.extended_openai_conversation_responses.conversation import (
    _ACTIVE_ARCHIVE,
    _ACTIVE_FUNCTION_GROUP_SESSION,
    _ACTIVE_MEMORY_SESSION,
    _ACTIVE_SCOPE,
    _ACTIVE_TEMPORARY_SCOPE,
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation_responses.guest_mode import (
    GuestCapabilityPolicy,
)
from custom_components.extended_openai_conversation_responses.local_intents import (
    LocalIntentResult,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    RuleEvaluation,
    RuleMatch,
)
from custom_components.extended_openai_conversation_responses.scope import user_scope
from homeassistant.components import conversation
from homeassistant.core import Context
from homeassistant.helpers import intent
from homeassistant.util import dt as dt_util


class _UsageRecorder:
    """Minimal deterministic usage context for orchestration assertions."""

    def __init__(self, *, successful: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self.successful = successful

    @asynccontextmanager
    async def async_run(self, **kwargs):
        self.calls.append(dict(kwargs))
        yield SimpleNamespace(run_id="run-1", successful=self.successful)


def _pipeline_fixture(monkeypatch, *, text: str = "hello"):
    entity = object.__new__(ExtendedOpenAIAgentEntity)
    entity.hass = SimpleNamespace(data={}, bus=SimpleNamespace(async_fire=MagicMock()))
    entity.entry = SimpleNamespace(entry_id="entry")
    entity.subentry = SimpleNamespace(subentry_id="agent", data={})
    entity._attr_entity_id = "conversation.agent"
    entity._continuity = SimpleNamespace(async_record_success=AsyncMock())
    entity._function_groups_runtime = None
    entity._request_rules = None
    entity._request_rule_runtime = None
    entity._usage = None
    entity._archive = None
    entity._async_begin_archive_session = AsyncMock(return_value=None)
    entity._async_handle_message_with_ha_tools = AsyncMock()

    user_input = SimpleNamespace(
        text=text,
        language="en",
        conversation_id=None,
        context=Context(user_id="alice"),
    )
    chat_log = SimpleNamespace(
        content=[
            conversation.SystemContent(content="system"),
            conversation.UserContent(content=text),
        ],
        conversation_id="conversation-1",
    )
    resolution = SimpleNamespace(
        conversation_id="conversation-1",
        key="device:kitchen",
        claim_token="claim-token",
        history=(),
    )
    policy = GuestCapabilityPolicy.unrestricted()
    scope = user_scope("alice", source="test", device_id="kitchen")
    llm_context = SimpleNamespace(context=user_input.context)

    monkeypatch.setattr(
        conversation_module,
        "async_get_chat_session",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace()),
    )
    monkeypatch.setattr(
        conversation_module,
        "async_get_chat_log",
        lambda *_args, **_kwargs: nullcontext(chat_log),
    )

    async def process():
        return await entity._async_process_claimed(
            user_input,
            llm_context,
            policy,
            scope,
            "kitchen",
            15,
            resolution,
        )

    return entity, user_input, chat_log, policy, process


@pytest.mark.parametrize(
    "mode",
    [
        DEFAULT_CONTINUE_CONVERSATION,
        CONTINUE_CONVERSATION_ALWAYS,
        CONTINUE_CONVERSATION_CONDITIONAL,
    ],
)
@pytest.mark.parametrize("successful", [False, True])
@pytest.mark.parametrize("with_usage", [False, True])
async def test_consumed_request_rule_bypasses_local_intent_and_provider(
    monkeypatch, mode, successful, with_usage
):
    entity, _user_input, _chat_log, _policy, process = _pipeline_fixture(
        monkeypatch, text="good night"
    )
    entity.subentry.data[CONF_CONTINUE_CONVERSATION] = mode
    if with_usage:
        entity._usage = _UsageRecorder()
        entity._usage.mark_current_run_failed = MagicMock()
    entity._request_rules = object()
    entity._request_rule_runtime = SimpleNamespace(effective_options=MagicMock())

    evaluation = RuleEvaluation(
        match=RuleMatch(
            {"id": "good-night", "name": "Good night"},
            "good night",
            False,
            100.0,
        ),
        consume=True,
        response="Handled locally",
        successful=successful,
    )
    evaluate_rule = AsyncMock(return_value=evaluation)
    try_local_intent = AsyncMock()
    monkeypatch.setattr(conversation_module, "async_evaluate_rule", evaluate_rule)
    monkeypatch.setattr(
        conversation_module, "async_try_handle_local_intent", try_local_intent
    )

    result = await process()

    assert result.response.speech["plain"]["speech"] == "Handled locally"
    assert result.continue_conversation is (
        successful and mode == CONTINUE_CONVERSATION_ALWAYS
    )
    evaluate_rule.assert_awaited_once()
    try_local_intent.assert_not_awaited()
    entity._async_handle_message_with_ha_tools.assert_not_awaited()
    assert entity._continuity.async_record_success.await_count == int(successful)
    payload = entity.hass.bus.async_fire.call_args.args[1]
    assert payload["status"] == "local"
    assert payload["handled_locally"] is True


@pytest.mark.parametrize("native_response", ["Native response", "Done"])
async def test_native_terminal_rule_never_calls_provider(monkeypatch, native_response):
    entity, _, _, _, process = _pipeline_fixture(monkeypatch)
    entity._request_rules = object()
    entity._request_rule_runtime = SimpleNamespace(effective_options=MagicMock())
    evaluation = RuleEvaluation(
        RuleMatch(
            {"id": "native", "name": "Native", "action": {"continue_to_ai": True}},
            "hello",
            False,
            100.0,
        ),
        consume=True,
        response=native_response,
        terminal=True,
    )
    monkeypatch.setattr(
        conversation_module, "async_evaluate_rule", AsyncMock(return_value=evaluation)
    )
    result = await process()
    assert result.response.speech["plain"]["speech"] == native_response
    entity._async_handle_message_with_ha_tools.assert_not_awaited()


async def test_successful_local_rule_continuation_calls_provider_once(monkeypatch):
    entity, user_input, _chat_log, _policy, process = _pipeline_fixture(
        monkeypatch, text="check the battery"
    )
    entity._request_rules = object()
    entity._request_rule_runtime = SimpleNamespace(
        effective_options=MagicMock(return_value={})
    )
    evaluation = RuleEvaluation(
        match=RuleMatch(
            {"id": "battery", "name": "Battery"}, "check the battery", False, 100.0
        ),
        consume=False,
    )
    monkeypatch.setattr(
        conversation_module, "async_evaluate_rule", AsyncMock(return_value=evaluation)
    )
    local_intent = AsyncMock()
    monkeypatch.setattr(
        conversation_module, "async_try_handle_local_intent", local_intent
    )
    response = intent.IntentResponse(language="en")
    response.async_set_speech("Battery checked")
    expected = conversation.ConversationResult(
        response=response, conversation_id="conversation-1"
    )
    entity._async_handle_message_with_ha_tools = AsyncMock(return_value=expected)
    result = await process()
    assert result is expected
    entity._async_handle_message_with_ha_tools.assert_awaited_once()
    assert (
        entity._async_handle_message_with_ha_tools.call_args.args[0].text
        == user_input.text
    )
    local_intent.assert_not_awaited()


@pytest.mark.parametrize(
    "mode",
    [
        DEFAULT_CONTINUE_CONVERSATION,
        CONTINUE_CONVERSATION_ALWAYS,
        CONTINUE_CONVERSATION_CONDITIONAL,
    ],
)
@pytest.mark.parametrize("successful", [False, True])
async def test_local_intent_bypasses_provider_and_records_continuity(
    monkeypatch, mode, successful
):
    entity, _user_input, _chat_log, _policy, process = _pipeline_fixture(
        monkeypatch, text="is the kitchen light on"
    )
    entity.subentry.data[CONF_CONTINUE_CONVERSATION] = mode
    local_response = intent.IntentResponse(language="en")
    if not successful:
        local_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, "Failed")
    local_response.async_set_speech("The kitchen light is on.")
    try_local_intent = AsyncMock(
        return_value=LocalIntentResult(
            response=local_response,
            intent_name="HassGetState",
        )
    )
    monkeypatch.setattr(
        conversation_module, "async_try_handle_local_intent", try_local_intent
    )

    result = await process()

    assert result.response is local_response
    assert result.continue_conversation is (
        successful and mode == CONTINUE_CONVERSATION_ALWAYS
    )
    try_local_intent.assert_awaited_once()
    assert try_local_intent.call_args.kwargs["guest_active"] is False
    entity._async_handle_message_with_ha_tools.assert_not_awaited()
    entity._continuity.async_record_success.assert_awaited_once()
    payload = entity.hass.bus.async_fire.call_args.args[1]
    assert payload["status"] == "local"
    assert payload["handled_locally"] is True


@pytest.mark.parametrize("successful", [False, True])
async def test_provider_result_records_usage_archive_and_successful_continuity(
    monkeypatch, successful
):
    entity, user_input, chat_log, policy, process = _pipeline_fixture(monkeypatch)
    usage = _UsageRecorder(successful=successful)
    archive = SimpleNamespace(async_record_turn=AsyncMock())
    archive_session = SimpleNamespace(session_id="archive-1")
    entity._usage = usage
    entity._archive = archive
    entity._async_begin_archive_session = AsyncMock(return_value=archive_session)
    entity._effective_guest_policy = MagicMock(return_value=policy)

    monkeypatch.setattr(
        conversation_module,
        "async_try_handle_local_intent",
        AsyncMock(return_value=None),
    )

    provider_response = intent.IntentResponse(language="en")
    provider_response.async_set_speech("Provider response")
    expected = conversation.ConversationResult(
        response=provider_response,
        conversation_id="conversation-1",
    )

    async def provider_success(_user_input, log, _request_options):
        log.content.append(
            conversation.AssistantContent(
                agent_id=entity.entity_id,
                content="Provider response",
            )
        )
        return expected

    entity._async_handle_message_with_ha_tools = AsyncMock(side_effect=provider_success)

    result = await process()

    assert result is expected
    assert usage.calls == [
        {
            "home_assistant_conversation_id": "conversation-1",
            "source_device_id": "kitchen",
        }
    ]
    archive.async_record_turn.assert_awaited_once_with(
        "archive-1",
        run_id="run-1",
        user_text=user_input.text,
        assistant_text="Provider response",
        successful=successful,
    )
    if successful:
        entity._continuity.async_record_success.assert_awaited_once_with(
            "device:kitchen",
            "claim-token",
            chat_log.content,
        )
    else:
        entity._continuity.async_record_success.assert_not_awaited()


async def test_unexpected_failure_restores_request_scoped_context(monkeypatch):
    entity, _user_input, _chat_log, _policy, process = _pipeline_fixture(monkeypatch)
    monkeypatch.setattr(
        conversation_module,
        "async_try_handle_local_intent",
        AsyncMock(return_value=None),
    )
    entity._async_handle_message_with_ha_tools = AsyncMock(
        side_effect=RuntimeError("provider wrapper failed")
    )

    outer_scope = user_scope("outer", source="test", device_id="hall")
    outer_function_group = object()
    tokens = [
        (_ACTIVE_SCOPE, _ACTIVE_SCOPE.set(outer_scope)),
        (_ACTIVE_ARCHIVE, _ACTIVE_ARCHIVE.set(("outer", "archive"))),
        (_ACTIVE_TEMPORARY_SCOPE, _ACTIVE_TEMPORARY_SCOPE.set("outer-temp")),
        (
            _ACTIVE_FUNCTION_GROUP_SESSION,
            _ACTIVE_FUNCTION_GROUP_SESSION.set(outer_function_group),
        ),
        (_ACTIVE_MEMORY_SESSION, _ACTIVE_MEMORY_SESSION.set(("outer-memory", 5))),
    ]
    try:
        with pytest.raises(RuntimeError, match="provider wrapper failed"):
            await process()

        assert _ACTIVE_SCOPE.get() is outer_scope
        assert _ACTIVE_ARCHIVE.get() == ("outer", "archive")
        assert _ACTIVE_TEMPORARY_SCOPE.get() == "outer-temp"
        assert _ACTIVE_FUNCTION_GROUP_SESSION.get() is outer_function_group
        assert _ACTIVE_MEMORY_SESSION.get() == ("outer-memory", 5)
    finally:
        for context_var, token in reversed(tokens):
            context_var.reset(token)


async def test_rejected_request_rule_does_not_continue_in_always_mode(monkeypatch):
    entity, _, _, _, process = _pipeline_fixture(monkeypatch)
    entity.subentry.data[CONF_CONTINUE_CONVERSATION] = CONTINUE_CONVERSATION_ALWAYS
    entity._request_rules = object()
    entity._request_rule_runtime = SimpleNamespace(effective_options=MagicMock())
    monkeypatch.setattr(
        conversation_module,
        "async_evaluate_rule",
        AsyncMock(side_effect=conversation_module.HomeAssistantError("Rejected")),
    )
    result = await process()
    assert "Rejected" in result.response.speech["plain"]["speech"]
    assert result.continue_conversation is False
    entity._async_handle_message_with_ha_tools.assert_not_awaited()
    entity._continuity.async_record_success.assert_not_awaited()


def _assist_fixture(monkeypatch):
    """Keep selection, claims and result construction real; recreate HA logs."""
    entity, _, _, policy, _ = _pipeline_fixture(monkeypatch)
    # Optional managers are already represented by the lightweight fixture.
    # Mark this options object reconciled, as HA setup does before requests.
    entity._extended_openai_runtime_config_data = entity.subentry.data
    entity._continuity = ConversationContinuity("agent")
    entity._resolve_live_guest_policy = MagicMock(return_value=policy)
    del entity._async_handle_message_with_ha_tools
    entity._configured_function_tools_from_data = MagicMock(return_value=[])
    entity._get_exposed_entities = MagicMock(return_value=[])
    entity._get_function_tools = MagicMock(return_value=[])
    entity._async_retrieve_memories = AsyncMock(return_value=[])
    entity._async_retrieve_temporary_memories = AsyncMock(return_value=[])
    entity._build_system_prompt = MagicMock(return_value="system")
    monkeypatch.setattr(
        conversation_module,
        "async_try_handle_local_intent",
        AsyncMock(return_value=None),
    )
    logs = []
    seen = []

    def new_log(_hass, _session, user_input):
        log = SimpleNamespace(
            content=[
                conversation.SystemContent(content="system"),
                conversation.UserContent(content=user_input.text),
            ],
            conversation_id=user_input.conversation_id or f"ha-{len(logs)}",
            continue_conversation=False,
        )
        logs.append(log)
        return nullcontext(log)

    monkeypatch.setattr(conversation_module, "async_get_chat_log", new_log)

    async def model(log, **kwargs):
        seen.append([item.content for item in log.content])
        log.content.append(
            conversation.AssistantContent(
                agent_id=entity.entity_id, content="Model reply"
            )
        )
        return False

    entity._async_handle_chat_log = AsyncMock(side_effect=model)

    async def invoke(
        text="hello",
        *,
        device="kitchen",
        satellite=None,
        incoming=None,
        direct=False,
    ):
        request = SimpleNamespace(
            text=text,
            language="en",
            conversation_id=incoming,
            context=Context(user_id="alice"),
            device_id=device,
            satellite_id=satellite,
        )
        request.as_llm_context = lambda _domain: SimpleNamespace(
            context=request.context
        )
        if direct:
            return await entity.async_process_direct(request)
        return await entity._async_process(request)

    return entity, invoke, logs, seen


@pytest.mark.parametrize("blank", ["   ", "\t", "\n", " \t\n "])
async def test_blank_assist_input_is_ignored_before_conversation_processing(
    monkeypatch, blank
):
    entity, invoke, logs, seen = _assist_fixture(monkeypatch)
    resolve = AsyncMock(wraps=entity._continuity.async_resolve)
    monkeypatch.setattr(entity._continuity, "async_resolve", resolve)
    entity._usage = SimpleNamespace(async_run=MagicMock())
    entity._archive = SimpleNamespace(async_record_turn=AsyncMock())

    result = await invoke(blank, incoming="existing-conversation")

    assert result.conversation_id == "existing-conversation"
    assert result.continue_conversation is False
    assert result.response.speech["plain"]["speech"] == ""
    assert logs == []
    assert seen == []
    resolve.assert_not_awaited()
    entity._usage.async_run.assert_not_called()
    entity._archive.async_record_turn.assert_not_awaited()


async def test_blank_assist_input_does_not_replay_established_history(monkeypatch):
    entity, invoke, logs, seen = _assist_fixture(monkeypatch)
    entity.subentry.data[CONF_CONVERSATION_CONTINUITY] = CONVERSATION_CONTINUITY_DEVICE

    first = await invoke("Remember the blue mug")
    assert len(logs) == 1
    assert len(seen) == 1

    resolve = AsyncMock(wraps=entity._continuity.async_resolve)
    monkeypatch.setattr(entity._continuity, "async_resolve", resolve)
    provider_calls = entity._async_handle_chat_log.await_count

    blank = await invoke("   ", incoming=first.conversation_id)

    assert blank.conversation_id == first.conversation_id
    assert blank.response.speech["plain"]["speech"] == ""
    assert len(logs) == 1
    assert len(seen) == 1
    assert entity._async_handle_chat_log.await_count == provider_calls
    resolve.assert_not_awaited()


@pytest.mark.parametrize(
    ("mode", "second_device", "expired", "resumes"),
    [
        (CONVERSATION_CONTINUITY_DEVICE, "kitchen", False, True),
        (CONVERSATION_CONTINUITY_DEVICE, "bedroom", False, False),
        (CONVERSATION_CONTINUITY_USER, "bedroom", False, True),
        (CONVERSATION_CONTINUITY_DEVICE, "kitchen", True, False),
    ],
)
async def test_separate_assist_calls_reinject_saved_history_into_recreated_chat_log(
    monkeypatch, mode, second_device, expired, resumes
):
    entity, invoke, logs, seen = _assist_fixture(monkeypatch)
    entity.subentry.data.update(
        {CONF_CONVERSATION_CONTINUITY: mode, CONF_CONVERSATION_TIMEOUT_MINUTES: 37}
    )
    first = await invoke("Remember the blue mug")
    if expired:
        for session in entity._continuity._sessions.values():
            session.last_active = dt_util.utcnow() - timedelta(minutes=38)
    second = await invoke("What colour?", device=second_device)
    assert logs[0] is not logs[1]
    assert (first.conversation_id == second.conversation_id) is resumes
    assert seen[0] == ["system", "Remember the blue mug"]
    assert seen[1] == (
        ["system", "Remember the blue mug", "Model reply", "What colour?"]
        if resumes
        else ["system", "What colour?"]
    )


@pytest.mark.parametrize(
    "mode",
    [
        CONVERSATION_CONTINUITY_HA_DEFAULT,
        CONVERSATION_CONTINUITY_DEVICE,
        CONVERSATION_CONTINUITY_USER,
    ],
)
@pytest.mark.parametrize("guest", [False, True])
@pytest.mark.parametrize("satellite", [None, "assist.kitchen"])
async def test_assist_passes_configured_continuity_scope_timeout_and_guest_namespace(
    monkeypatch, mode, guest, satellite
):
    entity, invoke, _, _ = _assist_fixture(monkeypatch)
    entity.subentry.data.update(
        {CONF_CONVERSATION_CONTINUITY: mode, CONF_CONVERSATION_TIMEOUT_MINUTES: 37}
    )
    entity._resolve_live_guest_policy.return_value = GuestCapabilityPolicy(guest)
    resolve = AsyncMock(wraps=entity._continuity.async_resolve)
    monkeypatch.setattr(entity._continuity, "async_resolve", resolve)
    await invoke(satellite=satellite, incoming="incoming-ha-id")
    args = resolve.await_args.args
    assert args[0] == mode
    assert args[1].user_id == "alice"
    # Production Voice Identity has always preferred HA's registry device ID when
    # Assist supplies both device_id and satellite_id. The explicit owner now makes
    # that installed runtime behavior visible to this orchestration fixture too.
    assert args[1].device_id == "kitchen"
    assert args[2:] == ("kitchen", "incoming-ha-id", 37)
    assert resolve.await_args.kwargs == {"namespace": "guest" if guest else None}


async def test_assist_owner_and_guest_histories_are_isolated_in_both_directions(
    monkeypatch,
):
    entity, invoke, _, seen = _assist_fixture(monkeypatch)
    entity.subentry.data[CONF_CONVERSATION_CONTINUITY] = CONVERSATION_CONTINUITY_DEVICE
    owner = await invoke("Owner secret")
    entity._resolve_live_guest_policy.return_value = GuestCapabilityPolicy(True)
    guest = await invoke("Guest secret", incoming=owner.conversation_id)
    await invoke("Guest follow-up")
    entity._resolve_live_guest_policy.return_value = (
        GuestCapabilityPolicy.unrestricted()
    )
    resumed_owner = await invoke("Owner follow-up", incoming=guest.conversation_id)
    assert owner.conversation_id != guest.conversation_id
    assert resumed_owner.conversation_id == owner.conversation_id
    assert seen[1] == ["system", "Guest secret"]
    assert seen[2] == ["system", "Guest secret", "Model reply", "Guest follow-up"]
    assert seen[3] == ["system", "Owner secret", "Model reply", "Owner follow-up"]


@pytest.mark.parametrize(
    ("mode", "ha_default", "decision", "expected"),
    [
        (DEFAULT_CONTINUE_CONVERSATION, False, None, False),
        (DEFAULT_CONTINUE_CONVERSATION, True, None, True),
        (CONTINUE_CONVERSATION_ALWAYS, False, None, True),
        (CONTINUE_CONVERSATION_ALWAYS, True, None, True),
        (CONTINUE_CONVERSATION_CONDITIONAL, True, False, False),
        (CONTINUE_CONVERSATION_CONDITIONAL, False, True, True),
    ],
)
async def test_assist_continue_mode_reaches_final_conversation_result(
    monkeypatch, mode, ha_default, decision, expected
):
    entity, invoke, _, _ = _assist_fixture(monkeypatch)
    entity.subentry.data[CONF_CONTINUE_CONVERSATION] = mode

    async def model(log, **kwargs):
        assert kwargs["conditional_continue"] is (
            mode == CONTINUE_CONVERSATION_CONDITIONAL
        )
        log.continue_conversation = ha_default
        log.content.append(
            conversation.AssistantContent(agent_id=entity.entity_id, content="Reply")
        )
        return decision

    entity._async_handle_chat_log.side_effect = model
    result = await invoke()
    assert result.response.speech["plain"]["speech"] == "Reply"
    assert result.continue_conversation is expected


async def test_assist_malformed_control_output_returns_error_without_listening(
    monkeypatch,
):
    """Adapter parse failures remain safe even if HA would keep listening."""
    from custom_components.extended_openai_conversation_responses.exceptions import (
        ParseArgumentsFailed,
    )

    entity, invoke, logs, _ = _assist_fixture(monkeypatch)
    entity.subentry.data[CONF_CONTINUE_CONVERSATION] = CONTINUE_CONVERSATION_CONDITIONAL

    async def invalid_finalizer(log, **kwargs):
        log.continue_conversation = True
        raise ParseArgumentsFailed('{"response":"Done","continue_conversation":"yes"}')

    entity._async_handle_chat_log.side_effect = invalid_finalizer
    result = await invoke()
    assert logs[0].continue_conversation is True
    assert result.response.error_code is intent.IntentResponseErrorCode.UNKNOWN
    assert result.continue_conversation is False


async def test_direct_assist_returns_consumed_rule_metadata(monkeypatch):
    """Direct callers receive the routing decision alongside the local response."""
    entity, invoke, _, _ = _assist_fixture(monkeypatch)
    entity._request_rules = object()
    entity._request_rule_runtime = SimpleNamespace(
        effective_options=MagicMock(return_value={})
    )
    evaluation = RuleEvaluation(
        match=RuleMatch(
            {"id": "scene", "name": "Set scene"},
            "set evening scene",
            False,
            100.0,
            slots={"scene": "evening"},
        ),
        consume=True,
        response="Evening scene set",
        successful=True,
    )
    monkeypatch.setattr(
        conversation_module,
        "async_evaluate_rule",
        AsyncMock(return_value=evaluation),
    )

    result, metadata = await invoke("set evening scene", direct=True)

    assert result.response.speech["plain"]["speech"] == "Evening scene set"
    assert metadata == {
        "handled_locally": True,
        "matched_rule": {"id": "scene", "name": "Set scene"},
        "captured_values": {"scene": "evening"},
    }


async def test_direct_assist_returns_local_intent_metadata(monkeypatch):
    """Direct local-intent responses identify the Home Assistant intent used."""
    _entity, invoke, _, _ = _assist_fixture(monkeypatch)
    local_response = intent.IntentResponse(language="en")
    local_response.async_set_speech("The kitchen light is on.")
    monkeypatch.setattr(
        conversation_module,
        "async_try_handle_local_intent",
        AsyncMock(
            return_value=LocalIntentResult(
                response=local_response,
                intent_name="HassGetState",
            )
        ),
    )

    result, metadata = await invoke("is the kitchen light on", direct=True)

    assert result.response is local_response
    assert metadata == {
        "handled_locally": True,
        "matched_intent": "HassGetState",
    }


async def test_fresh_conversation_request_resets_owned_state_after_response(
    monkeypatch,
):
    """A model-requested reset is finalized after the current response completes."""
    entity, _, chat_log, _, process = _pipeline_fixture(monkeypatch)
    monkeypatch.setattr(
        conversation_module,
        "async_try_handle_local_intent",
        AsyncMock(return_value=None),
    )
    reset_context = AsyncMock()
    monkeypatch.setattr(
        conversation_module, "async_reset_conversation_context", reset_context
    )

    async def schedule_reset(*_args):
        conversation_module.request_fresh_conversation(
            "state-session", "memory-session"
        )
        chat_log.content.append(
            conversation.AssistantContent(
                agent_id=entity.entity_id,
                content="I will start fresh next time.",
            )
        )
        return object()

    entity._async_handle_message_with_ha_tools = AsyncMock(side_effect=schedule_reset)

    await process()

    reset_context.assert_awaited_once_with(
        entity.hass,
        entity._continuity,
        "entry",
        "agent",
        continuity_key="device:kitchen",
        state_session_id="state-session",
        memory_session_id="memory-session",
    )


async def test_archive_turn_with_empty_chat_log_records_empty_assistant_text() -> None:
    """Archive recording remains best-effort when the provider produced no text."""
    record_turn = AsyncMock()
    entity = object.__new__(ExtendedOpenAIAgentEntity)
    entity.subentry = SimpleNamespace(data={})
    entity._archive = SimpleNamespace(async_record_turn=record_turn)
    entity._effective_guest_policy = MagicMock(
        return_value=SimpleNamespace(archive_retention=True)
    )

    await entity._async_archive_turn(
        SimpleNamespace(session_id="session-1"),
        "run-1",
        SimpleNamespace(text="Hello"),
        SimpleNamespace(content=[]),
        successful=False,
    )

    record_turn.assert_awaited_once_with(
        "session-1",
        run_id="run-1",
        user_text="Hello",
        assistant_text="",
        successful=False,
    )
