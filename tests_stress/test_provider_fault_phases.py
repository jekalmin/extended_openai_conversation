"""Nightly streaming and post-side-effect provider fault phases."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _chat_sse_tool_call,
    _prepare_service,
    _responses_sse_text,
    _responses_sse_tool_call,
    _tool_result_from_chat_request,
    _tool_result_from_responses_request,
)
from tests_real_ha.test_provider_wire_truncated_tool_stream import (
    _chat_sse_partial_tool_call,
    _responses_sse_partial_tool_call,
)
from tests_stress.conftest import record
from tests_stress.provider_fault_transport import ProviderFaultTransport, WireStep
from tests_stress.test_provider_fault_matrix import (
    _OWNER,
    _agent,
    _archived_successes,
    _say,
)


def _reply(mode: str, text: str) -> bytes:
    return (
        _chat_sse_text(text)
        if mode == API_MODE_CHAT_COMPLETIONS
        else _responses_sse_text(text)
    )


def _tool(mode: str) -> bytes:
    return (
        _chat_sse_tool_call()
        if mode == API_MODE_CHAT_COMPLETIONS
        else _responses_sse_tool_call()
    )


def _partial_assistant(mode: str) -> bytes:
    full = _reply(mode, "Partial assistant output")
    if mode == API_MODE_CHAT_COMPLETIONS:
        return full.split(b"data: [DONE]")[0]
    return b"\n\n".join(full.split(b"\n\n")[:2]) + b"\n\n"


def _partial_tool(mode: str) -> bytes:
    return (
        _chat_sse_partial_tool_call()
        if mode == API_MODE_CHAT_COMPLETIONS
        else _responses_sse_partial_tool_call()
    )


def _malformed_tool(mode: str) -> bytes:
    frames: list[bytes] = []
    for frame in _tool(mode).split(b"\n\n"):
        if not frame.startswith(b"data: {"):
            if frame:
                frames.append(frame)
            continue
        event = json.loads(frame[6:])
        if mode == API_MODE_CHAT_COMPLETIONS:
            event["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = (
                '{"list":'
            )
        else:
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                item["arguments"] = '{"list":'
            response = event.get("response")
            if isinstance(response, dict):
                for output in response.get("output", []):
                    if output.get("type") == "function_call":
                        output["arguments"] = '{"list":'
        frames.append(f"data: {json.dumps(event)}".encode())
    return b"\n\n".join(frames) + b"\n\n"


@pytest.mark.parametrize("mode", [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES])
@pytest.mark.parametrize(
    "phase", ["partial_assistant", "partial_tool", "malformed_tool"]
)
async def test_stream_failure_never_runs_a_partial_tool_and_next_turn_recovers(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    mode: str,
    phase: str,
) -> None:
    """Exercise SDK streaming and tool parsing through the public Assist path."""
    MockUser(id=_OWNER, name="Provider fault owner", is_owner=True).add_to_hass(hass)
    agent = await _agent(hass, mode, tools=True)
    calls = await _prepare_service(hass)
    first = (
        WireStep("sse", body=_malformed_tool(mode))
        if phase == "malformed_tool"
        else WireStep(
            "stream_break",
            body=_partial_assistant(mode)
            if phase == "partial_assistant"
            else _partial_tool(mode),
        )
    )
    wire = ProviderFaultTransport(
        [first, WireStep("sse", body=_reply(mode, "Recovered."))]
    )
    wire.install(monkeypatch, agent)
    record(stress_trace, "fault_injected", mode=mode, phase=phase, kind=first.kind)

    failed = await _say(hass, agent, f"Seeded {phase} fault")
    assert failed.response.error_code is not None
    assert calls == []
    assert _archived_successes(agent) == [False]
    assert agent._usage.runs[0].successful is False

    recovered = await _say(hass, agent, "Recover after interrupted provider stream")
    assert recovered.response.error_code is None
    assert recovered.response.as_dict()["speech"]["plain"]["speech"] == "Recovered."
    assert calls == []
    assert sorted(_archived_successes(agent)) == [False, True]
    assert agent._usage.totals.conversation_count == 2
    assert len(wire.requests) == 2
    record(stress_trace, "stream_phase", mode=mode, phase=phase)


@pytest.mark.parametrize("mode", [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES])
async def test_disconnect_after_tool_side_effect_never_replays_it(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    mode: str,
) -> None:
    """A failed continuation follows one real HA action and one tool result."""
    MockUser(id=_OWNER, name="Provider fault owner", is_owner=True).add_to_hass(hass)
    agent = await _agent(hass, mode, tools=True)
    calls = await _prepare_service(hass)
    wire = ProviderFaultTransport(
        [
            WireStep("sse", body=_tool(mode)),
            WireStep("transport", error="before_headers"),
            WireStep("sse", body=_reply(mode, "Recovered without replay.")),
        ]
    )
    wire.install(monkeypatch, agent)
    record(
        stress_trace,
        "fault_injected",
        mode=mode,
        phase="after_tool_side_effect",
        kind="disconnect_before_headers",
    )

    failed = await _say(hass, agent, "Turn off the test light")
    assert failed.response.error_code is not None
    assert len(calls) == 1
    tool_result: dict[str, Any] = (
        _tool_result_from_chat_request(wire.requests[1]["body"])
        if mode == API_MODE_CHAT_COMPLETIONS
        else _tool_result_from_responses_request(wire.requests[1]["body"])
    )
    assert tool_result["result"][0]["success"] is True
    assert _archived_successes(agent) == [False]
    assert agent._usage.runs[0].successful is False

    recovered = await _say(hass, agent, "Ask a fresh question")
    assert recovered.response.error_code is None
    assert (
        recovered.response.as_dict()["speech"]["plain"]["speech"]
        == "Recovered without replay."
    )
    assert len(calls) == 1
    assert len(wire.requests) == 3
    assert sorted(_archived_successes(agent)) == [False, True]
    record(stress_trace, "post_tool_disconnect", mode=mode, service_calls=len(calls))


@pytest.mark.parametrize("mode", [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES])
async def test_provider_replays_completed_call_id_after_lost_tool_result(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    mode: str,
) -> None:
    """A same-ID replay on the same conversation cannot repeat a HA side effect."""
    MockUser(id=_OWNER, name="Provider replay owner", is_owner=True).add_to_hass(hass)
    agent = await _agent(hass, mode, tools=True)
    calls = await _prepare_service(hass)
    wire = ProviderFaultTransport(
        [
            WireStep("sse", body=_tool(mode)),
            WireStep("transport", error="before_headers"),
            WireStep("sse", body=_tool(mode)),
            WireStep("sse", body=_reply(mode, "Recovered after rejected replay.")),
        ]
    )
    wire.install(monkeypatch, agent)

    conversation_id = uuid4().hex

    async def continue_turn(text: str) -> conversation.ConversationResult:
        return await conversation.async_converse(
            hass=hass,
            text=text,
            conversation_id=conversation_id,
            context=Context(),
            language="en",
            agent_id=agent.entry.entry_id,
        )

    failed = await continue_turn("Turn off the test light")
    assert failed.response.error_code is not None
    assert len(calls) == 1

    replay = await continue_turn("Retry the interrupted request")
    assert replay.response.error_code is not None
    assert len(calls) == 1
    recovered = await continue_turn("Report status")
    assert recovered.response.error_code is None
    assert recovered.response.as_dict()["speech"]["plain"]["speech"] == (
        "Recovered after rejected replay."
    )
    assert len(calls) == 1
    assert len(wire.requests) == 4
    record(
        stress_trace,
        "post_tool_replay",
        mode=mode,
        call_id="same",
        service_calls=len(calls),
        recovered=True,
    )


async def test_unknown_responses_event_is_ignored_before_valid_completion(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
) -> None:
    """A future provider event cannot invent a tool or corrupt the next result."""
    MockUser(id=_OWNER, name="Provider fault owner", is_owner=True).add_to_hass(hass)
    agent = await _agent(hass, API_MODE_RESPONSES, tools=True)
    calls = await _prepare_service(hass)
    unknown = b'data: {"type":"response.future_event","sequence_number":0}\n\n'
    wire = ProviderFaultTransport(
        [WireStep("sse", body=unknown + _responses_sse_text("Known completion wins."))]
    )
    wire.install(monkeypatch, agent)
    result = await _say(hass, agent, "Unknown event probe")
    assert result.response.error_code is None
    assert (
        result.response.as_dict()["speech"]["plain"]["speech"]
        == "Known completion wins."
    )
    assert calls == []
    assert agent._usage.totals.successful_request_count == 1
    record(stress_trace, "unknown_event_ignored")


@pytest.mark.parametrize("mode", [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES])
async def test_slow_provider_response_does_not_leave_agent_stuck(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    mode: str,
) -> None:
    MockUser(id=_OWNER, name="Provider fault owner", is_owner=True).add_to_hass(hass)
    agent = await _agent(hass, mode)
    wire = ProviderFaultTransport(
        [
            WireStep("sse", body=_reply(mode, "Slow answer."), delay=0.8),
            WireStep("sse", body=_reply(mode, "Next answer.")),
        ]
    )
    wire.install(monkeypatch, agent)
    first = await _say(hass, agent, "Slow provider")
    second = await _say(hass, agent, "Next provider")
    assert first.response.as_dict()["speech"]["plain"]["speech"] == "Slow answer."
    assert second.response.as_dict()["speech"]["plain"]["speech"] == "Next answer."
    assert agent._usage.totals.successful_request_count == 2
    assert agent._archive.stats()["turn_count"] == 2
    record(stress_trace, "slow_provider", mode=mode, delay=0.8)
