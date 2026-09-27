"""A consumer abandoning an active valid SDK stream closes its HTTP stream."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import gc
import json

import httpx
import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
)
from homeassistant.core import HomeAssistant
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _chat_sse_tool_call,
    _prepare_service,
    _raw_client,
    _responses_sse_text,
    _responses_sse_tool_call,
    _speech,
)
from tests_stress.conftest import record
from tests_stress.provider_fault_transport import GatedSSEStream
from tests_stress.test_provider_valid_chunking import _agent, _say
from tests_stress.test_runtime_soak import _resource_footprint

_OWNER = "chunk-boundary-owner"


def _frames(payload: bytes) -> list[bytes]:
    return [frame + b"\n\n" for frame in payload.strip().split(b"\n\n")]


def _event(value: dict) -> bytes:
    return b"data: " + json.dumps(value, separators=(",", ":")).encode() + b"\n\n"


def _partial_text(mode: str) -> list[bytes]:
    if mode == API_MODE_RESPONSES:
        frames = _frames(_responses_sse_text("Partial complete."))
        return [b"".join(frames[:2]), b"".join(frames[2:])]
    event = json.loads(_frames(_chat_sse_text("Partial"))[0][6:].strip())
    event["choices"][0]["finish_reason"] = None
    terminal = json.loads(_frames(_chat_sse_text(" complete."))[0][6:].strip())
    return [_event(event), _event(terminal) + b"data: [DONE]\n\n"]


def _partial_tool(mode: str) -> list[bytes]:
    if mode == API_MODE_RESPONSES:
        frames = _frames(_responses_sse_tool_call())
        added = json.loads(frames[0][6:].strip())
        partial = {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 1,
            "item_id": added["item"]["id"],
            "output_index": 0,
            "delta": '{"list":',
        }
        terminal = [json.loads(frame[6:].strip()) for frame in frames[1:]]
        arguments = terminal[0]["item"]["arguments"]
        remainder = {
            **partial,
            "sequence_number": 2,
            "delta": arguments[len(partial["delta"]) :],
        }
        completed_arguments = {
            "type": "response.function_call_arguments.done",
            "sequence_number": 3,
            "item_id": added["item"]["id"],
            "output_index": 0,
            "name": added["item"]["name"],
            "arguments": arguments,
        }
        for index, event in enumerate(terminal, start=4):
            event["sequence_number"] = index
        return [
            frames[0] + _event(partial),
            b"".join(
                _event(event) for event in [remainder, completed_arguments, *terminal]
            ),
        ]
    original = json.loads(_frames(_chat_sse_tool_call())[0][6:].strip())
    first = json.loads(json.dumps(original))
    first["choices"][0]["finish_reason"] = None
    first["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = '{"list":'
    last = json.loads(json.dumps(original))
    last["choices"][0]["delta"].pop("role", None)
    last["choices"][0]["delta"]["tool_calls"][0].pop("id")
    last["choices"][0]["delta"]["tool_calls"][0]["function"].pop("name")
    last["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = original[
        "choices"
    ][0]["delta"]["tool_calls"][0]["function"]["arguments"][len('{"list":') :]
    return [_event(first), _event(last) + b"data: [DONE]\n\n"]


@pytest.mark.parametrize("mode", [API_MODE_RESPONSES, API_MODE_CHAT_COMPLETIONS])
@pytest.mark.parametrize("phase", ["partial_text", "partial_tool", "tool_continuation"])
async def test_consumer_cancellation_closes_active_provider_stream(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    mode: str,
    phase: str,
) -> None:
    """No terminal bytes or partial Function Tool arguments may run after cancel."""
    MockUser(id=_OWNER, name="Stream owner", is_owner=True).add_to_hass(hass)
    agent = await _agent(hass, mode)
    service_calls = await _prepare_service(hass)
    raw = _raw_client(agent)
    monkeypatch.setattr(raw, "max_retries", 0)
    gated = GatedSSEStream(
        _partial_tool(mode) if phase == "partial_tool" else _partial_text(mode),
        gate_after=1,
    )
    requests: list[dict] = []
    streams: list[GatedSSEStream] = []

    async def send(request: httpx.Request, *args, **kwargs) -> httpx.Response:
        del args, kwargs
        body = json.loads(request.content)
        requests.append(body)
        if phase == "tool_continuation" and len(requests) == 1:
            payload = (
                _responses_sse_tool_call()
                if mode == API_MODE_RESPONSES
                else _chat_sse_tool_call()
            )
            stream = GatedSSEStream([payload])
        elif (phase == "tool_continuation" and len(requests) == 2) or (
            phase != "tool_continuation" and len(requests) == 1
        ):
            stream = gated
        else:
            payload = (
                _responses_sse_text("Next turn is healthy.")
                if mode == API_MODE_RESPONSES
                else _chat_sse_text("Next turn is healthy.")
            )
            stream = GatedSSEStream([payload])
        streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    monkeypatch.setattr(raw._client, "send", send)
    before = _resource_footprint(hass)
    active = asyncio.create_task(_say(hass, agent, "Cancel active stream"))
    try:
        await asyncio.wait_for(gated.delivered.wait(), timeout=10)
        assert not active.done()
        assert len(service_calls) == (1 if phase == "tool_continuation" else 0)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, timeout=10)
        await asyncio.wait_for(gated.closed.wait(), timeout=10)
    finally:
        gated.release.set()
        if not active.done():
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active
    assert gated.yielded == 1
    assert len(service_calls) == (1 if phase == "tool_continuation" else 0)
    assert not [
        turn
        for turns in agent._archive._turns.values()
        for turn in turns
        if turn.successful
    ]
    prior_requests = len(requests)
    recovered = await _say(hass, agent, "Next valid request")
    assert _speech(recovered) == "Next turn is healthy."
    assert len(requests) == prior_requests + 1
    assert len(service_calls) == (1 if phase == "tool_continuation" else 0)
    await hass.async_block_till_done()
    gc.collect()
    assert _resource_footprint(hass) == before
    assert all(stream.closed.is_set() for stream in streams)
    record(
        stress_trace,
        "active_stream_cancel",
        mode=mode,
        phase=phase,
        requests=len(requests),
    )
