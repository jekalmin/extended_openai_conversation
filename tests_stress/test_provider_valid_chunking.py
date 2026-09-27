"""Valid SDK SSE meaning is independent of HTTP byte chunk boundaries."""

from __future__ import annotations

from copy import deepcopy
import json

import httpx
import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_ARCHIVE_ENABLED,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
    DEFAULT_CONF_FUNCTION_TOOLS,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_provider_wire_e2e import (
    _TOOL_ARGUMENTS,
    _TOOL_CALL_ID,
    _chat_sse_text,
    _chat_sse_tool_call,
    _prepare_service,
    _raw_client,
    _responses_sse_text,
    _responses_sse_tool_call,
    _speech,
)
from tests_stress.conftest import record
from tests_stress.provider_fault_transport import GatedSSEStream, split_valid_sse

_OWNER = "chunk-boundary-owner"


def _utf8_sse(body: bytes) -> bytes:
    """Keep the SSE events identical while emitting literal UTF-8 JSON strings."""
    return b"".join(
        b"data: "
        + json.dumps(json.loads(line[6:]), ensure_ascii=False).encode()
        + b"\n\n"
        if line.startswith(b"data: ") and line != b"data: [DONE]"
        else line + b"\n\n"
        for line in body.strip().split(b"\n\n")
    )


def _split_patterns(body: bytes, seed: int) -> dict[str, list[bytes]]:
    def around(needle: bytes) -> list[int]:
        at = body.find(needle)
        assert at >= 0, needle
        return sorted({at + 1, at + len(needle) - 1})

    frames = [
        index + 2
        for index in range(len(body) - 1)
        if body[index : index + 2] == b"\n\n"
    ]
    line_ends = [index + 1 for index, byte in enumerate(body[:-1]) if byte == 10]
    line_starts = [0, *line_ends]
    line_parts = sorted(
        {
            start + (end - start) // divisor
            for start, end in zip(line_starts, [*line_ends, len(body)], strict=True)
            for divisor in (2, 3)
            if end - start >= 4
        }
    )
    utf8 = body.find("É".encode())
    escape = b"\\n" if b"\\n" in body else b'\\"'
    patterns = {
        "one_chunk": [body],
        "one_frame": split_valid_sse(body, positions=frames[:-1]),
        "line_midpoints": split_valid_sse(body, positions=line_parts),
        "one_byte": split_valid_sse(body, sizes=[1] * len(body)),
        "data_prefix": split_valid_sse(body, positions=[1, 2, 3, 4, 5]),
        "frame_delimiters": split_valid_sse(
            body,
            positions=sorted(
                {
                    point
                    for end in frames
                    for point in (end - 2, end - 1, end, end + 1)
                    if 0 < point < len(body)
                }
            ),
        ),
        "json_property": split_valid_sse(
            body, positions=around(b'"type"' if b'"type"' in body else b'"object"')
        ),
        "json_escape": split_valid_sse(body, positions=around(escape)),
        "seeded_alternating": split_valid_sse(body, seed=seed),
    }
    if utf8 >= 0:
        patterns["utf8_code_point"] = split_valid_sse(body, positions=[utf8 + 1])
    return patterns


async def _agent(hass: HomeAssistant, mode: str):
    entry = _make_entry(
        f"Valid SSE chunks {mode}",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: mode,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_ARCHIVE_ENABLED: True,
            CONF_FUNCTION_TOOLS: [deepcopy(DEFAULT_CONF_FUNCTION_TOOLS[0])],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent and agent._usage and agent._archive
    return agent


async def _say(hass: HomeAssistant, agent, text: str):
    return await conversation.async_converse(
        hass=hass,
        text=text,
        conversation_id=None,
        context=Context(user_id=_OWNER),
        language="en",
        agent_id=agent.entry.entry_id,
    )


@pytest.mark.parametrize("mode", [API_MODE_RESPONSES, API_MODE_CHAT_COMPLETIONS])
@pytest.mark.parametrize("with_tool", [False, True])
async def test_valid_sse_chunk_boundaries_preserve_assist_meaning(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_seed: int,
    stress_trace: list[dict],
    mode: str,
    with_tool: bool,
) -> None:
    """Only the real SDK's HTTP byte iterator changes between public turns."""
    MockUser(id=_OWNER, name="Chunk owner", is_owner=True).add_to_hass(hass)
    agent = await _agent(hass, mode)
    service_calls = await _prepare_service(hass)
    text = "Chunk-safe Éire 東京 🙂\\ncomplete."
    text_wire = _utf8_sse(
        _responses_sse_text(text)
        if mode == API_MODE_RESPONSES
        else _chat_sse_text(text)
    )
    tool_wire = _utf8_sse(
        _responses_sse_tool_call()
        if mode == API_MODE_RESPONSES
        else _chat_sse_tool_call()
    )
    wire = tool_wire if with_tool else text_wire
    patterns = _split_patterns(wire, stress_seed)
    current: list[bytes] = []
    requests: list[dict] = []
    streams: list[GatedSSEStream] = []

    async def send(request: httpx.Request, *args, **kwargs) -> httpx.Response:
        del args, kwargs
        body = json.loads(request.content)
        requests.append(body)
        chunks = current if not with_tool or len(requests) % 2 else [text_wire]
        stream = GatedSSEStream(chunks)
        streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    client = _raw_client(agent)
    monkeypatch.setattr(client, "max_retries", 0)
    monkeypatch.setattr(client._client, "send", send)
    for pattern, chunks in patterns.items():
        assert b"".join(chunks) == wire, pattern
        current = chunks
        before_calls = len(service_calls)
        before_requests = len(requests)
        result = await _say(hass, agent, f"Chunk pattern {pattern}")
        assert _speech(result) == text, (mode, with_tool, pattern)
        assert len(requests) - before_requests == (2 if with_tool else 1), pattern
        assert len(service_calls) - before_calls == (1 if with_tool else 0), pattern
        if with_tool:
            continuation = requests[-1]
            if mode == API_MODE_RESPONSES:
                call = next(
                    item
                    for item in continuation["input"]
                    if item.get("type") == "function_call"
                )
                output = next(
                    item
                    for item in continuation["input"]
                    if item.get("type") == "function_call_output"
                )
                assert output["call_id"] == _TOOL_CALL_ID
                assert call["name"] == "execute_services"
                assert json.loads(call["arguments"]) == _TOOL_ARGUMENTS
            else:
                assistant = next(
                    item
                    for item in continuation["messages"]
                    if item.get("role") == "assistant" and item.get("tool_calls")
                )
                call = assistant["tool_calls"][0]
                output = next(
                    item
                    for item in continuation["messages"]
                    if item.get("role") == "tool"
                )
                assert output["tool_call_id"] == _TOOL_CALL_ID
                assert call["function"]["name"] == "execute_services"
                assert json.loads(call["function"]["arguments"]) == _TOOL_ARGUMENTS
            assert service_calls[-1].data["entity_id"] == ["light.provider_wire"]
    assert len(streams) == len(patterns) * (2 if with_tool else 1)
    assert all(stream.closed.is_set() for stream in streams)
    assert agent._usage.totals.conversation_count == len(patterns)
    assert agent._usage.totals.failed_request_count == 0
    assert len(
        [
            turn
            for turns in agent._archive._turns.values()
            for turn in turns
            if turn.successful
        ]
    ) == len(patterns)
    record(
        stress_trace,
        "valid_sse_chunking",
        mode=mode,
        tool=with_tool,
        patterns=list(patterns),
    )
