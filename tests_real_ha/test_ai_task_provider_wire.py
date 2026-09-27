"""Public AI Task acceptance through the real SDK's serialized HTTP exchange."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from openai import BadRequestError
import pytest
from pytest_homeassistant_custom_component.common import MockUser
import voluptuous as vol

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    DOMAIN,
)
from homeassistant.components import ai_task
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from tests_real_ha.test_ai_task_runtime import CallerAPI, ContextProbeTool, _entry
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _chat_sse_tool_call,
    _install_wire,
    _raw_client,
    _responses_sse_text,
    _responses_sse_tool_call,
)

MODES = (API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES)


async def _task_entity(hass: HomeAssistant, mode: str) -> tuple[Any, str]:
    entry = _entry(mode)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    subentry = next(
        s for s in entry.subentries.values() if s.subentry_type == "ai_task_data"
    )
    entity_id = er.async_get(hass).async_get_entity_id(
        ai_task.DOMAIN, DOMAIN, subentry.subentry_id
    )
    assert entity_id is not None
    return entry, entity_id


def _wire(monkeypatch: pytest.MonkeyPatch, entry: Any, replies: list[Any]) -> Any:
    return _install_wire(
        monkeypatch, SimpleNamespace(_client=entry.runtime_data), replies
    )


def _text_reply(mode: str, text: str) -> bytes:
    return (
        _responses_sse_text(text)
        if mode == API_MODE_RESPONSES
        else _chat_sse_text(text)
    )


def _truncated_reply(mode: str) -> bytes:
    if mode == API_MODE_RESPONSES:
        frames = _responses_sse_text("partial").strip().split(b"\n\n")
        return b"\n\n".join(frames[:-1]) + b"\n\n"
    first = _chat_sse_text("partial").split(b"\n\n", 1)[0]
    event = json.loads(first.removeprefix(b"data: "))
    event["choices"][0]["finish_reason"] = None
    return f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()


def _assert_paths(wire: Any, mode: str, count: int) -> None:
    endpoint = "/v1/responses" if mode == API_MODE_RESPONSES else "/v1/chat/completions"
    assert [request["path"] for request in wire.requests] == [endpoint] * count


def _input_text(body: dict[str, Any]) -> str:
    return json.dumps(body.get("input", body.get("messages", [])), sort_keys=True)


@pytest.mark.parametrize("mode", MODES)
async def test_plain_and_structured_ai_tasks_cross_sdk_wire(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    entry, entity_id = await _task_entity(hass, mode)
    wire = _wire(
        monkeypatch,
        entry,
        [
            _text_reply(mode, "Plain result"),
            _text_reply(mode, '{"answer":"ready","count":2}'),
        ],
    )
    plain = await ai_task.async_generate_data(
        hass,
        task_name="Plain Wire Task",
        entity_id=entity_id,
        instructions="PLAIN_WIRE_INSTRUCTIONS",
    )
    schema = vol.Schema({vol.Required("answer"): str, vol.Required("count"): int})
    structured = await ai_task.async_generate_data(
        hass,
        task_name="Structured Wire Task",
        entity_id=entity_id,
        instructions="STRUCTURED_WIRE_INSTRUCTIONS",
        structure=schema,
    )
    assert plain.data == "Plain result"
    assert structured.data == {"answer": "ready", "count": 2}
    _assert_paths(wire, mode, 2)
    first, second = (request["body"] for request in wire.requests)
    input_key = "input" if mode == API_MODE_RESPONSES else "messages"
    assert (
        next(item for item in first[input_key] if item.get("role") == "user")["content"]
        == "PLAIN_WIRE_INSTRUCTIONS"
    )
    assert (
        next(item for item in second[input_key] if item.get("role") == "user")[
            "content"
        ]
        == "STRUCTURED_WIRE_INSTRUCTIONS"
    )
    assert first["stream"] is True
    assert second["stream"] is True
    assert "PLAIN_WIRE_INSTRUCTIONS" in _input_text(first)
    assert "STRUCTURED_WIRE_INSTRUCTIONS" in _input_text(second)
    assert "PLAIN_WIRE_INSTRUCTIONS" not in _input_text(second)
    output_format = (
        second["text"]["format"]
        if mode == API_MODE_RESPONSES
        else second["response_format"]["json_schema"]
    )
    assert output_format["strict"] is True
    assert set(output_format["schema"]["properties"]) == {"answer", "count"}
    assert output_format["schema"]["required"] == ["answer", "count"]


@pytest.mark.parametrize("mode", MODES)
async def test_caller_tool_round_trip_crosses_sdk_wire_with_ha_context(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    entry, entity_id = await _task_entity(hass, mode)
    probe = ContextProbeTool()
    caller = CallerAPI(hass=hass, id="wire-probe", name="Wire Probe")
    caller.tools = [probe]
    user = MockUser(id="ai-task-wire-user", name="AI Task Wire User")
    user.add_to_hass(hass)
    context = Context(user_id=user.id)
    requests: list[dict[str, Any]] = []

    async def send(request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        del args, kwargs
        body = json.loads(request.content)
        requests.append({"path": request.url.path, "body": body})
        if len(requests) == 1:
            tools = body["tools"]
            assert len(tools) == 1
            name = (
                tools[0]["name"]
                if mode == API_MODE_RESPONSES
                else tools[0]["function"]["name"]
            )
            schema = (
                tools[0]["parameters"]
                if mode == API_MODE_RESPONSES
                else tools[0]["function"]["parameters"]
            )
            assert "value" in schema["properties"]
            payload = (
                _responses_sse_tool_call(
                    "call-ai-task-wire", name, {"value": "from-wire"}
                )
                if mode == API_MODE_RESPONSES
                else _chat_sse_tool_call(
                    "call-ai-task-wire", name, {"value": "from-wire"}
                )
            )
        else:
            payload = _text_reply(
                mode, "Tool wire complete" if len(requests) == 2 else "Next independent"
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
            request=request,
        )

    raw = _raw_client(SimpleNamespace(_client=entry.runtime_data))
    monkeypatch.setattr(raw._client, "send", send)
    result = await ai_task.async_generate_data(
        hass,
        task_name="Caller Tool Wire Task",
        entity_id=entity_id,
        instructions="Call the context probe.",
        llm_api=caller,
        context=context,
    )
    assert result.data == "Tool wire complete"
    next_task = await ai_task.async_generate_data(
        hass,
        task_name="After Caller Tool",
        entity_id=entity_id,
        instructions="NEXT_INDEPENDENT_MARKER",
    )
    assert next_task.data == "Next independent"
    assert len(requests) == 3
    endpoint = "/v1/responses" if mode == API_MODE_RESPONSES else "/v1/chat/completions"
    assert [request["path"] for request in requests] == [endpoint] * 3
    assert len(probe.calls) == 1
    tool_input, llm_context = probe.calls[0]
    assert tool_input.tool_name == "context_probe"
    assert tool_input.tool_args == {"value": "from-wire"}
    assert llm_context.context is context
    second = requests[1]["body"]
    if mode == API_MODE_RESPONSES:
        function_call = next(
            item for item in second["input"] if item.get("type") == "function_call"
        )
        assert function_call["call_id"] == "call-ai-task-wire"
        output = next(
            item
            for item in second["input"]
            if item.get("type") == "function_call_output"
        )
        assert output["call_id"] == "call-ai-task-wire"
        serialized = output["output"]
    else:
        assistant = next(item for item in second["messages"] if item.get("tool_calls"))
        assert assistant["tool_calls"][0]["id"] == "call-ai-task-wire"
        output = next(item for item in second["messages"] if item.get("role") == "tool")
        assert output["tool_call_id"] == "call-ai-task-wire"
        serialized = output["content"]
    assert json.loads(serialized) == {"result": {"echo": "from-wire"}}
    last_body = requests[2]["body"]
    assert "tools" not in last_body
    assert "NEXT_INDEPENDENT_MARKER" in _input_text(last_body)
    assert "from-wire" not in json.dumps(last_body)
    assert "call-ai-task-wire" not in json.dumps(last_body)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "mime_type,filename", [("image/png", "probe.png"), ("application/pdf", "probe.pdf")]
)
async def test_local_attachment_bytes_cross_sdk_wire(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    mime_type: str,
    filename: str,
) -> None:
    source = tmp_path / filename
    content = b"EOAI local attachment bytes \x00\xff"
    source.write_bytes(content)
    hass.config.media_dirs["local"] = str(tmp_path)
    entry, entity_id = await _task_entity(hass, mode)
    wire = _wire(monkeypatch, entry, [_text_reply(mode, "Attachment inspected")])
    attachment = {
        "media_content_id": f"media-source://media_source/local/{filename}",
        "media_content_type": mime_type,
    }
    if mode == API_MODE_CHAT_COMPLETIONS and mime_type == "application/pdf":
        with pytest.raises(HomeAssistantError, match="supports image attachments"):
            await ai_task.async_generate_data(
                hass,
                task_name="Unsupported PDF",
                entity_id=entity_id,
                instructions="Inspect file",
                attachments=[attachment],
            )
        assert wire.requests == []
        return
    result = await ai_task.async_generate_data(
        hass,
        task_name="Attachment Wire Task",
        entity_id=entity_id,
        instructions="Inspect file",
        attachments=[attachment],
    )
    assert result.data == "Attachment inspected"
    _assert_paths(wire, mode, 1)
    user = next(
        item
        for item in wire.requests[0]["body"].get(
            "input", wire.requests[0]["body"].get("messages")
        )
        if item.get("role") == "user"
    )
    part = next(
        item
        for item in user["content"]
        if item["type"] != ("input_text" if mode == API_MODE_RESPONSES else "text")
    )
    if mime_type == "application/pdf":
        assert part["type"] == "input_file"
        assert part["filename"] == filename
        data_url = part["file_data"]
    elif mode == API_MODE_RESPONSES:
        assert part["type"] == "input_image"
        data_url = part["image_url"]
    else:
        assert part["type"] == "image_url"
        data_url = part["image_url"]["url"]
    assert data_url.startswith(f"data:{mime_type};base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == content


@pytest.mark.parametrize("mode", MODES)
async def test_unavailable_local_attachment_never_reaches_provider(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    hass.config.media_dirs["local"] = str(tmp_path)
    entry, entity_id = await _task_entity(hass, mode)
    wire = _wire(monkeypatch, entry, [])
    with pytest.raises(HomeAssistantError, match="does not exist"):
        await ai_task.async_generate_data(
            hass,
            task_name="Missing Local Attachment",
            entity_id=entity_id,
            instructions="Inspect absent image",
            attachments=[
                {
                    "media_content_id": "media-source://media_source/local/absent.png",
                    "media_content_type": "image/png",
                }
            ],
        )
    assert wire.requests == []


@pytest.mark.parametrize("mode", MODES)
async def test_failed_ai_task_does_not_leak_request_state_to_next_task(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    source = tmp_path / "private.png"
    source.write_bytes(b"PRIVATE_ATTACHMENT_MARKER")
    hass.config.media_dirs["local"] = str(tmp_path)
    entry, entity_id = await _task_entity(hass, mode)
    wire = _wire(
        monkeypatch,
        entry,
        [
            _text_reply(mode, "invalid json"),
            (
                400,
                {
                    "error": {
                        "message": "provider rejected request",
                        "type": "invalid_request_error",
                    }
                },
            ),
            _text_reply(mode, "Independent task complete"),
        ],
    )
    attachment = {
        "media_content_id": "media-source://media_source/local/private.png",
        "media_content_type": "image/png",
    }
    with pytest.raises(HomeAssistantError, match="structured response"):
        await ai_task.async_generate_data(
            hass,
            task_name="Private Structured Task",
            entity_id=entity_id,
            instructions="PRIVATE_INSTRUCTIONS_MARKER",
            structure=vol.Schema({vol.Required("answer"): str}),
            attachments=[attachment],
        )
    with pytest.raises(BadRequestError, match="provider rejected request"):
        await ai_task.async_generate_data(
            hass,
            task_name="Provider Failure Task",
            entity_id=entity_id,
            instructions="FAILED_TASK_MARKER",
        )
    recovered = await ai_task.async_generate_data(
        hass,
        task_name="Independent Task",
        entity_id=entity_id,
        instructions="INDEPENDENT_TASK_MARKER",
    )
    assert recovered.data == "Independent task complete"
    _assert_paths(wire, mode, 3)
    final_body = wire.requests[2]["body"]
    final_text = json.dumps(final_body, sort_keys=True)
    encoded_private = base64.b64encode(source.read_bytes()).decode()
    assert encoded_private in json.dumps(wire.requests[0]["body"])
    assert "INDEPENDENT_TASK_MARKER" in final_text
    for absent in (
        "PRIVATE_INSTRUCTIONS_MARKER",
        encoded_private,
        "FAILED_TASK_MARKER",
        "function_call_output",
    ):
        assert absent not in final_text
    assert "response_format" not in final_body
    assert (
        "text" not in final_body
        or final_body["text"].get("format", {}).get("type") != "json_schema"
    )


@pytest.mark.parametrize("mode", MODES)
async def test_truncated_provider_stream_cannot_complete_ai_task_or_poison_next(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    entry, entity_id = await _task_entity(hass, mode)
    wire = _wire(
        monkeypatch, entry, [_truncated_reply(mode), _text_reply(mode, "Recovered")]
    )
    with pytest.raises(HomeAssistantError):
        await ai_task.async_generate_data(
            hass,
            task_name="Truncated Provider Task",
            entity_id=entity_id,
            instructions="TRUNCATED_TASK_MARKER",
        )
    recovered = await ai_task.async_generate_data(
        hass,
        task_name="Recovered Provider Task",
        entity_id=entity_id,
        instructions="RECOVERED_TASK_MARKER",
    )
    assert recovered.data == "Recovered"
    _assert_paths(wire, mode, 2)
    second = _input_text(wire.requests[1]["body"])
    assert "RECOVERED_TASK_MARKER" in second
    assert "TRUNCATED_TASK_MARKER" not in second
