"""Real-HA recovery when shutdown interrupts an in-flight provider request."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tests_real_ha.process_harness import run_python_child

DOMAIN = "extended_openai_conversation_responses"
_CHILD_PHASE = "ACTIVE_REQUEST_SHUTDOWN_PHASE"
_CONFIG_DIR = "ACTIVE_REQUEST_SHUTDOWN_CONFIG_DIR"
_STATE_FILE = "active-request-shutdown-state.json"
_AFTER_TOOL = "ACTIVE_REQUEST_SHUTDOWN_AFTER_TOOL"


def _raw_client(agent: Any) -> Any:
    """Unwrap integration instrumentation while leaving the real OpenAI SDK intact."""
    client = agent._client
    while hasattr(client, "_delegate"):
        client = client._delegate
    return client


class _BlockingWire:
    """Hold the SDK's outbound request until Home Assistant shutdown cancels it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.requests: list[str] = []

    async def send(
        self, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        del args, kwargs
        if os.environ.get(_AFTER_TOOL) == "1" and not self.requests:
            from tests_real_ha.test_provider_wire_e2e import (
                _chat_sse_tool_call,
                _responses_sse_tool_call,
            )

            self.requests.append(request.url.path)
            content = (
                _chat_sse_tool_call()
                if request.url.path == "/v1/chat/completions"
                else _responses_sse_tool_call()
            )
            if request.url.path == "/v1/responses":
                events = [
                    json.loads(part.removeprefix("data: "))
                    for part in content.decode().split("\n\n")
                    if part
                ]
                events[-1]["response"]["usage"] = {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "total_tokens": 13,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens_details": {"reasoning_tokens": 0},
                }
                content = "".join(
                    f"data: {json.dumps(event)}\n\n" for event in events
                ).encode()
            else:
                usage_chunk = {
                    "id": "usage",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "gpt-5.6",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }
                content = content.replace(
                    b"data: [DONE]",
                    f"data: {json.dumps(usage_chunk)}\n\ndata: [DONE]".encode(),
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=content,
                request=request,
            )
        self.requests.append(request.url.path)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("blocking provider request unexpectedly resumed")


def _chat_sse_text(text: str) -> bytes:
    chunk = {
        "id": "chatcmpl-active-request-recovery",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5.6",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def _response_object(text: str) -> dict[str, Any]:
    item = {
        "id": "msg-active-request-recovery",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
    return {
        "id": "resp-active-request-recovery",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 500,
        "model": "gpt-5.6",
        "output": [item],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": None,
    }


def _responses_sse_text(text: str) -> bytes:
    response = _response_object(text)
    item = response["output"][0]
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []},
            "sequence_number": 0,
        },
        {
            "type": "response.output_text.delta",
            "content_index": 0,
            "delta": text,
            "item_id": item["id"],
            "logprobs": [],
            "output_index": 0,
            "sequence_number": 1,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
            "sequence_number": 2,
        },
        {
            "type": "response.completed",
            "response": response,
            "sequence_number": 3,
        },
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


class _RecoveryWire:
    """Return one real-SDK-compatible response after the restart."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[str] = []

    async def send(
        self, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        del args, kwargs
        self.requests.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            content = _chat_sse_text(self.text)
        elif request.url.path == "/v1/responses":
            content = _responses_sse_text(self.text)
        else:
            raise AssertionError(f"Unexpected provider path: {request.url.path}")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=content,
            request=request,
        )


async def _create_entry(hass: Any) -> Any:
    """Create a persisted entry through the integration's real config flow."""
    from homeassistant.config_entries import SOURCE_USER, ConfigEntryState
    from homeassistant.const import CONF_API_KEY, CONF_NAME
    from homeassistant.data_entry_flow import FlowResultType

    config_flow = importlib.import_module(f"custom_components.{DOMAIN}.config_flow")
    const = importlib.import_module(f"custom_components.{DOMAIN}.const")
    authenticate = AsyncMock(return_value=object())
    with patch.object(config_flow, "get_authenticated_client", authenticate):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": SOURCE_USER}
        )
        assert result["type"] is FlowResultType.FORM
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_NAME: "Active Request Shutdown Acceptance",
                CONF_API_KEY: "sk-active-request-shutdown",
                const.CONF_BASE_URL: const.DEFAULT_CONF_BASE_URL,
                const.CONF_SKIP_AUTHENTICATION: True,
                const.CONF_API_PROVIDER: "openai",
            },
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    entry = result["result"]
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    return entry


async def _interrupt_phase(hass: Any, config_dir: Path) -> None:
    """Stop HA while a real SDK request is blocked in its HTTP send call."""
    from homeassistant.components import conversation
    from homeassistant.core import Context

    entry = await _create_entry(hass)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None

    markers = []
    if os.environ.get(_AFTER_TOOL) == "1":
        from homeassistant.components.homeassistant.exposed_entities import (
            async_expose_entity,
        )

        hass.states.async_set("light.provider_wire", "on")
        async_expose_entity(hass, conversation.DOMAIN, "light.provider_wire", True)

        async def mark(call):
            markers.append(call.data)

        hass.services.async_register("light", "turn_off", mark)

    wire = _BlockingWire()
    raw_client = _raw_client(agent)
    with patch.object(raw_client._client, "send", wire.send):
        request_task = hass.async_create_task(
            conversation.async_converse(
                hass=hass,
                text="This request should still be in flight when Home Assistant stops",
                conversation_id=None,
                context=Context(),
                language="en",
                agent_id=entry.entry_id,
            )
        )
        async with asyncio.timeout(15):
            await wire.started.wait()

        assert wire.requests, "provider request never reached the real SDK HTTP seam"
        assert len(markers) == (1 if os.environ.get(_AFTER_TOOL) == "1" else 0)
        (config_dir / _STATE_FILE).write_text(
            json.dumps({"entry_id": entry.entry_id}), encoding="utf-8"
        )

        # This is the contract under test: shutdown must cancel the request that is
        # blocked below the OpenAI SDK instead of hanging indefinitely on it.
        async with asyncio.timeout(20):
            await hass.async_stop()

        assert wire.cancelled.is_set(), "provider send did not receive cancellation"
        assert request_task.done(), "conversation task survived Home Assistant shutdown"
        assert request_task.cancelled(), (
            "interrupted conversation did not end by cancellation"
        )
        assert agent._usage.totals.conversation_count == 1
        assert agent._usage.runs[-1].successful is False


async def _recovery_phase(hass: Any, config_dir: Path) -> None:
    """Cold-start the same config and prove a later conversation works normally."""
    from homeassistant.components import conversation
    from homeassistant.config_entries import ConfigEntryState
    from homeassistant.core import Context

    state = json.loads((config_dir / _STATE_FILE).read_text(encoding="utf-8"))
    entries = hass.config_entries.async_entries(DOMAIN)
    assert len(entries) == 1
    entry = entries[0]
    await hass.async_block_till_done()
    assert entry.entry_id == state["entry_id"]
    assert entry.state is ConfigEntryState.LOADED

    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    usage = agent._usage
    assert usage.totals.conversation_count == 1
    assert len(usage.runs) == 1
    interrupted = usage.runs[0]
    assert not interrupted.successful
    assert interrupted.completed_at
    assert interrupted.error_type == "CancelledError"
    assert interrupted.tool_call_count == (
        1 if os.environ.get(_AFTER_TOOL) == "1" else 0
    )
    assert interrupted.total_tokens == (13 if os.environ.get(_AFTER_TOOL) == "1" else 0)
    assert usage.totals.total_tokens == interrupted.total_tokens
    assert interrupted.failed_request_count == 1
    assert interrupted.request_count == (2 if os.environ.get(_AFTER_TOOL) == "1" else 1)
    wire = _RecoveryWire("Recovered cleanly after the interrupted request.")
    raw_client = _raw_client(agent)
    with patch.object(raw_client._client, "send", wire.send):
        result = await conversation.async_converse(
            hass=hass,
            text="Confirm the restarted conversation agent is healthy",
            conversation_id=None,
            context=Context(),
            language="en",
            agent_id=entry.entry_id,
        )

    assert result.response.error_code is None
    assert (
        result.response.as_dict()["speech"]["plain"]["speech"]
        == "Recovered cleanly after the interrupted request."
    )
    assert len(wire.requests) == 1


async def _child_main() -> None:
    """Run one independent Home Assistant process for one acceptance phase."""
    from homeassistant import bootstrap, runner
    from homeassistant.helpers import recorder as recorder_helper

    config_dir = Path(os.environ[_CONFIG_DIR]).resolve()
    phase = os.environ[_CHILD_PHASE]
    sys.path.insert(0, str(config_dir))
    hass = await bootstrap.async_setup_hass(
        runner.RuntimeConfig(config_dir=str(config_dir), skip_pip=True)
    )
    assert hass is not None

    # bootstrap.async_setup_hass() is being used directly rather than via HA's
    # normal runner. Mirror the runner's recorder initialization contract before
    # the config flow loads this integration and its recorder/history/energy
    # dependencies.
    if recorder_helper.DATA_RECORDER not in hass.data:
        recorder_helper.async_initialize_recorder(hass)

    await hass.async_start()

    if phase == "interrupt":
        # _interrupt_phase performs the actual Home Assistant shutdown itself so
        # cancellation happens while the provider request is demonstrably in flight.
        await _interrupt_phase(hass, config_dir)
        return

    try:
        if phase == "recover":
            await _recovery_phase(hass, config_dir)
        else:
            raise AssertionError(f"Unknown active-request shutdown phase: {phase}")
        await hass.async_block_till_done()
    finally:
        await hass.async_stop()


def _run_child(
    config_dir: Path, phase: str, after_tool: bool = False
) -> subprocess.CompletedProcess[str]:
    return run_python_child(
        __file__,
        cwd=config_dir,
        extra_env={
            _CHILD_PHASE: phase,
            _CONFIG_DIR: str(config_dir),
            _AFTER_TOOL: "1" if after_tool else "0",
        },
        timeout=90,
    )


def _assert_child_ok(result: subprocess.CompletedProcess[str], phase: str) -> None:
    assert result.returncode == 0, (
        f"active-request shutdown phase {phase!r} failed\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


@pytest.mark.parametrize("after_tool", [False, True])
@pytest.mark.usefixtures("socket_enabled")
def test_shutdown_cancels_inflight_provider_request_and_next_boot_is_healthy(
    tmp_path: Path,
    after_tool: bool,
    unused_tcp_port: int,
) -> None:
    """Interrupt a live request during HA shutdown, then cold-start and converse."""
    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "custom_components" / DOMAIN
    config_dir = tmp_path / "ha-config"
    destination = config_dir / "custom_components" / DOMAIN
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    (config_dir / "configuration.yaml").write_text(
        "homeassistant:\n  name: Active Request Shutdown Acceptance\n"
        f"http:\n  server_port: {unused_tcp_port}\n",
        encoding="utf-8",
    )

    interrupted = _run_child(config_dir, "interrupt", after_tool)
    _assert_child_ok(interrupted, "interrupt")
    assert (config_dir / ".storage" / "core.config_entries").exists()
    assert (config_dir / _STATE_FILE).exists()

    recovered = _run_child(config_dir, "recover", after_tool)
    _assert_child_ok(recovered, "recover")


if __name__ == "__main__" and os.environ.get(_CHILD_PHASE):
    asyncio.run(_child_main())
