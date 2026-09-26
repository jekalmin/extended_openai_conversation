"""Cancellation and slow-success boundaries through genuine HA conversations."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import gc
from typing import Any

import httpx
import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
)
from homeassistant.core import HomeAssistant
from tests_real_ha.test_provider_wire_e2e import (
    _agent,
    _chat_sse_text,
    _chat_sse_tool_call,
    _prepare_service,
    _raw_client,
    _say,
    _speech,
)
from tests_stress.conftest import record
from tests_stress.test_runtime_soak import _resource_footprint


@pytest.mark.parametrize("phase", ["provider", "service", "archive", "post_archive"])
async def test_cancelled_boundary_releases_request_and_fresh_turn_succeeds(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    phase: str,
) -> None:
    """A gated cancellation cannot replay a side effect or poison the next turn."""
    agent = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    service_calls = await _prepare_service(hass)
    entered = asyncio.Event()
    release = asyncio.Event()
    requests = 0

    async def gated() -> None:
        entered.set()
        await release.wait()

    if phase == "service":

        async def service(call: Any) -> None:
            service_calls.append(call)
            await gated()

        hass.services.async_register("light", "turn_off", service)
    elif phase == "archive":
        assert agent._archive is not None
        original_archive = agent._archive.async_record_turn

        async def archive(*args: Any, **kwargs: Any) -> Any:
            await gated()
            return await original_archive(*args, **kwargs)

        monkeypatch.setattr(agent._archive, "async_record_turn", archive)
    elif phase == "post_archive":
        assert agent._continuity is not None
        original_record = agent._continuity.async_record_success

        async def record_success(*args: Any, **kwargs: Any) -> Any:
            await gated()
            return await original_record(*args, **kwargs)

        monkeypatch.setattr(agent._continuity, "async_record_success", record_success)

    async def send(request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal requests
        del args, kwargs
        requests += 1
        if phase == "provider" and requests == 1:
            await gated()
        body = (
            _chat_sse_tool_call()
            if phase == "service" and requests == 1
            else _chat_sse_text(
                "Fresh turn succeeds." if requests > 1 else "First turn."
            )
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )

    monkeypatch.setattr(_raw_client(agent)._client, "send", send)
    before = _resource_footprint(hass)
    active = asyncio.create_task(_say(hass, agent))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, timeout=10)
    finally:
        release.set()
        if not active.done():
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active

    assert len(service_calls) == (1 if phase == "service" else 0)
    assert requests == 1
    recovered = await _say(hass, agent)
    assert _speech(recovered) == "Fresh turn succeeds."
    assert requests == 2
    assert len(service_calls) == (1 if phase == "service" else 0)
    await hass.async_block_till_done()
    gc.collect()
    assert _resource_footprint(hass) == before
    record(stress_trace, "cancel_boundary", phase=phase, provider_requests=requests)


@pytest.mark.parametrize("dependency", ["provider", "service", "archive"])
async def test_slow_success_does_not_serialize_other_agent(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    dependency: str,
) -> None:
    """An unrelated agent completes while one dependency remains gated."""
    slow = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    fast = await _agent(hass, API_MODE_CHAT_COMPLETIONS)
    service_calls = await _prepare_service(hass)
    entered = asyncio.Event()
    release = asyncio.Event()
    slow_requests = 0

    async def gated() -> None:
        entered.set()
        await release.wait()

    if dependency == "service":

        async def service(call: Any) -> None:
            service_calls.append(call)
            await gated()

        hass.services.async_register("light", "turn_off", service)
    elif dependency == "archive":
        assert slow._archive is not None
        original = slow._archive.async_record_turn

        async def archive(*args: Any, **kwargs: Any) -> Any:
            await gated()
            return await original(*args, **kwargs)

        monkeypatch.setattr(slow._archive, "async_record_turn", archive)

    async def slow_send(
        request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        nonlocal slow_requests
        del args, kwargs
        slow_requests += 1
        if dependency == "provider" and slow_requests == 1:
            await gated()
        body = (
            _chat_sse_tool_call()
            if dependency == "service" and slow_requests == 1
            else _chat_sse_text("Slow success.")
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )

    async def fast_send(
        request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        del args, kwargs
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_chat_sse_text("Independent success."),
            request=request,
        )

    monkeypatch.setattr(_raw_client(slow)._client, "send", slow_send)
    monkeypatch.setattr(_raw_client(fast)._client, "send", fast_send)
    blocked = asyncio.create_task(_say(hass, slow))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        independent = await asyncio.wait_for(_say(hass, fast), timeout=10)
        assert _speech(independent) == "Independent success."
        assert not blocked.done()
    finally:
        release.set()

    assert _speech(await asyncio.wait_for(blocked, timeout=10)) == "Slow success."
    assert slow_requests == (2 if dependency == "service" else 1)
    assert len(service_calls) == (1 if dependency == "service" else 0)
    record(stress_trace, "slow_success", dependency=dependency, agents=2)
