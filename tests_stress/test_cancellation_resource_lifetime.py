"""Cancellation and slow-success boundaries through genuine HA conversations."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
import gc
from typing import Any

import httpx
import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
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
    _chat_sse_text,
    _chat_sse_tool_call,
    _prepare_service,
    _raw_client,
    _say,
    _speech,
)
from tests_stress.conftest import record
from tests_stress.test_runtime_soak import _resource_footprint


async def _agent(hass: HomeAssistant, title: str) -> Any:
    entry = _make_entry(
        title,
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [deepcopy(DEFAULT_CONF_FUNCTION_TOOLS[0])],
            CONF_ARCHIVE_ENABLED: True,
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    return agent


async def _say_text(hass: HomeAssistant, agent: Any, text: str) -> Any:
    return await conversation.async_converse(
        hass=hass,
        text=text,
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=agent.entry.entry_id,
    )


@pytest.mark.parametrize(
    "phase",
    [
        "before_validation",
        "after_resolution",
        "provider",
        "pre_dispatch",
        "service",
        "archive",
        "post_archive",
    ],
)
async def test_cancelled_boundary_releases_request_and_fresh_turn_succeeds(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    phase: str,
) -> None:
    """A gated cancellation cannot replay a side effect or poison the next turn."""
    agent = await _agent(hass, "Cancellation boundary")
    service_calls = await _prepare_service(hass)
    entered = asyncio.Event()
    release = asyncio.Event()
    requests = 0
    cancelled = False

    async def gated() -> None:
        entered.set()
        await release.wait()

    if phase == "before_validation":
        import custom_components.extended_openai_conversation_responses.conversation as conversation_module

        original_reconcile = conversation_module.async_reconcile_runtime_configuration

        async def reconcile(*args: Any, **kwargs: Any) -> Any:
            await gated()
            return await original_reconcile(*args, **kwargs)

        monkeypatch.setattr(
            conversation_module, "async_reconcile_runtime_configuration", reconcile
        )
    elif phase == "after_resolution":
        original_begin = agent._async_begin_archive_session

        async def begin_archive(*args: Any, **kwargs: Any) -> Any:
            await gated()
            return await original_begin(*args, **kwargs)

        monkeypatch.setattr(agent, "_async_begin_archive_session", begin_archive)
    elif phase == "pre_dispatch":
        original_dispatch = agent._async_dispatch_function_tool

        async def dispatch(*args: Any, **kwargs: Any) -> Any:
            await gated()
            return await original_dispatch(*args, **kwargs)

        monkeypatch.setattr(agent, "_async_dispatch_function_tool", dispatch)
    elif phase == "service":

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
            if phase in {"service", "pre_dispatch"} and requests == 1
            else _chat_sse_text("Fresh turn succeeds." if cancelled else "First turn.")
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
        cancelled = True
    finally:
        release.set()
        if not active.done():
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active

    first_requests = 0 if phase in {"before_validation", "after_resolution"} else 1
    assert len(service_calls) == (1 if phase == "service" else 0)
    assert requests == first_requests
    recovered = await _say(hass, agent)
    assert _speech(recovered) == "Fresh turn succeeds."
    assert requests == first_requests + 1
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
    slow = await _agent(hass, "Slow dependency")
    fast = await _agent(hass, "Independent dependency")
    assert slow is not fast
    # HA intentionally shares its HTTP transport across integration entries.
    assert _raw_client(slow)._client is _raw_client(fast)._client
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
        archive_calls = 0

        async def archive(*args: Any, **kwargs: Any) -> Any:
            nonlocal archive_calls
            archive_calls += 1
            if archive_calls == 1:
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

    async def send(request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
        if "Slow request" in request.content.decode():
            return await slow_send(request, *args, **kwargs)
        assert any(
            marker in request.content.decode()
            for marker in ("Independent request", "Sibling request")
        )
        return await fast_send(request, *args, **kwargs)

    monkeypatch.setattr(_raw_client(slow)._client, "send", send)
    blocked = asyncio.create_task(_say_text(hass, slow, "Slow request"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        independent = await asyncio.wait_for(
            _say_text(hass, fast, "Independent request"), timeout=10
        )
        assert _speech(independent) == "Independent success."
        sibling = await asyncio.wait_for(
            _say_text(hass, slow, "Sibling request"), timeout=10
        )
        assert _speech(sibling) == "Independent success."
        assert not blocked.done()
    finally:
        release.set()

    assert _speech(await asyncio.wait_for(blocked, timeout=10)) == "Slow success."
    assert slow_requests == (2 if dependency == "service" else 1)
    assert len(service_calls) == (1 if dependency == "service" else 0)
    record(
        stress_trace, "slow_success", dependency=dependency, agents=2, conversations=3
    )
