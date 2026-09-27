"""EOAI stream cancellation and pool ownership over actual loopback sockets."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from typing import Any

from aiohttp import web
import httpx
import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses import helpers
from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
)
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _responses_sse_text,
    _speech,
)
from tests_stress.conftest import record
from tests_stress.test_active_provider_stream_cancellation import _partial_text


async def _say(
    hass: HomeAssistant, entry_id: str, marker: str
) -> conversation.ConversationResult:
    return await conversation.async_converse(
        hass=hass,
        text=marker,
        conversation_id=None,
        context=Context(user_id="real-pool-owner"),
        language="en",
        agent_id=entry_id,
    )


@pytest.mark.parametrize("mode", [API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES])
async def test_real_pool_wait_and_cancellation_release_sdk_stream(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    socket_enabled: Any,
    stress_trace: list[dict],
    mode: str,
) -> None:
    del socket_enabled
    MockUser(id="real-pool-owner", name="Pool owner", is_owner=True).add_to_hass(hass)
    arrived: list[str] = []
    stream_open = asyncio.Event()
    release_server = asyncio.Event()
    first, rest = _partial_text(mode)

    async def provider(request: web.Request) -> web.StreamResponse:
        assert request.path == (
            "/v1/responses" if mode == API_MODE_RESPONSES else "/v1/chat/completions"
        )
        body = await request.json()
        serialized = json.dumps(body)
        marker = next(
            marker
            for marker in ("Held", "CancelledWaiter", "Queued", "Healthy", "PostReload")
            if marker in serialized
        )
        arrived.append(marker)
        if marker != "Held":
            payload = (
                _responses_sse_text(marker)
                if mode == API_MODE_RESPONSES
                else _chat_sse_text(marker)
            )
            return web.Response(body=payload, content_type="text/event-stream")
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(first)
        stream_open.set()
        await release_server.wait()
        with suppress(ConnectionResetError):
            await response.write(rest)
            await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", provider)
    app.router.add_post("/v1/responses", provider)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"
    attempted = {
        marker: asyncio.Event()
        for marker in ("CancelledWaiter", "Queued", "PostReload")
    }

    async def mark_attempt(request: httpx.Request) -> None:
        body = json.dumps(json.loads(request.content))
        for marker, event in attempted.items():
            if marker in body:
                event.set()

    client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
        timeout=httpx.Timeout(10.0, pool=10.0),
        trust_env=False,
        event_hooks={"request": [mark_attempt]},
    )
    monkeypatch.setattr(helpers, "get_async_client", lambda _hass: client)
    entry = _make_entry(
        "Real provider pool",
        include_ai_task=False,
        base_url=url,
        conversation_options={CONF_API_MODE: mode, CONF_CHAT_MODEL: "gpt-5.6"},
    )
    active: asyncio.Task[Any] | None = None
    cancelled_waiter: asyncio.Task[Any] | None = None
    queued: asyncio.Task[Any] | None = None
    try:
        await _setup_entry(hass, entry)
        active = asyncio.create_task(_say(hass, entry.entry_id, "Held"))
        await asyncio.wait_for(stream_open.wait(), 10)
        cancelled_waiter = asyncio.create_task(
            _say(hass, entry.entry_id, "CancelledWaiter")
        )
        await asyncio.wait_for(attempted["CancelledWaiter"].wait(), 10)
        assert arrived == ["Held"]
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancelled_waiter, 10)
        queued = asyncio.create_task(_say(hass, entry.entry_id, "Queued"))
        await asyncio.wait_for(attempted["Queued"].wait(), 10)
        assert arrived == ["Held"]
        assert not queued.done()
        hass.states.async_set("sensor.real_pool_probe", "responsive")
        assert hass.states.get("sensor.real_pool_probe").state == "responsive"
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, 10)
        assert _speech(await asyncio.wait_for(queued, 10)) == "Queued"
        assert (
            _speech(await asyncio.wait_for(_say(hass, entry.entry_id, "Healthy"), 10))
            == "Healthy"
        )
        assert arrived == ["Held", "Queued", "Healthy"]

        # A config-entry reload replaces the agent while the old agent still owns
        # a live stream. Cancelling that stale work frees the shared HTTP pool for
        # the replacement agent without requiring the provider to finish the stream.
        stream_open.clear()
        old_agent = conversation.async_get_agent(hass, entry.entry_id)
        active = asyncio.create_task(_say(hass, entry.entry_id, "Held"))
        await asyncio.wait_for(stream_open.wait(), 10)
        assert await asyncio.wait_for(
            hass.config_entries.async_reload(entry.entry_id), 10
        )
        assert entry.state is ConfigEntryState.LOADED
        assert conversation.async_get_agent(hass, entry.entry_id) is not old_agent
        queued = asyncio.create_task(_say(hass, entry.entry_id, "PostReload"))
        await asyncio.wait_for(attempted["PostReload"].wait(), 10)
        assert arrived == ["Held", "Queued", "Healthy", "Held"]
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, 10)
        assert _speech(await asyncio.wait_for(queued, 10)) == "PostReload"
        assert arrived[-1] == "PostReload"
        record(stress_trace, "real_connection_pool", mode=mode, arrivals=arrived)
    finally:
        release_server.set()
        for task in (active, cancelled_waiter, queued):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        await client.aclose()
        await runner.cleanup()
