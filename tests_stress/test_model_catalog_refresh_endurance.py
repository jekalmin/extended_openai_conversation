"""Nightly model-catalog refresh races across HA WebSocket and public Assist."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockUser

from custom_components.extended_openai_conversation_responses import (
    model_catalog_manager as runtime,
)
from custom_components.extended_openai_conversation_responses.helpers import (
    get_reasoning_effort_options,
)
from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_backup_transfer_protocol import _user_token
from tests_real_ha.test_provider_wire_e2e import _chat_sse_text, _install_wire, _speech
from tests_stress.conftest import record


async def _say(hass: HomeAssistant, entry_id: str, index: int) -> Any:
    return await conversation.async_converse(
        hass=hass,
        text=f"Use the current model catalogue {index}",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry_id,
    )


async def _command(client: Any, action: str) -> dict[str, Any]:
    await client.send_json_auto_id(
        {"type": runtime.WS_CATALOG, "action": action, "model": "gpt-5.6"}
    )
    return await client.receive_json()


class _CatalogResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.status = status
        self.headers = {"ETag": f'"nightly-catalog-{len(payload)}"'}
        self.content = SimpleNamespace(iter_chunked=self.iter_chunked)
        self.payload = payload
        self.entered = entered
        self.release = release

    async def __aenter__(self):
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return self

    async def __aexit__(self, *args):
        return None

    async def iter_chunked(self, _size: int):
        midpoint = len(self.payload) // 2
        yield self.payload[:midpoint]
        yield self.payload[midpoint:]


class _CatalogSession:
    def __init__(self, responses: list[_CatalogResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, *args: Any, **kwargs: Any) -> _CatalogResponse:
        del args
        self.calls.append(kwargs)
        return next(self.responses)


@pytest.mark.asyncio
async def test_slow_invalid_and_failed_refresh_preserve_active_operations(
    hass: HomeAssistant,
    hass_ws_client: Any,
    monkeypatch: pytest.MonkeyPatch,
    stress_scale: int,
    stress_trace: list[dict],
) -> None:
    """Refresh publishes complete metadata and retains last good data on failure."""
    entry = _make_entry("Catalog refresh endurance", include_ai_task=False)
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    manager: runtime.ModelCatalogManager = hass.data[runtime.DATA_MANAGER]
    initial_efforts = get_reasoning_effort_options("gpt-5.6")
    assert "minimal" not in initial_efforts
    admin = await hass_ws_client(
        hass,
        await _user_token(hass, MockUser(id="catalog-endurance-admin", is_owner=True)),
    )
    cycles = 1 if stress_scale == 1 else 3
    wire = _install_wire(
        monkeypatch,
        agent,
        [_chat_sse_text("Catalogue request completed.") for _ in range(3 * cycles)],
    )
    try:
        for cycle in range(cycles):
            current = manager.catalog or BUNDLED_CATALOG
            candidate = deepcopy(current)
            candidate["catalog_version"] += 1
            model = next(
                item for item in candidate["models"] if item["id"] == "gpt-5.6"
            )
            if cycle == 0:
                model["reasoning"]["efforts"].append("minimal")
                model["reasoning"]["by_api"]["responses"]["efforts"].append("minimal")
                model["reasoning"]["by_api"]["chat_completions"]["efforts"].append(
                    "minimal"
                )
            else:
                model["display_name"] = f"Catalog endurance {cycle}"
            entered = asyncio.Event()
            release = asyncio.Event()
            session = _CatalogSession(
                [
                    _CatalogResponse(
                        json.dumps(candidate).encode(), entered=entered, release=release
                    ),
                    _CatalogResponse(
                        json.dumps({**candidate, "schema_version": 99}).encode()
                    ),
                    _CatalogResponse(json.dumps(candidate).encode()),
                    _CatalogResponse(b'{"invalid": true}'),
                    _CatalogResponse(b"", status=503),
                ]
            )
            monkeypatch.setattr(
                runtime,
                "async_get_clientsession",
                lambda _, current=session: current,
            )
            before = get_reasoning_effort_options("gpt-5.6")
            checking = asyncio.create_task(_command(admin, "check"))
            await asyncio.wait_for(entered.wait(), timeout=10)
            assert get_reasoning_effort_options("gpt-5.6") == before
            during = await _say(hass, entry.entry_id, 3 * cycle)
            assert _speech(during) == "Catalogue request completed."
            release.set()
            checked = await asyncio.wait_for(checking, timeout=10)
            assert checked["success"] is True, (cycle, checked, manager.status())
            assert get_reasoning_effort_options("gpt-5.6") == before

            applied = await _command(admin, "apply")
            assert applied["success"] is True
            assert manager.catalog == candidate
            assert conversation.async_get_agent(hass, entry.entry_id) is agent
            activated_efforts = get_reasoning_effort_options("gpt-5.6")
            after = await _say(hass, entry.entry_id, 3 * cycle + 1)
            assert _speech(after) == "Catalogue request completed."

            incompatible = await _command(admin, "check")
            assert incompatible["success"] is True
            assert (
                incompatible["result"]["incompatible_catalog"]["schema_version"] == 99
            )
            assert manager.catalog == candidate
            assert manager.etag is None
            assert get_reasoning_effort_options("gpt-5.6") == activated_efforts
            compatible = await _command(admin, "check")
            assert compatible["success"] is True
            assert compatible["result"]["incompatible_catalog"] is None
            assert session.calls[2]["headers"] == {}

            for outcome in ("invalid", "unavailable"):
                failed = await _command(admin, "check")
                assert failed["success"] is False, outcome
                assert manager.catalog == candidate
                assert manager.available_catalog is None
                assert get_reasoning_effort_options("gpt-5.6") == activated_efforts
            recovered = await _say(hass, entry.entry_id, 3 * cycle + 2)
            assert _speech(recovered) == "Catalogue request completed."
        assert len(wire.requests) == 3 * cycles
        assert "minimal" in get_reasoning_effort_options("gpt-5.6")
        record(
            stress_trace,
            "summary",
            layer="Real HA WebSocket and provider wire",
            catalog_refresh_cycles=cycles,
            catalog_failed_refreshes=2 * cycles,
            catalog_incompatible_refreshes=cycles,
            public_turns=3 * cycles,
        )
    finally:
        await manager.async_reset()
