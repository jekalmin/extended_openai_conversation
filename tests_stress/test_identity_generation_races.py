"""Enhanced-only request construction races against mutable config identity."""

from __future__ import annotations

import asyncio

import pytest

from custom_components.extended_openai_conversation_responses import (
    conversation as conversation_module,
)
from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    CONF_API_MODE,
    CONF_CHAT_MODEL,
    CONF_FUNCTION_TOOLS,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_ha_llm_tool_acceptance import AcceptanceAPI, AcceptanceEchoTool
from tests_real_ha.test_provider_wire_e2e import _install_wire
from tests_stress.conftest import record


@pytest.mark.parametrize("mutation", ["model_aba", "tool_delete_recreate", "model_tool_aba"])
async def test_catalogue_discovery_rejects_changed_config_generation(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    stress_trace: list[dict],
    mutation: str,
) -> None:
    """An A→B→A edit during genuine HA discovery cannot publish stale work."""
    tool = AcceptanceEchoTool()
    api = AcceptanceAPI(hass, tool)
    llm.async_register_api(hass, api)
    reference = {
        "type": "ha_llm",
        "source_type": "api",
        "source_id": f"{type(tool).__module__}.{type(tool).__qualname__}",
        "api_id": api.id,
        "tool_name": tool.name,
    }
    configured = {
        "spec": {"name": "ha_nightly_echo"},
        "function": reference,
        "enabled": True,
    }
    entry = _make_entry(
        "Nightly discovery generation",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: API_MODE_CHAT_COMPLETIONS,
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_FUNCTION_TOOLS: [configured],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    wire = _install_wire(monkeypatch, agent, [])
    subentry = next(item for item in entry.subentries.values() if item.subentry_type == "conversation")
    original_data = subentry.data
    entered, release = asyncio.Event(), asyncio.Event()
    discover = conversation_module.async_discover

    async def paused_discover(*args, **kwargs):
        entered.set()
        await release.wait()
        return await discover(*args, **kwargs)

    monkeypatch.setattr(conversation_module, "async_discover", paused_discover)
    pending = asyncio.create_task(conversation.async_converse(
        hass=hass,
        text="Find the HA tool",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
    ))
    await asyncio.wait_for(entered.wait(), timeout=10)

    def update(*, model: str, tools: list[dict]) -> None:
        current = next(item for item in entry.subentries.values() if item.subentry_type == "conversation")
        hass.config_entries.async_update_subentry(
            entry,
            current,
            data={**current.data, CONF_CHAT_MODEL: model, CONF_FUNCTION_TOOLS: tools},
        )

    if mutation == "model_aba":
        update(model="gpt-4.1-mini", tools=[configured])
    elif mutation == "tool_delete_recreate":
        update(model="gpt-5.6", tools=[])
    else:
        update(model="gpt-4.1-mini", tools=[])
    update(model="gpt-5.6", tools=[configured])
    current_data = next(item for item in entry.subentries.values() if item.subentry_type == "conversation").data
    assert current_data == original_data and current_data is not original_data
    release.set()
    result = await asyncio.wait_for(pending, timeout=10)
    assert result.response.error_code is not None
    assert not wire.requests
    assert not tool.calls
    await hass.async_block_till_done()
    assert conversation.async_get_agent(hass, entry.entry_id) is not agent
    record(stress_trace, "summary", mutation=mutation, stale_provider_requests=0)
