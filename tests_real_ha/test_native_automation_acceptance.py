"""A native automation tool must survive genuine HA validation and reload."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pytest_homeassistant_custom_component.common import MockUser
import voluptuous as vol
import yaml

from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_TOOLS,
)
from custom_components.extended_openai_conversation_responses.ha_tool_result_compat import (
    tool_result_data,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry

_TOOL = {
    "spec": {
        "name": "acceptance_add_automation",
        "description": "Create one real Home Assistant automation.",
        "parameters": {
            "type": "object",
            "properties": {"automation_config": {"type": "string"}},
            "required": ["automation_config"],
        },
    },
    "function": {"type": "native", "name": "add_automation"},
    "enabled": True,
}


def _automation(alias: str, event: str, marker: str) -> dict:
    return {
        "alias": alias,
        "triggers": [{"trigger": "event", "event_type": event}],
        "actions": [{"action": "acceptance_probe.record", "data": {"marker": marker}}],
    }


async def _call_tool(hass, agent, user, automation_config: str):
    from custom_components.extended_openai_conversation_responses.agent_config import (
        configured_function_tools_from_data,
    )

    tool = next(
        item
        for item in configured_function_tools_from_data(agent.subentry.data)
        if item["spec"]["name"] == _TOOL["spec"]["name"]
    )
    return tool_result_data(
        await agent._execute_function_tool(
            tool,
            llm.ToolInput(
                id="automation-acceptance-call",
                tool_name=_TOOL["spec"]["name"],
                tool_args={"automation_config": automation_config},
                external=True,
            ),
            SimpleNamespace(context=Context(user_id=user.id), device_id=None),
            [],
        )
    )


@pytest.mark.asyncio
async def test_native_add_automation_validates_reloads_and_triggers_once(
    hass: HomeAssistant,
) -> None:
    config_dir = hass.config.config_dir
    from pathlib import Path

    automation_file = Path(config_dir) / "automations.yaml"
    configuration_file = Path(config_dir) / "configuration.yaml"
    original = _automation(
        "Existing acceptance", "existing_acceptance_event", "existing"
    )
    automation_file.write_text(yaml.safe_dump([original]), encoding="utf-8")
    configuration_file.write_text(
        "automation: !include automations.yaml\n", encoding="utf-8"
    )

    calls: list[str] = []

    async def record(call) -> None:
        calls.append(call.data["marker"])

    hass.services.async_register("acceptance_probe", "record", record)
    assert await async_setup_component(hass, "automation", {"automation": [original]})
    await hass.async_block_till_done()

    entry = _make_entry(
        "Automation Tool Acceptance",
        include_ai_task=False,
        conversation_options={CONF_FUNCTION_TOOLS: yaml.safe_dump([_TOOL])},
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    user = MockUser(id="automation-acceptance-admin", is_owner=True)
    user.add_to_hass(hass)

    hass.bus.async_fire("existing_acceptance_event")
    await hass.async_block_till_done()
    assert calls == ["existing"]

    added = _automation("New acceptance", "new_acceptance_event", "new")
    result = await _call_tool(hass, agent, user, yaml.safe_dump(added))
    assert result == {"result": "Success"}
    saved = yaml.safe_load(automation_file.read_text(encoding="utf-8"))
    assert [item["alias"] for item in saved] == [
        "Existing acceptance",
        "New acceptance",
    ]
    hass.bus.async_fire("new_acceptance_event")
    hass.bus.async_fire("existing_acceptance_event")
    await hass.async_block_till_done()
    assert calls == ["existing", "new", "existing"]

    await hass.services.async_call("automation", "reload", blocking=True)
    hass.bus.async_fire("new_acceptance_event")
    await hass.async_block_till_done()
    assert calls == ["existing", "new", "existing", "new"]

    before_invalid = automation_file.read_bytes()
    invalid = {
        "alias": "Invalid acceptance",
        "triggers": [{"trigger": "event"}],
        "actions": added["actions"],
    }
    with pytest.raises(vol.Invalid):
        await _call_tool(hass, agent, user, yaml.safe_dump(invalid))
    assert automation_file.read_bytes() == before_invalid
    hass.bus.async_fire("new_acceptance_event")
    hass.bus.async_fire("existing_acceptance_event")
    await hass.async_block_till_done()
    assert calls == ["existing", "new", "existing", "new", "new", "existing"]
