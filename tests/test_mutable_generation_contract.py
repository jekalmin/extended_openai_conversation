"""Reviewed mutable-state boundaries and the pre-provider consistency fence."""

from __future__ import annotations

import asyncio
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from homeassistant.exceptions import HomeAssistantError

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ci" / "mutable_state_contract.json"


def test_every_registered_mutable_dependency_has_reviewed_contract_and_evidence() -> (
    None
):
    """The explicit critical-path inventory requires semantics and executable evidence."""
    records = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert len(records) == len({record["resource"] for record in records})
    assert {record["resource"] for record in records} >= {
        "agent_config",
        "effective_model",
        "function_tool",
        "function_group",
        "request_rules",
        "guest_policy",
        "ha_target",
        "ha_service",
        "provider_client",
        "model_catalogue",
    }
    for record in records:
        assert record["semantics"] in {
            "snapshot",
            "reresolve",
            "reject_on_change",
            "not_async_sensitive",
        }
        assert record["boundary"]
        assert record["reason"]
        assert record["evidence"]
        for evidence in record["evidence"]:
            path, test_name = evidence.split("::", 1)
            source = (ROOT / path).read_text(encoding="utf-8")
            assert f"def {test_name}(" in source


@pytest.mark.parametrize("restore_original", [False, True], ids=["A-B", "A-B-A"])
async def test_ha_tool_discovery_rejects_configuration_generation_change(
    monkeypatch: pytest.MonkeyPatch, restore_original: bool
) -> None:
    """A discovery await cannot combine tools from A with later request settings."""
    module = import_module(
        "custom_components.extended_openai_conversation_responses.conversation"
    )
    original = {
        "function_tools": [
            {"spec": {"name": "HA tool"}, "function": {"type": "ha_llm"}}
        ],
        "function_groups": [],
    }
    subentry = SimpleNamespace(data=original)
    entered, release = asyncio.Event(), asyncio.Event()
    generated = []

    async def discover(*_args):
        entered.set()
        await release.wait()
        return module.ToolSnapshot()

    async def generate(*_args):
        generated.append(True)
        return None

    monkeypatch.setattr(module, "async_discover", discover)
    agent = SimpleNamespace(
        hass=object(),
        subentry=subentry,
        _configured_function_tools_from_data=lambda data: data["function_tools"],
        _effective_guest_policy=lambda: SimpleNamespace(guest_active=False),
        _async_handle_message=generate,
    )
    user_input = SimpleNamespace(as_llm_context=lambda _domain: object())
    task = asyncio.create_task(
        module.ExtendedOpenAIAgentEntity._async_handle_message_with_ha_tools(
            agent, user_input, object()
        )
    )
    await asyncio.wait_for(entered.wait(), 10)
    subentry.data = {**original, "chat_model": "changed"}
    if restore_original:
        subentry.data = dict(original)
    release.set()
    with pytest.raises(HomeAssistantError, match=r"configuration changed.*retry"):
        await task
    assert generated == []
