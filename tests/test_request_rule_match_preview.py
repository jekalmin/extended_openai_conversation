"""Side-effect-free Request Rule management preview tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from custom_components.extended_openai_conversation_responses import management_ui
from custom_components.extended_openai_conversation_responses.request_rule_match_preview import (
    request_rule_match_preview,
)
from custom_components.extended_openai_conversation_responses.request_rule_patterns import (
    SentenceMatchLimitError,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    RuleMatch,
)
from homeassistant.exceptions import HomeAssistantError


def _local_match() -> RuleMatch:
    return RuleMatch(
        rule={
            "id": "good-night",
            "name": "Good night",
            "match_type": "sentence_pattern",
            "action_type": "local_action",
            "action": {"actions": [{"action": "script.turn_on"}, {"delay": 1}]},
        },
        phrase="good night {room}",
        fuzzy=False,
        score=100.0,
        slots={"room": "kitchen"},
    )


def test_preview_summarizes_match_without_action_payloads() -> None:
    result = request_rule_match_preview(_local_match())
    assert result == {
        "matched": True,
        "rule": {
            "id": "good-night",
            "name": "Good night",
            "match_type": "sentence_pattern",
            "action_type": "local_action",
        },
        "matched_phrase": "good night {room}",
        "fuzzy": False,
        "score": 100.0,
        "captured_values": {"room": "kitchen"},
        "would_do": {
            "type": "local_action",
            "action_count": 2,
            "consumed": True,
            "provider_input": "none",
        },
    }
    assert "actions" not in result["would_do"]


def test_preview_lists_function_alias_without_executing_or_inventing_result() -> None:
    match = _local_match()
    match.rule["action"] = {
        "actions": [
            {
                "action": "extended_openai_conversation_responses.call_function",
                "data": {
                    "function": "get_battery",
                    "arguments": {},
                    "result_alias": "battery",
                },
            }
        ],
        "continue_to_ai": True,
    }
    preview = request_rule_match_preview(match)
    assert preview["would_do"] == {
        "type": "local_action",
        "action_count": 1,
        "consumed": False,
        "provider_input": "original",
        "functions": [{"name": "get_battery", "result_alias": "battery"}],
    }
    assert "result" not in preview["would_do"]


def test_preview_summarizes_model_routing_and_no_match() -> None:
    match = RuleMatch(
        rule={
            "id": "think",
            "name": "Think carefully",
            "match_type": "starts_with",
            "action_type": "model_routing",
            "action": {
                "reset": False,
                "model": "gpt-5",
                "reasoning_effort": "high",
                "scope": "conversation",
            },
        },
        phrase="think carefully",
        fuzzy=True,
        score=93.47,
    )
    assert request_rule_match_preview(match)["would_do"] == {
        "type": "model_routing",
        "reset": False,
        "model": "gpt-5",
        "reasoning_effort": "high",
        "scope": "conversation",
        "consumed": False,
        "provider_input": "original",
    }
    assert request_rule_match_preview(match)["score"] == 93.5
    assert request_rule_match_preview(None) == {"matched": False}


@pytest.mark.parametrize("action", ["test_match"])
async def test_management_test_actions_never_delegate_to_real_processing(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    matched_text: list[str] = []

    class Rules:
        async def async_match(self, _hass, text: str):
            matched_text.append(text)
            return _local_match()

    monkeypatch.setattr(
        management_ui,
        "entry_and_agent",
        lambda _hass, _entry_id, _subentry_id: (object(), object()),
    )

    async def get_rules(_hass, _entry_id, _subentry_id):
        return Rules()

    monkeypatch.setattr(management_ui, "async_get_request_rules", get_rules)
    command = management_ui.async_management_command
    hass = SimpleNamespace(
        data={},
        services=SimpleNamespace(
            async_call=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Home Assistant services must not be called")
            )
        ),
    )
    result = await command(
        hass,
        "admin-user",
        True,
        {
            "section": "request_rules",
            "action": action,
            "entry_id": "entry",
            "subentry_id": "agent",
            "text": "  good night kitchen  ",
        },
    )
    assert result["matched"] is True
    assert result["rule"]["name"] == "Good night"
    assert matched_text == ["  good night kitchen  "]


async def test_match_preview_reports_bounded_match_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview should surface limits and never fall through to real processing."""

    class Rules:
        async def async_match(self, _hass, _text: str):
            raise SentenceMatchLimitError(
                "Request Rule matching supports at most 2048 characters"
            )

    monkeypatch.setattr(
        management_ui,
        "entry_and_agent",
        lambda _hass, _entry_id, _subentry_id: (object(), object()),
    )

    async def get_rules(_hass, _entry_id, _subentry_id):
        return Rules()

    monkeypatch.setattr(management_ui, "async_get_request_rules", get_rules)
    command = management_ui.async_management_command
    with pytest.raises(HomeAssistantError, match="2048 characters"):
        await command(
            SimpleNamespace(data={}),
            "admin-user",
            True,
            {
                "section": "request_rules",
                "action": "test_match",
                "entry_id": "entry",
                "subentry_id": "agent",
                "text": "oversized",
            },
        )


async def test_match_preview_requires_admin_and_text() -> None:

    command = management_ui.async_management_command
    with pytest.raises(HomeAssistantError, match="Administrator permission"):
        await command(
            SimpleNamespace(data={}),
            "user",
            False,
            {
                "section": "request_rules",
                "action": "test_match",
                "entry_id": "entry",
                "subentry_id": "agent",
                "text": "hello",
            },
        )


def test_preview_uses_explicit_routing_flow_not_match_type() -> None:
    match = RuleMatch(
        rule={
            "id": "standalone-broad-rule",
            "name": "Standalone broad rule",
            "match_type": "starts_with",
            "action_type": "model_routing",
            "action": {
                "reset": False,
                "model": "gpt-5",
                "reasoning_effort": "high",
                "scope": "conversation",
                "continue_to_ai": False,
            },
        },
        phrase="think carefully",
        fuzzy=False,
        score=100.0,
    )
    result = request_rule_match_preview(match)["would_do"]
    assert result["consumed"] is True
    assert result["provider_input"] == "none"
