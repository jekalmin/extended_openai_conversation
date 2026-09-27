"""Contract tests against Home Assistant's actual Script engine."""

import asyncio
from copy import deepcopy

import pytest

from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
    RequestRuleRuntime,
    RequestRules,
    _outcome_probes,
    async_evaluate_rule,
)
from homeassistant.core import SupportsResponse
from homeassistant.exceptions import HomeAssistantError


class MemoryStore:
    def __init__(self, rule):
        self.rule = rule

    async def async_load(self):
        return {"defaults": dict(DEFAULT_MATCHING), "rules": [deepcopy(self.rule)]}

    async def async_save(self, _value):
        pass


async def evaluate(hass, actions, *, handoff=False):
    rule = {
        "id": "native",
        "name": "Native",
        "enabled": True,
        "phrases": ["run"],
        "match_type": "equals",
        "action_type": "local_action",
        "action": {
            "actions": actions,
            "success_response": "Done",
            "failure_response": "Failed",
            "continue_to_ai": handoff,
        },
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": 0,
    }
    rules = RequestRules(MemoryStore(rule))
    await rules.async_initialize()
    return await async_evaluate_rule(hass, rules, RequestRuleRuntime(), "run", "native")


@pytest.mark.parametrize(
    ("actions", "handoff", "response", "consume", "successful"),
    [
        ([{"variables": {"x": 1}}], False, "Done", True, True),
        ([{"variables": {"x": 1}}], True, None, False, True),
        ([{"set_conversation_response": "first"}], False, "first", True, True),
        ([{"set_conversation_response": "first"}], True, "first", True, True),
        (
            [
                {"set_conversation_response": "first"},
                {"set_conversation_response": "second"},
            ],
            True,
            "second",
            True,
            True,
        ),
        (
            [{"stop": "finished"}, {"variables": {"later": True}}],
            False,
            "Done",
            True,
            True,
        ),
        ([{"stop": "finished"}], True, "Done", True, True),
        ([{"stop": "disabled", "enabled": False}], True, None, False, True),
        (
            [{"set_conversation_response": "before"}, {"stop": "finished"}],
            True,
            "before",
            True,
            True,
        ),
        ([{"stop": "failed", "error": True}], True, "Failed", True, False),
        (
            [
                {"set_conversation_response": "before"},
                {"stop": "failed", "error": True},
            ],
            True,
            "Failed",
            True,
            False,
        ),
        (
            [{"condition": "template", "value_template": "{{ false }}"}],
            True,
            "Failed",
            True,
            False,
        ),
        (
            [
                {
                    "wait_template": "{{ false }}",
                    "timeout": "00:00:00",
                    "continue_on_timeout": False,
                }
            ],
            True,
            "Failed",
            True,
            False,
        ),
        (
            [
                {
                    "wait_template": "{{ false }}",
                    "timeout": "00:00:00",
                    "continue_on_timeout": True,
                }
            ],
            False,
            "Done",
            True,
            True,
        ),
        (
            [
                {
                    "wait_template": "{{ false }}",
                    "timeout": "00:00:00",
                    "continue_on_timeout": False,
                    "enabled": False,
                }
            ],
            True,
            None,
            False,
            True,
        ),
        (
            [
                {
                    "if": [{"condition": "template", "value_template": "{{ false }}"}],
                    "then": [
                        {"condition": "template", "value_template": "{{ false }}"}
                    ],
                    "else": [{"variables": {"branch": 1}}],
                },
                {"set_conversation_response": "after"},
            ],
            True,
            "after",
            True,
            True,
        ),
        (
            [
                {
                    "choose": [
                        {
                            "conditions": [
                                {
                                    "condition": "template",
                                    "value_template": "{{ true }}",
                                }
                            ],
                            "sequence": [
                                {
                                    "condition": "template",
                                    "value_template": "{{ false }}",
                                }
                            ],
                        }
                    ]
                },
                {"set_conversation_response": "after"},
            ],
            True,
            "after",
            True,
            True,
        ),
        (
            [
                {
                    "if": [{"condition": "template", "value_template": "{{ true }}"}],
                    "then": [{"set_conversation_response": "nested"}],
                }
            ],
            True,
            "nested",
            True,
            True,
        ),
        (
            [{"parallel": [{"sequence": [{"set_conversation_response": "parallel"}]}]}],
            True,
            "parallel",
            True,
            True,
        ),
        (
            [
                {
                    "if": [{"condition": "template", "value_template": "{{ true }}"}],
                    "then": [{"stop": "nested"}],
                },
                {"set_conversation_response": "never"},
            ],
            True,
            "Done",
            True,
            True,
        ),
    ],
)
async def test_native_script_outcomes(
    hass, actions, handoff, response, consume, successful
):
    hass.loop = asyncio.get_running_loop()
    hass.async_create_task_internal.side_effect = lambda coro, **_kwargs: (
        asyncio.create_task(coro)
    )
    outcome = await evaluate(hass, actions, handoff=handoff)
    assert outcome is not None
    assert (outcome.response, outcome.consume, outcome.successful) == (
        response,
        consume,
        successful,
    )


async def test_continue_on_error_follows_native_script_policy(hass):
    hass.loop = asyncio.get_running_loop()
    hass.async_create_task_internal.side_effect = lambda coro, **_kwargs: (
        asyncio.create_task(coro)
    )
    hass.services.async_call.side_effect = HomeAssistantError("recoverable")
    outcome = await evaluate(
        hass,
        [
            {"action": "light.turn_on", "continue_on_error": True},
            {"set_conversation_response": "recovered"},
        ],
        handoff=True,
    )
    assert outcome is not None
    assert (outcome.response, outcome.consume, outcome.successful) == (
        "recovered",
        True,
        True,
    )


async def test_script_variables_feed_native_response_but_not_generic_success(hass):
    hass.loop = asyncio.get_running_loop()
    response = await evaluate(
        hass,
        [
            {"variables": {"message": "native"}},
            {"set_conversation_response": "{{ message }}"},
        ],
    )
    assert response is not None and response.response == "native"
    generic = await evaluate(hass, [{"variables": {"message": "native"}}])
    assert generic is not None and generic.response == "Done"


async def test_native_service_response_variable_flows_to_later_actions(hass):
    hass.loop = asyncio.get_running_loop()
    hass.async_create_task_internal.side_effect = lambda coro, **_kwargs: (
        asyncio.create_task(coro)
    )
    hass.services.supports_response.return_value = SupportsResponse.OPTIONAL
    hass.services.async_call.return_value = {"message": "from service"}
    outcome = await evaluate(
        hass,
        [
            {"action": "test.echo", "response_variable": "reply"},
            {"set_conversation_response": "{{ reply.message }}"},
        ],
    )
    assert outcome is not None and outcome.response == "from service"


def test_outcome_instrumentation_leaves_service_payloads_untouched():
    action = {"action": "test.echo", "data": {"items": [{"stop": "payload"}]}}
    instrumented, _, _ = _outcome_probes([action])
    assert instrumented[1] == action
