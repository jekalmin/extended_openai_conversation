"""Luna reasoning configuration and preservation of advanced settings."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from custom_components.extended_openai_conversation.config_flow import (
    ExtendedOpenAIAITaskSubentryFlowHandler,
    ExtendedOpenAISubentryFlowHandler,
)
from custom_components.extended_openai_conversation.const import (
    RESPONSES_MODEL_REASONING_EFFORT_OPTIONS,
)
from homeassistant.config_entries import ConfigEntryState


@pytest.mark.parametrize(
    "handler",
    [
        ExtendedOpenAISubentryFlowHandler,
        ExtendedOpenAIAITaskSubentryFlowHandler,
    ],
)
@pytest.mark.parametrize("model", RESPONSES_MODEL_REASONING_EFFORT_OPTIONS)
async def test_advanced_reasoning_levels(handler, model):
    flow = handler()
    flow._temp_data = {"chat_model": model}
    flow.options = {"reasoning_effort": "high"}
    with patch.object(flow, "async_show_form", side_effect=lambda **kwargs: kwargs):
        result = await flow.async_step_advanced()
    schema = result["data_schema"].schema
    key = next(key for key in schema if key.schema == "reasoning_effort")
    assert key.description["suggested_value"] == "high"
    assert [option["value"] for option in schema[key].config["options"]] == (
        RESPONSES_MODEL_REASONING_EFFORT_OPTIONS[model]
    )


@pytest.mark.parametrize(
    "handler",
    [
        ExtendedOpenAISubentryFlowHandler,
        ExtendedOpenAIAITaskSubentryFlowHandler,
    ],
)
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
async def test_switching_from_none_suggests_supported_effort(handler, model):
    flow = handler()
    flow._temp_data = {"chat_model": model}
    flow.options = {"chat_model": "gpt-6-luna", "reasoning_effort": "none"}
    with patch.object(flow, "async_show_form", side_effect=lambda **kwargs: kwargs):
        result = await flow.async_step_advanced()
    key = next(
        key for key in result["data_schema"].schema if key.schema == "reasoning_effort"
    )
    assert key.description["suggested_value"] == "low"
    # Merely opening the form does not change the saved setting.
    assert flow.options["reasoning_effort"] == "none"


@pytest.mark.parametrize(
    "handler",
    [
        ExtendedOpenAISubentryFlowHandler,
        ExtendedOpenAIAITaskSubentryFlowHandler,
    ],
)
async def test_simple_save_preserves_advanced_options(handler):
    flow = handler()
    flow.options = {
        "chat_model": "gpt-6-luna",
        "reasoning_effort": "medium",
        "service_tier": "auto",
        "max_tokens": 3000,
        "prompt": "Unchanged",
    }
    flow._available_skills = []
    entry = SimpleNamespace(state=ConfigEntryState.LOADED)
    with (
        patch.object(flow, "_get_entry", return_value=entry),
        patch.object(
            type(flow), "_is_new", new_callable=lambda: property(lambda self: False)
        ),
        patch.object(flow, "_get_reconfigure_subentry", return_value=SimpleNamespace()),
        patch.object(
            flow, "async_update_and_abort", side_effect=lambda *args, **kwargs: kwargs
        ),
    ):
        result = await flow.async_step_init(
            {"advanced_options": False, "max_tokens": 4000}
        )
    assert result["data"]["reasoning_effort"] == "medium"
    assert result["data"]["service_tier"] == "auto"
    assert result["data"]["prompt"] == "Unchanged"
    assert result["data"]["max_tokens"] == 4000
