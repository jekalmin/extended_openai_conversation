"""Catalogue-generated UI projection and request-validation agreement."""

from __future__ import annotations

from itertools import product

import pytest

from custom_components.extended_openai_conversation_responses.model_capabilities import (
    ModelCapabilityError,
    frontend_capabilities,
    select_api_path,
    validate_api_path,
)
from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)
from custom_components.extended_openai_conversation_responses.request import (
    build_provider_request_snapshot,
)
from homeassistant.exceptions import HomeAssistantError


@pytest.mark.parametrize("model", [item["id"] for item in BUNDLED_CATALOG["models"]])
def test_projected_api_and_tool_choices_match_backend_for_every_model(model):
    projection = frontend_capabilities(model)
    efforts = (None, *projection["reasoning_effort_options"])
    for effort, api, functions, web_search in product(
        efforts, ("responses", "chat_completions"), (False, True), (False, True)
    ):
        evaluation = projection["evaluations"][api][
            effort if effort is not None else "null"
        ]
        ui_selectable = bool(
            projection["api"][api]
            and evaluation["reasoning"]
            and (not functions or evaluation["function"])
            and (not web_search or evaluation["web_search"])
        )
        try:
            validate_api_path(model, api, functions, effort, web_search)
            backend_accepts = True
        except ModelCapabilityError:
            backend_accepts = False
        assert ui_selectable == backend_accepts, (
            model,
            api,
            effort,
            functions,
            web_search,
        )

        key = f"{effort if effort is not None else 'null'}:{int(functions)}:{int(web_search)}"
        try:
            resolved = select_api_path(model, "auto", functions, effort, web_search)
        except ModelCapabilityError:
            resolved = None
        assert projection["auto_paths"][key] == resolved, (model, key)


@pytest.mark.parametrize(
    "model", ["gpt-6-astra", "gpt-6-sol", "gpt-5-mini", "gpt-5.6", "gpt-4.1"]
)
def test_selectable_representative_choices_build_safe_requests(model):
    projection = frontend_capabilities(model)
    efforts = projection["reasoning_effort_options"] or [None]
    for effort, api, functions, web_search in product(
        efforts, ("auto", "responses", "chat_completions"), (False, True), (False, True)
    ):
        options = {"chat_model": model, "api_mode": api, "web_search": web_search}
        if effort is not None:
            options["reasoning_effort"] = effort
        try:
            snapshot = build_provider_request_snapshot(
                options, {"api_provider": "openai"}, tools_required=functions
            )
            accepted = True
        except HomeAssistantError:
            accepted = False
        if api == "auto":
            key = f"{effort if effort is not None else 'null'}:{int(functions)}:{int(web_search)}"
            expected = projection["auto_paths"][key]
        else:
            evaluation = projection["evaluations"][api][
                effort if effort is not None else "null"
            ]
            expected = (
                api
                if (
                    projection["api"][api]
                    and evaluation["reasoning"]
                    and (not functions or evaluation["function"])
                    and (not web_search or evaluation["web_search"])
                )
                else None
            )
        assert accepted == (expected is not None), (
            model,
            api,
            effort,
            functions,
            web_search,
        )
        if not accepted:
            continue
        assert snapshot.api_mode == expected
        assert "max_tokens" not in snapshot.api_kwargs
        assert "temperature" not in snapshot.api_kwargs
        assert "top_p" not in snapshot.api_kwargs
        assert bool(snapshot.provider_tools) == web_search
        if projection["service_tier_options"]:
            for tier in projection["service_tier_options"]:
                selected = build_provider_request_snapshot(
                    {**options, "service_tier": tier},
                    {"api_provider": "openai"},
                    tools_required=functions,
                )
                assert selected.api_kwargs["service_tier"] == tier
