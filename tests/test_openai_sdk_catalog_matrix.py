"""Catalogue-generated UI, backend, and real SDK request conformance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from custom_components.extended_openai_conversation_responses.const import (
    CONF_FUNCTION_TOOLS,
)
from custom_components.extended_openai_conversation_responses.model_capabilities import (
    frontend_capabilities,
)
from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)
from homeassistant.exceptions import HomeAssistantError
from tests.test_openai_sdk_wire import (
    _chat_log,
    _chat_text_stream,
    _client,
    _close_client,
    _entity,
    _json_body,
    _responses_text_stream,
    _stream_response,
    _tool,
    _Wire,
)


def _completed_reply(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/responses":
        return httpx.Response(
            200,
            json={
                "id": "resp-catalog-conformance",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4.1-mini",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-catalog-conformance",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Conformant",
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": None,
            },
            request=request,
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-catalog-conformance",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Conformant"},
                }
            ],
            "usage": None,
        },
        request=request,
    )


def _reply(streaming: bool) -> Callable[[httpx.Request], httpx.Response]:
    if not streaming:
        return _completed_reply

    def respond(request: httpx.Request) -> httpx.Response:
        return _stream_response(
            _responses_text_stream("Conformant")
            if request.url.path == "/v1/responses"
            else _chat_text_stream("Conformant")
        )(request)

    return respond


def _ui_path(projection: dict[str, Any], case: dict[str, Any]) -> str | None:
    api = case["api_mode"]
    effort = case["reasoning_effort"]
    functions = case["functions"]
    web_search = case["web_search"]
    if api == "auto":
        key = f"{effort if effort is not None else 'null'}:{int(functions)}:{int(web_search)}"
        return projection["auto_paths"][key]
    evaluation = projection["evaluations"][api][
        effort if effort is not None else "null"
    ]
    if (
        projection["api"][api]
        and evaluation["reasoning"]
        and (not functions or evaluation["function"])
        and (not web_search or evaluation["web_search"])
    ):
        return api
    return None


def _sampling_allowed(rule: dict[str, Any], effort: str | None) -> bool:
    return rule["support"] == "always" or (
        rule["support"] == "conditional" and effort in rule["allowed_reasoning_efforts"]
    )


def _cases(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    effort = metadata["recommended_profile"]["reasoning_effort"]
    if effort is None and metadata["reasoning"]["supported"]:
        effort = metadata["reasoning"]["efforts"][0]
    base = {
        "reasoning_effort": effort,
        "functions": False,
        "web_search": False,
        "sampling": False,
        "tier": None,
    }
    cases = [{**base, "api_mode": api} for api in ("responses", "chat_completions")]
    cases.append({**base, "api_mode": "auto", "functions": True})
    cases.append({**base, "api_mode": "auto", "web_search": True})
    cases.append(
        {
            **base,
            "api_mode": "auto",
            "sampling": True,
            "tier": metadata["service_tiers"][0] if metadata["service_tiers"] else None,
        }
    )
    if len(metadata["reasoning"]["efforts"]) > 1:
        cases.append(
            {
                **base,
                "api_mode": "auto",
                "reasoning_effort": metadata["reasoning"]["efforts"][-1],
                "sampling": True,
            }
        )
    return cases


@pytest.mark.parametrize(
    "model",
    [
        item["id"]
        for item in BUNDLED_CATALOG.resolved.values()
        if item["status"] == "current"
    ],
)
async def test_every_current_model_emits_only_catalogue_allowed_sdk_fields(
    hass, model: str
) -> None:
    """A bounded complete model set covers explicit, Auto, tools, and sampling."""
    metadata = BUNDLED_CATALOG.resolved[model]
    projection = frontend_capabilities(model)
    for case in _cases(metadata):
        label = (model, case)
        expected_api = _ui_path(projection, case)
        options: dict[str, Any] = {
            "chat_model": model,
            "api_mode": case["api_mode"],
            "max_tokens": 128,
            "web_search": case["web_search"],
        }
        if case["reasoning_effort"] is not None:
            options["reasoning_effort"] = case["reasoning_effort"]
        if case["functions"]:
            options[CONF_FUNCTION_TOOLS] = [_tool()]
        if case["sampling"]:
            options.update(temperature=0.25, top_p=0.75)
        if case["tier"] is not None:
            options["service_tier"] = case["tier"]
        wire = _Wire([_reply(metadata["streaming"])])
        client = _client(wire)
        entity = _entity(hass, client, case["api_mode"])
        entity.entry.data = {"api_provider": "openai"}
        entity.subentry.data = options
        chat_log = _chat_log(hass)
        try:
            if expected_api is None:
                with pytest.raises(HomeAssistantError):
                    await entity._async_handle_chat_log(
                        chat_log, [_tool()] if case["functions"] else [], []
                    )
                assert wire.requests == [], label
                continue
            await entity._async_handle_chat_log(
                chat_log, [_tool()] if case["functions"] else [], []
            )
        finally:
            await _close_client(client)
        assert len(wire.requests) == 1, label
        request = wire.requests[0]
        assert request.url.path == (
            "/v1/responses" if expected_api == "responses" else "/v1/chat/completions"
        ), label
        body = _json_body(request)
        assert body["model"] == model, label
        assert body["stream"] is metadata["streaming"], label
        assert "max_tokens" not in body, label
        token_field = metadata["output_tokens"][expected_api]
        assert body[token_field] == 128, label
        assert ("temperature" in body) == (
            case["sampling"]
            and _sampling_allowed(metadata["temperature"], case["reasoning_effort"])
        ), label
        assert ("top_p" in body) == (
            case["sampling"]
            and _sampling_allowed(metadata["top_p"], case["reasoning_effort"])
        ), label
        if case["reasoning_effort"] is not None:
            effort_field = (
                body["reasoning"]["effort"]
                if expected_api == "responses"
                else body["reasoning_effort"]
            )
            assert effort_field == case["reasoning_effort"], label
        if case["tier"] is not None:
            assert body["service_tier"] == case["tier"], label
        tools = body.get("tools", [])
        assert any(item["type"] == "function" for item in tools) == case["functions"], (
            label
        )
        assert (
            any(item["type"] == "web_search" for item in tools) == case["web_search"]
        ), label
        assert chat_log.content[-1].content == "Conformant", label
