"""Small, sanitized live OpenAI canaries selected from the current catalogue."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from custom_components.extended_openai_conversation_responses.model_capabilities import (
    capability_allowed,
    model_capability_snapshot,
)
from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)
from custom_components.extended_openai_conversation_responses.request import (
    build_provider_request_snapshot,
    format_function_tools,
)

_PREFERRED_BASIC = ("gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini")
_PREFERRED_REASONING = ("gpt-5-mini", "gpt-5.6-luna", "gpt-4.1-mini")
_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_FUNCTION = {
    "spec": {
        "name": "canary_echo",
        "description": "Echo a short value; this canary never runs the tool",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
    "function": {"type": "template", "value_template": "unused"},
}


@dataclass(frozen=True, slots=True)
class Canary:
    name: str
    model: str
    api: str
    effort: str | None


def _choose(
    api: str,
    *,
    tool: str | None = None,
    structured: bool = False,
    preferred: tuple[str, ...] = _PREFERRED_BASIC,
) -> tuple[str, str | None]:
    candidates = [
        *preferred,
        *sorted(BUNDLED_CATALOG.resolved),
    ]
    for model_id in dict.fromkeys(candidates):
        model = BUNDLED_CATALOG.resolved.get(model_id)
        if model is None or model["status"] != "current" or not model["api"][api]:
            continue
        if structured and not model["structured_outputs"]:
            continue
        efforts = model["reasoning"]["by_api"][api]["efforts"] or [None]
        preferred_effort = model["recommended_profile"]["reasoning_effort"]
        ordered = sorted(
            efforts,
            key=lambda effort: (
                effort != preferred_effort,
                effort not in {None, "low"},
            ),
        )
        for effort in ordered:
            with model_capability_snapshot(model_id, model):
                if tool and not capability_allowed(
                    model_id,
                    tool,
                    api,
                    effort=effort,
                    streaming=model["streaming"],
                ):
                    continue
            return model_id, effort
    raise ValueError(f"Current catalogue has no canary model for {api}/{tool}")


def plan() -> list[Canary]:
    """Select six low-cost checks from supported current catalogue claims."""
    cases = [
        ("responses_text", "responses", None, False, _PREFERRED_BASIC),
        ("chat_text", "chat_completions", None, False, _PREFERRED_BASIC),
        ("responses_stream", "responses", None, False, _PREFERRED_REASONING),
        ("function_tool", "responses", "function", False, _PREFERRED_BASIC),
        ("web_search", "responses", "web_search", False, _PREFERRED_REASONING),
        ("structured_output", "responses", None, True, _PREFERRED_BASIC),
    ]
    selected = []
    for name, api, tool, structured, preferred in cases:
        model, effort = _choose(
            api, tool=tool, structured=structured, preferred=preferred
        )
        selected.append(Canary(name, model, api, effort))
    return selected


def _failure_class(error: BaseException) -> str:
    if not isinstance(error, OpenAIError):
        return "contract_drift"
    status = getattr(error, "status_code", None)
    if status in {400, 422}:
        return "contract_drift"
    if status in {401, 403}:
        return "account_access"
    if status == 404:
        return "model_access_or_catalog_drift"
    if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
        return "provider_or_rate_limit"
    return "transport_or_provider"


def _safe_failure(name: str, model: str, error: BaseException) -> dict[str, Any]:
    """Store no provider body, prompt, response text, URL, or credential."""
    status = getattr(error, "status_code", None)
    return {
        "name": name,
        "model": model,
        "outcome": "failed",
        "failure_class": _failure_class(error),
        "error_type": type(error).__name__,
        "status_code": status if type(status) is int else None,
    }


async def _run_one(client: AsyncOpenAI, case: Canary) -> None:
    options: dict[str, Any] = {
        "chat_model": case.model,
        "api_mode": case.api,
        # Reasoning tokens share this ceiling with visible text and tool output.
        "max_tokens": 512 if case.name in {"responses_stream", "web_search"} else 128,
    }
    if case.effort is not None:
        options["reasoning_effort"] = case.effort
    if case.name == "web_search":
        options["web_search"] = True
    snapshot = build_provider_request_snapshot(
        options,
        {"api_provider": "openai"},
        tools_required=case.name == "function_tool",
    )
    assert snapshot.api_mode == case.api
    kwargs = {**snapshot.api_kwargs, "stream": case.name == "responses_stream"}
    if not kwargs["stream"]:
        kwargs.pop("stream_options", None)
    prompt = "Reply with one short sentence."
    if case.name == "function_tool":
        prompt = "Call canary_echo with value hello."
        kwargs["tools"] = format_function_tools([_FUNCTION], case.api)
        kwargs["tool_choice"] = "required"
    elif case.name == "web_search":
        prompt = "Search the web for the current UTC date and answer briefly."
        kwargs["tools"] = list(snapshot.provider_tools)
        kwargs["tool_choice"] = "required"
    elif case.name == "structured_output":
        prompt = "Return a JSON object whose answer is hello."
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": "canary_answer",
                "strict": True,
                "schema": _SCHEMA,
            }
        }
    if case.api == "chat_completions":
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], **kwargs
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Chat Completions returned no text")
        return
    response = await client.responses.create(
        input=[{"role": "user", "content": prompt}], **kwargs
    )
    if case.name == "responses_stream":
        terminal = False
        text_seen = False
        async for event in response:
            terminal |= event.type == "response.completed"
            text_seen |= event.type == "response.output_text.delta" and bool(
                getattr(event, "delta", "")
            )
        if not terminal or not text_seen:
            raise ValueError("Responses stream lacked text or terminal event")
        return
    if response.status != "completed":
        raise ValueError("Responses did not complete")
    if case.name == "function_tool":
        if not any(item.type == "function_call" for item in response.output):
            raise ValueError("Required Function Tool call was absent")
    elif case.name == "web_search":
        if not any(item.type == "web_search_call" for item in response.output):
            raise ValueError("Required Web Search call was absent")
    elif case.name == "structured_output":
        value = json.loads(response.output_text)
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            raise ValueError("Structured output violated its schema")
    elif not response.output_text:
        raise ValueError("Responses returned no text")


async def run(
    cases: list[Canary], *, plan_only: bool
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    if plan_only:
        return [
            {
                "name": case.name,
                "model": case.model,
                "api": case.api,
                "outcome": "planned",
            }
            for case in cases
        ], 0
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required for live canaries")
    async with AsyncOpenAI(api_key=key, max_retries=0, timeout=30) as client:
        for case in cases:
            try:
                await _run_one(client, case)
            except Exception as error:
                results.append(_safe_failure(case.name, case.model, error))
            else:
                results.append(
                    {"name": case.name, "model": case.model, "outcome": "passed"}
                )
    classifications = {item.get("failure_class") for item in results}
    return results, (
        1
        if "contract_drift" in classifications
        else 2
        if any(item["outcome"] == "failed" for item in results)
        else 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    cases = plan()
    results, status = asyncio.run(run(cases, plan_only=args.plan_only))
    args.report.write_text(
        json.dumps(
            {
                "catalog_version": BUNDLED_CATALOG["catalog_version"],
                "schema_version": BUNDLED_CATALOG["schema_version"],
                "checks": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raise SystemExit(status)


if __name__ == "__main__":
    main()
