"""Cost and evidence boundaries for the scheduled six-call OpenAI canary."""

from __future__ import annotations

from openai import OpenAIError

from ci.live_openai_canary import _failure_class, _safe_failure, plan, run
from custom_components.extended_openai_conversation_responses.model_capabilities import (
    capability_allowed,
    model_capability_snapshot,
)
from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)


def test_canary_plan_is_bounded_and_catalogue_supported() -> None:
    cases = plan()
    assert [case.name for case in cases] == [
        "responses_text",
        "chat_text",
        "responses_stream",
        "function_tool",
        "web_search",
        "structured_output",
    ]
    assert len(cases) == 6
    for case in cases:
        metadata = BUNDLED_CATALOG.resolved[case.model]
        assert metadata["status"] == "current"
        assert metadata["api"][case.api]
        assert metadata["streaming"]
        if case.name == "structured_output":
            assert metadata["structured_outputs"]
        if case.name in {"function_tool", "web_search"}:
            tool = "function" if case.name == "function_tool" else "web_search"
            with model_capability_snapshot(case.model, metadata):
                assert capability_allowed(
                    case.model,
                    tool,
                    case.api,
                    effort=case.effort,
                    streaming=metadata["streaming"],
                )


async def test_plan_only_never_needs_credentials_or_provider() -> None:
    result, status = await run(plan(), plan_only=True)
    assert status == 0
    assert len(result) == 6
    assert all(item["outcome"] == "planned" for item in result)


def test_canary_failure_report_excludes_provider_text_and_secrets() -> None:
    error = ValueError("secret sk-should-never-leak and provider response")
    safe = _safe_failure("structured_output", "gpt-4.1-mini", error)
    assert safe == {
        "name": "structured_output",
        "model": "gpt-4.1-mini",
        "outcome": "failed",
        "failure_class": "contract_drift",
        "error_type": "ValueError",
        "status_code": None,
    }


def test_canary_distinguishes_contract_account_and_transient_failures() -> None:
    class WireError(OpenAIError):
        def __init__(self, status_code: int) -> None:
            super().__init__("provider text must not enter reports")
            self.status_code = status_code

    assert _failure_class(WireError(400)) == "contract_drift"
    assert _failure_class(WireError(401)) == "account_access"
    assert _failure_class(WireError(404)) == "model_access_or_catalog_drift"
    assert _failure_class(WireError(429)) == "provider_or_rate_limit"
    assert _failure_class(WireError(503)) == "provider_or_rate_limit"
