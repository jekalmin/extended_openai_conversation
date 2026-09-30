"""Tests for the narrow OpenAI SDK compatibility patch."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import openai
from pydantic import BaseModel, ValidationError, create_model
import pytest

from custom_components.extended_openai_conversation_responses import openai_compat


def _install_usage_models(
    monkeypatch: pytest.MonkeyPatch, input_model: type[BaseModel]
) -> type[BaseModel]:
    """Install isolated usage models at the SDK import path used by the patch."""
    response_usage = create_model(
        "ResponseUsage",
        input_tokens_details=(input_model, ...),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai.types.responses.response_usage",
        SimpleNamespace(
            InputTokensDetails=input_model,
            ResponseUsage=response_usage,
        ),
    )
    return response_usage


def _prepare_patch(monkeypatch: pytest.MonkeyPatch, version: str = "2.45.0") -> None:
    """Reset the module guard and expose the requested SDK version."""
    monkeypatch.setattr(openai_compat, "_PATCHED", False)
    monkeypatch.setattr(openai, "__version__", version)


def test_required_245_usage_field_becomes_optional_and_patch_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2.45 provider usage without cache_write_tokens parses after one patch."""

    class InputTokensDetails(BaseModel):
        cache_write_tokens: int

    ResponseUsage = _install_usage_models(monkeypatch, InputTokensDetails)
    _prepare_patch(monkeypatch)

    with pytest.raises(ValidationError):
        ResponseUsage(input_tokens_details={})

    openai_compat.apply_openai_compatibility()

    parsed = ResponseUsage(input_tokens_details={})
    assert parsed.input_tokens_details.cache_write_tokens == 0
    assert openai_compat._PATCHED is True

    # A second call must be a no-op rather than reapplying the schema mutation.
    field = InputTokensDetails.model_fields["cache_write_tokens"]
    field.default = 7
    openai_compat.apply_openai_compatibility()
    assert field.default == 7


@pytest.mark.parametrize("version", ["2.21.0", "2.45.1", "3.10.0"])
def test_non_target_openai_version_is_not_patched(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    """The workaround must not leak into SDK releases other than 2.45.0."""

    class InputTokensDetails(BaseModel):
        cache_write_tokens: int

    ResponseUsage = _install_usage_models(monkeypatch, InputTokensDetails)
    _prepare_patch(monkeypatch, version=version)

    openai_compat.apply_openai_compatibility()

    assert openai_compat._PATCHED is False
    with pytest.raises(ValidationError):
        ResponseUsage(input_tokens_details={})


def test_already_optional_usage_field_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-compatible SDK schema keeps its existing default."""

    class InputTokensDetails(BaseModel):
        cache_write_tokens: int = 9

    ResponseUsage = _install_usage_models(monkeypatch, InputTokensDetails)
    _prepare_patch(monkeypatch)

    openai_compat.apply_openai_compatibility()

    parsed = ResponseUsage(input_tokens_details={})
    assert parsed.input_tokens_details.cache_write_tokens == 9
    assert openai_compat._PATCHED is True


def test_missing_usage_field_is_treated_as_already_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema without the provider-specific field needs no mutation."""

    class InputTokensDetails(BaseModel):
        cached_tokens: int = 0

    ResponseUsage = _install_usage_models(monkeypatch, InputTokensDetails)
    _prepare_patch(monkeypatch)

    openai_compat.apply_openai_compatibility()

    parsed = ResponseUsage(input_tokens_details={})
    assert parsed.input_tokens_details.cached_tokens == 0
    assert "cache_write_tokens" not in InputTokensDetails.model_fields
    assert openai_compat._PATCHED is True
