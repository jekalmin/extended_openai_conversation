"""Narrow compatibility fixes for the Home Assistant-pinned OpenAI SDK."""

from __future__ import annotations

import logging

import openai

_LOGGER = logging.getLogger(__name__)

_OPENAI_245 = "2.45.0"
_PATCHED = False


def apply_openai_compatibility() -> None:
    """Allow OpenAI-compatible providers to omit a 2.45-only usage field.

    HA stable releases using openai==2.45.0 need this shim. That release made
    ``cache_write_tokens`` required in Responses usage metadata even though many
    OpenAI-compatible providers do not return it. Keep the HA-compatible package
    version and make only that generated field tolerant on that exact SDK.
    """
    global _PATCHED
    if _PATCHED or getattr(openai, "__version__", None) != _OPENAI_245:
        return

    from openai.types.responses.response_usage import InputTokensDetails, ResponseUsage

    field = InputTokensDetails.model_fields.get("cache_write_tokens")
    if field is None or not field.is_required():
        _PATCHED = True
        return

    # Pydantic generated the SDK model from an API schema that treated this newly
    # added provider-specific counter as mandatory. A zero default preserves the
    # meaning of an omitted counter and matches the SDK issue's recommended shape.
    field.default = 0
    InputTokensDetails.model_rebuild(force=True)
    # Rebuild the parent so nested response parsing uses the relaxed child schema.
    ResponseUsage.model_rebuild(force=True)
    _PATCHED = True
    _LOGGER.debug(
        "Applied OpenAI 2.45 Responses usage compatibility for compatible providers"
    )
