"""Helper functions for Extended OpenAI Conversation (Responses) component."""

from __future__ import annotations

from functools import partial
from inspect import isawaitable
import logging
import re
from typing import Any, cast

from openai import AsyncAzureOpenAI, AsyncClient, AsyncOpenAI

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.template import Template

from .const import DEFAULT_API_PROVIDER, DEFAULT_CONF_BASE_URL
from .entity_context_cache import (
    get_entity_prompt_metadata,
    normalize_entity_aliases as normalize_entity_aliases,
)
from .ha_permissions import filter_entities_for_active_user
from .model_capabilities import select_api_path
from .model_catalog import compatibility_capabilities, model_metadata
from .provider_errors import provider_transport_error

_LOGGER = logging.getLogger(__name__)


AZURE_DOMAIN_PATTERN = r"\.(openai\.azure\.com|azure-api\.net|services\.ai\.azure\.com)"


def get_api_mode(configured_mode: str, model: str, tools_required: bool = False) -> str:
    """Resolve/validate API mode from authoritative model capability data."""
    return select_api_path(model, configured_mode, tools_required)


def get_model_config(model: str) -> dict[str, Any]:
    """Return legacy-compatible booleans derived from v2 capability data.

    `supports_max_tokens` is deliberately always false. Older call sites that still
    branch on this helper therefore normalize onto max_completion_tokens instead of
    ever emitting the deprecated max_tokens field.
    """
    return compatibility_capabilities(model)


def get_reasoning_effort_options(model: str) -> list[str]:
    """Return the exact model-specific reasoning enum from the active catalogue."""
    return list(model_metadata(model)["reasoning"]["efforts"])


def get_exposed_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Get Assist-exposed entities the authenticated caller may read."""
    states = [
        state
        for state in hass.states.async_all()
        if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
    ]
    exposed_entities = []
    for state in states:
        metadata = get_entity_prompt_metadata(hass, state.entity_id)
        exposed_entities.append(
            {
                "entity_id": state.entity_id,
                "name": state.name,
                "state": state.state,
                "aliases": list(metadata.aliases),
            }
        )
    return filter_entities_for_active_user(hass, exposed_entities)


def is_azure_url(base_url: str | None) -> bool:
    """Check if the base URL is an Azure OpenAI URL."""
    return bool(base_url and re.search(AZURE_DOMAIN_PATTERN, base_url))


def supports_openai_hosted_tools(
    api_provider: str | None, base_url: str | None
) -> bool:
    """Return whether the entry uses OpenAI's native hosted tools endpoint."""
    if api_provider not in {None, DEFAULT_API_PROVIDER}:
        return False
    return not base_url or base_url.rstrip("/") == DEFAULT_CONF_BASE_URL.rstrip("/")


def get_token_param_for_model(model: str) -> str:
    """Return the modern Chat Completions output-token field.

    This compatibility helper intentionally never returns deprecated `max_tokens`.
    Responses requests use max_output_tokens through the request builder instead.
    """
    del model
    return "max_completion_tokens"


def convert_to_template(
    settings: Any,
    template_keys: list[str] | None = None,
    hass: HomeAssistant | None = None,
) -> None:
    if template_keys is None:
        template_keys = ["data", "event_data", "target", "service"]
    _convert_to_template(settings, template_keys, hass, [])


def _convert_to_template(
    settings: Any,
    template_keys: list[str],
    hass: HomeAssistant | None,
    parents: list[str],
) -> None:
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, cast(HomeAssistant, hass))
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, hass, parents)
                parents.pop()
            if isinstance(value, list):
                parents.append(key)
                for item in value:
                    _convert_to_template(item, template_keys, hass, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, hass, parents)


async def get_authenticated_client(
    hass: HomeAssistant,
    api_key: str,
    base_url: str | None,
    api_version: str | None,
    organization: str | None,
    api_provider: str | None,
    skip_authentication: bool = False,
) -> AsyncClient:
    """Validate OpenAI authentication."""

    # Match HA core: retain its shared HTTPX client and TLS/connection policy.
    # SDK 3 supports this client at runtime, while typing targets HTTPX2.
    client: AsyncClient
    if base_url and (is_azure_url(base_url) or api_provider == "azure"):
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            organization=organization,
            http_client=cast(Any, get_async_client(hass)),
        )
    else:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            http_client=cast(Any, get_async_client(hass)),
        )

    if skip_authentication:
        return client

    try:
        response = await hass.async_add_executor_job(
            partial(client.models.list, timeout=10)
        )

        # SDK 2 returns an async paginator immediately; SDK 3 returns a
        # coroutine. Materialize either awaitable exactly once before iterating.
        page = await response if isawaitable(response) else response
        async for _ in page:
            break
    except (TimeoutError, ConnectionError) as err:
        raise provider_transport_error(err) from err
    return client
