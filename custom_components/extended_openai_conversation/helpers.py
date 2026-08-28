"""Helper functions for Extended OpenAI Conversation component."""

from __future__ import annotations

from functools import partial
import logging
import re
from typing import Any

from openai import AsyncAzureOpenAI, AsyncClient, AsyncOpenAI

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.template import Template

from .const import (
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TOKEN_PARAM,
    MODEL_CONFIG_PATTERNS,
    MODEL_TOKEN_PARAMETER_SUPPORT,
)

_LOGGER = logging.getLogger(__name__)


AZURE_DOMAIN_PATTERN = r"\.(openai\.azure\.com|azure-api\.net|services\.ai\.azure\.com)"


def get_model_config(model: str) -> dict[str, bool]:
    """Get model-specific parameter configuration."""
    # Check patterns in order; first match wins
    for entry in MODEL_CONFIG_PATTERNS:
        pattern = str(entry["pattern"])
        entry_config = entry["config"]
        if re.match(pattern, model, re.IGNORECASE):
            # Type assertion since we know the structure from MODEL_CONFIG_PATTERNS
            return (
                dict(entry_config)
                if isinstance(entry_config, dict)
                else DEFAULT_MODEL_CONFIG
            )

    # Default configuration for standard models (gpt-4, gpt-4o, etc.)
    return DEFAULT_MODEL_CONFIG


def get_exposed_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Get exposed entities."""
    states = [
        state
        for state in hass.states.async_all()
        if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
    ]
    entity_registry = er.async_get(hass)
    exposed_entities = []
    for state in states:
        entity_id = state.entity_id
        entity = entity_registry.async_get(entity_id)

        aliases: list[str] = []
        if entity and entity.aliases:
            aliases = [str(a) for a in entity.aliases]

        exposed_entities.append(
            {
                "entity_id": entity_id,
                "name": state.name,
                "state": state.state,
                "aliases": aliases,
            }
        )
    return exposed_entities


def is_azure_url(base_url: str | None) -> bool:
    """Check if the base URL is an Azure OpenAI URL."""
    return bool(base_url and re.search(AZURE_DOMAIN_PATTERN, base_url))


def get_token_param_for_model(model: str) -> str:
    """Return the token parameter name for a model."""
    model_lower = model.lower()
    for entry in MODEL_TOKEN_PARAMETER_SUPPORT:
        if re.search(entry["pattern"], model_lower):
            return entry["token_param"]
    return DEFAULT_TOKEN_PARAM


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
    hass: HomeAssistant,
    parents: list[str],
) -> None:
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = Template(value, hass)
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

    client: AsyncClient
    if base_url and (is_azure_url(base_url) or api_provider == "azure"):
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            organization=organization,
            http_client=get_async_client(hass),
        )
    else:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            http_client=get_async_client(hass),
        )

    if skip_authentication:
        return client

    response = await hass.async_add_executor_job(
        partial(client.models.list, timeout=10)
    )

    async for _ in response:
        break
    return client
