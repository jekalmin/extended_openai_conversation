"""Config flow for OpenAI Conversation integration."""
from __future__ import annotations

from functools import partial
import logging
import types
import yaml
from types import MappingProxyType
from typing import Any

import openai
from openai import error
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME, CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    TemplateSelector,
    AttributeSelector,
    BooleanSelector,
    NumberSelectorMode,
)

from .helpers import PineconeStorage
from .exceptions import PineconeAuthenticationError

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_FUNCTIONS,
    CONF_USE_INTERACTIVE,
    CONF_PINECONE_API_KEY,
    CONF_PINECONE_TOP_K,
    CONF_PINECONE_SCORE_THRESHOLD,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_USE_INTERACTIVE,
    DEFAULT_PINECONE_TOP_K,
    DOMAIN,
    DEFAULT_NAME,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): str,
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_PINECONE_API_KEY): str,
    }
)

DEFAULT_CONF_FUNCTIONS_STR = yaml.dump(DEFAULT_CONF_FUNCTIONS, sort_keys=False)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_FUNCTIONS: DEFAULT_CONF_FUNCTIONS_STR,
        CONF_USE_INTERACTIVE: DEFAULT_USE_INTERACTIVE,
        CONF_PINECONE_TOP_K: DEFAULT_PINECONE_TOP_K,
        CONF_PINECONE_SCORE_THRESHOLD: None,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    openai.api_key = data[CONF_API_KEY]
    await hass.async_add_executor_job(partial(openai.Engine.list, request_timeout=10))
    pinecone_api_key = data.get(CONF_PINECONE_API_KEY)
    if pinecone_api_key is not None:
        await PineconeStorage(hass, pinecone_api_key).validate()


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        except error.APIConnectionError:
            errors["base"] = "cannot_connect"
        except error.AuthenticationError:
            errors["base"] = "invalid_auth"
        except PineconeAuthenticationError:
            errors["base"] = "pinecone_invalid_auth"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


class OptionsFlow(config_entries.OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )
        schema = openai_config_option_schema(self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def openai_config_option_schema(options: MappingProxyType[str, Any]) -> dict:
    """Return a schema for OpenAI completion options."""
    if not options:
        options = DEFAULT_OPTIONS
    return {
        vol.Optional(
            CONF_PROMPT,
            description={"suggested_value": options[CONF_PROMPT]},
            default=DEFAULT_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={
                # New key in HA 2023.4
                "suggested_value": options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
            },
            default=DEFAULT_CHAT_MODEL,
        ): str,
        vol.Optional(
            CONF_MAX_TOKENS,
            description={"suggested_value": options[CONF_MAX_TOKENS]},
            default=DEFAULT_MAX_TOKENS,
        ): int,
        vol.Optional(
            CONF_TOP_P,
            description={"suggested_value": options[CONF_TOP_P]},
            default=DEFAULT_TOP_P,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options[CONF_TEMPERATURE]},
            default=DEFAULT_TEMPERATURE,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
        vol.Optional(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            description={
                "suggested_value": options[CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION]
            },
            default=DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        ): int,
        vol.Optional(
            CONF_FUNCTIONS,
            description={"suggested_value": options.get(CONF_FUNCTIONS)},
            default=DEFAULT_CONF_FUNCTIONS_STR,
        ): TemplateSelector(),
        vol.Optional(
            CONF_USE_INTERACTIVE,
            description={"suggested_value": options.get(CONF_USE_INTERACTIVE)},
            default=DEFAULT_USE_INTERACTIVE,
        ): BooleanSelector(),
        vol.Optional(
            CONF_PINECONE_TOP_K,
            description={"suggested_value": options.get(CONF_PINECONE_TOP_K)},
            default=DEFAULT_PINECONE_TOP_K,
        ): NumberSelector(
            NumberSelectorConfig(min=0, max=999999, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(
            CONF_PINECONE_SCORE_THRESHOLD,
            description={"suggested_value": options.get(CONF_PINECONE_SCORE_THRESHOLD)},
        ): NumberSelector(
            NumberSelectorConfig(min=0, max=1, step=0.001, mode=NumberSelectorMode.BOX)
        ),
    }
