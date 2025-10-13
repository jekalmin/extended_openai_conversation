"""Config flow for OpenAI Conversation integration."""
from __future__ import annotations

import logging
import types
from types import MappingProxyType
from typing import Any

try:  # pragma: no cover - fallback for older OpenAI SDKs
    from openai import APIConnectionError, AuthenticationError
except Exception:  # pragma: no cover - fallback when exceptions move
    APIConnectionError = Exception
    AuthenticationError = Exception
import voluptuous as vol
import yaml

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_BUDGET_PROFILE,
    CONF_BUDGET_RETRIEVED,
    CONF_BUDGET_SCRATCHPAD,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_DRY_RUN,
    CONF_ENABLE_STREAMING,
    CONF_FUNCTIONS,
    CONF_MAX_COMPLETION_TOKENS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_MODEL_STRATEGY,
    CONF_MEMORY_API_KEY,
    CONF_MEMORY_BASE_URL,
    CONF_MEMORY_DEFAULT_NAMESPACE,
    CONF_MEMORY_WRITE_PATH,
    CONF_MEMORY_SEARCH_PATH,
    CONF_ORGANIZATION,
    CONF_PROACTIVITY_ENABLED,
    CONF_PROACTIVITY_K,
    CONF_PROACTIVITY_MIN_SCORE,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_ROUTER_FORCE_TOOLS,
    CONF_ROUTER_SEARCH_REGEX,
    CONF_ROUTER_WRITE_REGEX,
    CONF_SKIP_AUTHENTICATION,
    CONF_SPEAK_CONFIRMATION_FIRST,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_STREAM_MIN_CHARS,
    CONF_USE_RESPONSES_API,
    CONF_USE_TOOLS,
    CONTEXT_TRUNCATE_STRATEGIES,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_ENABLE_STREAMING,
    DEFAULT_MODEL_STRATEGY,
    DEFAULT_MEMORY_DEFAULT_NAMESPACE,
    DEFAULT_MEMORY_WRITE_PATH,
    DEFAULT_MEMORY_SEARCH_PATH,
    DEFAULT_PROACTIVITY_ENABLED,
    DEFAULT_PROACTIVITY_K,
    DEFAULT_PROACTIVITY_MIN_SCORE,
    DEFAULT_ROUTER_FORCE_TOOLS,
    DEFAULT_ROUTER_SEARCH_REGEX,
    DEFAULT_ROUTER_WRITE_REGEX,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_BUDGET_PROFILE,
    DEFAULT_BUDGET_RETRIEVED,
    DEFAULT_BUDGET_SCRATCHPAD,
    DEFAULT_SPEAK_CONFIRMATION_FIRST,
    DEFAULT_STREAM_MIN_CHARS,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_RESPONSES_API,
    DEFAULT_USE_TOOLS,
    MODEL_STRATEGY_AUTO,
    MODEL_STRATEGY_OPTIONS,
    DOMAIN,
    REASONING_EFFORT_OPTIONS,
)
from .helpers import validate_authentication

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): str,
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_BASE_URL, default=DEFAULT_CONF_BASE_URL): str,
        vol.Optional(CONF_API_VERSION): str,
        vol.Optional(CONF_ORGANIZATION): str,
        vol.Optional(
            CONF_SKIP_AUTHENTICATION, default=DEFAULT_SKIP_AUTHENTICATION
        ): bool,
    }
)

DEFAULT_CONF_FUNCTIONS_STR = yaml.dump(DEFAULT_CONF_FUNCTIONS, sort_keys=False)

MODEL_STRATEGY_LABELS = {
    MODEL_STRATEGY_AUTO: "Auto (prefer Responses when supported)",
    MODEL_STRATEGY_FORCE_CHAT: "Force Chat Completions",
    MODEL_STRATEGY_FORCE_RESPONSES: "Force Responses",
}

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_FUNCTIONS: DEFAULT_CONF_FUNCTIONS_STR,
        CONF_ATTACH_USERNAME: DEFAULT_ATTACH_USERNAME,
        CONF_USE_TOOLS: DEFAULT_USE_TOOLS,
        CONF_USE_RESPONSES_API: DEFAULT_USE_RESPONSES_API,
        CONF_MODEL_STRATEGY: DEFAULT_MODEL_STRATEGY,
        CONF_REASONING_EFFORT: DEFAULT_REASONING_EFFORT,
        CONF_MAX_COMPLETION_TOKENS: None,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        CONF_ENABLE_STREAMING: DEFAULT_ENABLE_STREAMING,
        CONF_STREAM_MIN_CHARS: DEFAULT_STREAM_MIN_CHARS,
        CONF_SPEAK_CONFIRMATION_FIRST: DEFAULT_SPEAK_CONFIRMATION_FIRST,
        CONF_PROACTIVITY_ENABLED: DEFAULT_PROACTIVITY_ENABLED,
        CONF_PROACTIVITY_K: DEFAULT_PROACTIVITY_K,
        CONF_PROACTIVITY_MIN_SCORE: DEFAULT_PROACTIVITY_MIN_SCORE,
        CONF_BUDGET_PROFILE: DEFAULT_BUDGET_PROFILE,
        CONF_BUDGET_SCRATCHPAD: DEFAULT_BUDGET_SCRATCHPAD,
        CONF_BUDGET_RETRIEVED: DEFAULT_BUDGET_RETRIEVED,
        CONF_MEMORY_BASE_URL: "",
        CONF_MEMORY_API_KEY: "",
        CONF_MEMORY_DEFAULT_NAMESPACE: DEFAULT_MEMORY_DEFAULT_NAMESPACE,
        CONF_MEMORY_WRITE_PATH: DEFAULT_MEMORY_WRITE_PATH,
        CONF_MEMORY_SEARCH_PATH: DEFAULT_MEMORY_SEARCH_PATH,
        CONF_ROUTER_FORCE_TOOLS: DEFAULT_ROUTER_FORCE_TOOLS,
        CONF_ROUTER_WRITE_REGEX: DEFAULT_ROUTER_WRITE_REGEX,
        CONF_ROUTER_SEARCH_REGEX: DEFAULT_ROUTER_SEARCH_REGEX,
        CONF_DRY_RUN: False,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    api_key = data[CONF_API_KEY]
    base_url = data.get(CONF_BASE_URL)
    api_version = data.get(CONF_API_VERSION)
    organization = data.get(CONF_ORGANIZATION)
    skip_authentication = data.get(CONF_SKIP_AUTHENTICATION)

    if base_url == DEFAULT_CONF_BASE_URL:
        # Do not set base_url if using OpenAI for case of OpenAI's base_url change
        base_url = None
        data.pop(CONF_BASE_URL)

    await validate_authentication(
        hass=hass,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        organization=organization,
        skip_authentication=skip_authentication,
    )


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
        except APIConnectionError:
            errors["base"] = "cannot_connect"
        except AuthenticationError:
            errors["base"] = "invalid_auth"
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
        schema = self.openai_config_option_schema(self.config_entry.options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )

    def openai_config_option_schema(self, options: MappingProxyType[str, Any]) -> dict:
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
                CONF_ATTACH_USERNAME,
                description={"suggested_value": options.get(CONF_ATTACH_USERNAME)},
                default=DEFAULT_ATTACH_USERNAME,
            ): BooleanSelector(),
            vol.Optional(
                CONF_USE_TOOLS,
                description={"suggested_value": options.get(CONF_USE_TOOLS)},
                default=DEFAULT_USE_TOOLS,
            ): BooleanSelector(),
            vol.Optional(
                CONF_USE_RESPONSES_API,
                description={
                    "suggested_value": options.get(
                        CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API
                    )
                },
                default=DEFAULT_USE_RESPONSES_API,
            ): BooleanSelector(),
            vol.Optional(
                CONF_MODEL_STRATEGY,
                description={
                    "suggested_value": options.get(
                        CONF_MODEL_STRATEGY, DEFAULT_MODEL_STRATEGY
                    )
                },
                default=DEFAULT_MODEL_STRATEGY,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(
                            value=strategy,
                            label=MODEL_STRATEGY_LABELS.get(strategy, strategy),
                        )
                        for strategy in MODEL_STRATEGY_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_REASONING_EFFORT,
                description={
                    "suggested_value": options.get(
                        CONF_REASONING_EFFORT, DEFAULT_REASONING_EFFORT
                    )
                },
                default=DEFAULT_REASONING_EFFORT,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=effort, label=effort.title())
                        for effort in REASONING_EFFORT_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_MAX_COMPLETION_TOKENS,
                description={
                    "suggested_value": options.get(CONF_MAX_COMPLETION_TOKENS)
                },
            ): NumberSelector(NumberSelectorConfig(min=1)),
            vol.Optional(
                CONF_CONTEXT_THRESHOLD,
                description={"suggested_value": options.get(CONF_CONTEXT_THRESHOLD)},
                default=DEFAULT_CONTEXT_THRESHOLD,
            ): int,
            vol.Optional(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                description={
                    "suggested_value": options.get(CONF_CONTEXT_TRUNCATE_STRATEGY)
                },
                default=DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=strategy["key"], label=strategy["label"])
                        for strategy in CONTEXT_TRUNCATE_STRATEGIES
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_ENABLE_STREAMING,
                description={"suggested_value": options.get(CONF_ENABLE_STREAMING)},
                default=DEFAULT_ENABLE_STREAMING,
            ): BooleanSelector(),
            vol.Optional(
                CONF_STREAM_MIN_CHARS,
                description={
                    "suggested_value": options.get(
                        CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS
                    )
                },
                default=DEFAULT_STREAM_MIN_CHARS,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1000, step=10)),
            vol.Optional(
                CONF_SPEAK_CONFIRMATION_FIRST,
                description={
                    "suggested_value": options.get(
                        CONF_SPEAK_CONFIRMATION_FIRST,
                        DEFAULT_SPEAK_CONFIRMATION_FIRST,
                    )
                },
                default=DEFAULT_SPEAK_CONFIRMATION_FIRST,
            ): BooleanSelector(),
            vol.Optional(
                CONF_PROACTIVITY_ENABLED,
                description={
                    "suggested_value": options.get(
                        CONF_PROACTIVITY_ENABLED, DEFAULT_PROACTIVITY_ENABLED
                    )
                },
                default=DEFAULT_PROACTIVITY_ENABLED,
            ): BooleanSelector(),
            vol.Optional(
                CONF_PROACTIVITY_K,
                description={"suggested_value": options.get(CONF_PROACTIVITY_K)},
                default=DEFAULT_PROACTIVITY_K,
            ): int,
            vol.Optional(
                CONF_PROACTIVITY_MIN_SCORE,
                description={
                    "suggested_value": options.get(
                        CONF_PROACTIVITY_MIN_SCORE, DEFAULT_PROACTIVITY_MIN_SCORE
                    )
                },
                default=DEFAULT_PROACTIVITY_MIN_SCORE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.01)),
            vol.Optional(
                CONF_BUDGET_PROFILE,
                description={"suggested_value": options.get(CONF_BUDGET_PROFILE)},
                default=DEFAULT_BUDGET_PROFILE,
            ): int,
            vol.Optional(
                CONF_BUDGET_SCRATCHPAD,
                description={
                    "suggested_value": options.get(
                        CONF_BUDGET_SCRATCHPAD, DEFAULT_BUDGET_SCRATCHPAD
                    )
                },
                default=DEFAULT_BUDGET_SCRATCHPAD,
            ): int,
            vol.Optional(
                CONF_BUDGET_RETRIEVED,
                description={
                    "suggested_value": options.get(
                        CONF_BUDGET_RETRIEVED, DEFAULT_BUDGET_RETRIEVED
                    )
                },
                default=DEFAULT_BUDGET_RETRIEVED,
            ): int,
            vol.Optional(
                CONF_MEMORY_BASE_URL,
                description={"suggested_value": options.get(CONF_MEMORY_BASE_URL)},
            ): str,
            vol.Optional(
                CONF_MEMORY_API_KEY,
                description={"suggested_value": options.get(CONF_MEMORY_API_KEY)},
            ): str,
            vol.Optional(
                CONF_MEMORY_DEFAULT_NAMESPACE,
                description={
                    "suggested_value": options.get(
                        CONF_MEMORY_DEFAULT_NAMESPACE, DEFAULT_MEMORY_DEFAULT_NAMESPACE
                    )
                },
                default=DEFAULT_MEMORY_DEFAULT_NAMESPACE,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value="corpus", label="Corpus"),
                        SelectOptionDict(value="profile", label="Profile"),
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_MEMORY_WRITE_PATH,
                description={
                    "suggested_value": options.get(
                        CONF_MEMORY_WRITE_PATH, DEFAULT_MEMORY_WRITE_PATH
                    )
                },
                default=DEFAULT_MEMORY_WRITE_PATH,
            ): str,
            vol.Optional(
                CONF_MEMORY_SEARCH_PATH,
                description={
                    "suggested_value": options.get(
                        CONF_MEMORY_SEARCH_PATH, DEFAULT_MEMORY_SEARCH_PATH
                    )
                },
                default=DEFAULT_MEMORY_SEARCH_PATH,
            ): str,
            vol.Optional(
                CONF_ROUTER_FORCE_TOOLS,
                description={
                    "suggested_value": options.get(
                        CONF_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_FORCE_TOOLS
                    )
                },
                default=DEFAULT_ROUTER_FORCE_TOOLS,
            ): BooleanSelector(),
            vol.Optional(
                CONF_ROUTER_WRITE_REGEX,
                description={
                    "suggested_value": options.get(
                        CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX
                    )
                },
                default=DEFAULT_ROUTER_WRITE_REGEX,
            ): str,
            vol.Optional(
                CONF_ROUTER_SEARCH_REGEX,
                description={
                    "suggested_value": options.get(
                        CONF_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_SEARCH_REGEX
                    )
                },
                default=DEFAULT_ROUTER_SEARCH_REGEX,
            ): str,
            vol.Optional(
                CONF_DRY_RUN,
                description={"suggested_value": options.get(CONF_DRY_RUN)},
                default=False,
            ): BooleanSelector(),
        }
