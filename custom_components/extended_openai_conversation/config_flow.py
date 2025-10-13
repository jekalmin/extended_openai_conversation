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
from homeassistant.const import (
    CONF_API_KEY,
    CONF_ENABLED,
    CONF_NAME,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError

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
    CONF_MIN_LOG_LEVEL,
    CONF_MODEL_CAPABILITIES_YAML,
    CONF_NAME,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_ROUTER_FORCE_TOOLS,
    CONF_ROUTER_SEARCH_REGEX,
    CONF_ROUTER_WRITE_REGEX,
    CONF_SKIP_AUTHENTICATION,
    CONF_SPEAK_CONFIRMATION_FIRST,
    CONF_STREAM_MIN_CHARS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_RESPONSES_API,
    CONF_USE_TOOLS,
    CONF_PROACTIVITY_ENABLED,
    CONF_PROACTIVITY_K,
    CONF_PROACTIVITY_MIN_SCORE,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_MODEL,
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
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_RESPONSES_API,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    MODEL_STRATEGY_AUTO,
    MODEL_STRATEGY_OPTIONS,
    MODEL_STRATEGY_FORCE_CHAT,         # <-- added
    MODEL_STRATEGY_FORCE_RESPONSES,    # <-- added
    REASONING_EFFORT_OPTIONS,
)

_LOGGER = logging.getLogger(__name__)


def _make_model_strategy_labels() -> MappingProxyType[str, str]:
    # Nice labels for the options menu.
    labels = {
        MODEL_STRATEGY_AUTO: "Auto (prefer Responses when supported)",
        MODEL_STRATEGY_FORCE_CHAT: "Force Chat Completions",
        MODEL_STRATEGY_FORCE_RESPONSES: "Force Responses",
    }
    return MappingProxyType(labels)


MODEL_STRATEGY_LABELS = _make_model_strategy_labels()


def _selector_select(options: list[SelectOptionDict], mode: SelectSelectorMode = SelectSelectorMode.DROPDOWN) -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=options,
            multiple=False,
            custom_value=False,
            mode=mode,
        )
    )


def _options_for_enum(values: list[str], labels: dict[str, str] | None = None) -> list[SelectOptionDict]:
    out: list[SelectOptionDict] = []
    for v in values:
        out.append(
            SelectOptionDict(
                value=v,
                label=(labels.get(v, v) if labels else v),
            )
        )
    return out


def _default_options_dict() -> dict[str, Any]:
    return {
        CONF_NAME: DEFAULT_NAME,
        CONF_BASE_URL: DEFAULT_BASE_URL,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MODEL_STRATEGY: DEFAULT_MODEL_STRATEGY,
        CONF_ENABLE_STREAMING: DEFAULT_ENABLE_STREAMING,
        CONF_USE_TOOLS: DEFAULT_USE_TOOLS,
        CONF_USE_RESPONSES_API: DEFAULT_USE_RESPONSES_API,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_COMPLETION_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        CONF_ATTACH_USERNAME: DEFAULT_ATTACH_USERNAME,
        CONF_SPEAK_CONFIRMATION_FIRST: DEFAULT_SPEAK_CONFIRMATION_FIRST,
        CONF_STREAM_MIN_CHARS: DEFAULT_STREAM_MIN_CHARS,
        CONF_ROUTER_FORCE_TOOLS: DEFAULT_ROUTER_FORCE_TOOLS,
        CONF_ROUTER_SEARCH_REGEX: DEFAULT_ROUTER_SEARCH_REGEX,
        CONF_ROUTER_WRITE_REGEX: DEFAULT_ROUTER_WRITE_REGEX,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_BUDGET_PROFILE: DEFAULT_BUDGET_PROFILE,
        CONF_BUDGET_RETRIEVED: DEFAULT_BUDGET_RETRIEVED,
        CONF_BUDGET_SCRATCHPAD: DEFAULT_BUDGET_SCRATCHPAD,
        CONF_PROACTIVITY_ENABLED: DEFAULT_PROACTIVITY_ENABLED,
        CONF_PROACTIVITY_K: DEFAULT_PROACTIVITY_K,
        CONF_PROACTIVITY_MIN_SCORE: DEFAULT_PROACTIVITY_MIN_SCORE,
    }


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Extended OpenAI Conversation."""

    VERSION = 1

    def __init__(self) -> None:
        self._errors: dict[str, str] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial step."""
        if user_input is not None:
            # Create the entry immediately; most tuning happens in options.
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME) or DEFAULT_NAME,
                data={
                    CONF_NAME: user_input.get(CONF_NAME, DEFAULT_NAME) or DEFAULT_NAME,
                },
            )

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME, default=DEFAULT_NAME): str,
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=self._errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        return await self.async_step_core(user_input)

    async def async_step_core(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Core OpenAI settings."""
        if user_input is not None:
            # Persist options
            return self.async_create_entry(title="", data=user_input)

        options = {**_default_options_dict(), **self.config_entry.options}

        model_strategy_options = _options_for_enum(MODEL_STRATEGY_OPTIONS, labels=dict(MODEL_STRATEGY_LABELS))

        schema = vol.Schema(
            {
                vol.Required(CONF_BASE_URL, default=options.get(CONF_BASE_URL, DEFAULT_BASE_URL)): str,
                vol.Required(CONF_API_KEY): str,
                vol.Required(CONF_CHAT_MODEL, default=options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)): str,
                vol.Required(
                    CONF_MODEL_STRATEGY,
                    default=options.get(CONF_MODEL_STRATEGY, DEFAULT_MODEL_STRATEGY),
                ): _selector_select(model_strategy_options),
                vol.Required(
                    CONF_ENABLE_STREAMING,
                    default=options.get(CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING),
                ): BooleanSelector(),
                vol.Required(
                    CONF_USE_TOOLS,
                    default=options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS),
                ): BooleanSelector(),
                vol.Required(
                    CONF_USE_RESPONSES_API,
                    default=options.get(CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API),
                ): BooleanSelector(),
                vol.Required(
                    CONF_TEMPERATURE,
                    default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=2.0, step=0.1, mode="box")),
                vol.Required(
                    CONF_TOP_P,
                    default=options.get(CONF_TOP_P, DEFAULT_TOP_P),
                ): NumberSelector(NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="box")),
                vol.Required(
                    CONF_MAX_TOKENS,
                    default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): NumberSelector(NumberSelectorConfig(min=128, max=128000, step=64, mode="box")),
                vol.Required(
                    CONF_MAX_COMPLETION_TOKENS,
                    default=options.get(CONF_MAX_COMPLETION_TOKENS, DEFAULT_MAX_TOKENS),
                ): NumberSelector(NumberSelectorConfig(min=128, max=128000, step=64, mode="box")),
                vol.Required(
                    CONF_CONTEXT_THRESHOLD,
                    default=options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD),
                ): NumberSelector(NumberSelectorConfig(min=256, max=128000, step=64, mode="box")),
                vol.Required(
                    CONF_CONTEXT_TRUNCATE_STRATEGY,
                    default=options.get(CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(value="soft", label="Soft (prefer keeping system and latest turns)"),
                            SelectOptionDict(value="hard", label="Hard (strict token cap)"),
                        ],
                        multiple=False,
                        custom_value=False,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(
                    CONF_ATTACH_USERNAME,
                    default=options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME),
                ): BooleanSelector(),
                vol.Required(
                    CONF_SPEAK_CONFIRMATION_FIRST,
                    default=options.get(CONF_SPEAK_CONFIRMATION_FIRST, DEFAULT_SPEAK_CONFIRMATION_FIRST),
                ): BooleanSelector(),
                vol.Required(
                    CONF_STREAM_MIN_CHARS,
                    default=options.get(CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS),
                ): NumberSelector(NumberSelectorConfig(min=0, max=1000, step=10, mode="box")),
                vol.Optional(
                    CONF_PROMPT,
                    description={"suggested_value": options.get(CONF_PROMPT, "")},
                ): TemplateSelector(),
                vol.Required(
                    CONF_ROUTER_FORCE_TOOLS,
                    default=options.get(CONF_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_FORCE_TOOLS),
                ): BooleanSelector(),
                vol.Required(
                    CONF_ROUTER_SEARCH_REGEX,
                    default=options.get(CONF_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_SEARCH_REGEX),
                ): str,
                vol.Required(
                    CONF_ROUTER_WRITE_REGEX,
                    default=options.get(CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX),
                ): str,
                vol.Required(
                    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                    default=options.get(
                        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                        DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                    ),
                ): NumberSelector(NumberSelectorConfig(min=0, max=50, step=1, mode="box")),
                vol.Required(
                    CONF_BUDGET_PROFILE,
                    default=options.get(CONF_BUDGET_PROFILE, DEFAULT_BUDGET_PROFILE),
                ): NumberSelector(NumberSelectorConfig(min=0, max=32768, step=128, mode="box")),
                vol.Required(
                    CONF_BUDGET_RETRIEVED,
                    default=options.get(CONF_BUDGET_RETRIEVED, DEFAULT_BUDGET_RETRIEVED),
                ): NumberSelector(NumberSelectorConfig(min=0, max=32768, step=128, mode="box")),
                vol.Required(
                    CONF_BUDGET_SCRATCHPAD,
                    default=options.get(CONF_BUDGET_SCRATCHPAD, DEFAULT_BUDGET_SCRATCHPAD),
                ): NumberSelector(NumberSelectorConfig(min=0, max=32768, step=128, mode="box")),
                vol.Required(
                    CONF_PROACTIVITY_ENABLED,
                    default=options.get(CONF_PROACTIVITY_ENABLED, DEFAULT_PROACTIVITY_ENABLED),
                ): BooleanSelector(),
                vol.Required(
                    CONF_PROACTIVITY_K,
                    default=options.get(CONF_PROACTIVITY_K, DEFAULT_PROACTIVITY_K),
                ): NumberSelector(NumberSelectorConfig(min=0, max=50, step=1, mode="box")),
                vol.Required(
                    CONF_PROACTIVITY_MIN_SCORE,
                    default=options.get(CONF_PROACTIVITY_MIN_SCORE, DEFAULT_PROACTIVITY_MIN_SCORE),
                ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.01, mode="box")),
                vol.Optional(CONF_API_VERSION): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(value="v1", label="OpenAI v1 (2024/2025)"),
                            SelectOptionDict(value="legacy", label="Legacy (for older proxies)"),
                        ],
                        multiple=False,
                        custom_value=False,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_SKIP_AUTHENTICATION, default=False): BooleanSelector(),
                vol.Optional(CONF_ENABLE_STREAMING, default=options.get(CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING)): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="core", data_schema=schema, errors={})

    async def async_step_memory(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Optional memory service settings."""
        if user_input is not None:
            merged = {**self.config_entry.options, **user_input}
            return self.async_create_entry(title="", data=merged)

        options = {**_default_options_dict(), **self.config_entry.options}

        schema = vol.Schema(
            {
                vol.Optional(CONF_MEMORY_BASE_URL): str,
                vol.Optional(CONF_MEMORY_API_KEY): str,
                vol.Optional(CONF_MEMORY_DEFAULT_NAMESPACE, default=options.get(CONF_MEMORY_DEFAULT_NAMESPACE, DEFAULT_MEMORY_DEFAULT_NAMESPACE)): str,
                vol.Optional(CONF_MEMORY_WRITE_PATH, default=options.get(CONF_MEMORY_WRITE_PATH, DEFAULT_MEMORY_WRITE_PATH)): str,
                vol.Optional(CONF_MEMORY_SEARCH_PATH, default=options.get(CONF_MEMORY_SEARCH_PATH, DEFAULT_MEMORY_SEARCH_PATH)): str,
                vol.Optional(CONF_MIN_LOG_LEVEL, default="INFO"): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(value="DEBUG", label="DEBUG"),
                            SelectOptionDict(value="INFO", label="INFO"),
                            SelectOptionDict(value="WARNING", label="WARNING"),
                            SelectOptionDict(value="ERROR", label="ERROR"),
                        ],
                        multiple=False,
                        custom_value=False,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(CONF_MODEL_CAPABILITIES_YAML, description={"suggested_value": self.config_entry.options.get(CONF_MODEL_CAPABILITIES_YAML, "")}): str,
                vol.Optional(CONF_DRY_RUN, default=False): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="memory", data_schema=schema, errors={})

    async def async_step_advanced(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Advanced / expert settings."""
        if user_input is not None:
            merged = {**self.config_entry.options, **user_input}
            return self.async_create_entry(title="", data=merged)

        options = {**_default_options_dict(), **self.config_entry.options}

        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_REASONING_EFFORT,
                    description={"suggested_value": options.get(CONF_REASONING_EFFORT)},
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=_options_for_enum(REASONING_EFFORT_OPTIONS),
                        multiple=False,
                        custom_value=False,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
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
                    CONF_ROUTER_WRITE_REGEX,
                    description={
                        "suggested_value": options.get(
                            CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX
                        )
                    },
                    default=DEFAULT_ROUTER_WRITE_REGEX,
                ): str,
                vol.Optional(
                    CONF_DRY_RUN,
                    description={"suggested_value": options.get(CONF_DRY_RUN)},
                    default=False,
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="advanced", data_schema=schema, errors={})

