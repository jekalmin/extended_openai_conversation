from __future__ import annotations

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant

from .const import (
    DOMAIN,
    # Connection/auth
    CONF_API_KEY, CONF_BASE_URL, CONF_API_VERSION, CONF_ORGANIZATION,
    # Model & strategy
    CONF_CHAT_MODEL, CONF_MODEL_STRATEGY, MODEL_STRATEGY_FORCE_CHAT, MODEL_STRATEGY_FORCE_RESPONSES,
    DEFAULT_CHAT_MODEL, DEFAULT_MODEL_STRATEGY,
    # Responses vs Chat
    CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API,
    # Reasoning effort
    CONF_REASONING_EFFORT,
    # Generation
    CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING,
    CONF_TEMPERATURE, DEFAULT_TEMPERATURE,
    CONF_TOP_P, DEFAULT_TOP_P,
    CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS,
    CONF_MAX_COMPLETION_TOKENS,
    # Prompt & presentation
    CONF_PROMPT, DEFAULT_PROMPT,
    CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME,
    CONF_SPEAK_CONFIRMATION_FIRST, DEFAULT_SPEAK_CONFIRMATION_FIRST,
    CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS,
    # Context/truncation
    CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    # Router (kept visible but optional)
    CONF_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_FORCE_TOOLS,
    CONF_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_SEARCH_REGEX,
    CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
)

REASONING_EFFORT_CHOICES = ["", "low", "medium", "high"]  # "" = unset / default
TRUNCATE_STRATEGIES = ["keep_latest", "clear_all"]

def _opt(entry: config_entries.ConfigEntry, key: str, default):
    return entry.options.get(key, entry.data.get(key, default))

class ExtendedOpenAIConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Initial setup: collect connection + a default model."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            # Store these in entry.data; options can override later.
            return self.async_create_entry(
                title="Extended OpenAI Conversation",
                data=user_input,
            )

        schema = vol.Schema({
            vol.Required(CONF_API_KEY): str,
            vol.Optional(CONF_BASE_URL, default="https://api.openai.com/v1"): str,
            vol.Optional(CONF_API_VERSION, default=""): str,  # Azure or compat only
            vol.Optional(CONF_ORGANIZATION, default=""): str,
            vol.Optional(CONF_CHAT_MODEL, default=DEFAULT_CHAT_MODEL): str,
        })
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    @staticmethod
    def async_get_options_flow(entry: config_entries.ConfigEntry):
        return ExtendedOpenAIOptionsFlow(entry)

class ExtendedOpenAIOptionsFlow(config_entries.OptionsFlow):
    """Single-screen options; no deprecated attributes."""

    def __init__(self, entry: config_entries.ConfigEntry) -> None:
        self.entry = entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            # Persist in options only. __init__.py reads options first, then data.
            return self.async_create_entry(title="", data=user_input)

        # Defaults (prefer options, fallback to data, then library defaults)
        schema = vol.Schema({
            # Connection/auth (editable post-setup)
            vol.Optional(CONF_API_KEY, default=_opt(self.entry, CONF_API_KEY, "")): str,
            vol.Optional(CONF_BASE_URL, default=_opt(self.entry, CONF_BASE_URL, "https://api.openai.com/v1")): str,
            vol.Optional(CONF_API_VERSION, default=_opt(self.entry, CONF_API_VERSION, "")): str,
            vol.Optional(CONF_ORGANIZATION, default=_opt(self.entry, CONF_ORGANIZATION, "")): str,

            # Model & strategy
            vol.Optional(CONF_CHAT_MODEL, default=_opt(self.entry, CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)): str,
            vol.Optional(CONF_MODEL_STRATEGY, default=_opt(self.entry, CONF_MODEL_STRATEGY, DEFAULT_MODEL_STRATEGY)): vol.In(
                [MODEL_STRATEGY_FORCE_CHAT, MODEL_STRATEGY_FORCE_RESPONSES, DEFAULT_MODEL_STRATEGY]
            ),
            vol.Optional(CONF_USE_RESPONSES_API, default=_opt(self.entry, CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API)): bool,
            vol.Optional(CONF_REASONING_EFFORT, default=_opt(self.entry, CONF_REASONING_EFFORT, "")): vol.In(REASONING_EFFORT_CHOICES),

            # Generation (note: temperature/top_p are ignored by GPT‑5 reasoning models)
            vol.Optional(CONF_ENABLE_STREAMING, default=_opt(self.entry, CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING)): bool,
            vol.Optional(CONF_TEMPERATURE, default=_opt(self.entry, CONF_TEMPERATURE, DEFAULT_TEMPERATURE)): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(CONF_TOP_P, default=_opt(self.entry, CONF_TOP_P, DEFAULT_TOP_P)): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(CONF_MAX_TOKENS, default=_opt(self.entry, CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)): vol.All(int, vol.Range(min=0)),
            vol.Optional(CONF_MAX_COMPLETION_TOKENS, default=_opt(self.entry, CONF_MAX_COMPLETION_TOKENS, DEFAULT_MAX_TOKENS)): vol.All(int, vol.Range(min=0)),

            # Context/truncation
            vol.Optional(CONF_CONTEXT_THRESHOLD, default=_opt(self.entry, CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD)): vol.All(int, vol.Range(min=0)),
            vol.Optional(CONF_CONTEXT_TRUNCATE_STRATEGY, default=_opt(self.entry, CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY)): vol.In(TRUNCATE_STRATEGIES),

            # Prompt & presentation
            vol.Optional(CONF_PROMPT, default=_opt(self.entry, CONF_PROMPT, DEFAULT_PROMPT)): str,
            vol.Optional(CONF_ATTACH_USERNAME, default=_opt(self.entry, CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME)): bool,
            vol.Optional(CONF_SPEAK_CONFIRMATION_FIRST, default=_opt(self.entry, CONF_SPEAK_CONFIRMATION_FIRST, DEFAULT_SPEAK_CONFIRMATION_FIRST)): bool,
            vol.Optional(CONF_STREAM_MIN_CHARS, default=_opt(self.entry, CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS)): vol.All(int, vol.Range(min=0)),

            # Router (safe defaults; kept off unless you enable)
            vol.Optional(CONF_ROUTER_FORCE_TOOLS, default=_opt(self.entry, CONF_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_FORCE_TOOLS)): bool,
            vol.Optional(CONF_ROUTER_SEARCH_REGEX, default=_opt(self.entry, CONF_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_SEARCH_REGEX)): str,
            vol.Optional(CONF_ROUTER_WRITE_REGEX, default=_opt(self.entry, CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX)): str,
            vol.Optional(CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, default=_opt(self.entry, CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION)): vol.All(int, vol.Range(min=0)),
        })

        return self.async_show_form(step_id="init", data_schema=schema)
