"""Extended OpenAI Conversation – Home Assistant integration."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI
# Robust exception imports (SDKs move these around)
try:  # pragma: no cover
    from openai._exceptions import AuthenticationError, OpenAIError
except Exception:  # pragma: no cover
    try:
        from openai import AuthenticationError  # type: ignore[attr-defined]
    except Exception:  # very old SDKs
        class AuthenticationError(Exception):  # type: ignore[no-redef]
            ...
    class OpenAIError(Exception):  # type: ignore[no-redef]
        ...

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_NAME  # <-- from HA core
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    # configuration keys / options (defined in your const.py)
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
    CONF_MEMORY_API_KEY,
    CONF_MEMORY_BASE_URL,
    CONF_MEMORY_DEFAULT_NAMESPACE,
    CONF_MEMORY_SEARCH_PATH,
    CONF_MEMORY_WRITE_PATH,
    CONF_MODEL_STRATEGY,
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
    CONF_STREAM_MIN_CHARS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_RESPONSES_API,
    CONF_USE_TOOLS,
    # defaults
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_BUDGET_PROFILE,
    DEFAULT_BUDGET_RETRIEVED,
    DEFAULT_BUDGET_SCRATCHPAD,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_ENABLE_STREAMING,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_STRATEGY,
    DEFAULT_PROACTIVITY_ENABLED,
    DEFAULT_PROACTIVITY_K,
    DEFAULT_PROACTIVITY_MIN_SCORE,
    DEFAULT_ROUTER_FORCE_TOOLS,
    DEFAULT_ROUTER_SEARCH_REGEX,
    DEFAULT_ROUTER_WRITE_REGEX,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_SPEAK_CONFIRMATION_FIRST,
    DEFAULT_STREAM_MIN_CHARS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_RESPONSES_API,
    DEFAULT_USE_TOOLS,
    # misc
    DOMAIN,
    MODEL_STRATEGY_AUTO,
    MODEL_STRATEGY_FORCE_CHAT,
    MODEL_STRATEGY_FORCE_RESPONSES,
)
from .exceptions import TokenLengthExceededError
from .helpers import is_azure, validate_authentication
from .openai_support import detect_chat_tools_support, detect_responses_max_tokens_param
from .context_composer import compose_system_sections, estimate_tokens
from .memory_tools import (
    MemoryServiceConfig,
    build_memory_tool_definitions,
    dispatch_memory_tool,
    get_memory_service_config,
    async_memory_search,
)
from .router import classify_intent
from .responses_adapter import responses_to_chat_like
from .services import async_setup_services
from .model_capabilities import ModelCapabilities, detect_model_capabilities

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
DATA_AGENT = "agent"


async def async_setup(hass: HomeAssistant, config) -> bool:
    """Register services."""
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up from a config entry (Devices & Services)."""

    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY),
            base_url=entry.options.get(CONF_BASE_URL) or entry.data.get(CONF_BASE_URL),
            api_version=entry.options.get(CONF_API_VERSION) or entry.data.get(CONF_API_VERSION),
            organization=entry.options.get(CONF_ORGANIZATION) or entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.options.get(
                CONF_SKIP_AUTHENTICATION,
                entry.data.get(CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION),
            ),
        )
    except AuthenticationError as err:
        _LOGGER.error("Extended OpenAI Conversation: invalid API key: %s", err)
        return False
    except Exception as err:
        # e.g. network or transient OpenAI API outage
        raise ConfigEntryNotReady(err) from err

    agent = OpenAIAgent(hass, entry)

    hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    hass.data[DOMAIN][entry.entry_id][CONF_API_KEY] = (
        entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY)
    )
    hass.data[DOMAIN][entry.entry_id][DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the integration."""
    hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    conversation.async_unset_agent(hass, entry)
    return True


class OpenAIAgent(conversation.AbstractConversationAgent):
    """Conversation agent backed by OpenAI."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry

        base_url = entry.options.get(CONF_BASE_URL) or entry.data.get(CONF_BASE_URL)
        if is_azure(base_url):
            self.client = AsyncAzureOpenAI(
                api_key=entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY),
                azure_endpoint=base_url,
                api_version=entry.options.get(CONF_API_VERSION) or entry.data.get(CONF_API_VERSION),
                organization=entry.options.get(CONF_ORGANIZATION) or entry.data.get(CONF_ORGANIZATION),
                http_client=get_async_client(hass),
            )
        else:
            self.client = AsyncOpenAI(
                api_key=entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY),
                base_url=base_url,
                organization=entry.options.get(CONF_ORGANIZATION) or entry.data.get(CONF_ORGANIZATION),
                http_client=get_async_client(hass),
            )

        # Cache platform headers (library memoizes internally)
        _ = hass.async_add_executor_job(self.client.platform_headers)

        self._responses_max_tokens_param = detect_responses_max_tokens_param()
        self._chat_supports_tools = detect_chat_tools_support()
        self._model_capabilities_cache: dict[str, ModelCapabilities] = {}

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Handle a user utterance."""

        opts = self._options()
        system_sections = compose_system_sections(self.hass, opts)

        # Tool routing intent
        decision = classify_intent(
            user_input.text,
            opts.get(CONF_ROUTER_SEARCH_REGEX),
            opts.get(CONF_ROUTER_WRITE_REGEX),
            force_tools=opts.get(CONF_ROUTER_FORCE_TOOLS),
        )

        # Build messages
        messages: list[dict] = [
            {"role": "system", "content": system_sections},
            {"role": "user", "content": user_input.text},
        ]

        # Proactive memory retrieval (optional)
        if opts.get(CONF_PROACTIVITY_ENABLED, DEFAULT_PROACTIVITY_ENABLED):
            try:
                retrievals = await async_memory_search(
                    self.hass,
                    self._memory_config(opts),
                    {
                        "query": user_input.text,
                        "top_k": opts.get(CONF_PROACTIVITY_K, DEFAULT_PROACTIVITY_K),
                        "min_score": opts.get(
                            CONF_PROACTIVITY_MIN_SCORE, DEFAULT_PROACTIVITY_MIN_SCORE
                        ),
                    },
                )
                if retrievals:
                    messages.append(
                        {"role": "system", "name": "memory_context", "content": retrievals}
                    )
            except Exception as err:
                _LOGGER.debug("Memory search failed (non-fatal): %s", err)

        # Respect context budget (simple truncation strategies)
        try:
            messages = self._apply_truncation(
                messages,
                opts.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD),
                opts.get(CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY),
            )
        except TokenLengthExceededError as err:
            return self._as_error(str(err))

        # Decide API
        model = opts.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        caps = self._detect_capabilities(model)
        strategy = opts.get(CONF_MODEL_STRATEGY, MODEL_STRATEGY_AUTO)
        use_responses = self._should_use_responses_api(
            model, caps, strategy, opts.get(CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API)
        )

        # Generation params
        temperature = opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = opts.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        max_completion_tokens = opts.get(CONF_MAX_COMPLETION_TOKENS, DEFAULT_MAX_TOKENS)
        stream = opts.get(CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING)

        # Build tools if needed
        tools: list[dict] = []
        tool_funcs: list[Any] = []
        if decision.should_use_tools:
            tool_defs, tool_funcs = build_memory_tool_definitions(
                hass=self.hass,
                entry=self.entry,
                memory_config=self._memory_config(opts),
            )
            tools.extend(tool_defs)

        # Invoke model
        try:
            if use_responses:
                result = await self._call_responses(
                    model=model,
                    messages=messages,
                    tools=tools if decision.should_use_tools else None,
                    max_output_tokens=max_completion_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream,
                )
            else:
                result = await self._call_chat(
                    model=model,
                    messages=messages,
                    tools=tools if decision.should_use_tools else None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream,
                )
        except (AuthenticationError, OpenAIError) as err:
            return self._as_error(f"Model call failed: {err}")

        # Tool pass (if any tool calls returned)
        if decision.should_use_tools and result.tool_calls:
            try:
                tool_response = await dispatch_memory_tool(
                    self.hass, self.entry, tool_funcs, result.tool_calls
                )
                if tool_response:
                    messages.append({"role": "tool", "content": tool_response})
                    # quick second pass
                    second = await self._call_chat(
                        model=model,
                        messages=messages,
                        tools=None,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=False,
                    )
                    result = second
            except Exception as err:
                _LOGGER.exception("Tool execution failed: %s", err)

        return conversation.ConversationResult(
            response=conversation.AgentResponse(speech=result.text or "")
        )

    # ---------------- helpers ----------------

    def _options(self) -> dict:
        """Options with defaults (prefer entry.options; fall back to entry.data)."""
        return {
            CONF_NAME: self.entry.data.get(CONF_NAME, "Extended OpenAI Conversation"),
            CONF_BASE_URL: self.entry.options.get(CONF_BASE_URL) or self.entry.data.get(CONF_BASE_URL),
            CONF_CHAT_MODEL: self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            CONF_MODEL_STRATEGY: self.entry.options.get(CONF_MODEL_STRATEGY, DEFAULT_MODEL_STRATEGY),
            CONF_ENABLE_STREAMING: self.entry.options.get(CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING),
            CONF_USE_TOOLS: self.entry.options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS),
            CONF_USE_RESPONSES_API: self.entry.options.get(CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API),
            CONF_TEMPERATURE: self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            CONF_TOP_P: self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            CONF_MAX_TOKENS: self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            CONF_MAX_COMPLETION_TOKENS: self.entry.options.get(CONF_MAX_COMPLETION_TOKENS, DEFAULT_MAX_TOKENS),
            CONF_CONTEXT_THRESHOLD: self.entry.options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD),
            CONF_CONTEXT_TRUNCATE_STRATEGY: self.entry.options.get(
                CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
            ),
            CONF_ATTACH_USERNAME: self.entry.options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME),
            CONF_SPEAK_CONFIRMATION_FIRST: self.entry.options.get(
                CONF_SPEAK_CONFIRMATION_FIRST, DEFAULT_SPEAK_CONFIRMATION_FIRST
            ),
            CONF_STREAM_MIN_CHARS: self.entry.options.get(CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS),
            CONF_PROMPT: self.entry.options.get(CONF_PROMPT, ""),
            CONF_ROUTER_FORCE_TOOLS: self.entry.options.get(CONF_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_FORCE_TOOLS),
            CONF_ROUTER_SEARCH_REGEX: self.entry.options.get(CONF_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_SEARCH_REGEX),
            CONF_ROUTER_WRITE_REGEX: self.entry.options.get(CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX),
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: self.entry.options.get(
                CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION
            ),
            CONF_BUDGET_PROFILE: self.entry.options.get(CONF_BUDGET_PROFILE, DEFAULT_BUDGET_PROFILE),
            CONF_BUDGET_RETRIEVED: self.entry.options.get(CONF_BUDGET_RETRIEVED, DEFAULT_BUDGET_RETRIEVED),
            CONF_BUDGET_SCRATCHPAD: self.entry.options.get(CONF_BUDGET_SCRATCHPAD, DEFAULT_BUDGET_SCRATCHPAD),
            CONF_PROACTIVITY_ENABLED: self.entry.options.get(CONF_PROACTIVITY_ENABLED, DEFAULT_PROACTIVITY_ENABLED),
            CONF_PROACTIVITY_K: self.entry.options.get(CONF_PROACTIVITY_K, DEFAULT_PROACTIVITY_K),
            CONF_PROACTIVITY_MIN_SCORE: self.entry.options.get(CONF_PROACTIVITY_MIN_SCORE, DEFAULT_PROACTIVITY_MIN_SCORE),
            CONF_API_VERSION: self.entry.options.get(CONF_API_VERSION) or self.entry.data.get(CONF_API_VERSION),
            CONF_ORGANIZATION: self.entry.options.get(CONF_ORGANIZATION) or self.entry.data.get(CONF_ORGANIZATION),
            CONF_SKIP_AUTHENTICATION: self.entry.options.get(
                CONF_SKIP_AUTHENTICATION, self.entry.data.get(CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION)
            ),
            CONF_FUNCTIONS: self.entry.options.get(CONF_FUNCTIONS, []),
            # memory service (optional)
            CONF_MEMORY_BASE_URL: self.entry.options.get(CONF_MEMORY_BASE_URL),
            CONF_MEMORY_API_KEY: self.entry.options.get(CONF_MEMORY_API_KEY),
            CONF_MEMORY_DEFAULT_NAMESPACE: self.entry.options.get(
                CONF_MEMORY_DEFAULT_NAMESPACE, DEFAULT_MEMORY_DEFAULT_NAMESPACE
            ),
            CONF_MEMORY_WRITE_PATH: self.entry.options.get(CONF_MEMORY_WRITE_PATH, "/memories/write"),
            CONF_MEMORY_SEARCH_PATH: self.entry.options.get(CONF_MEMORY_SEARCH_PATH, "/memories/search"),
        }

    def _memory_config(self, opts: dict) -> MemoryServiceConfig | None:
        return get_memory_service_config(
            base_url=opts.get(CONF_MEMORY_BASE_URL),
            api_key=opts.get(CONF_MEMORY_API_KEY),
            default_namespace=opts.get(CONF_MEMORY_DEFAULT_NAMESPACE),
            write_path=opts.get(CONF_MEMORY_WRITE_PATH, "/memories/write"),
            search_path=opts.get(CONF_MEMORY_SEARCH_PATH, "/memories/search"),
        )

    def _detect_capabilities(self, model: str | None) -> ModelCapabilities:
        if not model:
            return ModelCapabilities()
        cached = self._model_capabilities_cache.get(model)
        if cached:
            return cached
        caps = detect_model_capabilities(model)
        self._model_capabilities_cache[model] = caps
        return caps

    def _should_use_responses_api(
        self, model: str, caps: ModelCapabilities, strategy: str, use_responses_flag: bool
    ) -> bool:
        if strategy == MODEL_STRATEGY_FORCE_RESPONSES:
            return caps.supports_responses
        if strategy == MODEL_STRATEGY_FORCE_CHAT:
            return False
        if not caps.supports_responses:
            return False
        return use_responses_flag

    def _apply_truncation(self, messages: list[dict], max_tokens: int, strategy_key: str) -> list[dict]:
        """Basic truncation: clear_all | keep_latest | soft(default)."""
        if max_tokens <= 0:
            return messages

        total = estimate_tokens(messages)
        if total <= max_tokens:
            return messages

        if strategy_key == "clear_all":
            return messages[:1] + messages[-1:]

        if strategy_key == "keep_latest":
            base = [messages[0]]
            for i in range(len(messages) - 1, 0, -1):
                if messages[i]["role"] == "user":
                    base.append(messages[i])
                    break
            return base

        # soft – drop oldest after system message
        base = [messages[0]] + messages[1:]
        while estimate_tokens(base) > max_tokens and len(base) > 2:
            base.pop(1)
        return base

    # --------- OpenAI wrappers ---------

    class _Result:
        def __init__(self, text: str, usage: Any | None = None, tool_calls=None):
            self.text = text
            self.usage = usage
            self.tool_calls = tool_calls or []

    async def _call_chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> "_Result":
        completion = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        choice = completion.choices[0]
        text = (getattr(choice, "message", None) and choice.message.content) or ""
        tool_calls = getattr(getattr(choice, "message", None), "tool_calls", None) or []
        return self._Result(text=text, usage=getattr(completion, "usage", None), tool_calls=tool_calls)

    async def _call_responses(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> "_Result":
        # Collapse chat messages into a single string for Responses API
        input_text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)

        params: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "temperature": temperature,
            "top_p": top_p,
        }
        # Use the correct max token param for the installed SDK
        params[detect_responses_max_tokens_param()] = max_output_tokens

        resp = await self.client.responses.create(**params)
        adapted = responses_to_chat_like(resp)
        return self._Result(
            text=adapted.get("text", ""),
            usage=adapted.get("usage"),
            tool_calls=adapted.get("tool_calls") or [],
        )

    # --------- Result helpers ---------

    def _as_error(self, message: str) -> conversation.ConversationResult:
        return conversation.ConversationResult(
            response=conversation.AgentResponse(speech=f"Error: {message}")
        )

