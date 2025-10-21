"""Extended OpenAI Conversation – GPT‑5 ready, non‑streaming Responses, HA/Assist‑compatible result shape, with entry migration."""

from __future__ import annotations

import logging
from typing import Any, Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI
# Exception compatibility across SDK versions
try:
    from openai._exceptions import AuthenticationError, OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    try:
        from openai import AuthenticationError  # type: ignore[attr-defined]
    except Exception:
        class AuthenticationError(Exception):  # type: ignore[no-redef]
            pass
    try:
        from openai import OpenAIError  # type: ignore[attr-defined]
    except Exception:
        class OpenAIError(Exception):  # type: ignore[no-redef]
            pass

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.components import conversation

from .const import (
    DOMAIN,
    # auth / endpoints
    CONF_BASE_URL, CONF_API_VERSION, CONF_API_KEY, CONF_ORGANIZATION, CONF_SKIP_AUTHENTICATION,
    # model + strategy
    CONF_CHAT_MODEL, CONF_MODEL_STRATEGY, MODEL_STRATEGY_FORCE_CHAT, MODEL_STRATEGY_FORCE_RESPONSES,
    DEFAULT_CHAT_MODEL, DEFAULT_MODEL_STRATEGY,
    # generation
    CONF_ENABLE_STREAMING, CONF_TEMPERATURE, CONF_TOP_P, CONF_MAX_TOKENS, CONF_MAX_COMPLETION_TOKENS,
    DEFAULT_ENABLE_STREAMING, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    # responses vs chat
    CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API,
    # routing
    CONF_USE_TOOLS, DEFAULT_USE_TOOLS,
    CONF_ROUTER_FORCE_TOOLS, CONF_ROUTER_SEARCH_REGEX, CONF_ROUTER_WRITE_REGEX,
    DEFAULT_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_WRITE_REGEX,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    # truncation
    CONF_CONTEXT_THRESHOLD, CONF_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    # persona / formatting
    CONF_PROMPT, DEFAULT_PROMPT,
    CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME,
    CONF_SPEAK_CONFIRMATION_FIRST, DEFAULT_SPEAK_CONFIRMATION_FIRST,
    CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS,
    # budgets (not actively used until memory is on)
    CONF_BUDGET_PROFILE, CONF_BUDGET_RETRIEVED, CONF_BUDGET_SCRATCHPAD,
    DEFAULT_BUDGET_PROFILE, DEFAULT_BUDGET_RETRIEVED, DEFAULT_BUDGET_SCRATCHPAD,
    # proactivity + memory (dormant)
    CONF_PROACTIVITY_ENABLED, CONF_PROACTIVITY_K, CONF_PROACTIVITY_MIN_SCORE,
    DEFAULT_PROACTIVITY_ENABLED, DEFAULT_PROACTIVITY_K, DEFAULT_PROACTIVITY_MIN_SCORE,
    CONF_MEMORY_BASE_URL, CONF_MEMORY_API_KEY, CONF_MEMORY_DEFAULT_NAMESPACE, CONF_MEMORY_SEARCH_PATH, CONF_MEMORY_WRITE_PATH,
    DEFAULT_MEMORY_DEFAULT_NAMESPACE,
    # misc
    CONF_NAME,
    # reasoning
    CONF_REASONING_EFFORT,
)
from .model_capabilities import ModelCapabilities, detect_model_capabilities
from .exceptions import TokenLengthExceededError
from .helpers import is_azure, validate_authentication
from .openai_support import detect_chat_tools_support, detect_responses_max_tokens_param
from .context_composer import compose_system_sections, estimate_tokens
from .router import classify_intent

_LOGGER = logging.getLogger(__name__)

# ---------- Response objects that match HA’s Assist expectations ----------

class _AgentResponseCompat:
    __slots__ = ("speech", "response_type", "language", "data")
    def __init__(self, text: str, *, response_type: str = "agent", language: Optional[str] = None, data: Optional[dict] = None):
        self.speech = {"plain": {"speech": text or "", "extra_data": None}}
        self.response_type = response_type
        self.language = language
        self.data = data or {}

    def as_dict(self) -> dict:
        return {
            "response_type": self.response_type,
            "speech": self.speech,
            "language": self.language,
            "data": self.data,
        }

class _ConversationResultCompat:
    __slots__ = ("response", "language", "conversation_id", "continue_conversation")
    def __init__(self, response: _AgentResponseCompat, *, language: Optional[str], conversation_id: Optional[str], continue_conversation: bool = False):
        self.response = response
        self.language = language
        self.conversation_id = conversation_id
        self.continue_conversation = continue_conversation

    def as_dict(self) -> dict:
        return {
            "response": self.response.as_dict(),
            "language": self.language,
            "conversation_id": self.conversation_id,
            "continue_conversation": self.continue_conversation,
        }

def _build_result(text: str, *, language: Optional[str], conversation_id: Optional[str]):
    cont = bool(text.strip().endswith("?"))
    return _ConversationResultCompat(
        _AgentResponseCompat(text=text, language=language),
        language=language,
        conversation_id=conversation_id,
        continue_conversation=cont,
    )

# ---------- HA setup + migration ----------

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
DATA_AGENT = "agent"

async def async_setup(hass: HomeAssistant, config) -> bool:
    return True

async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Normalize any previous config-entry versions to our target (1)."""
    target = 1
    if entry.version == target:
        return True
    _LOGGER.info("Migrating %s entry '%s' from v%s to v%s", DOMAIN, entry.title, entry.version, target)
    try:
        # Do not alter data/options; just normalize version.
        hass.config_entries.async_update_entry(entry, data=entry.data, options=entry.options, version=target)
        _LOGGER.info("Migration complete for '%s'", entry.title)
        return True
    except Exception:  # be explicit in logs; let HA show banner if False
        _LOGGER.exception("Migration failed for '%s'", entry.title)
        return False

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    api_key = entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY)
    base_url = entry.options.get(CONF_BASE_URL) or entry.data.get(CONF_BASE_URL)
    api_version = entry.options.get(CONF_API_VERSION) or entry.data.get(CONF_API_VERSION)
    organization = entry.options.get(CONF_ORGANIZATION) or entry.data.get(CONF_ORGANIZATION)
    skip_auth = entry.options.get(CONF_SKIP_AUTHENTICATION, entry.data.get(CONF_SKIP_AUTHENTICATION, False))

    if not skip_auth:
        try:
            await validate_authentication(
                hass=hass, api_key=api_key, base_url=base_url, api_version=api_version, organization=organization
            )
        except AuthenticationError as err:
            _LOGGER.error("Extended OpenAI Conversation: auth check failed: %s", err)

    agent = OpenAIAgent(hass, entry)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = {DATA_AGENT: agent}
    conversation.async_set_agent(hass, entry, agent)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    conversation.async_unset_agent(hass, entry)
    hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    return True

# ---------- Router compatibility ----------

def _classify_intent_compat(text: str, opts: dict) -> bool:
    """Support both new and legacy classify_intent signatures; fallback to regex."""
    try:
        decision = classify_intent(text)  # type: ignore[misc]
        if hasattr(decision, "should_use_tools"):
            return bool(decision.should_use_tools)
        if isinstance(decision, bool):
            return decision
    except TypeError:
        pass
    try:
        decision = classify_intent(
            text,
            opts.get(CONF_ROUTER_SEARCH_REGEX),
            opts.get(CONF_ROUTER_WRITE_REGEX),
            force_tools=opts.get(CONF_ROUTER_FORCE_TOOLS),
        )
        if hasattr(decision, "should_use_tools"):
            return bool(decision.should_use_tools)
        if isinstance(decision, bool):
            return decision
    except Exception:
        pass
    import re
    if opts.get(CONF_ROUTER_FORCE_TOOLS):
        return True
    try:
        if (pat := opts.get(CONF_ROUTER_WRITE_REGEX)) and re.search(pat, text or "", re.IGNORECASE):
            return True
        if (pat := opts.get(CONF_ROUTER_SEARCH_REGEX)) and re.search(pat, text or "", re.IGNORECASE):
            return True
    except re.error:
        return False
    return False

# ---------- Agent ----------

class OpenAIAgent(conversation.AbstractConversationAgent):
    """Conversation agent backed by OpenAI (Responses API preferred)."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry

        base_url = entry.options.get(CONF_BASE_URL) or entry.data.get(CONF_BASE_URL)
        api_key = entry.options.get(CONF_API_KEY) or entry.data.get(CONF_API_KEY)
        api_version = entry.options.get(CONF_API_VERSION) or entry.data.get(CONF_API_VERSION)
        organization = entry.options.get(CONF_ORGANIZATION) or entry.data.get(CONF_ORGANIZATION)

        http_client = get_async_client(hass)  # avoids blocking SSL CA load on event loop

        if is_azure(base_url):
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                organization=organization,
                http_client=http_client,
            )
            chat_create = self.client.chat.completions.create
            responses_create = self.client.responses.create
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=organization,
                http_client=http_client,
            )
            chat_create = self.client.chat.completions.create
            responses_create = self.client.responses.create

        self._responses_max_tokens_param = detect_responses_max_tokens_param(responses_create) or "max_output_tokens"
        self._chat_supports_tools = detect_chat_tools_support(chat_create)
        self._model_capabilities_cache: dict[str, ModelCapabilities] = {}

    @property
    def supported_languages(self) -> list[str]:
        return [
            "en","en-US","en-GB","en-CA","en-AU","es","es-ES","es-MX","fr","fr-FR","fr-CA","de","it","pt","pt-BR",
            "nl","sv","no","da","fi","pl","cs","sk","sl","hu","ro","bg","el","tr","ru","uk","he","ar","fa","hi","bn",
            "ur","zh","zh-CN","zh-TW","ja","ko","th","vi","id","ms","fil",
        ]

    async def async_process(self, user_input: conversation.ConversationInput) -> conversation.ConversationResult:
        opts = self._options()
        language = getattr(user_input, "language", None)
        conv_id = getattr(user_input, "conversation_id", None)

        # 1) Compose system prompt (memory/retrieval OFF)
        try:
            sections = self._compose_sections_compat(
                opts=opts,
                retrieved_text="",  # dormant memory
                budget_profile=opts.get(CONF_BUDGET_PROFILE, DEFAULT_BUDGET_PROFILE),
                budget_scratchpad=opts.get(CONF_BUDGET_SCRATCHPAD, DEFAULT_BUDGET_SCRATCHPAD),
                budget_retrieved=opts.get(CONF_BUDGET_RETRIEVED, DEFAULT_BUDGET_RETRIEVED),
            )
        except Exception as err:
            return _build_result(f"Error: Prompt assembly failed: {err}", language=language, conversation_id=conv_id)

        system_text = (getattr(sections, "system", "") or "").strip()
        extra = (getattr(sections, "content", "") or "").strip()
        if extra:
            system_text = (system_text + ("\n\n" if system_text else "") + extra).strip()

        # 2) Messages
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_input.text},
        ]

        # 3) Truncation
        try:
            messages = self._apply_truncation(
                messages,
                opts.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD),
                opts.get(CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY),
            )
        except TokenLengthExceededError as err:
            return _build_result(f"Error: {err}", language=language, conversation_id=conv_id)

        # 4) Tool routing decision (future; not enforced now)
        _ = _classify_intent_compat(user_input.text or "", opts)

        # 5) Choose API path
        model = opts.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        caps = self._detect_capabilities(model)
        use_responses = self._should_use_responses_api(
            model, caps, opts.get(CONF_MODEL_STRATEGY, DEFAULT_MODEL_STRATEGY), opts.get(CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API)
        )

        # 6) Common params
        temperature = opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        top_p = opts.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_tokens = opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        max_completion_tokens = opts.get(CONF_MAX_COMPLETION_TOKENS, DEFAULT_MAX_TOKENS)
        # Force non-stream for now to guarantee text
        stream = False

        # 7) Call model
        try:
            if use_responses:
                result = await self._call_responses(
                    model=model, messages=messages,
                    max_output_tokens=max_completion_tokens,
                    temperature=temperature, top_p=top_p, stream=stream,
                    reasoning_effort=(opts.get(CONF_REASONING_EFFORT) or "").strip().lower(),
                )
            else:
                result = await self._call_chat(
                    model=model, messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, stream=stream,
                )
        except OpenAIError as err:
            return _build_result(f"Error: Model call failed: {err}", language=language, conversation_id=conv_id)
        except Exception as err:
            _LOGGER.exception("Unexpected exception during model call")
            return _build_result(f"Error: Unexpected error: {err}", language=language, conversation_id=conv_id)

        text = result.text or ""
        return _build_result(text, language=language, conversation_id=conv_id)

    # ----- helpers -----

    def _options(self) -> dict:
        e = self.entry
        return {
            CONF_NAME: e.data.get(CONF_NAME, "Extended OpenAI Conversation"),
            CONF_BASE_URL: e.options.get(CONF_BASE_URL) or e.data.get(CONF_BASE_URL),
            CONF_CHAT_MODEL: e.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            CONF_MODEL_STRATEGY: e.options.get(CONF_MODEL_STRATEGY, DEFAULT_MODEL_STRATEGY),
            CONF_ENABLE_STREAMING: e.options.get(CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING),
            CONF_USE_TOOLS: e.options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS),
            CONF_USE_RESPONSES_API: e.options.get(CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API),
            CONF_TEMPERATURE: e.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            CONF_TOP_P: e.options.get(CONF_TOP_P, DEFAULT_TOP_P),
            CONF_MAX_TOKENS: e.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            CONF_MAX_COMPLETION_TOKENS: e.options.get(CONF_MAX_COMPLETION_TOKENS, DEFAULT_MAX_TOKENS),
            CONF_CONTEXT_THRESHOLD: e.options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD),
            CONF_CONTEXT_TRUNCATE_STRATEGY: e.options.get(CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY),
            CONF_ATTACH_USERNAME: e.options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME),
            CONF_SPEAK_CONFIRMATION_FIRST: e.options.get(CONF_SPEAK_CONFIRMATION_FIRST, DEFAULT_SPEAK_CONFIRMATION_FIRST),
            CONF_STREAM_MIN_CHARS: e.options.get(CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS),
            CONF_PROMPT: e.options.get(CONF_PROMPT, DEFAULT_PROMPT),
            CONF_ROUTER_FORCE_TOOLS: e.options.get(CONF_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_FORCE_TOOLS),
            CONF_ROUTER_SEARCH_REGEX: e.options.get(CONF_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_SEARCH_REGEX),
            CONF_ROUTER_WRITE_REGEX: e.options.get(CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX),
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: e.options.get(CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION, DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION),
            CONF_BUDGET_PROFILE: e.options.get(CONF_BUDGET_PROFILE, DEFAULT_BUDGET_PROFILE),
            CONF_BUDGET_RETRIEVED: e.options.get(CONF_BUDGET_RETRIEVED, DEFAULT_BUDGET_RETRIEVED),
            CONF_BUDGET_SCRATCHPAD: e.options.get(CONF_BUDGET_SCRATCHPAD, DEFAULT_BUDGET_SCRATCHPAD),
            CONF_PROACTIVITY_ENABLED: e.options.get(CONF_PROACTIVITY_ENABLED, DEFAULT_PROACTIVITY_ENABLED),
            CONF_PROACTIVITY_K: e.options.get(CONF_PROACTIVITY_K, DEFAULT_PROACTIVITY_K),
            CONF_PROACTIVITY_MIN_SCORE: e.options.get(CONF_PROACTIVITY_MIN_SCORE, DEFAULT_PROACTIVITY_MIN_SCORE),
            CONF_REASONING_EFFORT: e.options.get(CONF_REASONING_EFFORT),
            # memory placeholders (unused here)
            CONF_MEMORY_BASE_URL: e.options.get(CONF_MEMORY_BASE_URL),
            CONF_MEMORY_API_KEY: e.options.get(CONF_MEMORY_API_KEY),
            CONF_MEMORY_DEFAULT_NAMESPACE: e.options.get(CONF_MEMORY_DEFAULT_NAMESPACE, DEFAULT_MEMORY_DEFAULT_NAMESPACE),
            CONF_MEMORY_WRITE_PATH: e.options.get(CONF_MEMORY_WRITE_PATH, "/memories/write"),
            CONF_MEMORY_SEARCH_PATH: e.options.get(CONF_MEMORY_SEARCH_PATH, "/memories/search"),
        }

    def _detect_capabilities(self, model: str | None) -> ModelCapabilities:
        if not model:
            return ModelCapabilities()
        cached = self._model_capabilities_cache.get(model)
        if cached:
            return cached
        caps = detect_model_capabilities(model)
        self._model_capabilities_cache[model] = caps
        return caps

    def _should_use_responses_api(self, model: str, caps: ModelCapabilities, strategy: str, use_responses_flag: bool) -> bool:
        if (model or "").lower().startswith("gpt-5"):
            return True  # GPT‑5 → Responses API
        if strategy == MODEL_STRATEGY_FORCE_RESPONSES:
            return caps.supports_responses
        if strategy == MODEL_STRATEGY_FORCE_CHAT:
            return False
        return caps.supports_responses and use_responses_flag

    def _apply_truncation(self, messages: list[dict], max_tokens: int, strategy_key: str) -> list[dict]:
        if max_tokens <= 0:
            return messages
        total = estimate_tokens(messages)
        if total <= max_tokens:
            return messages
        if strategy_key == "clear_all":
            return messages[:1] + messages[-1:]
        if strategy_key == "keep_latest":
            return [messages[0], messages[-1]]
        return [messages[0], messages[-1]]

    class _Result:
        def __init__(self, text: str, usage: dict | None = None, tool_calls: list | None = None):
            self.text = text
            self.usage = usage or {}
            self.tool_calls = tool_calls or []

    async def _call_chat(self, *, model: str, messages: list[dict], max_tokens: int, temperature: float, top_p: float, stream: bool) -> "_Result":
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if not model.lower().startswith("gpt-5"):
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        comp = await self.client.chat.completions.create(**kwargs)
        choice = comp.choices[0]
        text = (getattr(choice, "message", None) and choice.message.content) or ""
        tool_calls = getattr(getattr(choice, "message", None), "tool_calls", None) or []
        return self._Result(text=text, usage=getattr(comp, "usage", None), tool_calls=tool_calls)

    async def _call_responses(self, *, model: str, messages: list[dict], max_output_tokens: int, temperature: float, top_p: float, stream: bool, reasoning_effort: str) -> "_Result":
        input_text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
        params: dict[str, Any] = {"model": model, "input": input_text}
        params[self._responses_max_tokens_param] = max_output_tokens
        is_reasoning = (model or "").lower().startswith("gpt-5")
        if not is_reasoning:
            params["temperature"] = temperature
            params["top_p"] = top_p
        if is_reasoning and reasoning_effort in ("low", "medium", "high"):
            params["reasoning"] = {"effort": reasoning_effort}
        # Force non-streaming until we implement event consumption
        resp = await self.client.responses.create(**params)

        text = getattr(resp, "output_text", None)
        if text is None:
            try:
                from .responses_adapter import responses_to_chat_like  # type: ignore
                text = (responses_to_chat_like(resp) or {}).get("text", "")
            except Exception:
                try:
                    output = getattr(resp, "output", None) or []
                    if output and hasattr(output[0], "content"):
                        content = output[0].content or []
                        if content and hasattr(content[0], "text"):
                            text = getattr(content[0].text, "value", None) or getattr(content[0], "text", "") or ""
                except Exception:
                    text = ""

        return self._Result(text=text or "", usage=getattr(resp, "usage", None), tool_calls=[])

    # --- system prompt composer compatibility ---

    class _Sections:
        __slots__ = ("system", "content")
        def __init__(self, system: str = "", content: str = ""):
            self.system = system
            self.content = content

    def _compose_sections_compat(
        self,
        *,
        opts: dict,
        retrieved_text: str,
        budget_profile: int,
        budget_scratchpad: int,
        budget_retrieved: int,
    ):
        """Try common call orders to support different forks/versions."""
        kw = {
            "budget_profile": budget_profile,
            "budget_scratchpad": budget_scratchpad,
            "budget_retrieved": budget_retrieved,
        }
        try:
            return compose_system_sections(opts, retrieved_text, **kw, hass=self.hass)  # type: ignore[misc]
        except TypeError:
            pass
        try:
            return compose_system_sections(self.hass, opts, retrieved_text, **kw)  # type: ignore[misc]
        except TypeError:
            pass
        try:
            return compose_system_sections(opts, self.hass, retrieved_text, **kw)  # type: ignore[misc]
        except TypeError:
            pass
        try:
            return compose_system_sections(self.hass, opts, retrieved_text)  # type: ignore[misc]
        except TypeError:
            pass
        try:
            return compose_system_sections(opts, retrieved_text)  # type: ignore[misc]
        except Exception as err:
            _LOGGER.debug("compose_system_sections fallback due to %r; returning minimal prompt", err)
            return self._Sections(system=(opts.get(CONF_PROMPT) or ""), content="")
