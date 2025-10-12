"""The OpenAI Conversation integration."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from typing import Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._exceptions import AuthenticationError, OpenAIError
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
import yaml

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_NAME, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

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
    CONF_MEMORY_DEFAULT_NAMESPACE,
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
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_ENABLE_STREAMING,
    DEFAULT_MODEL_STRATEGY,
    DEFAULT_MEMORY_DEFAULT_NAMESPACE,
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
    DEFAULT_PROMPT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SPEAK_CONFIRMATION_FIRST,
    DEFAULT_STREAM_MIN_CHARS,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_RESPONSES_API,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
    MODEL_STRATEGY_AUTO,
    MODEL_STRATEGY_FORCE_CHAT,
    MODEL_STRATEGY_FORCE_RESPONSES,
)
from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import get_function_executor, is_azure, validate_authentication
from .openai_support import (
    detect_chat_tools_support,
    detect_responses_max_tokens_param,
)
from .context_composer import compose_system_sections, estimate_tokens
from .memory_tools import (
    MEMORY_SEARCH_NAME,
    MEMORY_TOOL_SPECS,
    MEMORY_WRITE_NAME,
    MemoryServiceConfig,
    build_memory_tool_definitions,
    dispatch_memory_tool,
    get_memory_service_config,
    is_configured as memory_service_configured,
    async_memory_search,
)
from .router import RouterDecision, classify_intent
from .responses_adapter import model_is_reasoning, responses_to_chat_like
from .services import async_setup_services
from .model_capabilities import ModelCapabilities, detect_model_capabilities

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


# hass.data key for agent.
DATA_AGENT = "agent"

POLICY_TEMPLATE = """Memory Operations Policy:
- On every user turn, run memory.search first; ground answers in high-scoring results (≥ {min_score}).
- When the user provides a durable preference or fact, call memory.write before answering.
- Never role-play saves or recalls—use tools.
- Do not expose memory IDs; summarize politely.
- When forced by router, assume the argument `text` for memory.write is the normalized user statement.
- If streaming is enabled, stream a short confirmation line first, then continue."""


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
        )
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    agent = OpenAIAgent(hass, entry)

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}
        base_url = entry.data.get(CONF_BASE_URL)
        if is_azure(base_url):
            self.client = AsyncAzureOpenAI(
                api_key=entry.data[CONF_API_KEY],
                azure_endpoint=base_url,
                api_version=entry.data.get(CONF_API_VERSION),
                organization=entry.data.get(CONF_ORGANIZATION),
                http_client=get_async_client(hass),
            )
        else:
            self.client = AsyncOpenAI(
                api_key=entry.data[CONF_API_KEY],
                base_url=base_url,
                organization=entry.data.get(CONF_ORGANIZATION),
                http_client=get_async_client(hass),
            )
        # Cache current platform data which gets added to each request (caching done by library)
        _ = hass.async_add_executor_job(self.client.platform_headers)
        self._responses_max_tokens_param = self._detect_responses_max_tokens_param()
        self._chat_supports_tools = self._detect_chat_tools_support()
        self._active_memory_config: MemoryServiceConfig | None = None
        self._model_capabilities_cache: dict[str, ModelCapabilities] = {}
        self._capability_log_signatures: dict[str, tuple[str, ...]] = {}
        self._last_options_signature: tuple[str, ...] | None = None

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    def _should_use_responses_api(
        self,
        model: str,
        capabilities: ModelCapabilities,
        strategy: str,
        use_responses_flag: bool,
    ) -> bool:
        if strategy == MODEL_STRATEGY_FORCE_RESPONSES:
            return capabilities.supports_responses
        if strategy == MODEL_STRATEGY_FORCE_CHAT:
            return False
        if not capabilities.supports_responses:
            return False
        if capabilities.supports_reasoning:
            if use_responses_flag:
                return True
            if capabilities.prefers_responses or model_is_reasoning(model):
                return True
        return False

    def _detect_responses_max_tokens_param(self) -> str | None:
        create_method = getattr(getattr(self.client, "responses", None), "create", None)
        return detect_responses_max_tokens_param(create_method)

    def _detect_chat_tools_support(self) -> bool:
        chat_obj = getattr(self.client, "chat", None)
        completions_obj = getattr(chat_obj, "completions", None)
        create_method = getattr(completions_obj, "create", None)
        return detect_chat_tools_support(create_method)

    def _options_signature(self) -> tuple[str, ...]:
        return tuple(
            sorted((key, repr(value)) for key, value in self.entry.options.items())
        )

    def _get_model_capabilities(self, model: str) -> ModelCapabilities:
        capabilities = self._model_capabilities_cache.get(model)
        if capabilities is None:
            capabilities = detect_model_capabilities(model)
            self._model_capabilities_cache[model] = capabilities
        return capabilities

    def _maybe_log_capabilities(self, model: str, capabilities: ModelCapabilities) -> None:
        signature = self._last_options_signature
        if signature is None:
            return
        logged_signature = self._capability_log_signatures.get(model)
        if logged_signature == signature:
            return
        self._capability_log_signatures[model] = signature
        _LOGGER.info(
            "Model %s capabilities: responses=%s chat=%s reasoning=%s streaming=%s prefers_responses=%s",
            model,
            capabilities.supports_responses,
            capabilities.supports_chat,
            capabilities.supports_reasoning,
            capabilities.supports_streaming,
            capabilities.prefers_responses,
        )

    @staticmethod
    def _pipeline_is_tts(user_input: conversation.ConversationInput) -> bool:
        pipeline = getattr(user_input, "pipeline", None)
        if pipeline is None:
            pipeline = getattr(getattr(user_input, "context", None), "pipeline", None)
        end_stage = getattr(pipeline, "end_stage", None)
        if isinstance(end_stage, str):
            return end_stage.lower() == "tts"
        if hasattr(end_stage, "value"):
            return str(getattr(end_stage, "value")).lower() == "tts"
        if getattr(user_input, "tts_output", None) is not None:
            return True
        return False

    async def _emit_stream_chunk(
        self, user_input: conversation.ConversationInput, text: str
    ) -> None:
        sender = getattr(user_input, "async_stream_output", None)
        if sender is None:
            sender = getattr(user_input, "async_stream_token", None)
        if sender is None or not text:
            return
        if asyncio.iscoroutinefunction(sender):
            await sender(text)
        else:
            sender(text)

    def _messages_to_responses_input(self, messages: list[dict]) -> list[dict]:
        responses_messages: list[dict] = []
        for message in messages:
            cloned = deepcopy(message)
            content = cloned.get("content")
            if isinstance(content, list):
                processed_content = []
                for item in content:
                    if hasattr(item, "model_dump"):
                        processed_content.append(item.model_dump())
                    else:
                        processed_content.append(item)
                cloned["content"] = processed_content
            responses_messages.append(cloned)
        return responses_messages

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        exposed_entities = self.get_exposed_entities()

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            try:
                system_message = self._generate_system_message(
                    exposed_entities, user_input
                )
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [system_message]
        user_message = {"role": "user", "content": user_input.text}
        if self.entry.options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME):
            user = user_input.context.user_id
            if user is not None:
                user_message[ATTR_NAME] = user

        messages.append(user_message)

        try:
            query_response = await self.query(user_input, messages, exposed_entities, 0)
        except OpenAIError as err:
            _LOGGER.error(err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages.append(query_response.message.model_dump(exclude_none=True))
        self.history[conversation_id] = messages

        self.hass.bus.async_fire(
            EVENT_CONVERSATION_FINISHED,
            {
                "response": query_response.response.model_dump(),
                "user_input": user_input,
                "messages": messages,
            },
        )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(query_response.message.content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _generate_system_message(
        self, exposed_entities, user_input: conversation.ConversationInput
    ):
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        prompt = self._async_generate_prompt(raw_prompt, exposed_entities, user_input)
        return {"role": "system", "content": prompt}

    def _async_generate_prompt(
        self,
        raw_prompt: str,
        exposed_entities,
        user_input: conversation.ConversationInput,
    ) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": user_input.device_id,
            },
            parse_result=False,
        )

    def get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                }
            )
        return exposed_entities

    def get_functions(self):
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except:
            raise FunctionLoadFailed()

    async def truncate_message_history(
        self, messages, exposed_entities, user_input: conversation.ConversationInput
    ):
        """Truncate message history."""
        strategy = self.entry.options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        )

        if strategy == "clear":
            last_user_message_index = None
            for i in reversed(range(len(messages))):
                if messages[i]["role"] == "user":
                    last_user_message_index = i
                    break

            if last_user_message_index is not None:
                del messages[1:last_user_message_index]
                # refresh system prompt when all messages are deleted
                messages[0] = self._generate_system_message(
                    exposed_entities, user_input
                )

    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        """Process a sentence."""
        options = self.entry.options
        options_signature = self._options_signature()
        if options_signature != self._last_options_signature:
            self._capability_log_signatures.clear()
            self._last_options_signature = options_signature
        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        strategy = options.get(CONF_MODEL_STRATEGY, DEFAULT_MODEL_STRATEGY)
        stream_min_chars = options.get(
            CONF_STREAM_MIN_CHARS, DEFAULT_STREAM_MIN_CHARS
        )
        speak_confirmation_first = options.get(
            CONF_SPEAK_CONFIRMATION_FIRST, DEFAULT_SPEAK_CONFIRMATION_FIRST
        )
        use_responses_flag = options.get(
            CONF_USE_RESPONSES_API, DEFAULT_USE_RESPONSES_API
        )
        capabilities = self._get_model_capabilities(model)
        self._maybe_log_capabilities(model, capabilities)
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        use_tools = options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)
        context_threshold = options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD)
        enable_streaming = options.get(CONF_ENABLE_STREAMING, DEFAULT_ENABLE_STREAMING)
        proactivity_enabled = options.get(
            CONF_PROACTIVITY_ENABLED, DEFAULT_PROACTIVITY_ENABLED
        )
        proactivity_k = options.get(CONF_PROACTIVITY_K, DEFAULT_PROACTIVITY_K)
        proactivity_min_score = options.get(
            CONF_PROACTIVITY_MIN_SCORE, DEFAULT_PROACTIVITY_MIN_SCORE
        )
        router_force_tools = options.get(
            CONF_ROUTER_FORCE_TOOLS, DEFAULT_ROUTER_FORCE_TOOLS
        )
        write_pattern = options.get(CONF_ROUTER_WRITE_REGEX, DEFAULT_ROUTER_WRITE_REGEX)
        search_pattern = options.get(CONF_ROUTER_SEARCH_REGEX, DEFAULT_ROUTER_SEARCH_REGEX)

        budgets = {
            "profile": options.get(CONF_BUDGET_PROFILE, DEFAULT_BUDGET_PROFILE),
            "scratchpad": options.get(
                CONF_BUDGET_SCRATCHPAD, DEFAULT_BUDGET_SCRATCHPAD
            ),
            "retrieved": options.get(CONF_BUDGET_RETRIEVED, DEFAULT_BUDGET_RETRIEVED),
        }

        memory_config = get_memory_service_config(options)
        self._active_memory_config = memory_config

        router_decision = classify_intent(
            user_input.text or "",
            write_pattern=write_pattern,
            search_pattern=search_pattern,
            force_tools=router_force_tools,
        )
        _LOGGER.debug(
            "Router decision: forced=%s chat_choice=%s responses_choice=%s",
            router_decision.forced_tool,
            router_decision.chat_tool_choice,
            router_decision.responses_tool_choice,
        )

        if router_decision.normalized_text:
            normalized = router_decision.normalized_text
            messages[-1]["content"] = (
                f"{normalized}\n\n(Original utterance: {user_input.text})"
            )

        ha_functions = self.get_functions()
        ha_function_specs = [setting["spec"] for setting in ha_functions]
        all_function_specs = ha_function_specs + MEMORY_TOOL_SPECS

        function_call: str | dict = "auto"
        chat_tool_choice_value: dict | str = router_decision.chat_tool_choice
        responses_tool_choice_value: dict | str = router_decision.responses_tool_choice
        if router_decision.forced_tool:
            function_call = {"name": router_decision.forced_tool}
        elif n_requests >= options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        ):
            function_call = "none"
            chat_tool_choice_value = "none"
            responses_tool_choice_value = "none"
            _LOGGER.debug(
                "Tool-call cap reached (%s) for session %s; forcing tool_choice=none",
                options.get(
                    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                ),
                user_input.conversation_id,
            )

        chat_tool_kwargs: dict = {}
        responses_tool_kwargs: dict = {}

        if all_function_specs:
            tool_definitions = build_memory_tool_definitions()
            if ha_function_specs:
                tool_definitions = [
                    {"type": "function", "function": spec}
                    for spec in ha_function_specs
                ] + tool_definitions

            if use_tools and self._chat_supports_tools:
                chat_tool_kwargs = {"tools": tool_definitions}
                if isinstance(chat_tool_choice_value, (str, dict)):
                    chat_tool_kwargs["tool_choice"] = chat_tool_choice_value
            else:
                chat_tool_kwargs = {
                    "functions": all_function_specs,
                    "function_call": function_call,
                }

            responses_tool_kwargs = {"tools": tool_definitions}
            if isinstance(responses_tool_choice_value, (str, dict)):
                responses_tool_kwargs["tool_choice"] = responses_tool_choice_value
            elif isinstance(function_call, str) and function_call != "auto":
                responses_tool_kwargs["tool_choice"] = function_call

        scratchpad = None
        if len(messages) > 1:
            for existing in reversed(messages[:-1]):
                if existing.get("role") == "assistant" and existing.get("content"):
                    scratchpad = str(existing.get("content"))
                    break

        retrieved_snippets = ""
        profile_digest = ""

        proactive_tasks: list[asyncio.Task] = []
        if proactivity_enabled and memory_service_configured(memory_config):
            search_args = {
                "query": user_input.text,
                "k": proactivity_k,
                "min_score": proactivity_min_score,
            }
            proactive_tasks.append(
                asyncio.create_task(
                    async_memory_search(
                        self.hass,
                        memory_config,
                        search_args,
                        token_budget=budgets["retrieved"],
                    )
                )
            )
            profile_args = {
                "query": "profile summary",
                "k": 1,
                "min_score": proactivity_min_score,
                "namespaces": [options.get(
                    CONF_MEMORY_DEFAULT_NAMESPACE, DEFAULT_MEMORY_DEFAULT_NAMESPACE
                )]
                + ["profile"],
            }
            proactive_tasks.append(
                asyncio.create_task(
                    async_memory_search(
                        self.hass,
                        memory_config,
                        profile_args,
                        token_budget=budgets["profile"],
                    )
                )
            )

        if proactive_tasks:
            results = await asyncio.gather(*proactive_tasks, return_exceptions=True)
            for index, result in enumerate(results):
                if isinstance(result, Exception):
                    _LOGGER.debug("Proactivity task %s failed: %s", index, result)
                    continue
                text, _snippets = result
                if index == 0:
                    retrieved_snippets = text
                else:
                    profile_digest = text

        base_system_content = messages[0].get("content", "")
        for marker in ("\n\nMemory Operations Policy:", "\n\n[Context Composer]"):
            if marker in base_system_content:
                base_system_content = base_system_content.split(marker)[0]

        composition = compose_system_sections(
            profile_digest or None,
            scratchpad,
            retrieved_snippets or None,
            budget_profile=budgets["profile"],
            budget_scratchpad=budgets["scratchpad"],
            budget_retrieved=budgets["retrieved"],
        )

        policy_text = POLICY_TEMPLATE.format(min_score=proactivity_min_score)
        appended_parts = [policy_text]
        if composition.content:
            appended_parts.append(f"[Context Composer]\n{composition.content}")
        messages[0]["content"] = "\n\n".join(filter(None, [base_system_content] + appended_parts))

        _LOGGER.debug(
            "Context slices tokens: profile=%s scratchpad=%s retrieved=%s",
            composition.slices["profile"].tokens,
            composition.slices["scratchpad"].tokens,
            composition.slices["retrieved"].tokens,
        )
        for key, slice_info in composition.slices.items():
            if slice_info.trimmed:
                _LOGGER.debug(
                    "Context slice %s trimmed to %s tokens", key, slice_info.tokens
                )

        _LOGGER.debug("Prompt for %s: %s", model, json.dumps(messages))

        should_use_responses = self._should_use_responses_api(
            model, capabilities, strategy, use_responses_flag
        )

        responses_token_param: str | None = None
        responses_token_value: int | None = None
        max_completion_option = options.get(CONF_MAX_COMPLETION_TOKENS)
        if max_completion_option is not None:
            param_candidate = self._responses_max_tokens_param or "max_output_tokens"
            responses_token_param = param_candidate
            responses_token_value = int(max_completion_option)

        chat_token_param = "max_tokens"
        chat_token_value: int | None = int(max_tokens) if max_tokens is not None else None
        if capabilities.supports_reasoning:
            chat_token_param = "max_completion_tokens"
            fallback_value = options.get(CONF_MAX_COMPLETION_TOKENS, max_tokens)
            chat_token_value = (
                int(fallback_value)
                if fallback_value is not None
                else None
            )

        chosen_path = "responses" if should_use_responses else "chat"
        chosen_token_param = (
            responses_token_param if should_use_responses else chat_token_param
        )
        chosen_token_value = (
            responses_token_value if should_use_responses else chat_token_value
        )
        dry_run = options.get(CONF_DRY_RUN, False)
        would_emit_early_confirmation = (
            enable_streaming
            and speak_confirmation_first
            and router_decision.detected_tool == MEMORY_WRITE_NAME
            and self._pipeline_is_tts(user_input)
        )

        if dry_run:
            _LOGGER.debug(
                "Dry run active – skipping OpenAI call (forced tool=%s, path=%s)",
                router_decision.forced_tool,
                chosen_path,
            )
            return self._dry_run_response(
                model,
                router_decision,
                composition,
                path=chosen_path,
                token_param=chosen_token_param,
                token_value=chosen_token_value,
                chat_tool_choice=chat_tool_choice_value,
                responses_tool_choice=responses_tool_choice_value,
                early_confirmation=would_emit_early_confirmation and should_use_responses,
            )

        if should_use_responses:
            reasoning_effort = options.get(CONF_REASONING_EFFORT, DEFAULT_REASONING_EFFORT)
            _LOGGER.debug(
                "Using Responses API (effort=%s, streaming=%s)",
                reasoning_effort,
                enable_streaming,
            )
            responses_kwargs = {
                "model": model,
                "input": self._messages_to_responses_input(messages),
                "temperature": temperature,
                "top_p": top_p,
                "user": user_input.conversation_id,
            }
            if model_is_reasoning(model):
                responses_kwargs["reasoning"] = {"effort": reasoning_effort}
            responses_kwargs.update(responses_tool_kwargs)
            if responses_token_param and responses_token_value is not None:
                responses_kwargs[responses_token_param] = responses_token_value
                _LOGGER.debug(
                    "Responses max-token param %s=%s",
                    responses_token_param,
                    responses_token_value,
                )
            else:
                _LOGGER.debug(
                    "Responses max-token param not sent (param=%s, value=%s)",
                    responses_token_param or self._responses_max_tokens_param,
                    responses_token_value,
                )

            endpoint_name = "responses"
            try:
                aggregated_parts: list[str] = []
                response_obj = None
                chunk_count = 0
                ack_text: str | None = None
                if enable_streaming and hasattr(self.client.responses, "stream"):
                    ack_text = "Got it — saved." if would_emit_early_confirmation else None
                    if ack_text:
                        aggregated_parts.extend([ack_text, "\n\n"])
                    first_chunk_latency: float | None = None
                    start_time = time.monotonic()
                    buffer_released = False
                    pending_buffer = ""
                    buffer_threshold = max(0, int(stream_min_chars))
                    can_stream = any(
                        hasattr(user_input, attr)
                        for attr in ("async_stream_output", "async_stream_token")
                    )
                    total_stream_tokens = 0
                    chunk_count = 0
                    if ack_text and can_stream:
                        await self._emit_stream_chunk(user_input, ack_text)
                        _LOGGER.debug(
                            "Early confirmation emitted for session %s",
                            user_input.conversation_id,
                        )
                    async with self.client.responses.stream(
                        **responses_kwargs
                    ) as stream:
                        async for event in stream:
                            event_type = getattr(event, "type", None)
                            if event_type == "response.output_text.delta":
                                delta = getattr(event, "delta", "") or ""
                                if delta:
                                    if first_chunk_latency is None:
                                        first_chunk_latency = time.monotonic() - start_time
                                        _LOGGER.debug(
                                            "First streamed token latency %.2fs",
                                            first_chunk_latency,
                                        )
                                    aggregated_parts.append(delta)
                                    pending_buffer += delta
                                    total_stream_tokens += estimate_tokens(delta)
                                    chunk_count += 1
                                    if can_stream:
                                        if (
                                            not buffer_released
                                            and (
                                                buffer_threshold == 0
                                                or len(pending_buffer)
                                                >= buffer_threshold
                                            )
                                        ):
                                            buffer_released = True
                                            release_payload = pending_buffer
                                            await self._emit_stream_chunk(
                                                user_input, release_payload
                                            )
                                            _LOGGER.debug(
                                                "Streaming buffer released after %s chars",
                                                len(release_payload),
                                            )
                                            pending_buffer = ""
                                        elif buffer_released:
                                            await self._emit_stream_chunk(user_input, delta)
                            elif event_type == "response.output_text":
                                text_chunk = getattr(event, "text", "") or ""
                                if text_chunk:
                                    aggregated_parts.append(text_chunk)
                                    pending_buffer += text_chunk
                                    total_stream_tokens += estimate_tokens(text_chunk)
                                    if can_stream and buffer_released:
                                        await self._emit_stream_chunk(user_input, text_chunk)
                            elif event_type == "response.completed":
                                response_obj = getattr(event, "response", None)
                        if can_stream and pending_buffer:
                            await self._emit_stream_chunk(user_input, pending_buffer)
                        if (
                            response_obj is None
                            and hasattr(stream, "get_final_response")
                        ):
                            response_obj = await stream.get_final_response()
                    if response_obj is None:
                        response_obj = await self.client.responses.create(**responses_kwargs)
                else:
                    response_obj = await self.client.responses.create(**responses_kwargs)
                response_dict = responses_to_chat_like(response_obj)
                aggregated_text = "".join(part for part in aggregated_parts if part).strip()
                if aggregated_text:
                    try:
                        response_dict["choices"][0]["message"]["content"] = (
                            aggregated_text
                        )
                    except (KeyError, IndexError):
                        pass
                if enable_streaming and aggregated_parts:
                    total_stream_tokens = sum(
                        estimate_tokens(part)
                        for part in aggregated_parts
                        if isinstance(part, str)
                    )
                    if total_stream_tokens:
                        char_count = sum(
                            len(part)
                            for part in aggregated_parts
                            if isinstance(part, str)
                        )
                        _LOGGER.debug(
                            "Streaming stats: chars=%s (~%s tokens) chunks=%s",
                            char_count,
                            total_stream_tokens,
                            chunk_count if "chunk_count" in locals() else None,
                        )
            except Exception as err:  # noqa: BLE001 - surface friendly response
                _LOGGER.debug(
                    "Error calling %s endpoint: %s", endpoint_name, err
                )
                return self._friendly_error_response(model)
            response: ChatCompletion = ChatCompletion.model_validate(response_dict)
            if response.usage:
                _LOGGER.debug(
                    "Responses usage prompt=%s completion=%s total=%s",
                    getattr(response.usage, "prompt_tokens", None),
                    getattr(response.usage, "completion_tokens", None),
                    getattr(response.usage, "total_tokens", None),
                )
        else:
            _LOGGER.debug("Using Chat Completions")
            endpoint_name = "chat.completions"
            chat_kwargs = {
                "model": model,
                "messages": messages,
                "top_p": top_p,
                "temperature": temperature,
                "user": user_input.conversation_id,
                **chat_tool_kwargs,
            }
            if chat_token_value is not None and chat_token_param:
                chat_kwargs[chat_token_param] = chat_token_value
                _LOGGER.debug(
                    "Chat max-token param %s=%s",
                    chat_token_param,
                    chat_token_value,
                )
            else:
                _LOGGER.debug(
                    "Chat max-token param omitted (param=%s)", chat_token_param
                )
            try:
                response = await self.client.chat.completions.create(**chat_kwargs)
            except Exception as err:  # noqa: BLE001 - surface friendly response
                _LOGGER.debug(
                    "Error calling %s endpoint: %s", endpoint_name, err
                )
                return self._friendly_error_response(model)

        _LOGGER.debug(
            "Response payload %s", json.dumps(response.model_dump(exclude_none=True))
        )

        if response.usage.total_tokens > context_threshold:
            before_trim = len(messages)
            await self.truncate_message_history(
                messages, exposed_entities, user_input
            )
            removed = before_trim - len(messages)
            if removed > 0:
                _LOGGER.debug(
                    "Trimmed context after %s tokens (removed %s message(s))",
                    response.usage.total_tokens,
                    removed,
                )

        choice: Choice = response.choices[0]
        message = choice.message

        tool_calls = getattr(message, "tool_calls", []) or []
        tool_names = [
            getattr(getattr(call, "function", None), "name", None) for call in tool_calls
        ]
        tool_names_str = ", ".join(filter(None, tool_names)) or "none"
        _LOGGER.debug(
            "Model returned %d tool call(s): %s", len(tool_calls), tool_names_str
        )

        if choice.finish_reason == "function_call":
            return await self.execute_function_call(
                user_input, messages, message, exposed_entities, n_requests + 1
            )
        if choice.finish_reason == "tool_calls":
            return await self.execute_tool_calls(
                user_input, messages, message, exposed_entities, n_requests + 1
            )
        if choice.finish_reason == "length":
            raise TokenLengthExceededError(response.usage.completion_tokens)

        return OpenAIQueryResponse(response=response, message=message)

    def _dry_run_response(
        self,
        model: str,
        router_decision: RouterDecision,
        composition,
        *,
        path: str,
        token_param: str | None,
        token_value: int | None,
        chat_tool_choice: dict | str,
        responses_tool_choice: dict | str,
        early_confirmation: bool,
    ) -> "OpenAIQueryResponse":
        details = [
            "Dry run: no model request was issued.",
            f"Path: {path}",
            f"Forced tool: {router_decision.forced_tool or 'auto'}",
            f"Chat tool_choice: {chat_tool_choice}",
            f"Responses tool_choice: {responses_tool_choice}",
        ]
        if token_param and token_value is not None:
            details.append(f"Token param: {token_param}={token_value}")
        else:
            details.append(
                f"Token param: none (param={token_param or 'n/a'})"
            )
        details.append(f"Early confirmation: {early_confirmation}")
        for key in ("profile", "scratchpad", "retrieved"):
            slice_info = composition.slices.get(key)
            if slice_info is None:
                continue
            details.append(
                f"{slice_info.label}: {slice_info.tokens} tokens (trimmed={slice_info.trimmed})"
            )
        message_content = "\n".join(details)
        payload = {
            "id": "dry-run",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": message_content,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        response = ChatCompletion.model_validate(payload)
        return OpenAIQueryResponse(response=response, message=response.choices[0].message)

    def _friendly_error_response(self, model: str) -> "OpenAIQueryResponse":
        message_content = "I ran into a connection error reaching the model."
        payload = {
            "id": "error",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": message_content,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        response = ChatCompletion.model_validate(payload)
        return OpenAIQueryResponse(response=response, message=response.choices[0].message)

    async def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        function_name = message.function_call.name
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return await self.execute_function(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests,
                function,
            )
        if function_name in (MEMORY_WRITE_NAME, MEMORY_SEARCH_NAME):
            try:
                arguments = json.loads(message.function_call.arguments)
            except json.decoder.JSONDecodeError as err:
                raise ParseArgumentsFailed(message.function_call.arguments) from err
            result = await self._execute_memory_tool(function_name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": result,
                }
            )
            return await self.query(user_input, messages, exposed_entities, n_requests)
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        function,
    ) -> OpenAIQueryResponse:
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(message.function_call.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(message.function_call.arguments) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )

        messages.append(
            {
                "role": "function",
                "name": message.function_call.name,
                "content": str(result),
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_calls(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        messages.append(message.model_dump(exclude_none=True))
        for tool in message.tool_calls:
            function_name = tool.function.name
            function = next(
                (s for s in self.get_functions() if s["spec"]["name"] == function_name),
                None,
            )
            if function is not None:
                result = await self.execute_tool_function(
                    user_input,
                    tool,
                    exposed_entities,
                    function,
                )

                messages.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result),
                    }
                )
            elif function_name in (MEMORY_WRITE_NAME, MEMORY_SEARCH_NAME):
                try:
                    arguments = json.loads(tool.function.arguments)
                except json.decoder.JSONDecodeError as err:
                    raise ParseArgumentsFailed(tool.function.arguments) from err
                result = await self._execute_memory_tool(function_name, arguments)
                messages.append(
                    {
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(result),
                    }
                )
            else:
                raise FunctionNotFound(function_name)
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_function(
        self,
        user_input: conversation.ConversationInput,
        tool,
        exposed_entities,
        function,
    ) -> OpenAIQueryResponse:
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(tool.function.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(tool.function.arguments) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )
        return result

    async def _execute_memory_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        config = self._active_memory_config or get_memory_service_config(self.entry.options)
        try:
            return await dispatch_memory_tool(self.hass, config, tool_name, arguments)
        except Exception as err:  # noqa: BLE001 - graceful fallback
            _LOGGER.debug("Memory tool %s failed: %s", tool_name, err)
            return "Memory tool execution failed."


class OpenAIQueryResponse:
    """OpenAI query response value object."""

    def __init__(
        self, response: ChatCompletion, message: ChatCompletionMessage
    ) -> None:
        """Initialize OpenAI query response value object."""
        self.response = response
        self.message = message
