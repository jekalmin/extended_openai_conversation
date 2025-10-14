"""The OpenAI Conversation integration."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

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
from homeassistant.const import ATTR_NAME, CONF_API_KEY, MATCH_ALL, Platform
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
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_FUNCTIONS,
    CONF_PRIMARY_SUPPORTS_VISION,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_VISION_FALLBACK_API_KEY,
    CONF_VISION_FALLBACK_API_VERSION,
    CONF_VISION_FALLBACK_BASE_URL,
    CONF_VISION_FALLBACK_MODEL,
    CONF_VISION_FALLBACK_ORGANIZATION,
    CONF_USE_TOOLS,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_PRIMARY_SUPPORTS_VISION,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import create_openai_client, get_function_executor, validate_authentication
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


# hass.data key for agent.
DATA_AGENT = "agent"

PLATFORMS = [Platform.AI_TASK]


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
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    conversation.async_unset_agent(hass, entry)
    if unload_ok and entry.entry_id in hass.data.get(DOMAIN, {}):
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok


class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent."""

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        """Normalize optional string settings."""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return str(value)

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}
        self.primary_supports_vision = entry.options.get(
            CONF_PRIMARY_SUPPORTS_VISION, DEFAULT_PRIMARY_SUPPORTS_VISION
        )
        self.vision_model: str | None = None
        self.vision_client = None
        self._primary_base_url = entry.data.get(CONF_BASE_URL)
        self._vision_base_url: str | None = None
        base_url = entry.data.get(CONF_BASE_URL)
        self.client = create_openai_client(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=base_url,
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
        )
        # Cache current platform data which gets added to each request (caching done by library)
        _ = hass.async_add_executor_job(self.client.platform_headers)

        fallback_model = self._coerce_optional_str(
            entry.options.get(CONF_VISION_FALLBACK_MODEL)
        )
        if fallback_model:
            fallback_api_key = self._coerce_optional_str(
                entry.options.get(CONF_VISION_FALLBACK_API_KEY)
            ) or entry.data[CONF_API_KEY]
            fallback_base_url = self._coerce_optional_str(
                entry.options.get(CONF_VISION_FALLBACK_BASE_URL)
            )
            fallback_api_version = self._coerce_optional_str(
                entry.options.get(CONF_VISION_FALLBACK_API_VERSION)
            )
            fallback_organization = self._coerce_optional_str(
                entry.options.get(CONF_VISION_FALLBACK_ORGANIZATION)
            )

            fallback_base_url = fallback_base_url or base_url
            fallback_api_version = fallback_api_version or entry.data.get(
                CONF_API_VERSION
            )
            fallback_organization = fallback_organization or entry.data.get(
                CONF_ORGANIZATION
            )

            try:
                self.vision_client = create_openai_client(
                    hass=hass,
                    api_key=fallback_api_key,
                    base_url=fallback_base_url,
                    api_version=fallback_api_version,
                    organization=fallback_organization,
                )
                _ = hass.async_add_executor_job(self.vision_client.platform_headers)
                self.vision_model = fallback_model
                self._vision_base_url = fallback_base_url
            except Exception:  # pylint: disable=broad-except
                self.vision_client = None
                self.vision_model = None
                self._vision_base_url = None
                _LOGGER.exception("Failed to initialize fallback vision client")

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

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

    @staticmethod
    def _messages_require_vision(messages: list[dict[str, Any]]) -> bool:
        """Detect whether any messages include image content parts."""
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    def _apply_provider_overrides(
        self,
        payload: dict[str, Any],
        tool_kwargs: dict[str, Any],
        base_url: str | None,
    ) -> dict[str, Any]:
        """Tweaks request payload/tool configuration for specific providers."""
        if not base_url:
            return tool_kwargs
        base_url_lower = base_url.lower()
        if "groq.com" in base_url_lower:
            # Groq expects `max_completion_tokens` instead of `max_tokens`
            if "max_tokens" in payload and "max_completion_tokens" not in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
            # Groq OpenAI-compatible endpoint rejects the `user` field.
            payload.pop("user", None)
            if "functions" in tool_kwargs:
                functions = tool_kwargs.pop("functions")
                tool_kwargs["tools"] = [
                    {"type": "function", "function": func} for func in functions
                ]
                function_call = tool_kwargs.pop("function_call", "auto")
                if isinstance(function_call, dict):
                    tool_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": function_call,
                    }
                else:
                    tool_kwargs["tool_choice"] = function_call
        elif "api.z.ai" in base_url_lower or "zhipu" in base_url_lower or "bigmodel" in base_url_lower:
            # GLM API expects `user_id` instead of `user`
            if "user" in payload and payload["user"]:
                payload["user_id"] = payload.pop("user")
            else:
                payload.pop("user", None)
            # Ensure max_tokens fits their spec (no rename required but ensure int)
            if "max_tokens" in payload and payload["max_tokens"] is None:
                payload.pop("max_tokens")
            # Map functions/tool params to GLM's `tools` format
            if "functions" in tool_kwargs:
                functions = tool_kwargs.pop("functions")
                tool_kwargs["tools"] = [
                    {"type": "function", "function": func} for func in functions
                ]
                function_call = tool_kwargs.pop("function_call", None)
                if isinstance(function_call, dict):
                    tool_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": function_call,
                    }
                elif function_call is not None:
                    tool_kwargs["tool_choice"] = function_call
            # GLM coding endpoint differs; ensure base path is correct handled via config
        return tool_kwargs

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
        response_format: dict | None = None,
    ) -> OpenAIQueryResponse:
        """Process a sentence."""
        primary_model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        use_tools = self.entry.options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)
        context_threshold = self.entry.options.get(
            CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD
        )
        functions = list(map(lambda s: s["spec"], self.get_functions()))
        function_call = "auto"
        if n_requests == self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        ):
            function_call = "none"

        tool_kwargs = {"functions": functions, "function_call": function_call}
        if use_tools:
            tool_kwargs = {
                "tools": [{"type": "function", "function": func} for func in functions],
                "tool_choice": function_call,
            }

        if len(functions) == 0:
            tool_kwargs = {}

        def _redact_data_urls(message: dict[str, Any]) -> dict[str, Any]:
            """Mask inline base64 data before logging."""
            message_copy = dict(message)
            content = message_copy.get("content")
            if isinstance(content, list):
                redacted_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        redacted_parts.append(part)
                        continue
                    if part.get("type") != "image_url":
                        redacted_parts.append(part)
                        continue
                    image_url = dict(part.get("image_url", {}))
                    url = image_url.get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        image_url["url"] = "<base64 image>"
                        new_part = dict(part)
                        new_part["image_url"] = image_url
                        redacted_parts.append(new_part)
                    else:
                        redacted_parts.append(part)
                message_copy["content"] = redacted_parts
            return message_copy

        requires_vision = self._messages_require_vision(messages)
        client = self.client
        model = primary_model

        if requires_vision:
            if self.primary_supports_vision:
                _LOGGER.debug(
                    "Primary model %s marked as vision-capable, using primary client",
                    primary_model,
                )
            elif self.vision_client and self.vision_model:
                client = self.vision_client
                model = self.vision_model
                _LOGGER.debug(
                    "Routing request with attachments to fallback vision model %s",
                    self.vision_model,
                )
            else:
                raise HomeAssistantError(
                    "Image attachments are not supported by the configured chat model "
                    f"`{primary_model}`. Configure a vision-capable fallback model in the "
                    "integration options before sending images."
                )

        safe_messages = [_redact_data_urls(message) for message in messages]
        _LOGGER.info("Prompt for %s: %s", model, json.dumps(safe_messages))

        request_payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "user": user_input.conversation_id,
        }
        if response_format is not None:
            request_payload["response_format"] = response_format

        base_url_for_request = (
            self._vision_base_url if client is self.vision_client else self._primary_base_url
        )
        tool_kwargs = self._apply_provider_overrides(
            request_payload, tool_kwargs, base_url_for_request
        )

        response: ChatCompletion = await client.chat.completions.create(
            **request_payload,
            **tool_kwargs,
        )

        _LOGGER.info("Response %s", json.dumps(response.model_dump(exclude_none=True)))

        if response.usage.total_tokens > context_threshold:
            await self.truncate_message_history(messages, exposed_entities, user_input)

        choice: Choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "function_call":
            return await self.execute_function_call(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests + 1,
                response_format,
            )
        if choice.finish_reason == "tool_calls":
            return await self.execute_tool_calls(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests + 1,
                response_format,
            )
        if choice.finish_reason == "length":
            raise TokenLengthExceededError(response.usage.completion_tokens)

        return OpenAIQueryResponse(response=response, message=message)

    async def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        response_format: dict | None,
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
                response_format,
            )
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        function,
        response_format: dict | None,
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
        return await self.query(
            user_input,
            messages,
            exposed_entities,
            n_requests,
            response_format=response_format,
        )

    async def execute_tool_calls(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        response_format: dict | None,
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
            else:
                raise FunctionNotFound(function_name)
        return await self.query(
            user_input,
            messages,
            exposed_entities,
            n_requests,
            response_format=response_format,
        )

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


class OpenAIQueryResponse:
    """OpenAI query response value object."""

    def __init__(
        self, response: ChatCompletion, message: ChatCompletionMessage
    ) -> None:
        """Initialize OpenAI query response value object."""
        self.response = response
        self.message = message
