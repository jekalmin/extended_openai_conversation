"""The OpenAI Conversation integration."""
from __future__ import annotations

import json
import logging
from functools import partial
from typing import Literal

import openai
import yaml
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL
from homeassistant.core import (
    HomeAssistant,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    TemplateError,
    ServiceNotFound,
)
from homeassistant.helpers import (
    config_validation as cv,
    intent,
    template,
    entity_registry as er,
)
from homeassistant.util import ulid
from openai import error

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_FUNCTIONS,
    CONF_PINECONE_API_KEY,
    CONF_PINECONE_TOP_K,
    CONF_USE_INTERACTIVE,
    CONF_PINECONE_SCORE_THRESHOLD,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_PINECONE_TOP_K,
    DEFAULT_USE_INTERACTIVE,
    DOMAIN,
    DATA_AGENT,
    DATA_STORAGE,
    DATA_USE_STORAGE,
)
from .exceptions import (
    EntityNotFound,
    EntityNotExposed,
    CallServiceError,
    FunctionNotFound,
    NativeNotFound,
)
from .helpers import (
    FunctionExecutor,
    NativeFunctionExecutor,
    ScriptFunctionExecutor,
    TemplateFunctionExecutor,
    convert_to_template,
    PineconeStorage,
)
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


FUNCTION_EXECUTORS: dict[str, FunctionExecutor] = {
    "predefined": NativeFunctionExecutor(),
    "native": NativeFunctionExecutor(),
    "script": ScriptFunctionExecutor(),
    "template": TemplateFunctionExecutor(),
}


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    await async_setup_services(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    try:
        await hass.async_add_executor_job(
            partial(
                openai.Engine.list,
                api_key=entry.data[CONF_API_KEY],
                request_timeout=10,
            )
        )
    except error.AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except error.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    agent = OpenAIAgent(hass, entry)
    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent
    data[DATA_USE_STORAGE] = False

    pinecone_api_key = entry.data.get(CONF_PINECONE_API_KEY)
    if pinecone_api_key is not None:
        storage = PineconeStorage(
            hass=hass, api_key=entry.data.get(CONF_PINECONE_API_KEY)
        )
        data[DATA_STORAGE] = storage
        data[DATA_USE_STORAGE] = True

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

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        use_interactive = self.entry.options.get(
            CONF_USE_INTERACTIVE, DEFAULT_USE_INTERACTIVE
        )
        exposed_entities = self.get_exposed_entities()

        if user_input.conversation_id in self.history and use_interactive:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            try:
                prompt = await self._async_generate_prompt(
                    user_input, raw_prompt, exposed_entities
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
            messages = [{"role": "system", "content": prompt}]

        messages.append({"role": "user", "content": user_input.text})

        try:
            response = await self.query(user_input, messages, exposed_entities, 0)
        except error.OpenAIError as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except (
            EntityNotFound,
            ServiceNotFound,
            CallServiceError,
            EntityNotExposed,
            FunctionNotFound,
            NativeNotFound,
        ) as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages.append(response)
        self.history[conversation_id] = messages

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response["content"])
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    async def _async_generate_prompt(
        self,
        user_input: conversation.ConversationInput,
        raw_prompt: str,
        exposed_entities,
    ) -> str:
        """Generate a prompt for the user."""

        matched_entities = []
        use_storage = self.hass.data[DOMAIN][self.entry.entry_id][DATA_USE_STORAGE]
        if use_storage:
            storage = self.hass.data[DOMAIN][self.entry.entry_id][DATA_STORAGE]
            embedding_result = await self.embeddings(user_input.text)
            top_k = self.entry.options.get(CONF_PINECONE_TOP_K, DEFAULT_PINECONE_TOP_K)
            score_threshold = self.entry.options.get(CONF_PINECONE_SCORE_THRESHOLD)

            result = await storage.query(
                topK=top_k,
                vector=embedding_result["data"][0]["embedding"],
            )
            matched_entities = result.get("matches", [])

            if score_threshold:
                result["matches"] = [
                    match
                    for match in result["matches"]
                    if match["score"] >= score_threshold
                ]

        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "matched_entities": matched_entities,
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
                aliases = list(entity.aliases)

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
            if not function:
                return []
            result = yaml.safe_load(function)
            if result:
                for setting in result:
                    for function in setting["function"].values():
                        convert_to_template(function)
            return result
        except:
            _LOGGER.error("Failed to load functions")
            return []

    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages,
        exposed_entities,
        n_requests,
    ):
        """Process a sentence."""
        model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        functions = list(map(lambda s: s["spec"], self.get_functions()))
        function_call = "auto"
        if n_requests == self.entry.options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        ):
            function_call = "none"

        _LOGGER.info("Prompt for %s: %s", model, messages)

        response = await openai.ChatCompletion.acreate(
            api_key=self.entry.data[CONF_API_KEY],
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            user=user_input.conversation_id,
            functions=functions,
            function_call=function_call,
        )

        _LOGGER.info("Response %s", response)
        message = response["choices"][0]["message"]
        if message.get("function_call"):
            message = await self.execute_function_call(
                user_input, messages, message, exposed_entities, n_requests + 1
            )
        return message

    def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message,
        exposed_entities,
        n_requests,
    ):
        function_name = message["function_call"]["name"]
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return self.execute_function(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests,
                function,
            )
        raise FunctionNotFound(message["function_call"]["name"])

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message,
        exposed_entities,
        n_requests,
        function,
    ):
        function_executor = FUNCTION_EXECUTORS[function["function"]["type"]]
        arguments = json.loads(message["function_call"]["arguments"])

        result = await function_executor.execute(
            self.hass, function, arguments, user_input, exposed_entities
        )

        messages.append(
            {
                "role": "function",
                "name": message["function_call"]["name"],
                "content": str(result),
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def embeddings(self, text: str):
        return await openai.Embedding.acreate(
            api_key=self.entry.data[CONF_API_KEY],
            input=text,
            model="text-embedding-ada-002",
            request_timeout=3,
        )
