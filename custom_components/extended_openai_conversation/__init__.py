"""The OpenAI Conversation integration."""
from __future__ import annotations

from functools import partial
import logging
from typing import Literal
import json
import yaml

import openai
from openai import error


from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.util import ulid
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
    ServiceNotFound,
)


from homeassistant.helpers import (
    config_validation as cv,
    intent,
    template,
    entity_registry as er,
)

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_FUNCTIONS,
    CONF_CUSTOM_FUNCTIONS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DOMAIN,
    SERVICE_RELOAD,
)

from .exceptions import (
    EntityNotFound,
    EntityNotExposed,
    CallServiceError,
    FunctionNotFound,
)

from .helpers import (
    CustomFunctionExecutor,
    ScriptCustomFunctionExecutor,
    TemplateCustomFunctionExecutor,
    convert_to_template,
)


_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


FUNCTION_EXECUTORS: dict[str, CustomFunctionExecutor] = {
    "script": ScriptCustomFunctionExecutor(),
    "template": TemplateCustomFunctionExecutor(),
}

# hass.data key for logging information.
DATA_AGENT = "agent"


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
        exposed_entities = self.get_exposed_entities()

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            try:
                prompt = self._async_generate_prompt(raw_prompt, exposed_entities)
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

    def _async_generate_prompt(self, raw_prompt: str, exposed_entities) -> str:
        """Generate a prompt for the user."""
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
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

    def get_custom_functions(self):
        try:
            custom_functions = self.entry.options.get(CONF_CUSTOM_FUNCTIONS)
            if not custom_functions:
                return []
            result = yaml.safe_load(custom_functions)
            if result:
                for setting in result:
                    for function in setting["function"].values():
                        convert_to_template(function)
            return result
        except:
            _LOGGER.error("failed to load custom functions")
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
        custom_functions = self.get_custom_functions()
        functions = CONF_FUNCTIONS + list(map(lambda s: s["spec"], custom_functions))
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
        custom_functions = self.get_custom_functions()
        function_name = message["function_call"]["name"]
        custom_function = next(
            (s for s in custom_functions if s["spec"]["name"] == function_name),
            None,
        )
        if function_name == "execute_services":
            return self.execute_services(
                user_input, messages, message, exposed_entities, n_requests
            )
        if custom_function is not None:
            return self.execute_custom_function(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests,
                custom_function,
            )
        else:
            raise FunctionNotFound(message["function_call"]["name"])

    async def execute_services(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message,
        exposed_entities,
        n_requests,
    ):
        arguments = json.loads(message["function_call"]["arguments"])

        result = []
        for service_argument in arguments.get("list", []):
            domain = service_argument["domain"]
            service = service_argument["service"]
            service_data = service_argument.get(
                "service_data", service_argument.get("data", {})
            )
            entity_id = service_data.get("entity_id", service_argument.get("entity_id"))

            if isinstance(entity_id, str):
                entity_id = [e.strip() for e in entity_id.split(',')]
            service_data["entity_id"] = entity_id

            if entity_id is None:
                raise CallServiceError(domain, service, service_data)
            if not self.hass.services.has_service(domain, service):
                raise ServiceNotFound(domain, service)
            if any(self.hass.states.get(entity) is None for entity in entity_id):
                raise EntityNotFound(entity_id)
            exposed_entity_ids = map(lambda e: e["entity_id"], exposed_entities)
            if not set(entity_id).issubset(exposed_entity_ids):
                raise EntityNotExposed(entity_id)

            try:
                await self.hass.services.async_call(
                    domain=domain,
                    service=service,
                    service_data=service_data,
                )
                result.append(True)
            except HomeAssistantError:
                _LOGGER.error(e)
                result.append(False)

        messages.append(
            {
                "role": "function",
                "name": message["function_call"]["name"],
                "content": str(result),
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_custom_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message,
        exposed_entities,
        n_requests,
        custom_function,
    ):
        custom_function_executor = FUNCTION_EXECUTORS[
            custom_function["function"]["type"]
        ]
        arguments = json.loads(message["function_call"]["arguments"])

        result = await custom_function_executor.execute(
            self.hass, custom_function, arguments, user_input
        )

        messages.append(
            {
                "role": "function",
                "name": message["function_call"]["name"],
                "content": result,
            }
        )
        return await self.query(user_input, messages, exposed_entities, n_requests)
