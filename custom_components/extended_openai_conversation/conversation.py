"""Extended OpenAI Conversation agent entity."""

from __future__ import annotations

import logging
from typing import Literal

from openai import OpenAIError
import yaml

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    async_get_chat_log,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er, intent, llm, template
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import ExtendedOpenAIConfigEntry
from .const import (
    CONF_FUNCTIONS,
    CONF_PROMPT,
    CONF_SKILLS,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_PROMPT,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .entity import ExtendedOpenAIBaseLLMEntity
from .exceptions import FunctionLoadFailed, FunctionNotFound, InvalidFunction
from .helpers import get_exposed_entities, get_function_executor
from .skills import SkillManager

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ExtendedOpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the OpenAI Conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [ExtendedOpenAIAgentEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ExtendedOpenAIAgentEntity(
    ConversationEntity,
    conversation.AbstractConversationAgent,
    ExtendedOpenAIBaseLLMEntity,
):
    """Extended OpenAI conversation agent."""

    _attr_supports_streaming = True
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(
        self,
        entry: ExtendedOpenAIConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @property
    def skills(self) -> list[str]:
        """Get the enabled skills list for this entity."""
        return self.subentry.data.get(CONF_SKILLS, [])

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        self.skill_manager = await SkillManager.async_get_instance(self.hass)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Call the API."""
        # Create LLM context
        llm_context = user_input.as_llm_context(DOMAIN)

        # Get exposed entities for custom functions
        exposed_entities = self._get_exposed_entities()

        # Get custom functions
        custom_functions = self._get_functions()

        # Build custom prompt with exposed entities
        system_prompt = self._build_system_prompt(
            exposed_entities, llm_context, user_input
        )

        # Set system prompt in chat log
        chat_log.content[0] = conversation.SystemContent(content=system_prompt)

        # Call the LLM

        try:
            await self._async_handle_chat_log(
                chat_log,
                custom_functions=custom_functions,
                exposed_entities=exposed_entities,
                llm_context=llm_context,
            )
        except OpenAIError as err:
            _LOGGER.error(err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )

        # Fire conversation finished event
        self.hass.bus.async_fire(
            EVENT_CONVERSATION_FINISHED,
            {
                "user_input": user_input,
                "messages": [c.as_dict() for c in chat_log.content],
                "agent_id": self.subentry.subentry_id,
            },
        )

        # Build response from chat log
        intent_response = intent.IntentResponse(language=user_input.language)

        # Get last assistant message
        last_content = chat_log.content[-1]
        if isinstance(last_content, conversation.AssistantContent):
            intent_response.async_set_speech(last_content.content or "")
        else:
            intent_response.async_set_speech("")

        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    def _build_system_prompt(
        self,
        exposed_entities: list[dict],
        llm_context: llm.LLMContext,
        user_input: ConversationInput,
    ) -> str:
        """Build system prompt with exposed entities and skills."""
        raw_prompt = self.subentry.data.get(CONF_PROMPT, DEFAULT_PROMPT)

        rendered_prompt = template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": llm_context.device_id,
                "user_input": user_input,
            },
            parse_result=False,
        )

        # Add skills section
        skills_section = self.skill_manager.build_skills_prompt_section(self.skills)
        if skills_section:
            rendered_prompt += skills_section

        return rendered_prompt

    def _get_exposed_entities(self):
        return get_exposed_entities(self.hass)

    def _get_functions(self) -> list[dict]:
        """Get custom functions configuration including skill functions."""
        try:
            function = self.subentry.data.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )

            result = result or []

            # Add skill functions
            skill_functions = self.skill_manager.get_skill_functions(self.skills)
            for setting in skill_functions:
                function_executor = get_function_executor(
                    setting["function"]["type"]
                )
                setting["function"] = function_executor.to_arguments(
                    setting["function"]
                )
            result.extend(skill_functions)

            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except Exception as e:
            raise FunctionLoadFailed() from e
