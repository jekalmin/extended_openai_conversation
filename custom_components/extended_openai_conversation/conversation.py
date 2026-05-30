"""Extended OpenAI Conversation agent entity."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from openai import OpenAIError
from voluptuous_openapi import convert
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
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm, template
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import ExtendedOpenAIConfigEntry
from .const import (
    CONF_FUNCTION_TOOLS,
    CONF_LLM_HASS_API,
    CONF_PROMPT,
    CONF_SKILLS,
    DEFAULT_CONF_FUNCTION_TOOLS,
    DEFAULT_LLM_HASS_API,
    DEFAULT_PROMPT,
    DEFAULT_WORKING_DIRECTORY,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .entity import ExtendedOpenAIBaseLLMEntity, _adjust_schema
from .exceptions import FunctionLoadFailed, FunctionNotFound, InvalidFunction
from .functions import get_function
from .helpers import get_exposed_entities
from .skills import Skill, SkillManager

_LOGGER = logging.getLogger(__name__)


def _is_nullable(prop_schema: dict[str, Any]) -> bool:
    """Check if a property schema accepts null."""
    prop_type = prop_schema.get("type")
    if isinstance(prop_type, list):
        return "null" in prop_type
    return prop_type == "null"


def _make_nullable_optional(schema: dict[str, Any]) -> None:
    """Remove nullable properties from required list so LLM treats them as optional.

    Keep 'name' required since it's semantically required by HA's Assist API
    even though it's typed as nullable.
    """
    if schema.get("type") != "object" or "properties" not in schema:
        return
    required = schema.get("required", [])
    nullable = {
        name
        for name, prop in schema["properties"].items()
        if _is_nullable(prop) and name != "name"
    }
    if nullable:
        schema["required"] = [r for r in required if r not in nullable]
    for prop in schema["properties"].values():
        _make_nullable_optional(prop)


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
    skill_manager: SkillManager

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @property
    def skills(self) -> list[str]:
        """Get the enabled skills list for this entity."""
        return self.subentry.data.get(CONF_SKILLS, []) or []

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

        # Calculate skills directory based on working directory
        working_dir = DEFAULT_WORKING_DIRECTORY
        if Path(working_dir).is_absolute():
            skills_dir = Path(working_dir) / "skills"
        else:
            skills_dir = Path(self.hass.config.config_dir) / working_dir / "skills"

        self.skill_manager = await SkillManager.async_get_instance(
            self.hass, user_skills_dir=str(skills_dir)
        )

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

        # Fetch LLM API instance if configured
        llm_api_ids = self.subentry.data.get(CONF_LLM_HASS_API, DEFAULT_LLM_HASS_API)
        llm_api: llm.APIInstance | None = None
        if llm_api_ids:
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    llm_api_ids,
                    llm_context=llm_context,
                )
            except HomeAssistantError as err:
                _LOGGER.error("Error getting LLM API: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error preparing LLM API: {err}",
                )
                return ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )

        # Set LLM API on chat log for tool execution
        chat_log.llm_api = llm_api

        # Get exposed entities for function tools
        exposed_entities = self._get_exposed_entities()

        # Get function tools (custom + HA LLM API tools)
        function_tools = self._get_function_tools()
        if llm_api:
            function_tools.extend(self._convert_llm_api_tools(llm_api))

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
                function_tools=function_tools,
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
            _LOGGER.error("Error during conversation: %s", err, exc_info=True)
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
        raw_prompt: str = self.subentry.data.get(CONF_PROMPT, DEFAULT_PROMPT)

        result = template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": llm_context.device_id,
                "user_input": user_input,
                "skills": self._get_enabled_skills(),
            },
            parse_result=False,
        )

        return str(result)

    def _get_enabled_skills(self) -> list[Skill]:
        """Get enabled skills as list for template rendering."""
        enabled_skill_names = self.skills
        all_skills = self.skill_manager.get_all_skills()

        return [s for s in all_skills if s.name in enabled_skill_names]

    def _get_exposed_entities(self) -> list[dict[str, Any]]:
        return get_exposed_entities(self.hass)

    def _get_function_tools(self) -> list[dict[str, Any]]:
        """Get custom functions configuration.

        This overrides the base class method to import DEFAULT_CONF_FUNCTION_TOOLS
        locally to avoid circular imports.
        """

        try:
            function_tools_config = self.subentry.data.get(CONF_FUNCTION_TOOLS)
            function_tools: list[dict[str, Any]] | None = (
                yaml.safe_load(function_tools_config)
                if function_tools_config
                else DEFAULT_CONF_FUNCTION_TOOLS
            )
            if function_tools:
                for function_tool in function_tools:
                    if isinstance(function_tool, dict) and "function" in function_tool:
                        function_config = function_tool["function"]
                        if (
                            isinstance(function_config, dict)
                            and "type" in function_config
                        ):
                            function = get_function(function_config["type"])
                            function_tool["function"] = function.validate_schema(
                                function_config
                            )

            return function_tools or []
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except Exception as e:
            raise FunctionLoadFailed() from e

    def _convert_llm_api_tools(self, llm_api: llm.APIInstance) -> list[dict[str, Any]]:
        """Convert HA LLM API tools to function_tools format.

        This overrides the base class method to import _make_nullable_optional
        locally to avoid circular imports.
        """
        from .entity import _make_nullable_optional

        result: list[dict[str, Any]] = []
        for tool in llm_api.tools:
            schema = convert(
                tool.parameters,
                custom_serializer=llm_api.custom_serializer or llm.selector_serializer,
            )
            _adjust_schema(schema)
            _make_nullable_optional(schema)
            result.append(
                {
                    "spec": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                    "function": {
                        "type": "llm_api",
                        "tool_name": tool.name,
                    },
                }
            )
        return result
