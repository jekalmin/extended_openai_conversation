"""AI Task integration for Extended OpenAI Conversation."""

from __future__ import annotations

from json import JSONDecodeError
import logging
from typing import TYPE_CHECKING

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from .const import CONF_LLM_HASS_API, DEFAULT_LLM_HASS_API
from .entity import ExtendedOpenAIBaseLLMEntity

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigSubentry

    from . import ExtendedOpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [ExtendedOpenAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ExtendedOpenAITaskEntity(
    ai_task.AITaskEntity,
    ExtendedOpenAIBaseLLMEntity,
):
    """Extended OpenAI AI Task entity."""

    def __init__(
        self, entry: ExtendedOpenAIConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        super().__init__(entry, subentry)
        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        # Determine which LLM API to use: caller's takes precedence, fallback to subentry config
        llm_api: llm.APIInstance | None = None
        llm_context = llm.LLMContext(
            platform=self.platform.domain,
            context=None,
            language=None,
            assistant=None,
            device_id=None,
        )
        if task.llm_api:
            # Caller provided llm_api class — fetch by class ID
            llm_api = await llm.async_get_api(
                self.hass, task.llm_api.id, llm_context=llm_context
            )
        else:
            # Fallback to subentry config
            llm_api_ids = self.subentry.data.get(
                CONF_LLM_HASS_API, DEFAULT_LLM_HASS_API
            )
            if llm_api_ids:
                try:
                    llm_api = await llm.async_get_api(
                        self.hass, llm_api_ids, llm_context=llm_context
                    )
                except HomeAssistantError as err:
                    _LOGGER.error("Error getting LLM API: %s", err)

        # Set on chat_log for downstream tool execution
        chat_log.llm_api = llm_api

        # Build function tools (custom + HA LLM API tools)
        function_tools = self._get_function_tools()
        if llm_api:
            function_tools.extend(self._convert_llm_api_tools(llm_api))

        # Call shared handler with tools
        await self._async_handle_chat_log(
            chat_log,
            function_tools=function_tools,
            exposed_entities=[],
            llm_context=None,
            structure_name=task.name,
            structure=task.structure,
        )

        # If loop was exhausted without a final assistant response, force one
        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            _LOGGER.warning(
                "Tool loop exhausted without final response, forcing completion"
            )
            await self._async_handle_chat_log(
                chat_log,
                function_tools=[],
                exposed_entities=[],
                llm_context=None,
                structure_name=task.name,
                structure=task.structure,
            )

        # Extract response
        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        # Handle structured output
        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            _LOGGER.error(
                "Failed to parse JSON response: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError("Error with structured response") from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
