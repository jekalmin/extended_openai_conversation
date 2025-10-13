"""AI Task entity for the Extended OpenAI Conversation integration."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import DATA_AGENT
from .const import DOMAIN
from .helpers import chat_log_to_messages, create_conversation_input

if TYPE_CHECKING:
    from . import OpenAIAgent

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the AI Task entity."""
    integration_data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if not integration_data:
        _LOGGER.debug("No stored data found for entry %s, skipping AI task setup", entry.entry_id)
        return

    agent: OpenAIAgent | None = integration_data.get(DATA_AGENT)
    if agent is None:
        _LOGGER.debug("No agent cached for entry %s, skipping AI task setup", entry.entry_id)
        return

    async_add_entities(
        [ExtendedOpenAIAITaskEntity(entry, agent)],
    )


class ExtendedOpenAIAITaskEntity(ai_task.AITaskEntity):
    """AI Task entity that reuses the Extended OpenAI conversation agent."""

    _attr_should_poll = False
    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    def __init__(self, entry: ConfigEntry, agent: "OpenAIAgent") -> None:
        """Initialize the entity."""
        self._entry = entry
        self._agent = agent
        self._attr_name = f"{entry.title} AI Task"
        self._attr_unique_id = f"{entry.entry_id}_ai_task"

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task using the conversation agent."""
        instructions = task.instructions or ""
        user_id = None
        if chat_log.content and isinstance(chat_log.content[-1], conversation.UserContent):
            # Try to reuse the last user's metadata when possible.
            last_user = chat_log.content[-1]
            if getattr(last_user, "user_id", None):
                user_id = last_user.user_id

        if not chat_log.content or not (
            isinstance(chat_log.content[-1], conversation.UserContent)
            and chat_log.content[-1].content == instructions
        ):
            chat_log.async_add_user_content(conversation.UserContent(content=instructions))

        exposed_entities = self._agent.get_exposed_entities()
        synthetic_input = create_conversation_input(
            text=instructions,
            conversation_id=chat_log.conversation_id,
            language=getattr(chat_log, "language", None),
            user_id=user_id,
            device_id=None,
        )
        system_message = self._agent._generate_system_message(  # noqa: SLF001 - Reuse existing prompt generator
            exposed_entities,
            synthetic_input,
        )

        messages = chat_log_to_messages(system_message, chat_log)

        response_format: dict[str, str] | None = None
        if task.structure is not None:
            structure_prompt = _structure_to_prompt(task.structure)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Return a JSON object that validates against the provided Home Assistant schema. "
                        "Do not include explanations or additional commentary.\n"
                        f"{structure_prompt}"
                    ),
                }
            )
            response_format = {"type": "json_object"}

        query_response = await self._agent.query(
            synthetic_input,
            messages,
            exposed_entities,
            0,
            response_format=response_format,
        )

        message_text = _message_to_text(query_response.message)

        if task.structure is None:
            result_payload: Any = message_text
            chat_payload = message_text
        else:
            try:
                structured_payload = json.loads(message_text)
            except json.JSONDecodeError as err:
                raise HomeAssistantError(
                    f"Structured response was not valid JSON: {err}"
                ) from err

            try:
                validated_payload = task.structure(structured_payload)
            except vol.Invalid as err:
                raise HomeAssistantError(
                    f"Structured response did not match schema: {err}"
                ) from err

            result_payload = validated_payload
            chat_payload = json.dumps(validated_payload, indent=2)

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id or self._entry.entry_id,
                content=chat_payload,
            )
        )

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=result_payload,
        )


def _message_to_text(message) -> str:
    """Convert a ChatCompletionMessage into plain text."""
    content = getattr(message, "content", "")
    if isinstance(content, str) or content is None:
        return content or ""

    # Response parts can be Pydantic models or dict-like objects.
    parts: list[str] = []
    for part in content:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _structure_to_prompt(schema: vol.Schema) -> str:
    """Serialize a voluptuous schema into a prompt-friendly string."""
    schema_payload = getattr(schema, "schema", schema)

    def _default_serializer(value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "config"):
            return value.config
        return str(value)

    try:
        return json.dumps(schema_payload, indent=2, default=_default_serializer)
    except TypeError:
        return str(schema_payload)
