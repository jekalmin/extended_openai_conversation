"""Message truncation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
from typing import Any

from openai import AsyncClient

from homeassistant.components import conversation
from homeassistant.components.conversation import ChatLog

from .helpers import ConversationSummary, get_model_config
from .memory.manager import MemoryManager

_LOGGER = logging.getLogger(__name__)


class MessageTruncateStrategy(ABC):
    """Base class for message truncation strategies."""

    @abstractmethod
    async def truncate(self, chat_log: conversation.ChatLog) -> None:
        """Truncate message history."""

    def _find_last_user_message_index(
        self, messages: list[conversation.Content]
    ) -> int | None:
        """Find the index of the last user message."""
        for i in reversed(range(len(messages))):
            if isinstance(messages[i], conversation.UserContent):
                return i
        return None


class ClearMessageTruncateStrategy(MessageTruncateStrategy):
    """Clear all messages except system prompt and last user message."""

    async def truncate(self, chat_log: conversation.ChatLog) -> None:
        """Clear message history."""
        messages = chat_log.content
        last_user_message_index = self._find_last_user_message_index(messages)
        if last_user_message_index is not None:
            del messages[1:last_user_message_index]
        _LOGGER.info("Context threshold exceeded, conversation history cleared")


class CompactMessageTruncateStrategy(MessageTruncateStrategy):
    """Summarize conversation and optionally store to memory."""

    summarize_system_prompt = """
    You are a memory extraction assistant. Given a conversation, extract:
    1. A concise summary (2-3 sentences)
    2. Key facts the user would want remembered (preferences, decisions, important details)

    Respond in JSON format:
    {
    "summary": "...",
    "key_facts": ["fact1", "fact2", ...]
    }

    Only include genuinely important and reusable information. Skip transient details like weather queries or time checks.
    """

    def __init__(
        self,
        client: AsyncClient,
        model: str,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        """Initialize compact truncate strategy."""
        self._client = client
        self._model = model
        self._memory_manager = memory_manager

    async def truncate(self, chat_log: conversation.ChatLog) -> None:
        """Summarize and truncate message history."""
        summary = await self._summarize_and_flush(chat_log)

        messages = chat_log.content
        last_user_message_index = self._find_last_user_message_index(messages)
        if last_user_message_index is not None:
            del messages[1:last_user_message_index]
            if summary:
                messages.insert(
                    1,
                    conversation.SystemContent(
                        content=f"[Previous conversation summary]: {summary}"
                    ),
                )

        if self._memory_manager is not None:
            _LOGGER.info(
                "Context threshold exceeded, conversation summarized and stored to memory"
            )
        else:
            _LOGGER.info("Context threshold exceeded, conversation summarized")

    async def _summarize_and_flush(self, chat_log: conversation.ChatLog) -> str:
        """Generate summary and optionally flush to memory."""
        result = await self._summarize_conversation(chat_log)

        if self._memory_manager is not None:
            return await self._memory_manager.async_flush_memories(
                chat_log.conversation_id, result
            )

        return result.summary

    async def _summarize_conversation(self, chat_log: ChatLog) -> ConversationSummary:
        """Generate a summary and extract key facts from conversation.

        Returns a ConversationSummary containing summary text and key facts.
        """
        conversation_text = self._build_conversation_text(chat_log)

        if not conversation_text.strip():
            return ConversationSummary(summary="", key_facts=[])

        try:
            # Get model configuration to determine which token parameter to use
            model_config = get_model_config(self._model)

            # Build API parameters based on model configuration
            api_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self.summarize_system_prompt},
                    {
                        "role": "user",
                        "content": f"Extract key information from this conversation:\n\n{conversation_text}",
                    },
                ],
                "response_format": {"type": "json_object"},
            }

            # Add token limit parameter based on model support
            if model_config["supports_max_completion_tokens"]:
                api_kwargs["max_completion_tokens"] = 1000
            elif model_config["supports_max_tokens"]:
                api_kwargs["max_tokens"] = 1000

            # Add temperature if supported
            if model_config["supports_temperature"]:
                api_kwargs["temperature"] = 0.1

            response = await self._client.chat.completions.create(**api_kwargs)

            content = response.choices[0].message.content or "{}"
            data = json.loads(content)

            summary = str(data.get("summary", ""))
            key_facts = data.get("key_facts", [])

            _LOGGER.info("Generated conversation summary and extracted key facts")

            return ConversationSummary(summary=summary, key_facts=key_facts)

        except Exception:
            _LOGGER.warning("Failed to generate conversation summary", exc_info=True)
            return ConversationSummary(summary="", key_facts=[])

    def _build_conversation_text(self, chat_log: ChatLog) -> str:
        """Build a text representation of the conversation from chat log."""
        from homeassistant.components.conversation import (
            AssistantContent,
            SystemContent,
            ToolResultContent,
            UserContent,
        )

        lines: list[str] = []
        for content in chat_log.content:
            if isinstance(content, SystemContent):
                continue
            elif isinstance(content, UserContent):
                lines.append(f"User: {content.content}")
            elif isinstance(content, AssistantContent):
                if content.content:
                    lines.append(f"Assistant: {content.content}")
            elif isinstance(content, ToolResultContent):
                continue

        return "\n".join(lines)
