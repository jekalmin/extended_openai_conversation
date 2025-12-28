"""Base entity for Extended OpenAI Conversation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from openai import AsyncClient, AsyncStream
from openai._exceptions import OpenAIError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from .const import (
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOKEN_PARAM,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    MODEL_PARAMETER_SUPPORT,
    MODEL_TOKEN_PARAMETER_SUPPORT,
)
from .exceptions import FunctionNotFound, ParseArgumentsFailed, TokenLengthExceededError
from .helpers import get_function_executor

if TYPE_CHECKING:
    from . import ExtendedOpenAIConfigEntry

_LOGGER = logging.getLogger(__name__)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


def _convert_content_to_param(
    chat_content: list[conversation.Content],
) -> list[ChatCompletionMessageParam]:
    """Convert chat log content to OpenAI message format."""
    messages: list[ChatCompletionMessageParam] = []

    for content in chat_content:
        if content.role == "system":
            messages.append({"role": "system", "content": content.content})
        elif content.role == "user":
            messages.append({"role": "user", "content": content.content})
        elif content.role == "assistant":
            msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
            if content.content:
                msg["content"] = content.content
            if content.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args),
                        },
                    }
                    for tool_call in content.tool_calls
                ]
            messages.append(msg)
        elif content.role == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": content.tool_call_id,
                    "content": json.dumps(content.tool_result),
                }
            )

    return messages


class ExtendedOpenAIBaseLLMEntity(Entity):
    """Extended OpenAI base entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: ExtendedOpenAIConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenAI",
            model=subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _client(self) -> AsyncClient:
        """Return the OpenAI client."""
        return self.entry.runtime_data

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        custom_functions: list[dict[str, Any]],
        exposed_entities: list[dict[str, Any]],
        llm_context: llm.LLMContext | None = None,
    ) -> None:
        """Generate an answer for the chat log with streaming support."""
        options = self.subentry.data
        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = options.get(CONF_TOP_P, DEFAULT_TOP_P)
        temperature = options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        use_tools = options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)
        max_function_calls = options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )

        messages = _convert_content_to_param(chat_log.content)

        # Build tools list from custom functions only
        tools: list[ChatCompletionToolParam] = []

        if use_tools:
            for func_spec in custom_functions:
                tools.append(
                    ChatCompletionToolParam(
                        type="function",
                        function=func_spec["spec"],
                    )
                )

        # Determine token parameter based on model
        model_lower = model.lower()
        token_kwargs = {self.get_token_param_for_model(model): max_tokens}
        supports_top_p = True
        for entry in MODEL_PARAMETER_SUPPORT:
            if re.search(entry["pattern"], model_lower):
                supports_top_p = "top_p" not in entry["unsupported_params"]
                break
        top_p_kwargs = {"top_p": top_p} if supports_top_p else {}

        tool_kwargs: dict[str, Any] = {}
        if tools:
            tool_kwargs["tools"] = tools
            tool_kwargs["tool_choice"] = "auto"

        # To prevent infinite loops, we limit the number of iterations
        for n_requests in range(MAX_TOOL_ITERATIONS):
            # Update tool_choice based on function call count
            if tools and n_requests >= max_function_calls:
                tool_kwargs["tool_choice"] = "none"

            _LOGGER.info("Prompt for %s: %s", model, json.dumps(messages))

            try:
                stream = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    top_p=top_p,
                    temperature=temperature,
                    user=chat_log.conversation_id,
                    stream=True,
                    stream_options={"include_usage": True},
                    **token_kwargs,
                    **top_p_kwargs,
                    **tool_kwargs,
                )
            except OpenAIError as err:
                _LOGGER.error("Error talking to OpenAI: %s", err)
                raise HomeAssistantError("Error talking to OpenAI") from err

            # Process stream and collect tool calls
            pending_tool_calls: list[llm.ToolInput] = []

            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id, self._transform_stream(chat_log, stream)
            ):
                if isinstance(content, conversation.AssistantContent):
                    if content.tool_calls:
                        pending_tool_calls.extend(content.tool_calls)

            if pending_tool_calls:
                _LOGGER.info("Response Tool Calls %s", pending_tool_calls)

            # Execute custom functions
            for tool_call in pending_tool_calls:
                custom_func = next(
                    (
                        f
                        for f in (custom_functions)
                        if f["spec"]["name"] == tool_call.tool_name
                    ),
                    None,
                )

                if custom_func is None:
                    raise FunctionNotFound(tool_call.tool_name)

                tool_result_content = await self._execute_custom_function(
                    custom_func,
                    tool_call,
                    llm_context,
                    exposed_entities,
                )

                chat_log.async_add_assistant_content_without_tools(tool_result_content)

            # Update messages for next iteration
            messages = _convert_content_to_param(chat_log.content)

            # Check if we need to continue (if there are pending tool results)
            if not chat_log.unresponded_tool_results:
                break

    async def _transform_stream(
        self,
        chat_log: conversation.ChatLog,
        result: AsyncStream[ChatCompletionChunk],
    ) -> AsyncGenerator[
        conversation.AssistantContentDeltaDict | conversation.ToolResultContentDeltaDict
    ]:
        """Transform OpenAI stream to Home Assistant format."""
        current_tool_calls: dict[int, dict[str, Any]] = {}
        first_chunk = True

        async for chunk in result:
            _LOGGER.debug("Received chunk: %s", chunk)

            # Signal new assistant message on first chunk
            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            if not chunk.choices:
                # Track usage from final chunk if available
                if chunk.usage:
                    chat_log.async_trace(
                        {
                            "stats": {
                                "input_tokens": chunk.usage.prompt_tokens,
                                "output_tokens": chunk.usage.completion_tokens,
                            }
                        }
                    )
                    if chunk.usage.total_tokens > self.subentry.data.get(
                        CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD
                    ):
                        await self._truncate_message_history(chat_log)
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                yield {"content": delta.content}

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            current_tool_calls[idx]["name"] = (
                                tool_call_delta.function.name
                            )
                        if tool_call_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += (
                                tool_call_delta.function.arguments
                            )

            if current_tool_calls and (choice.finish_reason in {"tool_calls", "stop"}):
                # Yield all accumulated tool calls (marked as external since we handle them ourselves)
                tool_calls_list = []
                for idx in sorted(current_tool_calls.keys()):
                    tool_call = current_tool_calls[idx]
                    try:
                        args = json.loads(tool_call["arguments"])
                    except json.JSONDecodeError as err:
                        raise ParseArgumentsFailed(tool_call["arguments"]) from err
                    tool_calls_list.append(
                        llm.ToolInput(
                            id=tool_call["id"],
                            tool_name=tool_call["name"],
                            tool_args=args,
                            external=True,  # Mark as external so ChatLog doesn't try to execute
                        )
                    )
                if tool_calls_list:
                    yield {"tool_calls": tool_calls_list}
                current_tool_calls.clear()
            if choice.finish_reason == "length":
                raise TokenLengthExceededError(
                    self.subentry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
                )

            if choice.finish_reason == "stop":
                break

    async def _execute_custom_function(
        self,
        function_spec: dict[str, Any],
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> conversation.ToolResultContent:
        """Execute a custom function."""
        function = function_spec["function"]
        arguments: dict[str, Any] = tool_input.tool_args
        function_executor = get_function_executor(function["type"])

        if self.should_run_in_background(arguments):
            # create a delayed function and execute in background
            function_executor = get_function_executor("composite")
            self.entry.async_create_task(
                self.hass,
                function_executor.execute(
                    self.hass,
                    self.get_delayed_function(function, arguments),
                    arguments,
                    llm_context,
                    exposed_entities,
                ),
            )
            result = "Scheduled"
        else:
            result = await function_executor.execute(
                self.hass, function, arguments, llm_context, exposed_entities
            )

        return conversation.ToolResultContent(
            agent_id=self.entity_id,
            tool_call_id=tool_input.id,
            tool_name=tool_input.tool_name,
            tool_result={"result": str(result)},
        )

    def should_run_in_background(self, arguments) -> bool:
        """Check if function needs delay."""
        return isinstance(arguments, dict) and arguments.get("delay") is not None

    def get_delayed_function(self, function, arguments) -> dict:
        """Execute function with delay."""
        # create a composite function with delay in script function
        return {
            "type": "composite",
            "sequence": [
                {
                    "type": "script",
                    "sequence": [{"delay": arguments["delay"]}],
                },
                function,
            ],
        }

    async def _truncate_message_history(self, chat_log: conversation.ChatLog) -> None:
        """Truncate message history based on strategy."""
        options = self.subentry.data
        strategy = options.get(
            CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY
        )

        if strategy == "clear":
            # Keep only system prompt and last user message
            # This is handled by refreshing the LLM data
            _LOGGER.info("Context threshold exceeded, conversation history cleared")
            last_user_message_index = None
            messages = chat_log.content
            for i in reversed(range(len(messages))):
                if isinstance(messages[i], conversation.UserContent):
                    last_user_message_index = i
                    break

            if last_user_message_index is not None:
                del messages[1:last_user_message_index]

    def get_token_param_for_model(self, model: str) -> str:
        """Return the token parameter name for a model."""
        model_lower = model.lower()
        for entry in MODEL_TOKEN_PARAMETER_SUPPORT:
            if re.search(entry["pattern"], model_lower):
                return entry["token_param"]
        return DEFAULT_TOKEN_PARAM
