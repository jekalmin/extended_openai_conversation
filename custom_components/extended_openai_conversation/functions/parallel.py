"""Anonymous web search and page fetching through Parallel Search MCP."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4
from weakref import ref

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import chat_session, llm

from ..const import DOMAIN
from .base import Function

PARALLEL_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "extended_openai_conversation (https://github.com/jekalmin/extended_openai_conversation)"
_SESSION_KEY = f"{DOMAIN}.parallel_sessions"
_LOGGER = logging.getLogger(__name__)


def _http_url(value: str) -> str:
    """Accept HTTP or HTTPS URLs without embedded credentials."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not parsed.username
            and not parsed.password
        ):
            return value
    except ValueError:
        pass
    raise vol.Invalid("Expected an HTTP or HTTPS URL without credentials")


_TEXT = vol.All(str, vol.Length(min=1))
_SEARCH_ARGUMENTS = vol.Schema(
    {
        vol.Required("objective"): _TEXT,
        vol.Required("search_queries"): vol.All([_TEXT], vol.Length(min=1)),
    }
)
_FETCH_ARGUMENTS = vol.Schema(
    {
        vol.Required("urls"): vol.All(
            [vol.All(str, _http_url)], vol.Length(min=1, max=20)
        ),
        vol.Optional("objective"): vol.Any(None, vol.All(str, vol.Length(max=200))),
        vol.Optional("search_queries"): vol.Any(None, [_TEXT]),
        vol.Optional("full_content"): bool,
    }
)


def _conversation_session_id(hass: HomeAssistant) -> str:
    """Keep one opaque ID until Home Assistant cleans up the chat session."""
    session = chat_session.current_session.get()
    if session is None:
        # A standalone function invocation is its own research run.
        return str(uuid4())

    sessions = hass.data.setdefault(_SESSION_KEY, {})
    key = id(session)
    if key not in sessions:
        # Do not keep failed/cancelled first-turn sessions alive just for this ID.
        sessions[key] = (ref(session, lambda _: sessions.pop(key, None)), str(uuid4()))
        session.async_on_cleanup(lambda: sessions.pop(key, None))
    return str(sessions[key][1])


class ParallelFunction(Function):
    """Call the fixed anonymous endpoint using the MCP Streamable HTTP client."""

    def __init__(self) -> None:
        super().__init__(
            vol.Schema(
                {
                    vol.Required("name"): vol.In({"web_search", "web_fetch"}),
                    vol.Optional("timeout", default=30): vol.All(
                        vol.Coerce(float), vol.Range(min=1, max=120)
                    ),
                }
            )
        )

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        """Return MCP content to the conversation, including source URLs and errors."""
        name = function_config["name"]
        schema = _SEARCH_ARGUMENTS if name == "web_search" else _FETCH_ARGUMENTS
        try:
            params = schema(arguments)
        except vol.Invalid as err:
            return {"error": f"Invalid Parallel arguments: {err}"}
        params["session_id"] = _conversation_session_id(hass)
        timeout = function_config.get("timeout", 30)
        try:
            # No API credentials, cookies, redirects or configurable endpoint.
            # A connection belongs to this call; conversation identity is separate.
            async with (
                asyncio.timeout(timeout),
                create_mcp_http_client(headers={"User-Agent": USER_AGENT}) as client,
            ):
                # Use the SDK's HTTP client type across HA's MCP 1.x/2.x versions.
                client.timeout = timeout
                client.follow_redirects = False
                async with streamable_http_client(
                    PARALLEL_URL, http_client=client
                ) as streams:
                    # SDK 1.x also yields a transport session-ID callback. The
                    # tool's conversation ID is owned by HA, independently of it.
                    read, write = streams[:2]
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        await session.list_tools()
                        result = await session.call_tool(name, params)
                        return result.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
        except TimeoutError:
            return {"error": "Parallel web request timed out. Try again later."}
        except Exception:
            # MCP transports can raise nested exception groups. Cancellation is a
            # BaseException and deliberately propagates to the conversation caller.
            _LOGGER.debug("Parallel web request failed", exc_info=True)
            return {"error": "Parallel web request failed. Try again later."}
