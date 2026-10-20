"""Exercise Parallel through the conversation resolver, dispatcher and MCP client."""

import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import httpx2
from openai.types.chat import ChatCompletionChunk
import pytest
import yaml

from custom_components.extended_openai_conversation.const import (
    CONF_FUNCTION_TOOLS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_CONF_FUNCTION_TOOLS,
)
from custom_components.extended_openai_conversation.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation.functions import (
    get_function,
    parallel,
)
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context
from homeassistant.helpers import chat_session

SEARCH = {
    "objective": "Home Assistant release",
    "search_queries": ["Home Assistant release"],
}
FETCH = {"urls": ["https://www.home-assistant.io/blog/"]}
SOURCE = "https://www.home-assistant.io/blog/"


@pytest.fixture
def mcp_http(monkeypatch):
    """Use the real SDK and HTTP serialization with a deterministic MCP server."""
    requests = []
    clients = []
    state = {"mode": "ok"}
    original_client = httpx2.AsyncClient

    async def handle(request):
        body = json.loads(request.content) if request.content else {}
        requests.append((request, body))
        if request.method != "POST":
            return httpx2.Response(405)
        method = body.get("method")
        if "id" not in body:
            return httpx2.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        else:
            assert method == "tools/call"
            if state["mode"] == "timeout":
                raise httpx2.ReadTimeout("test timeout", request=request)
            if state["mode"] == "wait":
                state["started"].set()
                await asyncio.Event().wait()
            if state["mode"] == "http_error":
                return httpx2.Response(503)
            if state["mode"] == "rpc_error":
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {"code": -32603, "message": "Unavailable"},
                    },
                )
            if state["mode"] == "malformed":
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {"content": 42},
                    },
                )
            result = {
                "content": [{"type": "text", "text": f"Release details: {SOURCE}"}],
                "isError": False,
            }
            if state["mode"] == "quota":
                result = {
                    "content": [
                        {"type": "text", "text": "Free anonymous quota exceeded"}
                    ],
                    "isError": True,
                }
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": body["id"], "result": result}
        )

    def client(**kwargs):
        assert kwargs["headers"] == {"User-Agent": parallel.USER_AGENT}
        instance = original_client(transport=httpx2.MockTransport(handle), **kwargs)
        clients.append(instance)
        return instance

    monkeypatch.setattr(parallel, "create_mcp_http_client", client)
    # Ambient credentials must never reach the anonymous client.
    monkeypatch.setenv("PARALLEL_API_KEY", "must-not-be-sent")
    return requests, clients, state


def make_agent(hass, monkeypatch, options=None):
    entry = MagicMock()
    entry.runtime_data.chat.completions.create = AsyncMock()
    subentry = MagicMock()
    subentry.subentry_id = "test"
    subentry.title = "Test assistant"
    subentry.data = options or {}
    agent = ExtendedOpenAIAgentEntity(entry, subentry)
    agent.hass = hass
    agent.entity_id = "conversation.test"
    monkeypatch.setattr(
        agent, "_build_system_prompt", lambda *args: "Use web tools and cite sources."
    )
    monkeypatch.setattr(agent, "_get_exposed_entities", lambda: [])
    return agent


def scripted_response(name=None, arguments=None):
    """Return model chunks, not a live model; production stream handling is real."""
    delta = {"content": f"Release details are at {SOURCE}"}
    finish = "stop"
    if name:
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        }
        finish = "tool_calls"

    async def chunks():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "scripted",
                "created": 0,
                "model": "scripted-model",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    return chunks()


async def ask(agent, name="web_search", arguments=None, conversation_id=None):
    """Enter the normal HA conversation path with a scripted tool-using model."""
    agent._client.chat.completions.create.side_effect = [
        scripted_response(name, arguments or SEARCH),
        scripted_response(),
    ]
    return await agent.async_process(
        ConversationInput(
            text="Find current release details",
            context=Context(),
            language="en",
            device_id=None,
            satellite_id=None,
            agent_id=agent.entity_id,
            conversation_id=conversation_id,
        )
    )


def calls(mcp_http):
    return [
        body["params"] for _, body in mcp_http[0] if body.get("method") == "tools/call"
    ]


async def test_default_conversation_search_fetch_and_followup(
    hass, monkeypatch, mcp_http
):
    agent = make_agent(hass, monkeypatch)
    first = await ask(agent)
    outgoing = agent._client.chat.completions.create.call_args_list
    assert {t["function"]["name"] for t in outgoing[0].kwargs["tools"]} >= {
        "web_search",
        "web_fetch",
        "execute_services",
    }
    assert SOURCE in outgoing[1].kwargs["messages"][-1]["content"]
    assert SOURCE in first.response.speech["plain"]["speech"]
    await ask(agent, "web_fetch", FETCH, first.conversation_id)
    await ask(agent, conversation_id=first.conversation_id)
    ids = [c["arguments"]["session_id"] for c in calls(mcp_http)]
    assert len(ids) == 3 and len(set(ids)) == 1
    assert str(UUID(ids[0])) == ids[0]
    assert ids[0] != first.conversation_id
    assert [c["name"] for c in calls(mcp_http)] == [
        "web_search",
        "web_fetch",
        "web_search",
    ]
    for request, _ in mcp_http[0]:
        assert str(request.url) == parallel.PARALLEL_URL
        assert request.headers["user-agent"] == parallel.USER_AGENT
        assert "authorization" not in request.headers
    assert all(c.is_closed for c in mcp_http[1])
    assert all(not c.follow_redirects and c.timeout.read == 30 for c in mcp_http[1])


async def test_interleaved_conversations_retry_quota_and_new_chat(
    hass, monkeypatch, mcp_http
):
    a = make_agent(hass, monkeypatch)
    b = make_agent(hass, monkeypatch)
    first, second = await asyncio.gather(ask(a), ask(b))
    first_id, second_id = [c["arguments"]["session_id"] for c in calls(mcp_http)]
    assert first_id != second_id
    mcp_http[2]["mode"] = "quota"
    await ask(a, conversation_id=first.conversation_id)
    assert (
        "quota exceeded"
        in a._client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    )
    mcp_http[2]["mode"] = "ok"
    await ask(b, "web_fetch", FETCH, second.conversation_id)
    await ask(a, conversation_id=first.conversation_id)
    assert [c["arguments"]["session_id"] for c in calls(mcp_http)[2:]] == [
        first_id,
        second_id,
        first_id,
    ]
    # A new chat gets a fresh UUID. Session cleanup releases the old mapping.
    session = hass.data[chat_session.DATA_CHAT_SESSION][first.conversation_id]
    session.async_cleanup()
    assert id(session) not in hass.data[parallel._SESSION_KEY]
    await ask(a)
    assert calls(mcp_http)[-1]["arguments"]["session_id"] not in {first_id, second_id}


@pytest.mark.parametrize("configured", ["[]", "null", "# no tools\n[]"])
def test_explicit_disabled_tools(hass, monkeypatch, configured):
    agent = make_agent(hass, monkeypatch, {CONF_FUNCTION_TOOLS: configured})
    assert agent._get_function_tools() == []


def test_saved_provider_and_overrides_remain_authoritative(hass, monkeypatch):
    saved = [
        {
            "spec": {"name": "web_search", "parameters": {"type": "object"}},
            "function": {
                "type": "template",
                "value_template": "my configured provider",
            },
        }
    ]
    agent = make_agent(hass, monkeypatch, {CONF_FUNCTION_TOOLS: yaml.safe_dump(saved)})
    tools = agent._get_function_tools()
    assert len(tools) == 1 and tools[0]["function"]["type"] == "template"
    assert "parallel" not in str(tools)


def test_default_schema_is_not_mutated(hass, monkeypatch):
    before = deepcopy(DEFAULT_CONF_FUNCTION_TOOLS)
    make_agent(hass, monkeypatch)._get_function_tools()
    assert before == DEFAULT_CONF_FUNCTION_TOOLS


@pytest.mark.parametrize(
    "options",
    [{CONF_FUNCTION_TOOLS: "[]"}, {CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: 0}],
)
async def test_disabled_or_limited_tools_make_no_mcp_request(
    hass, monkeypatch, mcp_http, options
):
    agent = make_agent(hass, monkeypatch, options)
    agent._client.chat.completions.create.return_value = scripted_response()
    await agent.async_process(
        ConversationInput(
            text="Hello",
            context=Context(),
            language="en",
            device_id=None,
            satellite_id=None,
            agent_id=agent.entity_id,
            conversation_id=None,
        )
    )
    kwargs = agent._client.chat.completions.create.call_args.kwargs
    if CONF_FUNCTION_TOOLS in options:
        assert "tools" not in kwargs
    else:
        assert kwargs["tool_choice"] == "none"
    assert not mcp_http[0]


@pytest.mark.parametrize("mode", ["http_error", "timeout", "malformed", "rpc_error"])
async def test_errors_return_to_model_and_release_clients(
    hass, monkeypatch, mcp_http, mode
):
    mcp_http[2]["mode"] = mode
    agent = make_agent(hass, monkeypatch)
    await ask(agent)
    assert (
        "Parallel web request"
        in agent._client.chat.completions.create.call_args.kwargs["messages"][-1][
            "content"
        ]
    )
    assert all(c.is_closed for c in mcp_http[1])


async def test_deadline_and_cancellation_release_clients(hass, monkeypatch, mcp_http):
    # Configure a short deadline directly after normal schema resolution for speed.
    agent = make_agent(hass, monkeypatch)
    tools = agent._get_function_tools()
    tool = next(t for t in tools if t["spec"]["name"] == "web_search")
    tool["function"]["timeout"] = 0.05
    monkeypatch.setattr(agent, "_get_function_tools", lambda: tools)
    mcp_http[2].update(mode="wait", started=asyncio.Event())
    await ask(agent)
    assert (
        "timed out"
        in agent._client.chat.completions.create.call_args.kwargs["messages"][-1][
            "content"
        ]
    )
    tool["function"]["timeout"] = 30
    mcp_http[2]["started"].clear()
    task = asyncio.create_task(ask(agent))
    await mcp_http[2]["started"].wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(c.is_closed for c in mcp_http[1])


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com",
        "https://user:secret@example.com",
        "https://",
        "http://[invalid",
    ],
)
async def test_invalid_urls_never_reach_mcp(hass, monkeypatch, mcp_http, url):
    agent = make_agent(hass, monkeypatch)
    await ask(agent, "web_fetch", {"urls": [url]})
    assert (
        "Invalid Parallel arguments"
        in agent._client.chat.completions.create.call_args.kwargs["messages"][-1][
            "content"
        ]
    )
    assert not mcp_http[0]


async def test_model_cannot_override_session_or_endpoint(hass, monkeypatch, mcp_http):
    agent = make_agent(hass, monkeypatch)
    await ask(
        agent,
        arguments={**SEARCH, "session_id": "injected", "url": "https://invalid.test"},
    )
    assert (
        "Invalid Parallel arguments"
        in agent._client.chat.completions.create.call_args.kwargs["messages"][-1][
            "content"
        ]
    )
    assert not mcp_http[0]


def test_provider_configuration_uses_registry_schema():
    provider = get_function("parallel")
    assert (
        provider.validate_schema({"type": "parallel", "name": "web_search"})["timeout"]
        == 30
    )
