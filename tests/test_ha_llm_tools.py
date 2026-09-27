"""HA-owned tools retain identity, isolation and the existing exchange contract."""

import asyncio
import builtins
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol

from custom_components.extended_openai_conversation_responses import ha_llm_tools
from custom_components.extended_openai_conversation_responses.agent_config import (
    AgentConfigError,
    agent_config_snapshot,
    configured_function_tools_from_data,
    normalize_agent_config,
    validate_function_tools,
)
from custom_components.extended_openai_conversation_responses.backup import (
    _safe_configuration,
)
from custom_components.extended_openai_conversation_responses.conversation import (
    ExtendedOpenAIAgentEntity,
)
from custom_components.extended_openai_conversation_responses.entity import (
    ExtendedOpenAIBaseLLMEntity,
    _format_structured_output,
)
from custom_components.extended_openai_conversation_responses.function_call_budget import (
    FunctionCallBudget,
)
from custom_components.extended_openai_conversation_responses.function_groups import (
    assemble_function_tools,
)
from custom_components.extended_openai_conversation_responses.ha_llm_tools import (
    ToolSnapshot,
    async_discover,
    caller_api_tools,
    current_snapshot,
    new_reference_tool,
    tool_snapshot_scope,
    validate_reference,
)
from custom_components.extended_openai_conversation_responses.ha_tool_result_compat import (
    tool_result_data,
)
from custom_components.extended_openai_conversation_responses.parallel_tool_execution import (
    is_parallel_safe_integration_tool,
)
from custom_components.extended_openai_conversation_responses.request import (
    format_function_tools,
)
from custom_components.extended_openai_conversation_responses.tool_exchange import (
    async_execute_tool_exchange,
)
from homeassistant.components import conversation, llm as llm_component
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm


class Echo(llm.Tool):
    name = "echo"
    # HA dev now requires the providing integration to identify its tools.
    integration = "extended_openai_conversation_responses"
    description = "Echo the supplied value"
    parameters = vol.Schema({vol.Required("value"): str})

    def __init__(self):
        self.calls = []

    async def async_call(self, hass, tool_input, llm_context):
        self.calls.append((tool_input, llm_context))
        payload = {
            "value": tool_input.tool_args["value"],
            "user": llm_context.context.user_id,
        }
        tool_result_type = getattr(llm, "ToolResult", None)
        return (
            tool_result_type(data=payload) if tool_result_type is not None else payload
        )


class TestAPI(llm.API):
    __test__ = False

    def __init__(self, hass, api_id="remote", tools=None):
        super().__init__(hass=hass, id=api_id, name="Remote source")
        self.tools = tools if tools is not None else [Echo()]
        self.contexts = []
        self.fail = False

    async def async_get_api_instance(self, context):
        self.contexts.append(context)
        if self.fail:
            raise HomeAssistantError("offline")
        return llm.APIInstance(
            api=self,
            api_prompt="Remote instructions",
            llm_context=context,
            tools=self.tools,
        )


def context(user="alice", device="speaker"):
    return llm.LLMContext(
        platform="extended_openai_conversation_responses",
        context=Context(user_id=user),
        language="en",
        assistant="conversation",
        device_id=device,
    )


def reference(source="one", name="echo", api="assist", kind="platform"):
    return {
        "type": "ha_llm",
        "source_type": kind,
        "source_id": source,
        "api_id": api,
        "tool_name": name,
    }


def group(tool):
    return {
        "id": "remote",
        "name": "Remote",
        "description": "Remote capabilities",
        "functions": [tool["spec"]["name"]],
        "enabled": True,
        "loading_mode": "on_demand",
    }


def register(hass, api):
    llm.async_register_api(hass, api)
    return api


def test_reference_roundtrip_and_schema_ownership():
    tool = new_reference_tool(reference(), set())
    config = normalize_agent_config(
        {"functions": [tool], "function_groups": [group(tool)]}
    )
    assert configured_function_tools_from_data(config) == [tool]
    restored = agent_config_snapshot(_safe_configuration(config))
    assert restored["functions"] == [tool]
    assert restored["function_groups"][0]["functions"] == [tool["spec"]["name"]]
    assert set(restored["functions"][0]["spec"]) == {"name"}
    assert json.loads(json.dumps(restored))["functions"] == [tool]


@pytest.mark.parametrize(
    "change",
    [
        {"tool_name": "*"},
        {"source_id": "*"},
        {"source_type": "integration"},
        {"api_id": ""},
        {"future_tools": True},
        {"tool_name": None},
    ],
)
def test_reference_rejects_wildcards_and_invalid_shapes(change):
    with pytest.raises(ValueError):
        validate_reference({**reference(), **change})


@pytest.mark.parametrize("extra", ["description", "parameters", "strict"])
def test_saved_external_schema_is_rejected(extra):
    tool = new_reference_tool(reference(), set())
    tool["spec"][extra] = {} if extra == "parameters" else "external"
    with pytest.raises(AgentConfigError):
        validate_function_tools([tool])


def test_aliases_handle_custom_and_source_collisions():
    first = new_reference_tool(reference(), set())
    assert new_reference_tool(reference(), set()) == first
    occupied = {first["spec"]["name"]}
    collision = new_reference_tool(reference(), occupied)
    other = new_reference_tool(reference(source="two"), occupied)
    assert (
        len({first["spec"]["name"], collision["spec"]["name"], other["spec"]["name"]})
        == 3
    )
    assert all(len(tool["spec"]["name"]) <= 64 for tool in (first, collision, other))
    assert collision["function"] == first["function"]
    with pytest.raises(AgentConfigError, match="already configured"):
        validate_function_tools([first, collision])


async def test_registered_api_live_schema_missing_return_and_no_future_tools(hass):
    api = register(hass, TestAPI(hass))
    first = await async_discover(hass, context())
    live = next(iter(first.tools.values()))
    selected = [live.reference]
    tool = new_reference_tool(live.reference, set())
    extra = Echo()
    extra.name = "future"
    api.tools.append(extra)
    api.tools[0].parameters = vol.Schema({vol.Required("changed"): int})
    changed = await async_discover(hass, context(), selected)
    assert len(changed.tools) == 1
    assert "changed" in changed.project([tool])[0]["spec"]["parameters"]["properties"]
    api.fail = True
    assert not (await async_discover(hass, context(), selected)).tools
    assert validate_function_tools([tool]) == [tool]
    api.fail = False
    assert (await async_discover(hass, context(), selected)).tools
    api.tools = []
    assert not (await async_discover(hass, context(), selected)).tools


async def test_duplicate_native_api_names_fail_closed_and_partial_failure(hass):
    register(hass, TestAPI(hass, tools=[Echo(), Echo()]))
    broken = register(hass, TestAPI(hass, "offline"))
    broken.fail = True
    register(hass, TestAPI(hass, "usable"))
    snapshot = await async_discover(hass, context())
    assert len(snapshot.tools) == 1
    assert next(iter(snapshot.tools.values())).reference["api_id"] == "usable"
    assert snapshot.unavailable_sources == ["offline"]


async def test_real_platform_contract_domains_context_prompts_and_collisions(hass):
    register(hass, llm_component.AssistAPI(hass))
    seen = []

    def contribute(hass_arg, llm_context, api_id):
        seen.append((hass_arg, llm_context, api_id))
        return llm_component.LLMTools(
            tools=[Echo()], prompt=f"Use device {llm_context.device_id}"
        )

    def broken(*args):
        raise RuntimeError("offline")

    hass.data[llm_component.DATA_PLATFORMS] = SimpleNamespace(
        async_get_platforms=AsyncMock(
            return_value={
                "one": SimpleNamespace(async_get_tools=contribute),
                "two": SimpleNamespace(async_get_tools=contribute),
                "broken": SimpleNamespace(async_get_tools=broken),
            }
        )
    )
    actual = context("nonadmin", "phone")
    snapshot = await async_discover(hass, actual)
    assert len(snapshot.tools) == 2
    assert {live.reference["source_id"] for live in snapshot.tools.values()} == {
        "one",
        "two",
    }
    assert all(item == (hass, actual, "assist") for item in seen)
    assert snapshot.unavailable_sources == ["assist/broken"]
    occupied = set()
    configured = [
        new_reference_tool(live.reference, occupied) for live in snapshot.tools.values()
    ]
    effective = snapshot.project(configured)
    assert len({tool["spec"]["name"] for tool in effective}) == 2
    assert "Use device phone" in snapshot.prompt_for(effective[:1])
    assert not snapshot.prompt_for([])
    assert "phone" not in json.dumps(configured)


async def test_concurrent_request_scopes_never_cross_users_or_tools(hass):
    api = register(hass, TestAPI(hass))
    arrived = asyncio.Event()
    count = 0

    async def request(user, device):
        nonlocal count
        current = context(user, device)
        snapshot = await async_discover(hass, current)
        with tool_snapshot_scope(snapshot):
            count += 1
            if count == 2:
                arrived.set()
            await arrived.wait()
            live = next(iter(current_snapshot().tools.values()))
            assert live.instance.llm_context is current
            result = await live.async_call(
                llm.ToolInput(tool_name="alias", tool_args={"value": device}, id=user)
            )
            assert result == {"value": device, "user": user}
        assert not current_snapshot().tools

    await asyncio.gather(request("alice", "kitchen"), request("bob", "phone"))
    assert len(api.contexts) == 2
    assert {call[0].id for call in api.tools[0].calls} == {"alice", "bob"}
    assert all(call[0].external is False for call in api.tools[0].calls)


async def test_groups_hide_schema_and_prompt_until_loaded(hass):
    register(hass, TestAPI(hass))
    snapshot = await async_discover(hass, context())
    tool = new_reference_tool(next(iter(snapshot.tools.values())).reference, set())
    configured = snapshot.project([tool])
    hidden = assemble_function_tools(configured, [group(tool)], set())
    assert hidden.configured_schemas_sent == 0
    assert snapshot.prompt_for(hidden.tools) == ""
    loaded = assemble_function_tools(configured, [group(tool)], {"remote"})
    assert loaded.configured_schemas_sent == 1
    assert "Remote instructions" in snapshot.prompt_for(loaded.tools)
    assert loaded.serialized_configured_schema_characters > 0
    assert not assemble_function_tools(ToolSnapshot().project([tool]), [], set()).tools
    configured[0]["enabled"] = False
    assert not assemble_function_tools(configured, [], set()).tools


@pytest.mark.parametrize("mode", ["responses", "chat_completions"])
async def test_provider_formats_use_live_schema_without_runtime_objects(hass, mode):
    register(hass, TestAPI(hass))
    snapshot = await async_discover(hass, context())
    tool = new_reference_tool(next(iter(snapshot.tools.values())).reference, set())
    formatted = format_function_tools(snapshot.project([tool]), mode)[0]
    spec = formatted if mode == "responses" else formatted["function"]
    assert "value" in spec["parameters"]["properties"]
    assert spec["strict"] is False
    assert "ha_available" not in json.dumps(formatted)
    assert "source_id" not in json.dumps(formatted)


async def test_caller_api_uses_existing_exchange_budget_and_single_execution(hass):
    api = TestAPI(hass)
    ctx = context()
    snapshot, tools = caller_api_tools(await api.async_get_api_instance(ctx))
    assert not is_parallel_safe_integration_tool(tools[0])
    entity = object.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    hass.auth.async_get_user.return_value = SimpleNamespace(is_active=True)
    chat_log = conversation.ChatLog(hass, "test")
    calls = [
        llm.ToolInput(
            tool_name=tools[0]["spec"]["name"],
            tool_args={"value": "hello"},
            id=str(i),
            external=True,
        )
        for i in range(2)
    ]
    chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(agent_id="conversation.test", tool_calls=calls)
    )
    with (
        tool_snapshot_scope(snapshot),
        pytest.raises(HomeAssistantError, match="limit"),
    ):
        await async_execute_tool_exchange(
            entity, chat_log, calls, tools, FunctionCallBudget(1), ctx, []
        )
    assert len(api.tools[0].calls) == 1
    results = [
        item
        for item in chat_log.content
        if isinstance(item, conversation.ToolResultContent)
    ]
    assert [item.tool_call_id for item in results] == ["0", "1"]


async def test_guest_denies_external_even_if_policy_allows_name(hass):
    tool = new_reference_tool(reference(), set())
    policy = SimpleNamespace(
        guest_active=True,
        legacy_function_flags=False,
        allows_configured_tool=lambda name: True,
    )
    filtered, groups = ExtendedOpenAIAgentEntity._filter_guest_tools_and_groups(
        [tool], [], policy
    )
    assert filtered == groups == []
    entity = object.__new__(ExtendedOpenAIAgentEntity)
    entity.hass = hass
    entity.entity_id = "conversation.guest"
    entity._effective_guest_policy = lambda: policy
    result = await entity._execute_function_tool(
        tool, llm.ToolInput(tool_name=tool["spec"]["name"], tool_args={}), context(), []
    )
    assert tool_result_data(result)
    assert not current_snapshot().tools


async def test_conversation_discovers_configured_ha_tools_for_the_request(hass):
    """The conversation boundary resolves saved HA references in caller context."""
    api = register(hass, TestAPI(hass))
    preview = await async_discover(hass, context())
    saved = new_reference_tool(next(iter(preview.tools.values())).reference, set())
    entity = object.__new__(ExtendedOpenAIAgentEntity)
    entity.hass = hass
    entity.subentry = SimpleNamespace(data={})
    entity._configured_function_tools_from_data = MagicMock(return_value=[saved])
    entity._effective_guest_policy = MagicMock(
        return_value=SimpleNamespace(guest_active=False)
    )
    expected = object()
    observed_names: list[str] = []

    async def handle_message(*_args):
        observed_names.extend(
            tool.tool.name for tool in current_snapshot().tools.values()
        )
        return expected

    entity._async_handle_message = AsyncMock(side_effect=handle_message)
    user_input = SimpleNamespace(as_llm_context=lambda _domain: context())
    chat_log = object()

    result = await entity._async_handle_message_with_ha_tools(
        user_input, chat_log, {"model": "test"}
    )

    assert result is expected
    assert observed_names == ["echo"]
    assert len(api.contexts) == 2
    entity._async_handle_message.assert_awaited_once_with(
        user_input, chat_log, {"model": "test"}
    )


async def test_source_schema_validates_before_side_effect(hass):
    api = register(hass, TestAPI(hass))
    live = next(iter((await async_discover(hass, context())).tools.values()))
    with pytest.raises(vol.Invalid):
        await live.async_call(llm.ToolInput(tool_name="echo", tool_args={"value": 123}))
    assert not api.tools[0].calls


async def test_mixed_group_prompt_tracks_actual_provider_rounds(hass):
    from custom_components.extended_openai_conversation_responses.const import (
        FUNCTION_GROUP_LOADER_TOOL_NAME,
    )
    from tests.test_tool_exchange_protocol import (
        _chat_log,
        _entity,
        _final_stream,
        _function_call_stream,
    )

    api = register(hass, TestAPI(hass))
    snapshot = await async_discover(hass, context())
    saved = new_reference_tool(next(iter(snapshot.tools.values())).reference, set())
    projected = snapshot.project([saved])
    loaded = set()
    sent = []
    entity = _entity(
        hass,
        [
            _function_call_stream(
                [("load", FUNCTION_GROUP_LOADER_TOOL_NAME, {"groups": ["remote"]})]
            ),
            _final_stream(),
        ],
    )

    async def create(**kwargs):
        sent.append(deepcopy(kwargs))
        return (
            _function_call_stream(
                [("load", FUNCTION_GROUP_LOADER_TOOL_NAME, {"groups": ["remote"]})]
            )
            if len(sent) == 1
            else _final_stream()
        )

    entity._client.responses.create.side_effect = create

    def factory():
        return assemble_function_tools(projected, [group(saved)], loaded).tools

    def load(groups):
        loaded.update(groups)
        return {"loaded": groups}

    with tool_snapshot_scope(snapshot):
        await entity._async_handle_chat_log(
            _chat_log(hass),
            [],
            [],
            context(),
            function_tools_factory=factory,
            function_group_loader=load,
        )
    assert len(sent) == 2
    assert "Remote instructions" not in json.dumps(sent[0])
    assert "Remote instructions" in json.dumps(sent[1])
    assert saved["spec"]["name"] not in json.dumps(sent[0])
    assert saved["spec"]["name"] in json.dumps(sent[1])
    assert not api.tools[0].calls


async def test_deactivation_during_discovery_denies_before_tool_call(hass):
    api = register(hass, TestAPI(hass))
    snapshot = await async_discover(hass, context())
    saved = new_reference_tool(next(iter(snapshot.tools.values())).reference, set())
    entity = object.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    guest = False
    entity._effective_guest_policy = lambda: SimpleNamespace(guest_active=guest)
    original = api.async_get_api_instance

    async def activate(ctx):
        nonlocal guest
        result = await original(ctx)
        guest = True
        return result

    api.async_get_api_instance = activate
    hass.auth.async_get_user.return_value = SimpleNamespace(is_active=True)
    result = await entity._execute_function_tool(
        saved,
        llm.ToolInput(tool_name=saved["spec"]["name"], tool_args={"value": "x"}),
        context(),
        [],
    )
    assert "Guest Mode" in str(tool_result_data(result))
    assert not api.tools[0].calls


async def test_removed_user_is_not_authorized_by_management_preview(hass):
    api = register(hass, TestAPI(hass))
    preview = await async_discover(hass, context("admin"))
    saved = new_reference_tool(next(iter(preview.tools.values())).reference, set())
    entity = object.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    hass.auth.async_get_user.return_value = None
    result = await entity._execute_function_tool(
        saved,
        llm.ToolInput(tool_name=saved["spec"]["name"], tool_args={"value": "x"}),
        context("removed"),
        [],
    )
    assert "no longer active" in str(tool_result_data(result))
    assert not api.tools[0].calls
    assert api.contexts[-1].context.user_id == "removed"


async def test_executor_does_not_interpret_external_delay_schema(hass, monkeypatch):
    from custom_components.extended_openai_conversation_responses import entity as base

    saved = new_reference_tool(reference(), set())
    saved["spec"]["parameters"] = {"anyOf": [{"type": "object"}]}
    live = SimpleNamespace(async_call=AsyncMock(return_value="external result"))
    snapshot = SimpleNamespace(caller_provided=True, tools={"reference": live})
    monkeypatch.setattr(base, "current_snapshot", lambda: snapshot)
    monkeypatch.setattr(base, "reference_key", lambda _ref: "reference")
    from custom_components.extended_openai_conversation_responses import (
        function_tool_resolution,
    )

    monkeypatch.setattr(
        function_tool_resolution,
        "latest_function_tool_for_execution",
        lambda *_args: saved,
    )
    monkeypatch.setattr(
        base,
        "async_execution_arguments",
        AsyncMock(
            side_effect=AssertionError("HA schema entered configured validation")
        ),
    )
    entity = object.__new__(ExtendedOpenAIBaseLLMEntity)
    entity.hass = hass
    entity.entity_id = "conversation.external"
    call = llm.ToolInput(
        tool_name=saved["spec"]["name"], tool_args={"delay": {"seconds": 10}}
    )
    result = await entity._execute_function_tool(saved, call, context(), [])
    assert tool_result_data(result) == {"result": "external result"}
    live.async_call.assert_awaited_once_with(call)


async def test_merged_display_names_are_not_discovered_for_persistence(hass):
    first = TestAPI(hass, "a")
    second = TestAPI(hass, "b")
    merged = llm.MergedAPI([first, second])
    register(hass, merged)

    # An explicitly empty selection is a hard no-op and must not instantiate
    # any API, while merged display names remain non-persistable.
    assert not (await async_discover(hass, context(), [])).tools
    assert first.contexts == []
    assert second.contexts == []
    assert not (await async_discover(hass, context())).tools

    # Caller-provided APIs are already request-bound, so ephemeral merged names
    # remain usable without becoming agent configuration references.
    instance = await merged.async_get_api_instance(context())
    snapshot, tools = caller_api_tools(instance)
    # Some Core versions namespace by display label; never choose first-wins.
    expected = 0 if instance.tools[0].name == instance.tools[1].name else 2
    assert len(snapshot.tools) == len(tools) == expected


@pytest.mark.parametrize("mode", ["responses", "chat_completions"])
async def test_real_provider_exchange_executes_ha_tool_once_and_removes_budgeted_prompt(
    hass, mode
):
    from openai.types.chat import ChatCompletionChunk

    from tests.test_tool_exchange_protocol import (
        FakeStream,
        _chat_log,
        _entity,
        _final_stream,
        _function_call_stream,
    )

    api = TestAPI(hass)
    ctx = context()
    instance = await api.async_get_api_instance(ctx)
    snapshot, tools = caller_api_tools(instance)
    name = tools[0]["spec"]["name"]
    if mode == "responses":
        streams = [
            _function_call_stream([("call", name, {"value": "hello"})]),
            _final_stream(),
        ]
    else:

        def chunk(delta, finish):
            return ChatCompletionChunk.model_validate(
                {
                    "id": "chat",
                    "created": 0,
                    "model": "test",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )

        streams = [
            FakeStream(
                [
                    chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": '{"value":"hello"}',
                                    },
                                }
                            ]
                        },
                        "tool_calls",
                    )
                ]
            ),
            FakeStream([chunk({"content": "Done"}, "stop")]),
        ]
    entity = _entity(hass, [], limit=1)
    entity.subentry.data["api_mode"] = mode
    sent = []

    async def create(**kwargs):
        sent.append(deepcopy(kwargs))
        return streams[len(sent) - 1]

    entity._client.responses.create.side_effect = create
    entity._client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=AsyncMock(side_effect=create))
    )
    hass.auth.async_get_user.return_value = SimpleNamespace(is_active=True)
    chat_log = _chat_log(hass)
    chat_log.llm_api = instance
    with tool_snapshot_scope(snapshot):
        await entity._async_handle_chat_log(chat_log, tools, [], ctx)
    assert len(sent) == 2
    assert len(api.tools[0].calls) == 1
    assert api.tools[0].calls[0][1] is ctx
    assert "Remote instructions" in json.dumps(sent[0])
    assert "Remote instructions" not in json.dumps(sent[1])
    results = [
        item
        for item in chat_log.content
        if isinstance(item, conversation.ToolResultContent)
    ]
    assert [item.tool_call_id for item in results] == ["call"]


async def test_ai_task_supplied_api_is_used_without_custom_functions(hass):
    from custom_components.extended_openai_conversation_responses.ai_task import (
        ExtendedOpenAITaskEntity,
    )

    api = TestAPI(hass)
    instance = await api.async_get_api_instance(context())
    entity = object.__new__(ExtendedOpenAITaskEntity)
    entity.hass = hass
    chat_log = SimpleNamespace(
        llm_api=instance,
        async_provide_llm_data=AsyncMock(),
        content=[conversation.AssistantContent(agent_id="task", content="Done")],
        conversation_id="task",
    )

    async def handle(log, **kwargs):
        assert kwargs["llm_context"] is instance.llm_context
        assert len(kwargs["function_tools"]) == 1
        assert kwargs["function_tools"][0]["function"]["type"] == "ha_llm"
        assert current_snapshot().caller_provided

    entity._async_handle_chat_log = AsyncMock(side_effect=handle)
    result = await entity._async_generate_data(
        SimpleNamespace(name="Task", structure=None), chat_log
    )
    assert result.data == "Done"
    chat_log.async_provide_llm_data.assert_awaited_once()
    assert not current_snapshot().tools


async def test_cached_source_instance_cannot_lend_its_admin_context(hass):
    api = register(hass, TestAPI(hass))
    cached = await api.async_get_api_instance(context("admin"))
    api.async_get_api_instance = AsyncMock(return_value=cached)
    alice = context("alice")
    bob = context("bob")
    first, second = await asyncio.gather(
        async_discover(hass, alice), async_discover(hass, bob)
    )
    first_live = next(iter(first.tools.values()))
    second_live = next(iter(second.tools.values()))
    assert first_live.instance is not second_live.instance
    assert first_live.instance.llm_context is alice
    assert second_live.instance.llm_context is bob
    assert cached.llm_context.context.user_id == "admin"


async def test_mutated_dispatch_list_cannot_switch_the_validated_tool(hass):
    api = register(hass, TestAPI(hass))
    snapshot = await async_discover(hass, context())
    live = next(iter(snapshot.tools.values()))
    live.instance.tools[:] = [Echo()]
    with pytest.raises(HomeAssistantError, match="changed before dispatch"):
        await live.async_call(
            llm.ToolInput(tool_name="echo", tool_args={"value": "hello"})
        )
    assert not api.tools[0].calls


async def test_ai_task_structured_output_uses_core_serializer_contract(hass):
    api = TestAPI(hass)
    instance = await api.async_get_api_instance(context())
    marker = object()

    def serializer(value):
        if value is marker:
            return {"type": "string", "enum": ["ready", "waiting"]}
        return llm.selector_serializer(value)

    instance.custom_serializer = serializer
    schema = vol.Schema({vol.Required("state"): marker, vol.Required("count"): int})
    result = _format_structured_output(schema, instance)
    assert result["properties"]["state"]["enum"] == ["ready", "waiting"]
    assert result["properties"]["count"]["type"] == "integer"
    assert set(result["required"]) == {"state", "count"}


# Consolidated HA LLM discovery/schema regressions.
async def test_schema_falls_back_when_ha_converter_is_unavailable(hass, monkeypatch):
    """Older HA converter availability still yields a live OpenAPI schema."""
    monkeypatch.setattr(llm, "to_openapi", None, raising=False)
    register(hass, TestAPI(hass))

    snapshot = await ha_llm_tools.async_discover(hass, context())

    live = next(iter(snapshot.tools.values()))
    assert live.spec["parameters"]["properties"]["value"]["type"] == "string"


async def test_namespaced_tools_are_preview_filtered_but_allowed_for_caller_api(hass):
    """Merged display names are not persisted, while caller-owned instances remain usable."""
    api = TestAPI(hass, tools=[llm.NamespacedTool("remote", Echo())])
    register(hass, api)

    preview = await ha_llm_tools.async_discover(hass, context())
    caller_snapshot, caller_tools = ha_llm_tools.caller_api_tools(
        await api.async_get_api_instance(context())
    )

    assert preview.tools == {}
    assert caller_snapshot.caller_provided is True
    assert len(caller_snapshot.tools) == 1
    assert len(caller_tools) == 1
    assert caller_tools[0]["ha_available"] is True


async def test_bad_tool_schema_is_isolated_from_other_tools(hass, monkeypatch, caplog):
    """One converter failure must not make a sibling HA tool unavailable."""
    broken = Echo()
    broken.name = "broken"
    usable = Echo()
    usable.name = "usable"
    register(hass, TestAPI(hass, tools=[broken, usable]))
    original_schema = ha_llm_tools._schema

    def schema(tool, serializer):
        if tool.name == "broken":
            raise ValueError("unsupported schema")
        return original_schema(tool, serializer)

    monkeypatch.setattr(ha_llm_tools, "_schema", schema)

    snapshot = await ha_llm_tools.async_discover(hass, context())

    assert [live.tool.name for live in snapshot.tools.values()] == ["usable"]
    assert "HA LLM tool schema unavailable" in caplog.text


async def test_selected_api_skips_unreferenced_registered_api(hass):
    """Reference-scoped discovery must not instantiate unrelated custom APIs."""
    selected_api = register(hass, TestAPI(hass, "selected"))
    unrelated = register(hass, TestAPI(hass, "unrelated"))
    preview = await ha_llm_tools.async_discover(hass, context())
    selected_reference = next(
        live.reference
        for live in preview.tools.values()
        if live.reference["api_id"] == "selected"
    )
    selected_api.contexts.clear()
    unrelated.contexts.clear()

    snapshot = await ha_llm_tools.async_discover(
        hass, context(), [selected_reference]
    )

    assert len(snapshot.tools) == 1
    assert next(iter(snapshot.tools.values())).reference["api_id"] == "selected"
    assert len(selected_api.contexts) == 1
    assert unrelated.contexts == []


async def test_assist_registry_unavailable_is_reported_without_raising(hass):
    """A missing contributor registry marks Assist unavailable rather than remapping it."""
    register(hass, llm_component.AssistAPI(hass))

    snapshot = await ha_llm_tools.async_discover(hass, context())

    assert snapshot.tools == {}
    assert snapshot.unavailable_sources == ["assist"]


async def test_platform_selection_and_empty_contribution_are_isolated(hass):
    """Only selected contributor domains run, and a platform may contribute no tools."""
    register(hass, llm_component.AssistAPI(hass))
    selected = MagicMock(
        return_value=llm_component.LLMTools(tools=[Echo()], prompt="selected")
    )
    ignored = MagicMock(
        return_value=llm_component.LLMTools(tools=[Echo()], prompt="ignored")
    )
    empty = MagicMock(return_value=None)
    hass.data[llm_component.DATA_PLATFORMS] = SimpleNamespace(
        async_get_platforms=AsyncMock(
            return_value={
                "empty": SimpleNamespace(async_get_tools=empty),
                "ignored": SimpleNamespace(async_get_tools=ignored),
                "selected": SimpleNamespace(async_get_tools=selected),
            }
        )
    )

    scoped = await ha_llm_tools.async_discover(
        hass, context(), [reference(source="selected")]
    )

    assert len(scoped.tools) == 1
    assert next(iter(scoped.tools.values())).reference["source_id"] == "selected"
    selected.assert_called_once()
    ignored.assert_not_called()
    empty.assert_not_called()

    preview = await ha_llm_tools.async_discover(hass, context())

    assert {live.reference["source_id"] for live in preview.tools.values()} == {
        "ignored",
        "selected",
    }
    assert preview.unavailable_sources == []
    empty.assert_called_once()


async def test_discovery_without_component_module_still_supports_custom_api(
    hass, monkeypatch
):
    """Optional platform discovery must not be required for opaque custom APIs."""
    api = register(hass, TestAPI(hass))
    real_import = builtins.__import__

    def import_without_component_llm(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "homeassistant.components" and "llm" in fromlist:
            raise ImportError("component unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_component_llm)

    snapshot = await ha_llm_tools.async_discover(hass, context())

    assert len(snapshot.tools) == 1
    assert next(iter(snapshot.tools.values())).reference["api_id"] == api.id


# Consolidated HA LLM serializer/prompt compatibility regressions.
def test_serializer_compat_preserves_ha_function_identity(monkeypatch) -> None:
    """Calling the local compatibility helper never replaces HA functions."""

    def existing(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(ha_llm_tools.llm, "to_openapi", existing, raising=False)

    ha_llm_tools.compatible_to_openapi({})

    assert ha_llm_tools.llm.to_openapi is existing


def test_serializer_compat_falls_back_when_probatio_is_unavailable(
    monkeypatch,
) -> None:
    """Keep HA untouched if the compatibility dependencies cannot be imported."""
    monkeypatch.setattr(ha_llm_tools.llm, "to_openapi", None, raising=False)
    monkeypatch.setitem(sys.modules, "probatio", None)

    ha_llm_tools.compatible_to_openapi({})

    assert ha_llm_tools.llm.to_openapi is None


def test_serializer_compat_selects_converter_and_translates_unsupported(
    monkeypatch,
) -> None:
    """Bridge unsupported sentinels in both Probatio and voluptuous directions."""
    probatio = ModuleType("probatio")
    voluptuous_openapi = ModuleType("voluptuous_openapi")

    probatio_unsupported = object()
    voluptuous_unsupported = object()

    class ProbatioSchema:
        pass

    probatio.Schema = ProbatioSchema  # type: ignore[attr-defined]
    probatio.UNSUPPORTED = probatio_unsupported  # type: ignore[attr-defined]
    voluptuous_openapi.UNSUPPORTED = voluptuous_unsupported  # type: ignore[attr-defined]

    calls: list[str] = []

    def probatio_convert(
        _schema: Any, *, custom_serializer: Any = None, **_kwargs: Any
    ) -> Any:
        calls.append("probatio")
        return custom_serializer("value")

    def voluptuous_convert(
        _schema: Any, *, custom_serializer: Any = None, **_kwargs: Any
    ) -> Any:
        calls.append("voluptuous")
        return custom_serializer("value")

    probatio.to_openapi = probatio_convert  # type: ignore[attr-defined]
    voluptuous_openapi.convert = voluptuous_convert  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "probatio", probatio)
    monkeypatch.setitem(sys.modules, "voluptuous_openapi", voluptuous_openapi)
    monkeypatch.setattr(ha_llm_tools.llm, "to_openapi", None, raising=False)

    converter = ha_llm_tools.compatible_to_openapi

    assert (
        converter(
            ProbatioSchema(), custom_serializer=lambda _value: voluptuous_unsupported
        )
        is probatio_unsupported
    )
    assert (
        converter(object(), custom_serializer=lambda _value: probatio_unsupported)
        is voluptuous_unsupported
    )
    assert calls == ["probatio", "voluptuous"]


def test_prompt_for_live_tool_without_source_prompt_still_adds_alias() -> None:
    """A prompt-less live tool still contributes its request-local name mapping."""
    reference = {
        "type": ha_llm_tools.TOOL_TYPE,
        "source_type": "api",
        "source_id": "example.Tool",
        "api_id": "example-api",
        "tool_name": "turn_on",
    }
    live = ha_llm_tools.LiveTool(
        reference=reference,
        tool=SimpleNamespace(name="turn_on"),
        instance=SimpleNamespace(),
        source_label="Example API",
        prompt="",
        spec={},
    )
    snapshot = ha_llm_tools.ToolSnapshot(tools={ha_llm_tools.reference_key(reference): live})

    rendered = snapshot.prompt_for(
        [
            {
                "spec": {"name": "ha_local_name"},
                "function": reference,
                "enabled": True,
            }
        ]
    )

    assert rendered == (
        "HA tool names in this request (source name = callable name):\n"
        "Example API: turn_on = ha_local_name"
    )
