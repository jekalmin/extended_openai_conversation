"""Tests for management frontend bootstrap and network optimizations."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import yaml

from homeassistant.exceptions import HomeAssistantError

from custom_components.extended_openai_conversation_responses import (
    agent_config,
    debug_ui,
    frontend_assets,
    management_function_repair as function_repair,
    management_loading_performance as loading,
    management_ui,
)
from custom_components.extended_openai_conversation_responses.agent_config import (
    agent_config_defaults,
    agent_config_snapshot,
)
from custom_components.extended_openai_conversation_responses.const import DOMAIN
from custom_components.extended_openai_conversation_responses.debug_ui import (
    async_setup_debug_ui,
)
from custom_components.extended_openai_conversation_responses.management_function_repair import (
    async_function_repair as _async_function_repair,
)
from custom_components.extended_openai_conversation_responses.management_loading_performance import (
    _agent_snapshot,
    async_agent_catalog,
    async_overview_summary,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    async_setup_management_ui,
)


class _HashableNamespace(SimpleNamespace):
    __hash__ = object.__hash__


class _Auth:
    async def async_get_users(self):
        return [SimpleNamespace(id="admin", name="Admin")]

    async def async_get_user(self, user_id):
        return SimpleNamespace(id=user_id, name="User")


class _ConfigEntries:
    def __init__(self, entry):
        self.entry = entry
        self.updates = 0

    def async_entries(self, domain):
        assert domain == DOMAIN
        return [self.entry]

    def async_get_entry(self, entry_id):
        return self.entry if entry_id == self.entry.entry_id else None

    def async_update_subentry(self, entry, subentry, *, data, title=None):
        assert entry is self.entry
        subentry.data = data
        if title is not None:
            subentry.title = title
        self.updates += 1


def _hass_with_agent():
    subentry = SimpleNamespace(
        subentry_id="agent-1",
        subentry_type="conversation",
        title="Jarvis",
        data=agent_config_defaults(),
    )
    entry = SimpleNamespace(
        entry_id="entry-1",
        domain=DOMAIN,
        title="Provider",
        data={},
        subentries={subentry.subentry_id: subentry},
    )
    hass = _HashableNamespace(
        data={},
        auth=_Auth(),
        config_entries=_ConfigEntries(entry),
    )
    return hass, entry, subentry


def _persisted_invalid_function_tools() -> str:
    return yaml.safe_dump(
        [
            {
                "spec": {
                    "name": "invalid_phone_tool",
                    "description": "Persisted schema containing malformed supported vocabulary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "phone": {
                                "type": "string",
                                "enum": ["home", "mobile"],
                                "minLength": "legacy",
                            }
                        },
                    },
                },
                "function": {"type": "native", "name": "execute_service"},
            }
        ],
        sort_keys=False,
    )


def _unavailable_native_tools() -> tuple[str, list[dict]]:
    """A valid sibling plus unavailable native implementations and references."""
    valid = yaml.safe_load(_persisted_invalid_function_tools())[0]
    valid["spec"]["name"] = "valid_phone_tool"
    valid["spec"]["parameters"]["properties"]["phone"]["minLength"] = 1
    missing = []
    for index in range(3):
        tool = deepcopy(valid)
        tool["spec"]["name"] = f"unavailable_phone_tool_{index}"
        tool["function"]["name"] = f"removed_native_implementation_{index}"
        missing.append(tool)
    groups = [
        {
            "id": "phone_tools", "name": "Phone Tools",
            "description": "Phone-related tools", "loading_mode": "always",
            "functions": [valid["spec"]["name"], missing[0]["spec"]["name"]],
        }
    ]
    return yaml.safe_dump([valid, *missing], sort_keys=False), groups


def test_runtime_function_quarantine_keeps_valid_siblings() -> None:
    invalid = yaml.safe_load(_persisted_invalid_function_tools())[0]
    valid = yaml.safe_load(_persisted_invalid_function_tools())[0]
    valid["spec"]["name"] = "valid_phone_tool"
    valid["spec"]["parameters"]["properties"]["phone"]["minLength"] = 1

    tools = quarantine._runtime_configured_function_tools(
        {"functions": yaml.safe_dump([valid, invalid], sort_keys=False)}
    )

    assert [tool["spec"]["name"] for tool in tools] == ["valid_phone_tool"]
    assert quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.get() == frozenset(
        {"invalid_phone_tool"}
    )
    assert quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.get() is False

    quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.set(frozenset())
    quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.set(False)


def test_runtime_group_quarantine_drops_only_quarantined_references(
    monkeypatch,
) -> None:
    from custom_components.extended_openai_conversation_responses import agent_config

    captured = {}

    def validate(groups, function_tools):
        captured["groups"] = groups
        captured["function_tools"] = function_tools
        return groups

    monkeypatch.setattr(agent_config, "validate_function_groups", validate)
    quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.set(frozenset({"broken_tool"}))
    quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.set(False)
    tools = [{"spec": {"name": "good_tool"}}]

    result = quarantine._runtime_validate_function_groups(
        [
            {
                "id": "test",
                "functions": ["good_tool", "broken_tool", "unrelated_missing_tool"],
            }
        ],
        tools,
    )

    assert result[0]["functions"] == ["good_tool", "unrelated_missing_tool"]
    assert captured["function_tools"] is tools

    quarantine._RUNTIME_QUARANTINED_FUNCTION_NAMES.set(frozenset())
    quarantine._RUNTIME_QUARANTINE_ALL_FUNCTIONS.set(False)


async def test_frontend_manifest_load_is_executor_backed_and_cached(
    monkeypatch,
) -> None:
    urls = {
        "management": f"/{DOMAIN}/frontend/assets/management-test.js",
    }
    loader = MagicMock(return_value=urls)
    register_static = AsyncMock()
    executor = AsyncMock(side_effect=lambda callback: callback())
    hass = SimpleNamespace(
        data={},
        async_add_executor_job=executor,
        http=SimpleNamespace(async_register_static_paths=register_static),
    )
    monkeypatch.setattr(frontend_assets, "_load_entry_urls_sync", loader)

    await frontend_assets.async_register_frontend_assets(hass)
    await frontend_assets.async_register_frontend_assets(hass)

    executor.assert_awaited_once_with(loader)
    loader.assert_called_once_with()
    register_static.assert_awaited_once()
    assert frontend_assets.frontend_entry_url(hass, "management") == urls["management"]


def test_cached_setup_uses_shared_production_asset_boundary() -> None:
    hass = SimpleNamespace(
        data={
            f"{DOMAIN}.frontend_entry_urls": {
                "management": f"/{DOMAIN}/frontend/assets/management-test.js"
            }
        }
    )
    assert management_ui.frontend_entry_url(hass, "management").startswith(
        f"/{DOMAIN}/frontend/assets/management-"
    )


def test_agent_snapshot_accepts_frontend_normalized_function_tools() -> None:
    hass, entry, subentry = _hass_with_agent()
    config = agent_config_snapshot(subentry.data)
    assert isinstance(config["functions"], list)

    result = _agent_snapshot(
        hass,
        entry,
        subentry,
        config=config,
        title="Updated Jarvis",
    )

    assert result["title"] == "Updated Jarvis"
    assert result["function_count"] == sum(
        tool.get("enabled", True) is True for tool in config["functions"]
    )


def test_agent_snapshot_projects_configured_temporary_memory_mode() -> None:
    hass, entry, subentry = _hass_with_agent()
    subentry.data["temporary_memory"] = "eager"

    result = _agent_snapshot(hass, entry, subentry)

    assert result["temporary_memory"] == "eager"


def test_agent_snapshot_keeps_invalid_function_tool_agent_discoverable() -> None:
    hass, entry, subentry = _hass_with_agent()
    subentry.data["functions"] = _persisted_invalid_function_tools()
    function_repair._health_cache.clear()

    result = _agent_snapshot(hass, entry, subentry)

    assert result["title"] == "Jarvis"
    assert result["function_count"] is None
    assert "configuration_issue" not in result

    function_repair.management_function_tool_health(dict(subentry.data))
    result = _agent_snapshot(hass, entry, subentry)
    assert result["function_count"] == 0
    assert result["configuration_issue"]["field"] == "functions"
    assert result["configuration_issue"]["repairable"] is True
    assert "minLength" in result["configuration_issue"]["message"]


def test_invalid_function_tool_health_is_cached_per_persisted_revision(
    monkeypatch,
) -> None:
    function_repair._cached_isolated_function_tools.cache_clear()
    function_repair._cached_function_tool_state.cache_clear()
    function_repair._health_cache.clear()
    original = function_repair._isolate_function_tools_uncached
    calls = 0

    def counted(options):
        nonlocal calls
        calls += 1
        return original(options)

    monkeypatch.setattr(function_repair, "_isolate_function_tools_uncached", counted)
    options = {"functions": _persisted_invalid_function_tools()}

    first = function_repair.management_function_tool_health(options)
    second = function_repair.management_function_tool_health(options)

    assert calls == 1
    assert first == second
    assert first["invalid_count"] == 1
    assert "minLength" in first["validation_error"]


def test_function_health_peek_tracks_tool_mutations_and_restore(monkeypatch) -> None:
    monkeypatch.setattr(function_repair, "_health_cache", function_repair.OrderedDict())
    projections = Mock(side_effect=lambda options: {
        "enabled_count": len(options["functions"]),
        "validation_error": None,
    })
    monkeypatch.setattr(function_repair, "_uncached_function_tool_health", projections)
    original = {"functions": [{"enabled": True}], "function_groups": []}
    disabled = {"functions": [{"enabled": False}], "function_groups": []}
    removed = {"functions": [], "function_groups": []}

    assert function_repair.peek_function_tool_health(original) is None
    assert function_repair.management_function_tool_health(original)["enabled_count"] == 1
    assert function_repair.management_function_tool_health(original)["enabled_count"] == 1
    assert projections.call_count == 1
    assert function_repair.peek_function_tool_health(disabled) is None
    function_repair.management_function_tool_health(disabled)
    assert function_repair.peek_function_tool_health(removed) is None
    function_repair.management_function_tool_health(removed)
    assert projections.call_count == 3

    # A group-only change leaves the Function Tool projection unchanged.
    grouped = {**original, "function_groups": [{"id": "group"}]}
    assert function_repair.peek_function_tool_health(grouped)["enabled_count"] == 1
    # Restore/import of the original persisted tools can reuse their old projection.
    assert function_repair.peek_function_tool_health(original)["enabled_count"] == 1


def test_agent_config_revision_does_not_validate_persisted_config(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data["functions"] = _persisted_invalid_function_tools()
    monkeypatch.setattr(
        function_repair,
        "agent_config_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("revision checks must not validate configuration")
        ),
    )

    revision = function_repair.agent_config_revision(subentry.data, subentry.title)

    assert isinstance(revision, str)
    assert len(revision) == 64


async def test_configuration_get_caches_normalized_persisted_snapshot(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data = {**subentry.data, "max_tokens": 777}
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    management_ui._cached_configuration_defaults()
    original = function_repair.agent_config_snapshot
    tool_validator = Mock(wraps=agent_config.validate_function_tools)
    monkeypatch.setattr(agent_config, "validate_function_tools", tool_validator)
    original_revision = function_repair.agent_config_revision_from_snapshot
    calls = 0
    revision_calls = 0

    def counted(data):
        nonlocal calls
        calls += 1
        return original(data)

    def counted_revision(data, title):
        nonlocal revision_calls
        revision_calls += 1
        return original_revision(data, title)

    monkeypatch.setattr(function_repair, "agent_config_snapshot", counted)
    monkeypatch.setattr(
        function_repair, "agent_config_revision_from_snapshot", counted_revision
    )
    monkeypatch.setattr(
        management_ui,
        "decorate_configuration_result",
        lambda _hass, _entry_data, result, **_kwargs: result,
    )

    message = {
        "entry_id": "entry-1",
        "subentry_id": "agent-1",
        "section": "configuration",
        "action": "get",
    }
    first = await management_ui.async_management_command(
        hass, "admin", True, message
    )
    second = await management_ui.async_management_command(
        hass, "admin", True, message
    )

    assert calls == 1
    assert tool_validator.call_count == 1
    assert revision_calls == 1
    assert first["_performance"]["snapshot_cache_hit"] is False
    assert second["_performance"]["snapshot_cache_hit"] is True
    assert first["config"] == second["config"]
    assert first["revision"] == second["revision"]
    first["config"]["chat_model"] = "changed locally"
    assert second["config"]["chat_model"] != "changed locally"


async def test_configuration_get_reuses_validated_exact_defaults(monkeypatch, caplog) -> None:
    hass, _entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    management_ui._cached_configuration_defaults.cache_clear()
    build_defaults = Mock(wraps=management_ui.agent_config_snapshot)
    build_persisted = Mock(wraps=function_repair.agent_config_snapshot)
    monkeypatch.setattr(management_ui, "agent_config_snapshot", build_defaults)
    monkeypatch.setattr(function_repair, "agent_config_snapshot", build_persisted)
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1",
        "section": "configuration", "action": "get",
    }

    first = await management_ui.async_management_command(hass, "admin", True, message)
    second = await management_ui.async_management_command(hass, "admin", True, message)

    build_defaults.assert_called_once()
    build_persisted.assert_not_called()
    assert first["_performance"]["default_snapshot_reused"] is True
    assert second["_performance"]["snapshot_cache_hit"] is True
    assert second["_performance"]["default_snapshot_reused"] is False
    assert first["config"] == second["config"]
    first["config"]["chat_model"] = "changed locally"
    assert second["config"]["chat_model"] != "changed locally"
    assert not [record for record in caplog.records if record.levelno >= 30]

    subentry.data = {**subentry.data, "max_tokens": 777}
    changed = await management_ui.async_management_command(hass, "admin", True, message)
    build_persisted.assert_called_once()
    assert changed["config"]["max_tokens"] == 777
    assert changed["_performance"]["snapshot_cache_hit"] is False


def test_persisted_projection_tracks_title_and_authoritative_data_replacement(
    monkeypatch,
) -> None:
    _hass, _entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    original = function_repair.persisted_config_projection(subentry)
    assert function_repair.persisted_config_projection(subentry) is original

    subentry.title = "Renamed"
    renamed = function_repair.persisted_config_projection(subentry)
    assert renamed.revision != original.revision
    assert renamed.snapshot is None
    with pytest.raises(HomeAssistantError, match="changed in another tab"):
        function_repair.require_agent_config_revision(subentry, original.revision)

    # Import/restore and config-flow updates replace the persisted data mapping.
    subentry.data = {**subentry.data, "max_tokens": 777}
    replaced = function_repair.persisted_config_projection(subentry)
    assert replaced.revision != renamed.revision
    assert replaced.snapshot is None
    assert function_repair.normalized_persisted_config_snapshot(replaced)[0]["max_tokens"] == 777

    recreated = SimpleNamespace(
        subentry_id=subentry.subentry_id,
        title=subentry.title,
        data=agent_config_defaults(),
    )
    new_projection = function_repair.persisted_config_projection(recreated)
    assert new_projection is not replaced
    assert new_projection.revision != replaced.revision


@pytest.mark.parametrize(
    ("field", "intermediate"),
    [
        ("functions", []),
        ("function_groups", [{"id": "lights", "name": "Lighting", "functions": ["demo"]}]),
        ("guest_mode_enabled", False),
        ("voice_device_mappings", {"kitchen": "user:one"}),
        ("exposed_entities_enabled", False),
    ],
)
async def test_agent_revision_rejects_aba_while_writer_is_suspended(
    monkeypatch, field, intermediate
) -> None:
    """An in-flight management writer cannot accept a restored old value."""
    _hass, _entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    original = deepcopy(subentry.data)
    assert original[field] != intermediate
    expected = function_repair.persisted_config_projection(subentry).revision
    entered, resume = asyncio.Event(), asyncio.Event()

    async def stale_writer() -> None:
        entered.set()
        await resume.wait()
        function_repair.require_agent_config_revision(subentry, expected)

    task = asyncio.create_task(stale_writer())
    await entered.wait()
    subentry.data = {**original, field: intermediate}
    assert function_repair.persisted_config_projection(subentry).revision != expected
    subentry.data = deepcopy(original)
    resume.set()
    with pytest.raises(HomeAssistantError, match="changed in another tab"):
        await task
    assert dict(subentry.data) == original


def test_agent_revision_lineage_survives_projection_cache_eviction(monkeypatch) -> None:
    """A busy installation must not turn an evicted stale token valid again."""
    class Agent:
        pass

    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    monkeypatch.setattr(function_repair, "_revision_lineages", {})
    monkeypatch.setattr(function_repair, "_PROJECTION_CACHE_LIMIT", 2)
    target = Agent()
    target.data = agent_config_defaults()
    target.title = "ABA target"
    target.subentry_id = "aba"
    original = deepcopy(target.data)
    stale = function_repair.persisted_config_projection(target).revision
    others = []
    for index in range(3):
        other = Agent()
        other.data = agent_config_defaults()
        other.title = f"Other {index}"
        other.subentry_id = f"other-{index}"
        others.append(other)
        function_repair.persisted_config_projection(other)
    assert id(target) not in function_repair._persisted_projections

    target.data = {**original, "guest_mode_enabled": False}
    target.data = deepcopy(original)
    with pytest.raises(HomeAssistantError, match="changed in another tab"):
        function_repair.require_agent_config_revision(target, stale)


async def test_import_replaces_cached_configuration_projection(monkeypatch) -> None:
    hass, _entry, _subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    monkeypatch.setattr(
        management_ui,
        "decorate_configuration_result",
        lambda _hass, _entry_data, result, **_kwargs: result,
    )
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1",
        "section": "configuration", "action": "get",
    }
    before = await management_ui.async_management_command(hass, "admin", True, message)
    replacement = agent_config_defaults()
    replacement["max_tokens"] = 777
    imported = await management_ui.async_management_command(
        hass, "admin", True, {
            **message,
            "action": "import",
            "confirm": True,
            "document": {
                "schema": "extended_openai_conversation.agent",
                "version": management_ui.AGENT_CONFIG_EXPORT_VERSION,
                "title": "Imported",
                "config": replacement,
            },
        },
    )
    after = await management_ui.async_management_command(hass, "admin", True, message)
    assert after["title"] == "Imported"
    assert after["config"]["max_tokens"] == 777
    assert after["revision"] == imported["revision"]
    assert after["revision"] != before["revision"]


def test_function_mutation_seeds_next_persisted_read(monkeypatch) -> None:
    hass, entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    before = function_repair.persisted_config_projection(subentry)
    tools = yaml.safe_load(subentry.data["functions"])
    tools[0]["enabled"] = False
    groups = subentry.data["function_groups"]
    saved = function_repair.persist_valid_function_configuration(
        hass, entry, subentry, tools, groups,
        expected_revision=before.revision,
    )
    monkeypatch.setattr(
        function_repair,
        "agent_config_snapshot",
        Mock(side_effect=AssertionError("mutation response should seed the next read")),
    )
    after = function_repair.persisted_config_projection(subentry)
    assert after.revision == saved["revision"]
    assert after.revision != before.revision


async def test_guest_mode_primary_get_skips_heavy_catalogues(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data[management_ui.CONF_GUEST_POLICY_VERSION] = (
        management_ui.GUEST_POLICY_VERSION
    )
    guest = SimpleNamespace(
        status=lambda: {"state": "inactive", "currently_active": False}
    )
    monkeypatch.setattr(management_ui, "async_get_guest_mode", AsyncMock(return_value=guest))
    monkeypatch.setattr(
        management_ui,
        "async_get_knowledge",
        AsyncMock(side_effect=AssertionError("Knowledge must stay off the primary path")),
    )
    monkeypatch.setattr(
        management_ui,
        "get_exposed_entities",
        MagicMock(side_effect=AssertionError("entity catalog must stay off primary path")),
    )
    monkeypatch.setattr(
        management_ui,
        "configured_function_tools_from_data",
        MagicMock(side_effect=AssertionError("tool catalog must stay off primary path")),
    )

    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "guest_mode",
            "action": "get",
        },
    )

    assert result["status"]["state"] == "inactive"
    assert result["legacy_policy"] is False
    assert "policy" not in result
    assert "knowledge_sources" not in result


async def test_guest_mode_details_reuses_one_exposed_entity_projection(monkeypatch) -> None:
    hass, _entry, _subentry = _hass_with_agent()
    guest = SimpleNamespace(
        status=lambda: {"state": "inactive", "currently_active": False}
    )
    policy = SimpleNamespace(as_diagnostics=lambda: {"guest_active": False})
    library = SimpleNamespace(async_list=AsyncMock(return_value=[]))
    exposed = [{"entity_id": "light.kitchen"}, {"entity_id": "switch.fan"}]
    exposed_loader = MagicMock(return_value=exposed)

    monkeypatch.setattr(management_ui, "async_get_guest_mode", AsyncMock(return_value=guest))
    monkeypatch.setattr(management_ui, "async_get_knowledge", AsyncMock(return_value=library))
    monkeypatch.setattr(management_ui, "configured_function_tools_from_data", lambda _data: [])
    monkeypatch.setattr(management_ui, "validate_function_groups", lambda _groups, _tools: [])
    monkeypatch.setattr(management_ui, "get_exposed_entities", exposed_loader)

    def resolve_policy(_hass, _options, _manager, _tools, *, exposed_entities=None):
        assert exposed_entities is exposed
        return policy

    monkeypatch.setattr(management_ui, "resolve_guest_policy", resolve_policy)

    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "guest_mode",
            "action": "details",
        },
    )

    exposed_loader.assert_called_once_with(hass)
    assert result["domains"] == ["light", "switch"]
    assert result["policy"]["guest_active"] is False
    assert result["_performance"]["total_ms"] >= 0


async def test_agent_catalog_does_not_initialize_per_agent_managers(
    monkeypatch,
) -> None:
    hass, _entry, _subentry = _hass_with_agent()
    for name in (
        "async_get_usage",
        "async_get_memory",
        "async_get_knowledge",
        "async_get_guest_mode",
        "async_get_archive",
        "async_scope_catalog_projection",
    ):
        monkeypatch.setattr(
            "custom_components.extended_openai_conversation_responses."
            f"management_loading_performance.{name}",
            AsyncMock(side_effect=AssertionError(f"{name} should not be called")),
        )

    result = await async_agent_catalog(hass, "admin", True)

    assert [agent["title"] for agent in result["agents"]] == ["Jarvis"]
    assert result["agents"][0]["model"] == agent_config_defaults()["chat_model"]
    assert result["is_admin"] is True
    assert result["_performance"]["agent_count"] == 1
    assert result["_performance"]["total_ms"] >= 0
    assert result["_performance"]["snapshots"][0]["total_ms"] >= 0


async def test_cold_catalog_skips_tool_validation_for_multiple_agents(monkeypatch) -> None:
    hass, entry, subentry = _hass_with_agent()
    second = SimpleNamespace(
        subentry_id="agent-2", subentry_type="conversation", title="Second",
        data={**agent_config_defaults(), "functions": _persisted_invalid_function_tools()},
    )
    entry.subentries[second.subentry_id] = second
    function_repair._health_cache.clear()
    monkeypatch.setattr(
        loading, "management_function_tool_health",
        Mock(side_effect=AssertionError("catalog must not validate tools")),
    )
    monkeypatch.setattr(
        function_repair, "_uncached_function_tool_health",
        Mock(side_effect=AssertionError("catalog must only peek")),
    )

    result = await async_agent_catalog(hass, "admin", True)

    assert len(result["agents"]) == 2
    assert all(agent["function_count"] is None for agent in result["agents"])


async def test_agent_catalog_keeps_invalid_function_tool_agent_visible(
    monkeypatch,
) -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data["functions"] = _persisted_invalid_function_tools()
    function_repair._health_cache.clear()
    result = await async_agent_catalog(hass, "admin", True)

    assert [agent["subentry_id"] for agent in result["agents"]] == ["agent-1"]
    assert "configuration_issue" not in result["agents"][0]
    function_repair.management_function_tool_health(dict(subentry.data))
    result = await async_agent_catalog(hass, "admin", True)
    issue = result["agents"][0]["configuration_issue"]
    assert issue["field"] == "functions"
    assert issue["repairable"] is True
    assert "minLength" in issue["message"]


async def test_function_repair_get_exposes_invalid_persisted_tools_without_normalizing() -> (
    None
):
    hass, _entry, subentry = _hass_with_agent()
    subentry.data["functions"] = _persisted_invalid_function_tools()

    result = await _async_function_repair(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "action": "get",
        },
    )

    assert (
        result["tools"][0]["spec"]["parameters"]["properties"]["phone"]["minLength"]
        == "legacy"
    )
    assert "minLength" in result["validation_error"]
    assert isinstance(result["revision"], str)
    assert hass.config_entries.updates == 0


async def test_function_repair_save_is_atomic_and_preserves_unrelated_data() -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data["functions"] = _persisted_invalid_function_tools()
    subentry.data["repair_sentinel"] = {"nested": ["leave", "untouched"]}
    original_sentinel = subentry.data["repair_sentinel"]
    repair = await _async_function_repair(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "action": "get",
        },
    )

    result = await _async_function_repair(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "action": "save",
            "revision": repair["revision"],
            "tools": [],
        },
    )

    assert result["valid"] is True
    assert yaml.safe_load(subentry.data["functions"]) == []
    assert subentry.data["repair_sentinel"] is original_sentinel
    assert subentry.data["repair_sentinel"] == {"nested": ["leave", "untouched"]}
    assert hass.config_entries.updates == 1


async def test_full_get_recovers_from_malformed_tools_and_repair(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    subentry.data = {**subentry.data, "functions": _persisted_invalid_function_tools()}
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1",
        "section": "configuration", "action": "get",
    }
    broken = await management_ui.async_management_command(hass, "admin", True, message)
    assert broken["function_repair"]["invalid_count"] == 1
    assert broken["_performance"]["function_issue_check_ms"] >= 0
    assert broken["_performance"]["repair_projection_ms"] >= 0

    repaired = await _async_function_repair(
        hass, "admin", True,
        {
            "entry_id": "entry-1", "subentry_id": "agent-1", "action": "save",
            "revision": broken["revision"], "tools": [],
        },
    )
    normal = await management_ui.async_management_command(hass, "admin", True, message)
    assert repaired["valid"] is True
    assert "function_repair" not in normal
    assert normal["config"]["functions"] == []
    assert normal["revision"] == repaired["revision"]


async def test_unavailable_native_full_get_reuses_one_quarantine_state(monkeypatch) -> None:
    hass, entry, subentry = _hass_with_agent()
    raw_tools, groups = _unavailable_native_tools()
    subentry.data = {
        **subentry.data, "functions": raw_tools, "function_groups": groups,
    }
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    function_repair._cached_isolated_function_tools.cache_clear()
    isolate = Mock(wraps=function_repair._isolate_function_tools_uncached)
    monkeypatch.setattr(function_repair, "_isolate_function_tools_uncached", isolate)
    strict = Mock(side_effect=AssertionError("repairable native tools must skip strict snapshot"))
    monkeypatch.setattr(function_repair, "agent_config_snapshot", strict)
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1",
        "section": "configuration", "action": "get",
    }

    cold = await management_ui.async_management_command(hass, "admin", True, message)
    repeated = await management_ui.async_management_command(hass, "admin", True, message)
    legacy = function_repair._safe_configuration_payload(
        hass, management_ui, loading, entry, subentry
    )
    for field in ("config", "defaults", "options", "function_repair"):
        assert cold[field] == legacy[field]

    strict.assert_not_called()
    assert isolate.call_count == 1
    assert cold["_performance"]["strict_snapshot_skipped_for_repair"] is True
    assert cold["_performance"]["repair_state_cache_hit"] is False
    assert repeated["_performance"]["repair_state_cache_hit"] is True
    assert cold["function_repair"]["invalid_count"] == 3
    assert [tool["spec"]["name"] for tool in cold["config"]["functions"]] == [
        "valid_phone_tool"
    ]
    assert cold["function_repair"]["group_issues"][0]["unavailable_functions"] == [
        "unavailable_phone_tool_0"
    ]
    assert cold["function_repair"]["persisted_groups"] == groups
    assert subentry.data["functions"] == raw_tools
    cold["function_repair"]["invalid_tools"].clear()
    cold["config"]["functions"].clear()
    assert len(repeated["function_repair"]["invalid_tools"]) == 3
    assert len(repeated["config"]["functions"]) == 1

    # A non-Function replacement keeps the validated quarantine state.
    subentry.data = {**subentry.data, "max_tokens": 777}
    unrelated = await management_ui.async_management_command(hass, "admin", True, message)
    assert unrelated["_performance"]["repair_state_cache_hit"] is True
    assert unrelated["config"]["max_tokens"] == 777
    assert isolate.call_count == 1

    # Both authoritative Function fields cause a fresh isolation when changed.
    updated_tools = yaml.safe_load(raw_tools)
    updated_tools[-1]["spec"]["name"] = "renamed_unavailable"
    subentry.data = {
        **subentry.data,
        "functions": yaml.safe_dump(updated_tools, sort_keys=False),
    }
    await management_ui.async_management_command(hass, "admin", True, message)
    assert isolate.call_count == 2
    subentry.data = {**subentry.data, "function_groups": []}
    regrouped = await management_ui.async_management_command(hass, "admin", True, message)
    assert regrouped["_performance"]["repair_state_cache_hit"] is False
    assert regrouped["function_repair"]["group_issues"] == []
    assert isolate.call_count == 2  # Tool isolation is keyed only to Function Tools.


async def test_repairable_read_routes_have_distinct_backend_work(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    raw_tools, groups = _unavailable_native_tools()
    subentry.data = {**subentry.data, "functions": raw_tools, "function_groups": groups}
    original_data = deepcopy(subentry.data)
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    message = {"entry_id": "entry-1", "subentry_id": "agent-1"}
    normal = await management_ui.async_management_command(
        hass, "admin", True, {**message, "section": "configuration", "action": "get"}
    )
    assert normal["function_repair"]["group_issues"]
    assert "_performance" in normal

    normalize = Mock(wraps=loading._snapshot_normalized_configuration)
    monkeypatch.setattr(loading, "_snapshot_normalized_configuration", normalize)
    repeated = await management_ui.async_management_command(
        hass, "admin", True, {**message, "section": "configuration", "action": "get"}
    )
    assert repeated["_performance"]["repair_state_cache_hit"] is True
    normalize.assert_not_called()

    legacy = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "section": "function_repair", "action": "configuration_get"},
    )
    assert normalize.call_count == 2  # Full config and defaults are rebuilt.
    assert "_performance" not in legacy
    for field in ("config", "function_repair", "revision"):
        assert legacy[field] == repeated[field]

    normalize.reset_mock()
    retention = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "section": "configuration", "action": "retention_get"},
    )
    normalize.assert_not_called()
    assert retention["projection"] == "retention"
    assert set(retention["config"]) == {
        "usage_request_retention_days", "usage_run_retention_days"
    }
    assert "function_repair" not in retention
    assert "defaults" not in retention
    assert "model_capabilities" not in retention
    assert retention["revision"] == repeated["revision"]
    assert subentry.data == original_data


async def test_valid_native_full_get_keeps_strict_snapshot_path(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    raw_tools, _groups = _unavailable_native_tools()
    subentry.data = {
        **subentry.data,
        "functions": yaml.safe_dump(yaml.safe_load(raw_tools)[:1], sort_keys=False),
        "function_groups": [],
    }
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    strict = Mock(wraps=function_repair.agent_config_snapshot)
    monkeypatch.setattr(function_repair, "agent_config_snapshot", strict)
    monkeypatch.setattr(
        management_ui, "repair_state_for_projection",
        Mock(side_effect=AssertionError("valid tools must not enter repair isolation")),
    )
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1",
        "section": "configuration", "action": "get",
    }

    cold = await management_ui.async_management_command(hass, "admin", True, message)
    warm = await management_ui.async_management_command(hass, "admin", True, message)

    strict.assert_called_once()
    assert "function_repair" not in cold
    assert cold["_performance"]["snapshot_cache_hit"] is False
    assert warm["_performance"]["snapshot_cache_hit"] is True
    assert cold["config"] == warm["config"]


async def test_repair_preflight_keeps_valid_sibling_collection_checks(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    raw_tools, _groups = _unavailable_native_tools()
    tools = yaml.safe_load(raw_tools)
    duplicate = deepcopy(tools[0])
    tools.insert(1, duplicate)
    subentry.data = {
        **subentry.data,
        "functions": yaml.safe_dump(tools, sort_keys=False),
        "function_groups": [],
    }
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    function_repair._cached_isolated_function_tools.cache_clear()

    with pytest.raises(agent_config.AgentConfigError, match="duplicate tool name"):
        await management_ui.async_management_command(
            hass, "admin", True,
            {
                "entry_id": "entry-1", "subentry_id": "agent-1",
                "section": "configuration", "action": "get",
            },
        )


async def test_delete_quarantined_tools_immediately_restores_normal_get(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    raw_tools, groups = _unavailable_native_tools()
    subentry.data = {
        **subentry.data, "functions": raw_tools, "function_groups": groups,
    }
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1",
        "section": "configuration", "action": "get",
    }
    current = await management_ui.async_management_command(hass, "admin", True, message)
    for remaining in (2, 1, 0):
        await management_ui.async_management_command(
            hass, "admin", True,
            {
                **message, "section": "function_repair", "action": "delete_one",
                "index": 1, "revision": current["revision"],
            },
        )
        next_read = await management_ui.async_management_command(
            hass, "admin", True, message
        )
        assert next_read["revision"] != current["revision"]
        if remaining:
            assert next_read["function_repair"]["invalid_count"] == remaining
            assert next_read["_performance"]["repair_state_cache_hit"] is False
        else:
            assert "function_repair" not in next_read
            assert next_read["config"]["functions"][0]["spec"]["name"] == "valid_phone_tool"
            assert next_read["config"]["function_groups"][0]["functions"] == [
                "valid_phone_tool"
            ]
        current = next_read


async def test_function_repair_rejects_still_invalid_tools_without_persisting() -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data["functions"] = _persisted_invalid_function_tools()
    repair = await _async_function_repair(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "action": "get",
        },
    )
    invalid = yaml.safe_load(_persisted_invalid_function_tools())

    with pytest.raises(Exception, match="minLength"):
        await _async_function_repair(
            hass,
            "admin",
            True,
            {
                "entry_id": "entry-1",
                "subentry_id": "agent-1",
                "action": "save",
                "revision": repair["revision"],
                "tools": invalid,
            },
        )

    assert hass.config_entries.updates == 0
    assert "minLength" in subentry.data["functions"]


async def test_overview_summary_loads_selected_agent_managers_once(monkeypatch) -> None:
    hass, _entry, _subentry = _hass_with_agent()
    usage = SimpleNamespace(
        as_dict=lambda: {"total_tokens": 200},
        today_summary=lambda: {"total_tokens": 20},
        month_summary=lambda: {"total_tokens": 80},
        latest_run=None,
    )
    memory = SimpleNamespace(
        memory_count=7,
        stats=MagicMock(side_effect=AssertionError("overview must not build memory stats")),
    )
    knowledge = SimpleNamespace(source_count=3)
    guest = SimpleNamespace(
        status=lambda: {"state": "scheduled", "currently_active": False}
    )
    mocks = {
        "async_get_usage": AsyncMock(return_value=usage),
        "async_get_memory": AsyncMock(return_value=memory),
        "async_get_knowledge": AsyncMock(return_value=knowledge),
        "async_get_guest_mode": AsyncMock(return_value=guest),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(
            "custom_components.extended_openai_conversation_responses."
            f"management_loading_performance.{name}",
            mock,
        )

    result = await async_overview_summary(hass, _entry, _subentry, is_admin=True)

    assert result["usage"]["today"]["total_tokens"] == 20
    assert result["agent"]["memory_count"] == 7
    assert result["agent"]["knowledge_source_count"] == 3
    assert result["agent"]["guest_mode"]["state"] == "scheduled"
    memory.stats.assert_not_called()
    assert result["load_errors"] == []
    performance = result["_performance"]
    assert performance["total_ms"] >= 0
    assert set(("usage_load_ms", "memory_load_ms", "knowledge_load_ms", "guest_load_ms")) <= performance.keys()
    assert performance["agent_snapshot"]["total_ms"] >= 0
    for mock in mocks.values():
        mock.assert_awaited_once()


async def test_overview_reuses_one_function_tool_health_projection(monkeypatch) -> None:
    hass, entry, subentry = _hass_with_agent()
    usage = SimpleNamespace(
        as_dict=lambda: {"total_tokens": 0},
        today_summary=lambda: {"total_tokens": 0},
        month_summary=lambda: {"total_tokens": 0},
        latest_run=None,
    )
    monkeypatch.setattr(loading, "async_get_usage", AsyncMock(return_value=usage))
    monkeypatch.setattr(
        loading,
        "async_get_memory",
        AsyncMock(return_value=SimpleNamespace(memory_count=0)),
    )
    monkeypatch.setattr(
        loading,
        "async_get_knowledge",
        AsyncMock(return_value=SimpleNamespace(source_count=0)),
    )
    monkeypatch.setattr(
        loading,
        "async_get_guest_mode",
        AsyncMock(
            return_value=SimpleNamespace(
                status=lambda: {"state": "inactive", "currently_active": False}
            )
        ),
    )
    health = {
        "usable_count": 2,
        "enabled_count": 1,
        "invalid_count": 0,
        "total_count": 2,
        "isolatable": False,
        "validation_error": None,
        "invalid_names": [],
    }
    projection = MagicMock(return_value=health)
    monkeypatch.setattr(loading, "management_function_tool_health", projection)

    result = await async_overview_summary(hass, entry, subentry, is_admin=True)

    projection.assert_called_once_with(dict(subentry.data))
    assert result["agent"]["function_count"] == 1
    assert result["setup_health"]["function_tools"] is health


async def test_configuration_get_reports_phase_timings(monkeypatch) -> None:
    hass, _entry, _subentry = _hass_with_agent()
    monkeypatch.setattr(management_ui, "local_handling_snapshot", lambda *_args: {})
    monkeypatch.setattr(
        management_ui,
        "decorate_configuration_result",
        lambda _hass, _entry_data, result, **_kwargs: result,
    )

    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "get",
        },
    )

    performance = result["_performance"]
    assert performance["total_ms"] >= 0
    assert set(
        (
            "config_snapshot_ms",
            "snapshot_build_ms",
            "snapshot_copy_ms",
            "revision_ms",
            "defaults_snapshot_ms",
            "defaults_cache_hit",
            "options_ms",
            "options_cache_hit",
            "model_capabilities_ms",
            "decoration_ms",
            "request_total_ms",
            "dispatch_ms",
            "agent_resolution_ms",
            "handler_ms",
            "response_assembly_ms",
            "maintenance_lease_ms",
            "command_total_ms",
        )
    ) <= performance.keys()


async def test_configuration_save_normalizes_once(monkeypatch) -> None:
    """The combined Save path validates and persists one normalized candidate."""
    hass, _entry, subentry = _hass_with_agent()
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    original_merge = management_ui.merge_agent_config
    merge_calls = 0

    def counted_merge(current, updates):
        nonlocal merge_calls
        merge_calls += 1
        return original_merge(current, updates)

    monkeypatch.setattr(management_ui, "merge_agent_config", counted_merge)
    monkeypatch.setattr(
        loading,
        "validate_function_tools",
        Mock(side_effect=AssertionError("save response revalidated Function Tools")),
    )
    monkeypatch.setattr(
        loading,
        "validate_function_groups",
        Mock(side_effect=AssertionError("save response revalidated Function Groups")),
    )
    monkeypatch.setattr(
        management_ui,
        "local_handling_snapshot",
        lambda _hass, _entry, _subentry, _snapshot: {},
    )

    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "save",
            "config": {"chat_model": "gpt-5-mini"},
            "title": "Updated Jarvis",
        },
    )

    assert result["valid"] is True
    assert result["config"]["chat_model"] == "gpt-5-mini"
    assert isinstance(result["config"]["functions"], list)
    assert result["title"] == "Updated Jarvis"
    assert subentry.data["chat_model"] == "gpt-5-mini"
    assert hass.config_entries.updates == 1
    assert merge_calls == 1
    monkeypatch.setattr(
        function_repair,
        "agent_config_snapshot",
        Mock(side_effect=AssertionError("save response should seed the next read")),
    )
    monkeypatch.setattr(
        management_ui,
        "decorate_configuration_result",
        lambda _hass, _entry_data, result, **_kwargs: result,
    )
    fetched = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "get",
        },
    )
    assert fetched["revision"] == result["revision"]
    assert fetched["config"] == result["config"]


async def test_unrelated_save_reuses_valid_functions_and_seeds_fresh_snapshot(
    monkeypatch,
) -> None:
    hass, _entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1", "section": "configuration",
    }
    current = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    tools = Mock(wraps=agent_config.validate_function_tools)
    groups = Mock(wraps=agent_config.validate_function_groups)
    monkeypatch.setattr(agent_config, "validate_function_tools", tools)
    monkeypatch.setattr(agent_config, "validate_function_groups", groups)
    saved = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "revision": current["revision"],
         "config": {"prompt": "New authoritative prompt"}},
    )
    fetched = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )

    tools.assert_not_called()
    groups.assert_not_called()
    assert saved["_performance"]["function_tools_reused"] is True
    assert saved["_performance"]["function_groups_reused"] is True
    assert saved["_performance"]["model_capabilities_cache_hit"] is True
    assert {"merge_validation_ms", "subentry_update_ms", "response_snapshot_ms",
            "projection_seed_ms", "command_total_ms"} <= saved["_performance"].keys()
    assert saved["revision"] == fetched["revision"]
    assert saved["config"]["prompt"] == fetched["config"]["prompt"] == subentry.data["prompt"]
    saved["config"]["functions"].clear()
    assert fetched["config"]["functions"]
    after_response_mutation = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    assert after_response_mutation["config"]["functions"]
    assert subentry.data["functions"]


async def test_unrelated_save_preserves_authoritative_valid_function_yaml(
    monkeypatch,
) -> None:
    hass, _entry, subentry = _hass_with_agent()
    tools, _groups = _unavailable_native_tools()
    raw_valid = yaml.safe_dump(yaml.safe_load(tools)[:1], sort_keys=True)
    subentry.data = {**subentry.data, "functions": raw_valid}
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1", "section": "configuration",
    }
    await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    saved = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "config": {"prompt": "Unrelated"}},
    )
    assert saved["valid"] is True
    assert subentry.data["functions"] == raw_valid
    assert saved["config"]["functions"][0]["spec"]["name"] == "valid_phone_tool"


async def test_unrelated_save_preserves_quarantine_without_reisolation(
    monkeypatch,
) -> None:
    hass, _entry, subentry = _hass_with_agent()
    raw_tools, raw_groups = _unavailable_native_tools()
    subentry.data = {
        **subentry.data, "functions": raw_tools, "function_groups": raw_groups,
    }
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    function_repair._cached_isolated_function_tools.cache_clear()
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    management_ui._configuration_defaults()
    isolate = Mock(wraps=function_repair._isolate_function_tools_uncached)
    monkeypatch.setattr(function_repair, "_isolate_function_tools_uncached", isolate)
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1", "section": "configuration",
    }
    current = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    assert isolate.call_count == 1
    tools = Mock(wraps=agent_config.validate_function_tools)
    groups = Mock(wraps=agent_config.validate_function_groups)
    monkeypatch.setattr(agent_config, "validate_function_tools", tools)
    monkeypatch.setattr(agent_config, "validate_function_groups", groups)
    saved = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "revision": current["revision"],
         "config": {"prompt": "Safe unrelated edit"}},
    )
    next_read = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    tools.assert_not_called()
    groups.assert_not_called()
    assert isolate.call_count == 1
    assert subentry.data["functions"] == raw_tools
    assert subentry.data["function_groups"] == raw_groups
    assert saved["function_repair"]["invalid_count"] == 3
    assert saved["function_repair"]["persisted_groups"] == raw_groups
    assert saved["config"]["prompt"] == next_read["config"]["prompt"]
    assert next_read["_performance"]["repair_state_cache_hit"] is True
    assert next_read["revision"] == saved["revision"]
    saved["function_repair"]["invalid_tools"].clear()
    saved["config"]["functions"].clear()
    assert next_read["function_repair"]["invalid_count"] == 3
    assert len(next_read["config"]["functions"]) == 1
    after_response_mutation = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    assert len(after_response_mutation["function_repair"]["invalid_tools"]) == 3
    assert len(after_response_mutation["config"]["functions"]) == 1


async def test_function_mutations_revalidate_only_changed_dependencies(
    monkeypatch,
) -> None:
    hass, _entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1", "section": "configuration",
    }
    await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    tools = Mock(wraps=agent_config.validate_function_tools)
    groups = Mock(wraps=agent_config.validate_function_groups)
    monkeypatch.setattr(agent_config, "validate_function_tools", tools)
    monkeypatch.setattr(agent_config, "validate_function_groups", groups)
    candidate = yaml.safe_load(subentry.data["functions"])
    candidate[0]["spec"]["name"] = "renamed_tool"
    tool_save = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "config": {
            "functions": yaml.safe_dump(candidate, sort_keys=False),
        }},
    )
    assert tool_save["valid"] is True
    assert tool_save["_performance"]["function_tools_reused"] is False
    assert tools.call_count == 1
    assert groups.call_count == 1
    tools.reset_mock()
    groups.reset_mock()
    group_save = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "config": {"function_groups": [{
            "id": "renamed", "name": "Renamed", "description": "Renamed tools",
            "loading_mode": "always", "functions": ["renamed_tool"],
        }]}},
    )
    assert group_save["valid"] is True
    assert group_save["_performance"]["function_tools_reused"] is True
    assert group_save["_performance"]["function_groups_reused"] is False
    tools.assert_not_called()
    groups.assert_called_once()
    fetched = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    assert fetched["revision"] == group_save["revision"]
    assert fetched["config"]["function_groups"] == group_save["config"]["function_groups"]
    assert fetched["config"]["function_groups"][0]["functions"] == ["renamed_tool"]


async def test_save_model_metadata_reuses_only_unchanged_model(monkeypatch) -> None:
    hass, _entry, _subentry = _hass_with_agent()
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    management_ui._cached_model_capabilities.cache_clear()
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1", "section": "configuration",
    }
    await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    capabilities = Mock(wraps=management_ui.model_capabilities)
    monkeypatch.setattr(management_ui, "model_capabilities", capabilities)
    ordinary = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "config": {"prompt": "Changed"}},
    )
    capabilities.assert_not_called()
    assert ordinary["_performance"]["model_capabilities_cache_hit"] is True
    model = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "config": {"chat_model": "gpt-5"}},
    )
    capabilities.assert_called_once_with("gpt-5")
    assert model["_performance"]["model_capabilities_cache_hit"] is False
    expected = model["model_capabilities"]["supports_temperature"]
    model["model_capabilities"]["supports_temperature"] = not expected
    repeated = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "config": {"prompt": "Changed again"}},
    )
    assert repeated["model_capabilities"]["supports_temperature"] is expected
    assert capabilities.call_count == 1


async def test_cold_repairable_save_uses_one_quarantine_state(monkeypatch, caplog) -> None:
    hass, _entry, subentry = _hass_with_agent()
    raw_tools, raw_groups = _unavailable_native_tools()
    subentry.data = {
        **subentry.data, "functions": raw_tools, "function_groups": raw_groups,
    }
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    function_repair._cached_isolated_function_tools.cache_clear()
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    management_ui._configuration_defaults()
    isolate = Mock(wraps=function_repair._isolate_function_tools_uncached)
    monkeypatch.setattr(function_repair, "_isolate_function_tools_uncached", isolate)
    strict = Mock(side_effect=AssertionError("repairable save must not validate raw tools"))
    monkeypatch.setattr(agent_config, "validate_function_tools", strict)
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1", "section": "configuration",
    }
    saved = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "config": {"prompt": "Cold safe edit"}},
    )
    next_read = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    assert saved["valid"] is True
    assert isolate.call_count == 1
    strict.assert_not_called()
    assert next_read["function_repair"]["invalid_count"] == 3
    assert next_read["_performance"]["repair_state_cache_hit"] is True
    assert subentry.data["functions"] == raw_tools
    assert not [record for record in caplog.records if record.levelno >= 30]


async def test_nonisolatable_functions_cannot_be_bypassed_on_save(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data = {**subentry.data, "functions": "not a list"}
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    message = {
        "entry_id": "entry-1", "subentry_id": "agent-1", "section": "configuration",
    }
    current = await management_ui.async_management_command(
        hass, "admin", True, {**message, "action": "get"}
    )
    assert current["function_repair"]["isolatable"] is False
    attempted = await management_ui.async_management_command(
        hass, "admin", True,
        {**message, "action": "save", "revision": current["revision"],
         "config": {"prompt": "must not persist"}},
    )
    assert attempted["valid"] is False
    assert "functions" in attempted["errors"]
    assert hass.config_entries.updates == 0


async def test_retention_projection_reads_only_needed_fields(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    normalizer = Mock(side_effect=AssertionError("retention must not normalize"))
    revision_hash = Mock(wraps=function_repair.agent_config_revision_from_snapshot)
    monkeypatch.setattr(function_repair, "agent_config_snapshot", normalizer)
    monkeypatch.setattr(
        function_repair,
        "validate_function_tools",
        Mock(side_effect=AssertionError("retention must not validate tools")),
    )
    monkeypatch.setattr(
        function_repair, "agent_config_revision_from_snapshot", revision_hash
    )
    monkeypatch.setattr(
        management_ui,
        "_configuration_defaults",
        Mock(side_effect=AssertionError("retention does not need full defaults")),
    )
    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "retention_get",
        },
    )
    fields = {"usage_request_retention_days", "usage_run_retention_days"}
    assert result["projection"] == "retention"
    assert set(result["config"]) == fields
    assert set(result["options"]) == fields
    assert result["title"] == subentry.title
    assert isinstance(result["revision"], str)
    assert "defaults" not in result
    assert "model_capabilities" not in result
    repeated = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "retention_get",
        },
    )
    assert repeated["revision"] == result["revision"]
    normalizer.assert_not_called()
    revision_hash.assert_called_once()
    assert result["_performance"]["projection_cache_hit"] is False
    assert repeated["_performance"]["projection_cache_hit"] is True


async def test_malformed_tools_do_not_block_cold_retention_projection(monkeypatch) -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data = {**subentry.data, "functions": _persisted_invalid_function_tools()}
    monkeypatch.setattr(
        function_repair, "_persisted_projections", function_repair.OrderedDict()
    )
    monkeypatch.setattr(
        function_repair, "agent_config_snapshot",
        Mock(side_effect=AssertionError("retention must not validate tools")),
    )
    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "retention_get",
        },
    )
    assert result["revision"] == function_repair.repair_revision(subentry)
    assert result["projection"] == "retention"


async def test_configuration_patch_preserves_omitted_fields_and_skips_local_snapshot(
    monkeypatch,
) -> None:
    hass, _entry, subentry = _hass_with_agent()
    from custom_components.extended_openai_conversation_responses import (
        management_configuration_guidance as guidance,
    )

    monkeypatch.setattr(guidance, "exposed_attribute_catalog", lambda *_: {})
    before = agent_config_snapshot(subentry.data)
    local = Mock(return_value={"supported": True})
    monkeypatch.setattr(management_ui, "local_handling_snapshot", local)
    revision = management_ui._agent_config_revision(subentry.data, subentry.title)
    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "save",
            "revision": revision,
            "config": {"max_tokens": 750},
        },
    )
    assert result["valid"] is True
    assert result["config"]["max_tokens"] == 750
    assert result["config"]["chat_model"] == before["chat_model"]
    assert subentry.data["chat_model"] == before["chat_model"]
    assert result["revision"] != revision
    assert "local_handling" not in result
    local.assert_not_called()
    assert hass.config_entries.updates == 1

    with pytest.raises(HomeAssistantError, match="changed"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            {
                "entry_id": "entry-1",
                "subentry_id": "agent-1",
                "section": "configuration",
                "action": "save",
                "revision": revision,
                "config": {"max_tokens": 800},
            },
        )
    assert subentry.data["max_tokens"] == 750
    assert hass.config_entries.updates == 1

    local_saved = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "save",
            "config": {"local_intent_exclusions": ["HassTurnOn"]},
        },
    )
    local.assert_called_once()

    updated = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "update",
            "revision": local_saved["revision"],
            "config": {"max_tokens": 900},
        },
    )
    assert updated["config"]["max_tokens"] == 900
    assert "local_handling" not in updated
    local.assert_called_once()
    await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "save",
            "config": {"local_intent_exclusions": ["HassTurnOn"]},
        },
    )
    local.assert_called_once()


async def test_configuration_patch_validates_complete_merged_candidate() -> None:
    hass, _entry, subentry = _hass_with_agent()
    subentry.data["speech_regex_replacements"] = [{"pattern": "[", "replacement": ""}]
    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "save",
            "config": {"max_tokens": 750},
        },
    )
    assert result["valid"] is False
    assert "speech_regex_replacements[0].pattern" in result["errors"]
    assert hass.config_entries.updates == 0


async def test_configuration_save_validation_failure_does_not_persist(
    monkeypatch,
) -> None:
    """A single failed normalization remains frontend-friendly and write-free."""
    hass, _entry, _subentry = _hass_with_agent()
    original_merge = management_ui.merge_agent_config
    merge_calls = 0

    def counted_merge(current, updates):
        nonlocal merge_calls
        merge_calls += 1
        return original_merge(current, updates)

    monkeypatch.setattr(management_ui, "merge_agent_config", counted_merge)

    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "save",
            "config": {
                "speech_regex_replacements": [{"pattern": "[", "replacement": ""}]
            },
        },
    )

    assert result["valid"] is False
    assert "speech_regex_replacements[0].pattern" in result["errors"]
    assert hass.config_entries.updates == 0
    assert merge_calls == 1


@pytest.mark.parametrize("field", ["functions", "function_groups"])
async def test_configuration_save_rejects_malformed_function_configuration(field) -> None:
    hass, _entry, _subentry = _hass_with_agent()

    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        {
            "entry_id": "entry-1",
            "subentry_id": "agent-1",
            "section": "configuration",
            "action": "save",
            "config": {field: "not a list"},
        },
    )

    assert result["valid"] is False
    assert field in result["errors"]
    assert hass.config_entries.updates == 0


async def test_management_setup_retry_resumes_after_panel_failure(monkeypatch) -> None:
    """A failed panel registration must not poison setup or duplicate earlier steps."""
    setup_key = "test.management_ui_setup"
    hass = SimpleNamespace(data={})
    asset_register = AsyncMock()
    websocket_register = MagicMock()
    panel_register = AsyncMock(side_effect=[RuntimeError("panel unavailable"), None])
    monkeypatch.setattr(management_ui, "_UI_SETUP", setup_key)
    monkeypatch.setattr(management_ui, "async_register_frontend_assets", asset_register)
    monkeypatch.setattr(
        management_ui,
        "frontend_entry_url",
        lambda _hass, name: f"/built/{name}.js",
    )
    monkeypatch.setattr(
        management_ui.websocket_api, "async_register_command", websocket_register
    )
    monkeypatch.setattr(
        management_ui.panel_custom, "async_register_panel", panel_register
    )

    with pytest.raises(RuntimeError, match="panel unavailable"):
        await async_setup_management_ui(hass)

    assert setup_key not in hass.data
    await async_setup_management_ui(hass)

    assert hass.data[setup_key] is True
    assert asset_register.await_count == 1
    assert websocket_register.call_count == 1
    assert panel_register.await_count == 2
    assert (
        panel_register.await_args_list[0].kwargs["module_url"] == "/built/management.js"
    )


async def test_debug_setup_retry_resumes_after_websocket_failure(monkeypatch) -> None:
    """A failed debug websocket registration retries without duplicating assets."""
    setup_key = "test.debug_ui_setup"
    hass = SimpleNamespace(data={})
    asset_register = AsyncMock()
    websocket_register = MagicMock(side_effect=[RuntimeError("ws unavailable"), None])
    monkeypatch.setattr(debug_ui, "_DEBUG_UI_SETUP", setup_key)
    monkeypatch.setattr(debug_ui, "async_register_frontend_assets", asset_register)
    monkeypatch.setattr(
        debug_ui.websocket_api, "async_register_command", websocket_register
    )

    with pytest.raises(RuntimeError, match="ws unavailable"):
        await async_setup_debug_ui(hass)

    assert setup_key not in hass.data
    await async_setup_debug_ui(hass)

    assert hass.data[setup_key] is True
    assert asset_register.await_count == 1
    assert websocket_register.call_count == 2


from custom_components.extended_openai_conversation_responses import (
    function_tool_quarantine as quarantine,
)


async def test_overview_primary_does_not_initialize_storage_managers(monkeypatch) -> None:
    hass, entry, subentry = _hass_with_agent()
    for name in ("async_get_usage", "async_get_memory", "async_get_knowledge", "async_get_guest_mode"):
        monkeypatch.setattr(
            loading,
            name,
            AsyncMock(side_effect=AssertionError(f"{name} should stay cold")),
        )
    function_repair._health_cache.clear()
    monkeypatch.setattr(
        loading, "management_function_tool_health",
        Mock(side_effect=AssertionError("primary must not validate tools")),
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.management_setup_health._exposed_entity_count",
        Mock(side_effect=AssertionError("primary must not scan states")),
    )

    result = await loading.async_overview_primary(
        hass, entry, subentry, is_admin=True
    )

    assert result["loading"] == {
        "usage": True,
        "memory": True,
        "knowledge": True,
        "guest_mode": True,
        "setup_health": True,
    }
    assert result["usage"] == {}
    assert result["setup_health"]["function_tools"]["loading"] is True
    assert result["setup_health"]["exposed_entity_count_loading"] is True


async def test_overview_selected_agent_resolves_health_once(monkeypatch) -> None:
    hass, entry, subentry = _hass_with_agent()
    function_repair._health_cache.clear()
    health = {"enabled_count": 2, "usable_count": 2, "validation_error": None}
    projection = Mock(return_value=health)
    monkeypatch.setattr(loading, "management_function_tool_health", projection)
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.management_setup_health._exposed_entity_count",
        Mock(return_value=4),
    )

    primary = await loading.async_overview_primary(hass, entry, subentry, is_admin=True)
    assert primary["agent"]["function_count"] is None
    detail = await loading.async_overview_detail(
        hass, entry, subentry, is_admin=True, kind="setup_health"
    )
    assert detail["setup_health"]["function_tools"] == health
    assert detail["setup_health"]["exposed_entity_count"] == 4
    projection.assert_called_once_with(dict(subentry.data))


async def test_overview_detail_loads_only_requested_manager(monkeypatch) -> None:
    hass, entry, subentry = _hass_with_agent()
    usage = AsyncMock(side_effect=AssertionError("usage should stay cold"))
    memory = AsyncMock(return_value=SimpleNamespace(memory_count=3))
    knowledge = AsyncMock(side_effect=AssertionError("knowledge should stay cold"))
    guest = AsyncMock(side_effect=AssertionError("guest should stay cold"))
    monkeypatch.setattr(loading, "async_get_usage", usage)
    monkeypatch.setattr(loading, "async_get_memory", memory)
    monkeypatch.setattr(loading, "async_get_knowledge", knowledge)
    monkeypatch.setattr(loading, "async_get_guest_mode", guest)

    result = await loading.async_overview_detail(
        hass, entry, subentry, is_admin=True, kind="memory"
    )

    assert result["agent"]["memory_count"] == 3
    memory.assert_awaited_once_with(hass, "entry-1", "agent-1")
    usage.assert_not_awaited()
    knowledge.assert_not_awaited()
    guest.assert_not_awaited()


async def test_overview_summary_isolates_each_manager_failure(monkeypatch) -> None:
    """A failed optional manager must degrade the summary instead of failing it."""
    hass, entry, subentry = _hass_with_agent()
    monkeypatch.setattr(
        loading,
        "settings_snapshot",
        lambda config: {"chat_model": config["chat_model"]},
    )
    monkeypatch.setattr(loading, "get_loaded_guest_mode", lambda *_args: None)

    failures = {
        "async_get_usage": RuntimeError("usage failed"),
        "async_get_memory": RuntimeError("memory failed"),
        "async_get_knowledge": RuntimeError("knowledge failed"),
        "async_get_guest_mode": RuntimeError("guest failed"),
    }
    for name, error in failures.items():
        monkeypatch.setattr(loading, name, AsyncMock(side_effect=error))

    result = await loading.async_overview_summary(
        hass,
        entry,
        subentry,
        is_admin=True,
    )

    assert result["usage"] == {}
    assert result["agent"]["memory_count"] == 0
    assert result["agent"]["knowledge_source_count"] == 0
    assert result["agent"]["tokens_today"] == 0
    assert result["agent"]["guest_mode"]["state"] == "unloaded"
    assert [error["key"] for error in result["load_errors"]] == [
        "usage",
        "memories",
        "knowledge",
        "guest_mode",
    ]
    assert [error["message"] for error in result["load_errors"]] == [
        "usage failed",
        "memory failed",
        "knowledge failed",
        "guest failed",
    ]


async def test_overview_summary_uses_exception_type_when_message_is_empty(
    monkeypatch,
) -> None:
    """Empty exception messages still produce useful load-error diagnostics."""
    hass, entry, subentry = _hass_with_agent()
    knowledge = SimpleNamespace(source_count=0)
    guest = SimpleNamespace(status=lambda: {"state": "off", "currently_active": False})
    monkeypatch.setattr(loading, "settings_snapshot", lambda config: config)
    monkeypatch.setattr(
        loading, "async_get_usage", AsyncMock(side_effect=RuntimeError())
    )
    monkeypatch.setattr(
        loading,
        "async_get_memory",
        AsyncMock(return_value=SimpleNamespace(memory_count=0)),
    )
    monkeypatch.setattr(
        loading, "async_get_knowledge", AsyncMock(return_value=knowledge)
    )
    monkeypatch.setattr(loading, "async_get_guest_mode", AsyncMock(return_value=guest))

    result = await loading.async_overview_summary(
        hass,
        entry,
        subentry,
        is_admin=True,
    )

    assert result["load_errors"] == [
        {"key": "usage", "label": "Usage", "message": "RuntimeError"}
    ]


@pytest.mark.parametrize("title", ["   ", 123])
async def test_save_configuration_rejects_bad_title_without_writing(
    hass, management_message, title
):
    result = await management_ui.async_management_command(
        hass,
        "admin",
        True,
        management_message("configuration", "save", title=title, config={}),
    )
    assert result == {"valid": False, "errors": {"title": "must not be empty"}}
    hass.config_entries.async_update_subentry.assert_not_called()


async def test_save_configuration_rejects_non_object_config(hass, management_message):
    with pytest.raises(HomeAssistantError, match="config must be an object"):
        await management_ui.async_management_command(
            hass,
            "admin",
            True,
            management_message("configuration", "save", config=["not", "an", "object"]),
        )
    hass.config_entries.async_update_subentry.assert_not_called()


async def test_owned_command_routes_directly_to_loading_and_save(
    hass, management_message, management_agent, monkeypatch
):
    agents = AsyncMock(return_value={"path": "agents"})
    overview = AsyncMock(return_value={"path": "overview"})
    save = AsyncMock(return_value={"path": "save"})
    monkeypatch.setattr(loading, "async_agent_catalog", agents)
    monkeypatch.setattr(loading, "async_overview_summary", overview)
    monkeypatch.setattr(management_ui, "_async_save_configuration", save)
    assert await management_ui.async_management_command(
        hass, "user", False, {"action": "agents"}
    ) == {"path": "agents"}
    assert await management_ui.async_management_command(
        hass, "user", False, management_message("overview", "summary")
    ) == {"path": "overview"}
    assert await management_ui.async_management_command(
        hass, "user", True, management_message("configuration", "save")
    ) == {"path": "save"}
    agents.assert_awaited_once_with(hass, "user", False)
    overview.assert_awaited_once_with(hass, *management_agent, is_admin=False)
    save.assert_awaited_once()
    assert save.await_args.args[0].subentry is management_agent[1]


async def test_cached_management_setup_is_noop_when_complete(monkeypatch) -> None:
    """A completed setup marker prevents duplicate registrations."""
    setup_key = "test.management.complete"
    monkeypatch.setattr(management_ui, "_UI_SETUP", setup_key)

    await async_setup_management_ui(SimpleNamespace(data={setup_key: True}))


async def test_cached_management_setup_respects_completed_step_markers(
    monkeypatch,
) -> None:
    """Retry state skips completed static/websocket steps and resumes at panel."""
    setup_key = "test.management.partial"
    static_key = f"{setup_key}.static_paths"
    websocket_key = f"{setup_key}.websocket"
    panel_key = f"{setup_key}.panel"
    static_paths = AsyncMock(side_effect=AssertionError("static paths repeated"))
    websocket_register = MagicMock(side_effect=AssertionError("websocket repeated"))
    panel_register = AsyncMock()
    hass = SimpleNamespace(
        data={
            static_key: True,
            websocket_key: True,
            frontend_assets._FRONTEND_ENTRY_URLS: {
                "management": f"/{DOMAIN}/frontend/assets/management-test.js"
            },
        },
        http=SimpleNamespace(async_register_static_paths=static_paths),
    )
    monkeypatch.setattr(management_ui, "_UI_SETUP", setup_key)
    monkeypatch.setattr(
        management_ui.websocket_api, "async_register_command", websocket_register
    )
    monkeypatch.setattr(
        management_ui.panel_custom, "async_register_panel", panel_register
    )

    await async_setup_management_ui(hass)

    assert hass.data[panel_key] is True
    assert hass.data[setup_key] is True
    static_paths.assert_not_awaited()
    websocket_register.assert_not_called()
    panel_register.assert_awaited_once()


async def test_cached_debug_setup_is_noop_when_complete(monkeypatch) -> None:
    """A completed debug setup marker prevents duplicate registrations."""
    setup_key = "test.debug.complete"
    monkeypatch.setattr(debug_ui, "_DEBUG_UI_SETUP", setup_key)

    await async_setup_debug_ui(SimpleNamespace(data={setup_key: True}))


async def test_cached_debug_setup_respects_completed_step_markers(monkeypatch) -> None:
    """A retry can finish without repeating already completed debug steps."""
    setup_key = "test.debug.partial"
    static_key = f"{setup_key}.static_paths"
    websocket_key = f"{setup_key}.websocket"
    static_paths = AsyncMock(side_effect=AssertionError("static paths repeated"))
    websocket_register = MagicMock(side_effect=AssertionError("websocket repeated"))
    hass = SimpleNamespace(
        data={static_key: True, websocket_key: True},
        http=SimpleNamespace(async_register_static_paths=static_paths),
    )
    monkeypatch.setattr(debug_ui, "_DEBUG_UI_SETUP", setup_key)
    monkeypatch.setattr(
        debug_ui.websocket_api, "async_register_command", websocket_register
    )

    await async_setup_debug_ui(hass)

    assert hass.data[setup_key] is True
    static_paths.assert_not_awaited()
    websocket_register.assert_not_called()



async def test_scope_catalog_archive_only_skips_memory_managers(monkeypatch) -> None:
    """Archive scope requests must not initialize persistent or temporary memory."""
    archive_counts = {"user:admin": 3}
    archive = SimpleNamespace(scope_counts=lambda: archive_counts)
    projection = AsyncMock(return_value=[{"scope_id": "user:admin"}])
    archive_loader = AsyncMock(return_value=archive)
    memory_loader = AsyncMock(side_effect=AssertionError("persistent memory should stay cold"))
    temporary_loader = AsyncMock(side_effect=AssertionError("temporary memory should stay cold"))
    monkeypatch.setattr(loading, "async_get_archive", archive_loader)
    monkeypatch.setattr(loading, "async_get_memory", memory_loader)
    monkeypatch.setattr(loading, "async_get_temporary_memory", temporary_loader)
    monkeypatch.setattr(loading, "async_scope_catalog_projection", projection)

    hass = SimpleNamespace()
    result = await loading.async_scope_catalog(
        hass,
        "admin",
        True,
        "entry-1",
        "agent-1",
        scope_kind="archive",
    )

    assert result == {"scopes": [{"scope_id": "user:admin"}]}
    archive_loader.assert_awaited_once_with(hass, "entry-1", "agent-1")
    memory_loader.assert_not_awaited()
    temporary_loader.assert_not_awaited()
    projection.assert_awaited_once_with(
        hass,
        "admin",
        True,
        {},
        archive_counts,
        {},
    )


async def test_scope_catalog_memory_kinds_load_only_requested_manager(monkeypatch) -> None:
    """Persistent and Temporary Memory scope requests stay independent."""
    persistent = SimpleNamespace(scope_counts=lambda: {"admin": 2})
    temporary = SimpleNamespace(owner_counts=lambda: {"user:admin": 4})
    memory_loader = AsyncMock(return_value=persistent)
    temporary_loader = AsyncMock(return_value=temporary)
    archive_loader = AsyncMock(side_effect=AssertionError("archive should stay cold"))
    projection = AsyncMock(side_effect=[
        [{"scope_id": "persistent"}],
        [{"scope_id": "temporary"}],
    ])
    monkeypatch.setattr(loading, "async_get_memory", memory_loader)
    monkeypatch.setattr(loading, "async_get_temporary_memory", temporary_loader)
    monkeypatch.setattr(loading, "async_get_archive", archive_loader)
    monkeypatch.setattr(loading, "async_scope_catalog_projection", projection)

    hass = SimpleNamespace()
    persistent_result = await loading.async_scope_catalog(
        hass, "admin", True, "entry-1", "agent-1", scope_kind="memory"
    )
    temporary_result = await loading.async_scope_catalog(
        hass, "admin", True, "entry-1", "agent-1", scope_kind="temporary"
    )

    assert persistent_result == {"scopes": [{"scope_id": "persistent"}]}
    assert temporary_result == {"scopes": [{"scope_id": "temporary"}]}
    memory_loader.assert_awaited_once_with(hass, "entry-1", "agent-1")
    temporary_loader.assert_awaited_once_with(hass, "entry-1", "agent-1")
    archive_loader.assert_not_awaited()
    assert projection.await_args_list[0].args[3:] == ({"admin": 2}, {}, {})
    assert projection.await_args_list[1].args[3:] == ({}, {}, {"user:admin": 4})


async def test_scope_catalog_loads_all_scope_managers_concurrently(monkeypatch) -> None:
    """Independent scope managers must start together rather than as a waterfall."""
    memory_started = asyncio.Event()
    archive_started = asyncio.Event()
    temporary_started = asyncio.Event()
    memory_counts = {"all": 4, "user": 3}
    archive_counts = {"all": 7, "user": 5}
    temporary_counts = {"user:user": 2}

    async def wait_for_peers(*events):
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in events)),
            timeout=1,
        )

    async def get_memory(_hass, entry_id, subentry_id):
        assert (entry_id, subentry_id) == ("entry-1", "agent-1")
        memory_started.set()
        await wait_for_peers(archive_started, temporary_started)
        return SimpleNamespace(scope_counts=lambda: memory_counts)

    async def get_archive(_hass, entry_id, subentry_id):
        assert (entry_id, subentry_id) == ("entry-1", "agent-1")
        archive_started.set()
        await wait_for_peers(memory_started, temporary_started)
        return SimpleNamespace(scope_counts=lambda: archive_counts)

    async def get_temporary(_hass, entry_id, subentry_id):
        assert (entry_id, subentry_id) == ("entry-1", "agent-1")
        temporary_started.set()
        await wait_for_peers(memory_started, archive_started)
        return SimpleNamespace(owner_counts=lambda: temporary_counts)

    scope_catalog = AsyncMock(return_value=[{"id": "all", "label": "All"}])
    monkeypatch.setattr(loading, "async_scope_catalog_projection", scope_catalog)
    monkeypatch.setattr(loading, "async_get_memory", get_memory)
    monkeypatch.setattr(loading, "async_get_archive", get_archive)
    monkeypatch.setattr(loading, "async_get_temporary_memory", get_temporary)

    hass = SimpleNamespace()
    result = await loading.async_scope_catalog(
        hass,
        "admin",
        True,
        "entry-1",
        "agent-1",
    )

    assert result == {"scopes": [{"id": "all", "label": "All"}]}
    scope_catalog.assert_awaited_once_with(
        hass,
        "admin",
        True,
        memory_counts,
        archive_counts,
        temporary_counts,
    )
