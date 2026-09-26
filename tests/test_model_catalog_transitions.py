"""Model Catalog refresh lifecycle, capability transitions and saved-configuration safeguards."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.extended_openai_conversation_responses import (
    model_catalog as catalog,
    model_catalog as data,
    model_catalog_manager as runtime,
)
from custom_components.extended_openai_conversation_responses.const import (
    CONF_CHAT_MODEL,
    CONF_REASONING_EFFORT,
    DOMAIN,
)


@pytest.fixture(autouse=True)
def reset_catalog():
    """Do not let module-global catalogue state leak between tests."""
    data.activate_catalog(None)
    yield
    data.activate_catalog(None)


def _label_candidate():
    value = deepcopy(data.BUNDLED_CATALOG)
    value["catalog_version"] += 1
    value["models"][0]["display_name"] = "Astra (catalog update)"
    return value


class MemoryStore:
    """Copy persisted state and inject save failures without publishing partial writes."""

    def __init__(self, saved=None):
        self.saved = deepcopy(saved)
        self.fail = False
        self.save_calls = 0
        self.load_calls = 0

    async def async_load(self):
        self.load_calls += 1
        return deepcopy(self.saved)

    async def async_save(self, value):
        self.save_calls += 1
        if self.fail:
            raise OSError("disk full")
        self.saved = deepcopy(value)


@pytest.fixture
def check_manager(hass):
    result = runtime.ModelCatalogManager(hass)
    result.store = MemoryStore()
    return result


def _chunked_transport(monkeypatch, raw, *, status=200, etag='"v3"', error=None):
    async def chunks(_size):
        for offset in range(0, len(raw), 100):
            yield raw[offset : offset + 100]

    response = SimpleNamespace(
        status=status,
        headers={"ETag": etag},
        content=SimpleNamespace(iter_chunked=chunks),
    )
    context = AsyncMock()
    context.__aenter__.return_value = response
    if error:
        context.__aenter__.side_effect = error
    session = SimpleNamespace(get=Mock(return_value=context))
    monkeypatch.setattr(runtime, "async_get_clientsession", lambda _: session)
    return session.get


def _catalog() -> dict:
    value = deepcopy(data.BUNDLED_CATALOG)
    value["catalog_version"] += 1
    return value


def _model(value: dict) -> dict:
    return value["models"][0]


def _set_temperature(model: dict, support: str, allowed, send_policy: str) -> None:
    model["temperature"] = {
        "support": support,
        "allowed_reasoning_efforts": allowed,
        "send_policy": send_policy,
    }


def _stored_manager(hass) -> runtime.ModelCatalogManager:
    manager = runtime.ModelCatalogManager(hass)
    manager.store = MemoryStore()
    return manager


def _downloaded_candidate() -> dict:
    candidate = deepcopy(data.BUNDLED_CATALOG)
    candidate["catalog_version"] += 1
    candidate["models"][0]["display_name"] = "Downloaded catalogue"
    return candidate


def _subentry(*, model="gpt-5.6", effort=None):
    values = {CONF_CHAT_MODEL: model}
    if effort is not None:
        values[CONF_REASONING_EFFORT] = effort
    return SimpleNamespace(
        subentry_type="conversation",
        subentry_id="agent",
        data=values,
    )


def _entry(subentry) -> SimpleNamespace:
    return SimpleNamespace(entry_id="entry", subentries={"agent": subentry})


def _versioned_candidate(*, increment: int = 1) -> dict:
    value = deepcopy(catalog.BUNDLED_CATALOG)
    value["catalog_version"] += increment
    value["models"][0]["display_name"] = f"Updated {increment}"
    return value


@pytest.fixture
def manager(hass):
    result = runtime.ModelCatalogManager(hass)
    result.store = MemoryStore()
    return result


def candidate():
    """Return a valid downloaded catalogue with a download-only effort."""
    value = deepcopy(data.BUNDLED_CATALOG)
    value["catalog_version"] += 1
    model = next(item for item in value["models"] if item["id"] == "gpt-5.6")
    model["reasoning"]["efforts"].append("minimal")
    model["reasoning"]["by_api"]["responses"]["efforts"].append("minimal")
    return value


async def set_saved_conversation(monkeypatch, manager, *, model="gpt-5.6", effort=None):
    """Expose one persisted conversation subentry through HA's config-entry shape."""
    subentry_data = {CONF_CHAT_MODEL: model}
    if effort is not None:
        subentry_data[CONF_REASONING_EFFORT] = effort
    subentry = SimpleNamespace(
        subentry_type="conversation",
        subentry_id="conversation-1",
        data=subentry_data,
    )
    entry = SimpleNamespace(entry_id="entry-1", subentries={"conversation-1": subentry})
    monkeypatch.setattr(
        manager.hass.config_entries, "async_entries", lambda _domain: [entry]
    )

    async def no_rules(_hass, _entry_id, _subentry_id):
        return SimpleNamespace(snapshot=lambda: {"rules": []})

    monkeypatch.setattr(runtime, "async_get_request_rules", no_rules)


def _expanded_candidate() -> dict:
    candidate = deepcopy(data.BUNDLED_CATALOG)
    candidate["catalog_version"] += 1
    model = next(item for item in candidate["models"] if item["id"] == "gpt-5.6")
    model["reasoning"]["efforts"].append("minimal")
    model["reasoning"]["by_api"]["responses"]["efforts"].append("minimal")
    return candidate


def _transport(monkeypatch, raw: bytes, *, etag: str = '"catalog"') -> Mock:
    async def chunks(_size):
        yield raw

    response = SimpleNamespace(
        status=200,
        headers={"ETag": etag},
        content=SimpleNamespace(iter_chunked=chunks),
    )
    context = AsyncMock()
    context.__aenter__.return_value = response
    get = Mock(return_value=context)
    monkeypatch.setattr(
        runtime, "async_get_clientsession", lambda _: SimpleNamespace(get=get)
    )
    return get


async def test_check_stages_update_without_changing_active_catalog(
    check_manager, monkeypatch
):
    value = _label_candidate()
    get = _chunked_transport(monkeypatch, json.dumps(value).encode())

    result = await check_manager.async_check(force=True)

    assert result["source"] == "bundled"
    assert result["schema_version"] == 6
    assert result["update_available"] is True
    assert result["available_catalog_version"] == value["catalog_version"]
    assert result["last_error"] is None
    assert get.call_args.kwargs == {"headers": {}, "allow_redirects": False}
    assert check_manager.catalog is None
    assert check_manager.available_catalog == value
    assert check_manager.store.saved["catalog"] is None
    assert check_manager.store.saved["available_catalog"] == value
    assert data.model_metadata("gpt-6-astra")["display_name"] == "gpt-6-astra"


async def test_apply_activates_staged_catalog_and_clears_pending_state(
    check_manager, monkeypatch
):
    value = _label_candidate()
    _chunked_transport(monkeypatch, json.dumps(value).encode())
    await check_manager.async_check(force=True)

    result = await check_manager.async_apply_update()

    assert result["source"] == "downloaded"
    assert result["catalog_version"] == value["catalog_version"]
    assert result["update_available"] is False
    assert result["available_catalog_version"] is None
    assert result["last_error"] is None
    assert check_manager.catalog == value
    assert check_manager.available_catalog is None
    assert check_manager.store.saved["catalog"] == value
    assert check_manager.store.saved["available_catalog"] is None
    assert check_manager.store.load_calls == 0
    assert (
        data.model_metadata("gpt-6-astra")["display_name"] == "Astra (catalog update)"
    )


async def test_restoring_bundled_data_is_persistent_across_restart(
    check_manager, monkeypatch
):
    value = _label_candidate()
    _chunked_transport(monkeypatch, json.dumps(value).encode())
    await check_manager.async_check(force=True)
    await check_manager.async_apply_update()

    result = await check_manager.async_reset()

    assert result["source"] == "bundled"
    assert result["update_available"] is True
    assert check_manager.catalog is None
    assert check_manager.available_catalog == value
    assert check_manager.store.saved["catalog"] is None
    assert check_manager.store.saved["available_catalog"] == value
    assert check_manager.store.load_calls == 0
    assert data.model_metadata("gpt-6-astra")["display_name"] == "gpt-6-astra"

    restarted = runtime.ModelCatalogManager(check_manager.hass)
    restarted.store = check_manager.store
    await restarted.async_load()

    assert restarted.status()["source"] == "bundled"
    assert restarted.status()["update_available"] is True
    assert restarted.catalog is None
    assert restarted.available_catalog == value
    assert data.model_metadata("gpt-6-astra")["display_name"] == "gpt-6-astra"


async def test_restart_does_not_resurrect_stale_pending_state(
    check_manager, monkeypatch
):
    value = _label_candidate()
    _chunked_transport(monkeypatch, json.dumps(value).encode())
    await check_manager.async_check(force=True)
    await check_manager.async_apply_update()

    assert check_manager.store.saved["available_catalog"] is None

    restarted = runtime.ModelCatalogManager(check_manager.hass)
    restarted.store = check_manager.store
    await restarted.async_load()

    assert restarted.catalog == value
    assert restarted.available_catalog is None
    assert restarted.status()["update_available"] is False
    assert (
        data.model_metadata("gpt-6-astra")["display_name"] == "Astra (catalog update)"
    )

    # Be defensive about storage left by an interrupted/older write: an update at
    # or below the active version must not be surfaced as pending after restart.
    restarted.store.saved["available_catalog"] = deepcopy(value)
    second_restart = runtime.ModelCatalogManager(check_manager.hass)
    second_restart.store = restarted.store
    await second_restart.async_load()

    assert second_restart.catalog == value
    assert second_restart.available_catalog is None
    assert second_restart.status()["update_available"] is False


async def test_daily_cadence_and_etag_apply_to_checks(check_manager, monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr(runtime.time, "time", lambda: now)
    get = _chunked_transport(monkeypatch, json.dumps(_label_candidate()).encode())
    await check_manager.async_check()
    now += runtime.UPDATE_INTERVAL - 1
    await check_manager.async_check()
    assert get.call_count == 1

    now += 1
    get = _chunked_transport(monkeypatch, b"", status=304)
    await check_manager.async_check()
    assert get.call_args.kwargs["headers"] == {"If-None-Match": '"v3"'}
    assert check_manager.status()["last_error"] is None
    assert check_manager.status()["update_available"] is True


async def test_reset_waits_for_inflight_check(check_manager, monkeypatch):
    _chunked_transport(monkeypatch, json.dumps(_label_candidate()).encode())
    started, release = asyncio.Event(), asyncio.Event()
    save = check_manager.store.async_save

    async def paused_save(value):
        started.set()
        await release.wait()
        await save(value)

    monkeypatch.setattr(check_manager.store, "async_save", paused_save)
    check = asyncio.create_task(check_manager.async_check(force=True))
    await started.wait()
    assert data.model_metadata("gpt-6-astra")["display_name"] == "gpt-6-astra"
    reset = asyncio.create_task(check_manager.async_reset())
    release.set()
    await asyncio.gather(check, reset)
    assert check_manager.catalog is None
    assert check_manager.available_catalog == _label_candidate()
    assert check_manager.store.saved["catalog"] is None
    assert check_manager.store.saved["available_catalog"] == _label_candidate()


def test_sampling_rank_orders_supported_states() -> None:
    assert data._sampling_rank({"support": "always"}) == (3, frozenset())
    assert data._sampling_rank(
        {"support": "conditional", "allowed_reasoning_efforts": ["low", "high"]}
    ) == (2, frozenset({"low", "high"}))
    assert data._sampling_rank({"support": "undocumented"}) == (1, frozenset())
    assert data._sampling_rank({"support": "never"}) == (0, frozenset())


def test_transition_rejects_removed_reasoning_effort() -> None:
    candidate = _catalog()
    _model(candidate)["reasoning"]["efforts"].remove("max")
    _model(candidate)["reasoning"]["by_api"]["responses"]["efforts"].remove("max")
    _model(candidate)["reasoning"]["by_api"]["chat_completions"]["efforts"].remove("max")

    with pytest.raises(ValueError, match="cannot remove reasoning effort choices"):
        data.validate_catalog_transition(None, candidate)


def test_transition_rejects_removed_api_path() -> None:
    candidate = _catalog()
    model = _model(candidate)
    model["api"]["responses"] = False
    model["reasoning"]["by_api"]["responses"]["efforts"] = []
    model["auto_api"] = "chat_completions"
    model["function_calling"].update(
        responses=False, chat_completions=True, preferred_api="chat_completions"
    )
    model["tools"]["function"]["responses"] = {"support": "never"}
    model["tools"]["web_search"]["responses"] = {"support": "never"}
    model["recommended_profile"]["api"] = "chat_completions"

    with pytest.raises(ValueError, match="cannot remove an API path"):
        data.validate_catalog_transition(None, candidate)


def test_transition_rejects_removed_function_calling_support() -> None:
    candidate = _catalog()
    model = _model(candidate)
    model["function_calling"]["responses"] = False
    model["function_calling"]["preferred_api"] = "chat_completions"

    with pytest.raises(ValueError, match="cannot remove function-calling support"):
        data.validate_catalog_transition(None, candidate)


def test_transition_rejects_lower_max_output_limit() -> None:
    candidate = _catalog()
    _model(candidate)["limits"]["max_output_tokens"] -= 1

    with pytest.raises(ValueError, match="cannot lower max output"):
        data.validate_catalog_transition(None, candidate)


def test_transition_rejects_sampling_support_downgrade() -> None:
    current = _catalog()
    candidate = _catalog()
    _set_temperature(_model(current), "always", None, "omit_unless_configured")
    _set_temperature(
        _model(candidate), "conditional", ["low"], "omit_unless_configured"
    )

    with pytest.raises(ValueError, match="cannot narrow temperature"):
        data.validate_catalog_transition(current, candidate)


def test_transition_rejects_narrower_conditional_sampling_efforts() -> None:
    current = _catalog()
    candidate = _catalog()
    _set_temperature(
        _model(current), "conditional", ["low", "medium"], "omit_unless_configured"
    )
    _set_temperature(
        _model(candidate), "conditional", ["low"], "omit_unless_configured"
    )

    with pytest.raises(ValueError, match="cannot narrow temperature"):
        data.validate_catalog_transition(current, candidate)


async def test_irrelevant_subentries_do_not_load_request_rules(
    hass, monkeypatch
) -> None:
    manager = _stored_manager(hass)
    manager.catalog = _downloaded_candidate()
    hass.config_entries.async_entries.return_value = [
        SimpleNamespace(
            entry_id="entry",
            subentries={
                "task": SimpleNamespace(
                    subentry_type="ai_task", subentry_id="task", data={}
                )
            },
        )
    ]
    get_rules = AsyncMock()
    monkeypatch.setattr(runtime, "async_get_request_rules", get_rules)

    assert await manager._bundled_reset_would_invalidate_saved_reasoning() is False
    get_rules.assert_not_awaited()


@pytest.mark.parametrize(
    "action",
    [
        {"model": "gpt-5.6", "reasoning_effort": "not-a-bundled-effort"},
        {"reasoning_effort": "not-a-bundled-effort"},
    ],
    ids=["fixed-model", "all-models"],
)
async def test_request_rule_reasoning_can_block_reset(
    hass, monkeypatch, action
) -> None:
    manager = _stored_manager(hass)
    manager.catalog = _downloaded_candidate()
    hass.config_entries.async_entries.return_value = [_entry(_subentry())]
    rules = SimpleNamespace(
        snapshot=lambda: {"rules": [{"action_type": "model_routing", "action": action}]}
    )
    monkeypatch.setattr(
        runtime, "async_get_request_rules", AsyncMock(return_value=rules)
    )

    assert await manager._bundled_reset_would_invalidate_saved_reasoning() is True


async def test_dynamic_and_reset_rules_do_not_block_catalog_reset(
    hass, monkeypatch
) -> None:
    manager = _stored_manager(hass)
    manager.catalog = _downloaded_candidate()
    hass.config_entries.async_entries.return_value = [_entry(_subentry())]
    rules = SimpleNamespace(
        snapshot=lambda: {
            "rules": [
                {"action_type": "other", "action": {}},
                {"action_type": "model_routing", "action": {"reset": True}},
                {
                    "action_type": "model_routing",
                    "action": {"reasoning_effort": "{effort}"},
                },
                {
                    "action_type": "model_routing",
                    "action": {
                        "model": "{model}",
                        "reasoning_effort": "medium",
                    },
                },
            ]
        }
    )
    monkeypatch.setattr(
        runtime, "async_get_request_rules", AsyncMock(return_value=rules)
    )

    assert await manager._bundled_reset_would_invalidate_saved_reasoning() is False


async def test_no_downloaded_catalog_never_blocks_reset(hass) -> None:
    manager = _stored_manager(hass)

    assert await manager._bundled_reset_would_invalidate_saved_reasoning() is False


@pytest.mark.asyncio
async def test_valid_model_routing_rule_continues_scanning(hass, monkeypatch) -> None:
    manager = runtime.ModelCatalogManager(hass)
    manager.catalog = _versioned_candidate()
    subentry = SimpleNamespace(
        subentry_id="agent",
        subentry_type="conversation",
        data={CONF_CHAT_MODEL: "gpt-5.6", CONF_REASONING_EFFORT: "medium"},
    )
    entry = SimpleNamespace(entry_id="entry", subentries={"agent": subentry})
    hass.config_entries.async_entries.return_value = [entry]
    rules = SimpleNamespace(
        snapshot=lambda: {
            "rules": [
                {
                    "action_type": "model_routing",
                    "action": {"model": "gpt-5.6", "reasoning_effort": "medium"},
                },
                {"action_type": "other", "action": {}},
            ]
        }
    )
    get_rules = AsyncMock(return_value=rules)
    monkeypatch.setattr(runtime, "async_get_request_rules", get_rules)

    assert await manager._bundled_reset_would_invalidate_saved_reasoning() is False
    get_rules.assert_awaited_once_with(hass, "entry", "agent")


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gpt-5.6", None, False),
        ("gpt-5.6", "high", False),
        ("gpt-5.6", "minimal", True),
        ("gpt-5.6", 42, False),
        ("private-model", "xhigh", True),
    ],
    ids=[
        "no-reasoning",
        "bundled-compatible",
        "download-only-reasoning",
        "legacy-malformed-reasoning",
        "model-absent-from-bundled-catalog",
    ],
)
async def test_bundled_reset_safeguard_checks_saved_conversation_reasoning(
    manager, monkeypatch, model, effort, expected
):
    manager.catalog = candidate()
    await set_saved_conversation(monkeypatch, manager, model=model, effort=effort)

    assert await manager._bundled_reset_would_invalidate_saved_reasoning() is expected


async def test_concurrent_ordinary_refreshes_share_one_fetch(manager, monkeypatch):
    raw = json.dumps(candidate()).encode()
    started = asyncio.Event()
    release = asyncio.Event()

    async def chunks(_size):
        started.set()
        await release.wait()
        yield raw

    response = SimpleNamespace(
        status=200,
        headers={"ETag": '"v2"'},
        content=SimpleNamespace(iter_chunked=chunks),
    )
    context = AsyncMock()
    context.__aenter__.return_value = response
    get = Mock(return_value=context)
    monkeypatch.setattr(
        runtime, "async_get_clientsession", lambda _: SimpleNamespace(get=get)
    )

    first = asyncio.create_task(manager.async_update())
    await started.wait()
    second = asyncio.create_task(manager.async_update())
    await asyncio.sleep(0)
    assert not second.done()

    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert get.call_count == 1
    assert first_result == second_result
    assert first_result["source"] == "bundled"
    assert first_result["update_available"] is True
    assert first_result["last_error"] is None
    assert manager.catalog is None
    assert manager.available_catalog == candidate()
    assert manager.store.saved["catalog"] is None
    assert manager.store.saved["available_catalog"] == candidate()


def test_hot_transition_cannot_remove_previously_valid_reasoning_choice() -> None:
    current = _expanded_candidate()
    candidate = deepcopy(current)
    candidate["catalog_version"] += 1
    model = next(item for item in candidate["models"] if item["id"] == "gpt-5.6")
    model["reasoning"]["efforts"].remove("minimal")
    model["reasoning"]["by_api"]["responses"]["efforts"].remove("minimal")

    with pytest.raises(ValueError, match="cannot remove reasoning effort choices"):
        data.validate_catalog_transition(current, candidate)


def test_hot_transition_cannot_remove_reasoning_capability() -> None:
    candidate = deepcopy(data.BUNDLED_CATALOG)
    candidate["catalog_version"] += 1
    model = next(item for item in candidate["models"] if item["id"] == "gpt-5.6")
    model["reasoning"]["supported"] = False
    model["reasoning"]["efforts"] = []
    model["reasoning"]["openai_default"] = None
    for api in ("responses", "chat_completions"):
        model["reasoning"]["by_api"][api]["efforts"] = []
    model["recommended_profile"]["reasoning_effort"] = None
    for parameter in ("temperature", "top_p"):
        model[parameter] = {
            "support": "undocumented",
            "allowed_reasoning_efforts": None,
            "send_policy": "omit",
        }
    model["function_calling"]["chat_completions"] = False
    model["tools"]["function"]["chat_completions"] = {"support": "never"}

    with pytest.raises(ValueError, match="cannot remove reasoning effort choices"):
        data.validate_catalog_transition(None, candidate)


def test_new_exact_model_does_not_inherit_unknown_model_capabilities() -> None:
    """An unknown ID has no permissive family fallback to preserve in v2."""
    assert data.model_metadata("private-model")["status"] == "unknown"
    assert data.model_metadata("private-model")["reasoning"]["efforts"] == []

    candidate = deepcopy(data.BUNDLED_CATALOG)
    candidate["catalog_version"] += 1
    model = deepcopy(
        next(item for item in candidate["models"] if item["id"] == "gpt-4.1")
    )
    model.update(id="private-model", display_name="private-model")
    candidate["models"].append(model)

    data.validate_catalog_transition(None, candidate)


async def test_manager_rejects_narrowing_and_keeps_last_good_catalog(
    hass, monkeypatch
) -> None:
    manager = runtime.ModelCatalogManager(hass)
    manager.store = MemoryStore()
    current = _expanded_candidate()
    _transport(monkeypatch, json.dumps(current).encode(), etag='"v2"')
    checked = await manager.async_update(force=True)
    assert checked["last_error"] is None
    assert checked["update_available"] is True
    applied = await manager.async_apply_update()
    assert applied["last_error"] is None
    assert applied["source"] == "downloaded"

    candidate = deepcopy(current)
    candidate["catalog_version"] += 1
    model = next(item for item in candidate["models"] if item["id"] == "gpt-5.6")
    model["reasoning"]["efforts"].remove("minimal")
    model["reasoning"]["by_api"]["responses"]["efforts"].remove("minimal")
    _transport(monkeypatch, json.dumps(candidate).encode(), etag='"v3"')

    result = await manager.async_update(force=True)
    assert result["last_error"]
    assert manager.catalog == current
    assert manager.store.saved["catalog"] == current
    assert data.model_metadata("gpt-5.6")["reasoning"]["efforts"][-1] == "minimal"


async def test_restart_rejects_stored_override_that_narrows_bundled_choices(
    hass,
) -> None:
    candidate = deepcopy(data.BUNDLED_CATALOG)
    candidate["catalog_version"] += 1
    model = next(item for item in candidate["models"] if item["id"] == "gpt-5.6")
    model["reasoning"]["efforts"] = ["low"]
    model["reasoning"]["by_api"]["responses"]["efforts"] = ["low"]
    model["reasoning"]["by_api"]["chat_completions"]["efforts"] = ["low"]

    manager = runtime.ModelCatalogManager(hass)
    manager.store = MemoryStore()
    manager.store.saved = {
        "catalog": candidate,
        "etag": '"bad"',
        "last_checked": 0,
    }
    await manager.async_load()

    assert manager.catalog is None
    assert manager.status()["source"] == "bundled"
    assert manager.status()["last_error"]
    assert data.model_metadata("gpt-5.6")["reasoning"]["efforts"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


async def test_reset_is_blocked_when_saved_agent_uses_download_only_choice(
    hass,
) -> None:
    current = _expanded_candidate()
    subentry = SimpleNamespace(
        data={
            CONF_CHAT_MODEL: "gpt-5.6",
            CONF_REASONING_EFFORT: "minimal",
        },
        subentry_id="conversation-subentry",
        subentry_type="conversation",
    )
    entry = SimpleNamespace(
        entry_id="catalog-entry",
        subentries={subentry.subentry_id: subentry},
    )
    hass.config_entries.async_entries.return_value = [entry]

    manager = runtime.ModelCatalogManager(hass)
    manager.store = MemoryStore()
    manager.catalog = current
    manager.etag = '"v2"'
    manager.store.saved = {
        "catalog": current,
        "etag": manager.etag,
        "last_checked": 0,
    }
    data.activate_catalog(current)

    result = await manager.async_reset()

    hass.config_entries.async_entries.assert_called_once_with(DOMAIN)
    assert result["source"] == "downloaded"
    assert "saved configuration" in result["last_error"]
    assert "blocked" in result["last_error"]
    assert manager.etag == '"v2"'
    assert manager.catalog == current
    assert manager.store.saved["catalog"] == current
    assert data.model_metadata("gpt-5.6")["reasoning"]["efforts"][-1] == "minimal"
