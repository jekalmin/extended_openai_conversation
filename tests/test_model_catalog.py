"""Model Catalog loading, strict validation, lookup and manager integration contracts."""

from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.extended_openai_conversation_responses import (
    model_catalog as data,
    model_catalog_manager as runtime,
)


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    """Keep catalogue publication local to each test."""
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

    async def async_load(self):
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


def _websocket_handler():
    """Return the undecorated handler when HA decorators expose wrapped callables."""
    return inspect.unwrap(runtime.websocket_catalog)


def test_bundled_catalog_is_current_schema_and_parses_exactly():
    parsed = data.parse_catalog(Path(data.__file__).with_suffix(".json").read_bytes())
    assert parsed == data.BUNDLED_CATALOG
    assert parsed["schema_version"] == data.CURRENT_SCHEMA_VERSION
    assert parsed["catalog_version"] >= 7


def test_previous_supported_schema_remains_readable():
    prior = deepcopy(data.BUNDLED_CATALOG)
    prior["schema_version"] = 5
    prior["catalog_version"] -= 1
    prior.pop("compatibility")
    assert data.validate_catalog(prior)["schema_version"] == 5


def test_future_schema_and_newer_release_have_distinct_error():
    for mutation in (
        lambda value: value.update(schema_version=99),
        lambda value: value["compatibility"].update(minimum_eoai_version="99.0.0"),
    ):
        candidate = deepcopy(data.BUNDLED_CATALOG)
        mutation(candidate)
        with pytest.raises(
            data.IncompatibleCatalogError, match="requires a newer version"
        ):
            data.validate_catalog(candidate)


def test_invalid_schema_number_is_not_an_incompatible_update():
    candidate = deepcopy(data.BUNDLED_CATALOG)
    candidate["schema_version"] = 0
    with pytest.raises(ValueError, match="Invalid model catalogue version"):
        data.validate_catalog(candidate)


def test_required_current_models_and_invalid_aliases():
    expected = {
        "gpt-6-astra",
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
    }
    current = {item["id"] for item in data.catalog_picker_models()}
    assert expected <= current
    assert not {"o2", "o4", "gpt-5.3"} & current


def test_picker_hides_deprecated_unless_already_selected():
    normal = data.catalog_picker_models()
    assert all(item["status"] == "current" for item in normal)
    assert "gpt-4-turbo" not in {item["id"] for item in normal}

    existing = data.catalog_picker_models(selected_model="gpt-4-turbo")
    deprecated = next(item for item in existing if item["id"] == "gpt-4-turbo")
    assert deprecated["status"] == "deprecated"


def test_unknown_model_is_preserved_with_conservative_capabilities():
    metadata = data.model_metadata("my-future-openai-model")
    assert metadata["id"] == "my-future-openai-model"
    assert metadata["status"] == "unknown"
    assert metadata["temperature"]["support"] == "undocumented"
    assert metadata["top_p"]["support"] == "undocumented"
    assert metadata["function_calling"]["responses"] is False
    assert metadata["function_calling"]["chat_completions"] is False


def test_current_models_have_exact_non_streaming_exceptions():
    for raw in data.BUNDLED_CATALOG["models"]:
        item = data.model_metadata(raw["id"])
        if item["status"] == "current":
            assert item["streaming"] is (item["id"] not in {"gpt-5.5-pro", "o3-pro"}), (
                item["id"]
            )


def test_bundled_service_tiers_match_current_openai_support() -> None:
    expected_latest = {
        "gpt-6-astra": ["auto", "default", "flex", "fast", "priority"],
        "gpt-6-sol": ["auto", "default", "flex", "fast", "priority"],
        "gpt-6-luna": ["auto", "default", "flex", "fast", "priority"],
        "gpt-5.6": ["auto", "default", "flex", "fast", "priority", "ultrafast"],
        "gpt-5.6-sol": ["auto", "default", "flex", "fast", "priority", "ultrafast"],
        "gpt-5.6-terra": ["auto", "default", "flex", "fast", "priority"],
        "gpt-5.6-luna": ["auto", "default", "flex", "fast", "priority"],
        "gpt-5.5": ["auto", "default", "flex"],
        "gpt-5.4": ["auto", "default", "flex"],
    }
    by_id = {item["id"]: item for item in data.BUNDLED_CATALOG["models"]}
    for model_id, tiers in expected_latest.items():
        assert by_id[model_id]["service_tiers"] == tiers
    for model_id in ("gpt-5-mini", "gpt-4.1", "gpt-4o", "o3"):
        assert by_id[model_id]["service_tiers"] == ["auto", "default"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(schema_version=1),
        lambda v: v.update(schema_version=True),
        lambda v: v.update(catalog_version="next"),
        lambda v: v.update(models=[]),
        lambda v: v["models"].append(deepcopy(v["models"][0])),
        lambda v: v["models"][0].update(status="retired"),
        lambda v: v["models"][0]["temperature"].update(support="maybe"),
        lambda v: v["models"][0]["temperature"].update(send_policy="send"),
        lambda v: v["models"][0]["reasoning"].update(efforts=["run code"]),
        lambda v: v["models"][0]["api"].update(responses=1),
        lambda v: v["models"][0]["limits"].update(max_output_tokens=0),
    ],
)
def test_invalid_schema_v4_catalog_is_rejected(mutate):
    value = _label_candidate()
    mutate(value)
    with pytest.raises(ValueError):
        data.validate_catalog(value)


async def test_stored_v1_catalog_is_migrated_to_authoritative_v4(check_manager):
    check_manager.store.saved = {
        "catalog": {
            "schema_version": 1,
            "catalog_version": 1,
            "defaults": {},
            "models": [],
        },
        "etag": '"old"',
        "last_checked": 0,
    }
    await check_manager.async_load()
    assert check_manager.status()["schema_version"] == 6
    assert data.model_metadata("gpt-5.6")["reasoning"]["efforts"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


async def test_corrupt_storage_falls_back_to_bundled(check_manager):
    check_manager.store.saved = {"catalog": {"schema_version": 99}}
    await check_manager.async_load()
    assert check_manager.status()["source"] == "bundled"
    assert check_manager.status()["schema_version"] == 6
    assert check_manager.last_error


async def test_setup_is_shared_and_timer_is_removed_on_stop(hass, monkeypatch):
    check_manager = runtime.ModelCatalogManager(hass)
    check_manager.store = MemoryStore()
    monkeypatch.setattr(runtime, "ModelCatalogManager", lambda _: check_manager)
    interval, cancel = Mock(), Mock()
    interval.return_value = cancel
    monkeypatch.setattr(runtime, "async_track_time_interval", interval)
    monkeypatch.setattr(runtime.websocket_api, "async_register_command", Mock())
    await runtime.async_setup_model_catalog(hass)
    await runtime.async_setup_model_catalog(hass)
    assert interval.call_count == 1
    assert interval.call_args.args[2].total_seconds() == 3600
    check_manager.async_check = AsyncMock()
    await interval.call_args.args[1](None)
    check_manager.async_check.assert_awaited_once()
    hass.bus.async_listen_once.call_args.args[1](None)
    cancel.assert_called_once()


@pytest.mark.parametrize(
    ("allowed", "match"),
    [
        ([], "Invalid conditional temperature reasoning efforts"),
        (["low", "low"], "Invalid conditional temperature reasoning efforts"),
        (["none"], "Invalid conditional temperature reasoning efforts"),
    ],
)
def test_conditional_sampling_requires_nonempty_unique_supported_efforts(
    allowed, match
) -> None:
    value = _catalog()
    _set_temperature(_model(value), "conditional", allowed, "omit_unless_configured")

    with pytest.raises(ValueError, match=match):
        data.validate_catalog(value)


def test_nonconditional_sampling_requires_null_allowed_efforts() -> None:
    value = _catalog()
    _set_temperature(_model(value), "always", ["low"], "omit_unless_configured")

    with pytest.raises(ValueError, match="allowed_reasoning_efforts must be null"):
        data.validate_catalog(value)


def test_unsupported_sampling_must_be_omitted() -> None:
    value = _catalog()
    _set_temperature(_model(value), "never", None, "omit_unless_configured")

    with pytest.raises(ValueError, match="must be omitted"):
        data.validate_catalog(value)


def test_auto_api_must_reference_an_enabled_api() -> None:
    value = _catalog()
    model = _model(value)
    model["api"]["responses"] = False

    with pytest.raises(ValueError, match="Invalid Auto API preference"):
        data.validate_catalog(value)


def test_function_calling_flags_must_be_real_booleans() -> None:
    value = _catalog()
    _model(value)["function_calling"]["responses"] = 1

    with pytest.raises(ValueError, match="Unexpected or missing catalogue fields"):
        data.validate_catalog(value)


def test_function_preference_must_reference_enabled_api() -> None:
    value = _catalog()
    model = _model(value)
    model["auto_api"] = None
    model["api"]["responses"] = False
    model["function_calling"]["preferred_api"] = "responses"
    model["recommended_profile"]["api"] = "chat_completions"

    with pytest.raises(ValueError, match="Invalid preferred function-calling API"):
        data.validate_catalog(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda reasoning: reasoning.update(supported="yes"),
        lambda reasoning: reasoning.update(efforts="low"),
        lambda reasoning: reasoning.update(efforts=["low", "low"]),
        lambda reasoning: reasoning.update(supported=False),
    ],
)
def test_reasoning_capability_shape_is_strict(mutate) -> None:
    value = _catalog()
    mutate(_model(value)["reasoning"])

    with pytest.raises(ValueError, match="Invalid reasoning"):
        data.validate_catalog(value)


def test_openai_reasoning_default_must_be_an_allowed_effort() -> None:
    value = _catalog()
    _model(value)["reasoning"]["openai_default"] = "none"

    with pytest.raises(ValueError, match="Invalid OpenAI reasoning default"):
        data.validate_catalog(value)


def test_streaming_flag_must_be_boolean() -> None:
    value = _catalog()
    _model(value)["streaming"] = 1

    with pytest.raises(ValueError, match="Streaming capability must be boolean"):
        data.validate_catalog(value)


def test_output_token_mapping_is_exact() -> None:
    value = _catalog()
    _model(value)["output_tokens"]["legacy_max_tokens"] = "max_tokens"

    with pytest.raises(ValueError, match="Invalid output-token parameter mapping"):
        data.validate_catalog(value)


def test_recommended_api_must_be_available() -> None:
    value = _catalog()
    model = _model(value)
    model["recommended_profile"]["api"] = "invalid"

    with pytest.raises(ValueError, match="Invalid recommended API"):
        data.validate_catalog(value)


def test_recommended_reasoning_effort_must_be_supported() -> None:
    value = _catalog()
    _model(value)["recommended_profile"]["reasoning_effort"] = "none"

    with pytest.raises(ValueError, match="Invalid recommended reasoning effort"):
        data.validate_catalog(value)


def test_recommended_sampling_profile_must_omit_sampling() -> None:
    value = _catalog()
    _model(value)["recommended_profile"]["temperature"] = 0.7

    with pytest.raises(ValueError, match="Recommended sampling profile must omit"):
        data.validate_catalog(value)


def test_service_tiers_and_cache_flag_are_strictly_validated() -> None:
    value = _catalog()
    _model(value)["service_tiers"] = ["auto", "auto"]
    with pytest.raises(ValueError, match="Invalid service-tier"):
        data.validate_catalog(value)

    value = _catalog()
    _model(value)["service_tiers"] = ["turbo"]
    with pytest.raises(ValueError, match="Invalid service-tier"):
        data.validate_catalog(value)

    value = _catalog()
    _model(value)["explicit_prompt_cache"] = 1
    with pytest.raises(ValueError, match="Invalid service-tier"):
        data.validate_catalog(value)


def test_alias_target_and_lifecycle_note_are_strictly_validated() -> None:
    value = _catalog()
    _model(value)["alias_of"] = "INVALID MODEL ID"
    with pytest.raises(ValueError, match="Invalid alias target"):
        data.validate_catalog(value)

    value = _catalog()
    _model(value)["lifecycle_note"] = "x" * 513
    with pytest.raises(ValueError, match="Invalid lifecycle note"):
        data.validate_catalog(value)


def test_defaults_must_describe_unknown_models() -> None:
    value = _catalog()
    value["defaults"]["status"] = "current"

    with pytest.raises(ValueError, match="defaults must describe unknown models"):
        data.validate_catalog(value)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda model: model.update(display_name=""), "Invalid display name"),
        (lambda model: model.update(kind="family"), "Invalid model kind"),
    ],
)
def test_model_wrapper_metadata_is_strict(mutate, match) -> None:
    value = _catalog()
    mutate(_model(value))

    with pytest.raises(ValueError, match=match):
        data.validate_catalog(value)


def test_alias_must_reference_a_different_catalog_model() -> None:
    value = _catalog()
    model = _model(value)
    model["alias_of"] = model["id"]

    with pytest.raises(ValueError, match="Alias target must reference another"):
        data.validate_catalog(value)


def test_v1_migration_rejects_non_v1_and_preserves_monotonic_version() -> None:
    with pytest.raises(ValueError, match="Not a model catalogue v1 document"):
        data.migrate_catalog_v1({"schema_version": 2})

    migrated = data.migrate_catalog_v1(
        {
            "schema_version": 1,
            "catalog_version": data.BUNDLED_CATALOG["catalog_version"] + 5,
        }
    )
    assert migrated["schema_version"] == 6
    assert migrated["catalog_version"] == data.BUNDLED_CATALOG["catalog_version"] + 6


def test_validate_or_migrate_marks_legacy_schemas_as_migrated() -> None:
    migrated, changed = data.validate_or_migrate_catalog({"schema_version": 1})
    assert changed is True
    assert migrated["schema_version"] == 6

    legacy_v2 = deepcopy(data.BUNDLED_CATALOG)
    legacy_v2["models"] = [
        deepcopy(item) for item in data.BUNDLED_CATALOG.resolved.values()
    ]
    legacy_v2["schema_version"] = 2
    legacy_v2["catalog_version"] = 2
    legacy_v2["defaults"]["service_tier"] = bool(
        legacy_v2["defaults"].pop("service_tiers")
    )
    for model in legacy_v2["models"]:
        model["service_tier"] = bool(model.pop("service_tiers"))
    migrated, changed = data.validate_or_migrate_catalog(legacy_v2)
    assert changed is True
    assert migrated["schema_version"] == 5
    assert (
        migrated["models"][0]["service_tiers"]
        == data.BUNDLED_CATALOG["models"][0]["service_tiers"]
    )

    current = _catalog()
    validated, changed = data.validate_or_migrate_catalog(current)
    assert changed is False
    assert validated == current
    assert validated is not current


def test_catalog_picker_preserves_selected_custom_model_case_insensitively() -> None:
    picked = data.catalog_picker_models(selected_model="MY-CUSTOM-MODEL")
    selected = next(item for item in picked if item["id"] == "my-custom-model")
    assert selected["status"] == "unknown"
    assert selected["display_name"] == "my-custom-model"


def test_activate_catalog_can_restore_bundled_state() -> None:
    custom = _catalog()
    _model(custom)["display_name"] = "Updated Astra"
    data.activate_catalog(custom)
    assert data.model_metadata("gpt-6-astra")["display_name"] == "Updated Astra"

    data.activate_catalog(None)
    assert data.model_metadata("gpt-6-astra")["display_name"] == "gpt-6-astra"


@pytest.mark.parametrize(
    "saved",
    [
        {"catalog": None, "etag": None, "last_checked": True},
        {"catalog": None, "etag": None, "last_checked": time.time() + 3600},
        {"catalog": None, "etag": "bad\netag", "last_checked": 0},
        {"catalog": None, "etag": "x" * 257, "last_checked": 0},
    ],
    ids=["boolean-check-time", "future-check-time", "newline-etag", "oversized-etag"],
)
async def test_load_rejects_invalid_storage_metadata(hass, saved) -> None:
    manager = _stored_manager(hass)
    manager.store = MemoryStore(saved)

    await manager.async_load()

    assert manager.catalog is None
    assert (
        manager.last_error
        == "Stored model data could not be loaded; using bundled data."
    )
    assert manager.status()["source"] == "bundled"


async def test_load_discards_download_older_than_bundled(hass, monkeypatch) -> None:
    manager = _stored_manager(hass)
    old = deepcopy(data.BUNDLED_CATALOG)
    old["catalog_version"] -= 1
    manager.store = MemoryStore({"catalog": old, "etag": '"stale"', "last_checked": 0})
    monkeypatch.setattr(
        runtime,
        "validate_or_migrate_catalog",
        lambda value: (deepcopy(value), False),
    )

    await manager.async_load()

    assert manager.catalog is None
    assert manager.etag is None
    assert manager.last_error is None


@pytest.mark.asyncio
async def test_initialize_accepts_stored_metadata_without_catalog(
    hass, monkeypatch
) -> None:
    manager = runtime.ModelCatalogManager(hass)
    manager.store = SimpleNamespace(
        async_load=AsyncMock(
            return_value={"catalog": None, "etag": None, "last_checked": 0}
        )
    )
    validate_transition = Mock()
    activate = Mock()
    monkeypatch.setattr(runtime, "validate_catalog_transition", validate_transition)
    monkeypatch.setattr(runtime, "activate_catalog", activate)

    await manager.async_load()

    assert manager.catalog is None
    assert manager.etag is None
    assert manager.last_checked == 0
    assert manager.last_error is None
    validate_transition.assert_not_called()
    activate.assert_called_once_with(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["lookup", "reset", "update"])
async def test_websocket_actions_return_complete_catalog_payload(
    hass, monkeypatch, action: str
) -> None:
    status = {
        "source": "downloaded",
        "catalog_version": 7,
        "schema_version": 5,
        "update_available": False,
        "available_catalog_version": None,
        "last_checked": 123.0,
        "last_error": None,
    }
    manager = SimpleNamespace(
        catalog={"catalog_version": 7},
        status=Mock(return_value=status),
        async_check=AsyncMock(return_value=status),
        async_reset=AsyncMock(return_value=status),
    )
    hass.data[runtime.DATA_MANAGER] = manager
    connection = SimpleNamespace(send_result=Mock(), send_error=Mock())
    metadata = {"id": "gpt-test", "reasoning": {"efforts": ["low", "high"]}}
    capabilities = {"responses": True}
    picker = [{"id": "gpt-test"}]
    monkeypatch.setattr(runtime, "model_metadata", Mock(return_value=metadata))
    monkeypatch.setattr(
        runtime, "frontend_capabilities", Mock(return_value=capabilities)
    )
    monkeypatch.setattr(runtime, "catalog_picker_models", Mock(return_value=picker))

    await _websocket_handler()(
        hass,
        connection,
        {"id": 42, "action": action, "model": "gpt-test"},
    )

    connection.send_error.assert_not_called()
    connection.send_result.assert_called_once_with(
        42,
        {
            **status,
            "model_capabilities": capabilities,
            "model_metadata": metadata,
            "catalog_models": picker,
            "reasoning_effort_options": ["low", "high"],
        },
    )
    if action == "lookup":
        manager.async_check.assert_not_awaited()
        manager.async_reset.assert_not_awaited()
    elif action == "reset":
        manager.async_reset.assert_awaited_once_with()
        manager.async_check.assert_not_awaited()
    else:
        manager.async_check.assert_awaited_once_with(force=True)
        manager.async_reset.assert_not_awaited()
