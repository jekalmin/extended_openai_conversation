"""Model Catalog refresh failures, persistence atomicity, publication and retry contracts."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiohttp import ClientError
import pytest

from custom_components.extended_openai_conversation_responses import (
    model_catalog as catalog,
    model_catalog as data,
    model_catalog_manager as runtime,
)


@pytest.fixture(autouse=True)
def isolated_catalog():
    """Keep catalogue publication local to each test."""
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


def _stored_manager(hass) -> runtime.ModelCatalogManager:
    manager = runtime.ModelCatalogManager(hass)
    manager.store = MemoryStore()
    return manager


def _downloaded_candidate() -> dict:
    candidate = deepcopy(data.BUNDLED_CATALOG)
    candidate["catalog_version"] += 1
    candidate["models"][0]["display_name"] = "Downloaded catalogue"
    return candidate


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(schema_version=value["schema_version"] + 1),
        lambda value: value["compatibility"].update(minimum_eoai_version="99.0.0"),
    ],
)
async def test_incompatible_check_preserves_catalog_and_etag_recovery(
    manager, monkeypatch, change
):
    candidate = _downloaded_candidate()
    change(candidate)
    manager._candidate_preserves_saved_requests = AsyncMock()
    get = _install_transport(monkeypatch, candidate, etag='"incompatible"')

    result = await manager.async_check(force=True)
    assert result["incompatible_catalog"] == {
        "schema_version": candidate["schema_version"],
        "catalog_version": candidate["catalog_version"],
        "minimum_eoai_version": candidate["compatibility"]["minimum_eoai_version"],
    }
    assert result["last_error"] is None
    assert result["source"] == "bundled"
    assert result["update_available"] is False
    assert manager.store.saved["catalog"] is None
    assert manager.store.saved["available_catalog"] is None
    assert manager.store.saved["etag"] is None
    assert manager.etag is None
    manager._candidate_preserves_saved_requests.assert_not_called()
    assert data.model_metadata("gpt-6-astra")["display_name"] == "gpt-6-astra"

    restarted = runtime.ModelCatalogManager(manager.hass)
    restarted.store = MemoryStore(manager.store.saved)
    await restarted.async_load()
    assert restarted.status()["incompatible_catalog"] == result["incompatible_catalog"]
    assert restarted.etag is None

    compatible = _downloaded_candidate()
    _install_transport(monkeypatch, compatible, etag='"compatible"')
    recovered = await restarted.async_check(force=True)
    assert recovered["incompatible_catalog"] is None
    assert recovered["update_available"] is True
    assert recovered["available_catalog_version"] == compatible["catalog_version"]
    assert get.call_args.kwargs["headers"] == {}


def _versioned_candidate(*, increment: int = 1) -> dict:
    value = deepcopy(catalog.BUNDLED_CATALOG)
    value["catalog_version"] += increment
    value["models"][0]["display_name"] = f"Updated {increment}"
    return value


class _ResponseContext:
    def __init__(self, response) -> None:
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, *_args) -> None:
        return None


def _install_transport(
    monkeypatch, payload: dict, *, etag: str | None = '"next"'
) -> Mock:
    raw = json.dumps(payload).encode()

    async def chunks(_size: int):
        yield raw

    response = SimpleNamespace(
        status=200,
        headers={} if etag is None else {"ETag": etag},
        content=SimpleNamespace(iter_chunked=chunks),
    )
    get = Mock(return_value=_ResponseContext(response))
    monkeypatch.setattr(
        runtime,
        "async_get_clientsession",
        lambda _hass: SimpleNamespace(get=get),
    )
    return get


def _websocket_handler():
    """Return the undecorated handler when HA decorators expose wrapped callables."""
    return inspect.unwrap(runtime.websocket_catalog)


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


def transport(monkeypatch, raw=b"", *, status=200, error=None, etag='"v2"'):
    async def chunks(_size):
        for offset in range(0, len(raw), 100):
            yield raw[offset : offset + 100]

    response = SimpleNamespace(
        status=status,
        headers={"ETag": etag} if etag else {},
        content=SimpleNamespace(iter_chunked=chunks),
    )
    context = AsyncMock()
    context.__aenter__.return_value = response
    if error is not None:
        context.__aenter__.side_effect = error
    get = Mock(return_value=context)
    monkeypatch.setattr(
        runtime, "async_get_clientsession", lambda _: SimpleNamespace(get=get)
    )
    return get


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"[]",
        b'{"schema_version":2,"schema_version":2}',
        b" " * (data.MAX_CATALOG_BYTES + 1),
    ],
    ids=["truncated", "array", "duplicate-key", "oversized"],
)
async def test_malformed_and_oversized_download_rejected(
    check_manager, monkeypatch, raw
):
    _chunked_transport(monkeypatch, raw)
    assert (await check_manager.async_check(force=True))["last_error"]
    assert check_manager.catalog is None
    assert check_manager.available_catalog is None
    assert data.model_metadata("gpt-5.6")["status"] == "current"


@pytest.mark.parametrize(
    "failure",
    ["http-rate-limit", "http-server", "http-client", "storage", "old", "same-version"],
)
async def test_failed_check_never_replaces_current_data(
    check_manager, monkeypatch, failure
):
    first = _label_candidate()
    _chunked_transport(monkeypatch, json.dumps(first).encode())
    await check_manager.async_check(force=True)
    await check_manager.async_apply_update()
    previous = deepcopy(check_manager.catalog)
    previous_etag = check_manager.etag

    value = deepcopy(first)
    value["catalog_version"] += 1
    value["models"][0]["display_name"] = "Another label"
    if failure == "old":
        value["catalog_version"] = data.BUNDLED_CATALOG["catalog_version"] - 1
    elif failure == "same-version":
        value["catalog_version"] = previous["catalog_version"]
    elif failure == "storage":
        check_manager.store.fail = True

    _chunked_transport(
        monkeypatch,
        json.dumps(value).encode(),
        status={"http-rate-limit": 429, "http-server": 500, "http-client": 400}.get(
            failure, 200
        ),
    )
    result = await check_manager.async_check(force=True)
    assert (
        result["last_error"]
        == "Model data check failed; the current catalogue was kept."
    )
    assert check_manager.etag == previous_etag
    assert check_manager.catalog == previous
    assert check_manager.available_catalog is None
    assert (
        data.model_metadata("gpt-6-astra")["display_name"] == "Astra (catalog update)"
    )
    if not check_manager.store.fail:
        assert check_manager.store.saved["catalog"] == previous
        assert check_manager.store.saved["available_catalog"] is None


async def test_migrated_load_survives_rewrite_failure(hass, monkeypatch) -> None:
    manager = _stored_manager(hass)
    candidate = _downloaded_candidate()
    manager.store = MemoryStore(
        {"catalog": {"legacy": True}, "etag": None, "last_checked": 0}
    )
    manager.store.fail = True
    monkeypatch.setattr(
        runtime,
        "validate_or_migrate_catalog",
        lambda _value: (deepcopy(candidate), True),
    )

    await manager.async_load()

    assert manager.catalog == candidate
    assert manager.last_error is None
    assert data.model_metadata(candidate["models"][0]["id"])["display_name"] == (
        "Downloaded catalogue"
    )


async def test_unsolicited_not_modified_is_rejected(hass, monkeypatch) -> None:
    manager = _stored_manager(hass)
    response = SimpleNamespace(status=304, headers={}, content=None)
    context = AsyncMock()
    context.__aenter__.return_value = response
    session = SimpleNamespace(get=lambda *_args, **_kwargs: context)
    monkeypatch.setattr(runtime, "async_get_clientsession", lambda _hass: session)

    status = await manager.async_update(force=True)

    assert status["source"] == "bundled"
    assert status["last_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("etag", ["x" * 257, '"bad\nvalue"', '"bad\rvalue"'])
async def test_update_rejects_invalid_response_etag(
    hass, monkeypatch, etag: str
) -> None:
    manager = runtime.ModelCatalogManager(hass)
    _install_transport(monkeypatch, _versioned_candidate(), etag=etag)
    save = AsyncMock()
    activate = Mock()
    monkeypatch.setattr(manager, "_save", save)
    monkeypatch.setattr(runtime, "activate_catalog", activate)

    result = await manager.async_update(force=True)

    assert (
        result["last_error"]
        == "Model data check failed; the current catalogue was kept."
    )
    assert manager.catalog is None
    assert manager.etag is None
    activate.assert_not_called()
    assert save.await_count == 1
    saved_catalog, saved_available, saved_etag, _checked = save.await_args.args
    assert saved_catalog is None
    assert saved_available is None
    assert saved_etag is None


@pytest.mark.asyncio
async def test_update_rejects_catalogue_version_rollback(hass, monkeypatch) -> None:
    manager = runtime.ModelCatalogManager(hass)
    current = _versioned_candidate(increment=2)
    manager.catalog = current
    manager.etag = '"current"'
    _install_transport(monkeypatch, _versioned_candidate(increment=1), etag='"older"')
    save = AsyncMock()
    activate = Mock()
    monkeypatch.setattr(manager, "_save", save)
    monkeypatch.setattr(runtime, "activate_catalog", activate)

    result = await manager.async_update(force=True)

    assert (
        result["last_error"]
        == "Model data check failed; the current catalogue was kept."
    )
    assert manager.catalog == current
    assert manager.etag == '"current"'
    activate.assert_not_called()
    assert save.await_count == 1
    saved_catalog, saved_available, saved_etag, _checked = save.await_args.args
    assert saved_catalog == current
    assert saved_available is None
    assert saved_etag == '"current"'


@pytest.mark.asyncio
async def test_websocket_update_failure_sends_error_without_metadata_work(
    hass, monkeypatch
) -> None:
    failed_status = {
        "source": "bundled",
        "catalog_version": 2,
        "schema_version": 2,
        "update_available": False,
        "available_catalog_version": None,
        "last_checked": 123.0,
        "last_error": "refresh failed",
    }
    manager = SimpleNamespace(
        catalog=None,
        status=Mock(),
        async_check=AsyncMock(return_value=failed_status),
        async_reset=AsyncMock(),
    )
    hass.data[runtime.DATA_MANAGER] = manager
    connection = SimpleNamespace(send_result=Mock(), send_error=Mock())
    monkeypatch.setattr(
        runtime,
        "model_metadata",
        lambda *_args: pytest.fail(
            "metadata should not be evaluated after update failure"
        ),
    )
    monkeypatch.setattr(
        runtime,
        "frontend_capabilities",
        lambda *_args: pytest.fail(
            "capabilities should not be evaluated after update failure"
        ),
    )
    monkeypatch.setattr(
        runtime,
        "catalog_picker_models",
        lambda *_args: pytest.fail(
            "picker should not be evaluated after update failure"
        ),
    )

    await _websocket_handler()(
        hass,
        connection,
        {"id": 99, "action": "update", "model": "gpt-test"},
    )

    manager.async_check.assert_awaited_once_with(force=True)
    connection.send_error.assert_called_once_with(
        99, "model_catalog_check_failed", "refresh failed"
    )
    connection.send_result.assert_not_called()
    manager.status.assert_not_called()


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (200, TimeoutError()),
        (200, ClientError("offline")),
        (429, None),
        (503, None),
    ],
    ids=["timeout", "client-error", "rate-limit", "server-error"],
)
async def test_transient_refresh_failures_do_not_warn_or_error(
    manager, monkeypatch, caplog, status, error
):
    get = transport(monkeypatch, status=status, error=error)

    with caplog.at_level(logging.DEBUG, logger=runtime.__name__):
        result = await manager.async_update()

    assert result["last_error"]
    assert result["source"] == "bundled"
    assert manager.store.saved["catalog"] is None
    assert not [
        record
        for record in caplog.records
        if record.name == runtime.__name__ and record.levelno >= logging.WARNING
    ]

    # A background outage is remembered, so the hourly scheduler does not retry
    # until the normal daily refresh interval has elapsed.
    await manager.async_update()
    assert get.call_count == 1


async def test_permanent_http_failure_retains_warning_visibility(
    manager, monkeypatch, caplog
):
    transport(monkeypatch, status=404)

    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        result = await manager.async_update()

    assert result["last_error"]
    assert result["source"] == "bundled"
    assert any(
        record.name == runtime.__name__ and record.levelno == logging.WARNING
        for record in caplog.records
    )


async def test_reset_persistence_failure_keeps_published_catalog_and_can_retry(
    manager, monkeypatch
):
    downloaded = candidate()
    manager.catalog = downloaded
    manager.etag = '"v2"'
    manager.store.saved = {
        "catalog": deepcopy(downloaded),
        "etag": manager.etag,
        "last_checked": manager.last_checked,
    }
    runtime.activate_catalog(downloaded)
    monkeypatch.setattr(
        manager.hass.config_entries, "async_entries", lambda _domain: []
    )
    manager.store.fail = True

    failed = await manager.async_reset()

    assert failed["source"] == "downloaded"
    assert "reset failed" in failed["last_error"]
    assert manager.catalog == downloaded
    assert manager.etag == '"v2"'
    assert manager.store.saved["catalog"] == downloaded
    assert runtime.model_metadata("gpt-5.6")["reasoning"]["efforts"][-1] == "minimal"

    manager.store.fail = False
    succeeded = await manager.async_reset()

    assert succeeded["source"] == "bundled"
    assert succeeded["last_error"] is None
    assert manager.catalog is None
    assert manager.available_catalog == downloaded
    assert manager.etag == '"v2"'
    assert manager.store.saved["catalog"] is None
    assert manager.store.saved["available_catalog"] == downloaded
    assert runtime.model_metadata("gpt-5.6")["reasoning"]["efforts"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]


async def test_failed_refresh_keeps_downloaded_catalog_and_forced_retry_succeeds(
    manager, monkeypatch
):
    downloaded = candidate()
    transport(monkeypatch, json.dumps(downloaded).encode(), etag='"v2"')
    await manager.async_check(force=True)
    await manager.async_apply_update()
    manager.last_checked = 0.0
    failed_get = transport(monkeypatch, error=TimeoutError())

    failed = await manager.async_update()

    assert failed["source"] == "downloaded"
    assert failed["last_error"]
    assert manager.catalog == downloaded
    assert manager.store.saved["catalog"] == downloaded
    assert failed_get.call_count == 1
    assert manager.available_catalog is None
    assert manager.store.saved["available_catalog"] is None
    assert manager.etag == '"v2"'
    assert runtime.model_metadata("gpt-5.6")["reasoning"]["efforts"][-1] == "minimal"

    refreshed = candidate()
    refreshed["catalog_version"] += 1
    refreshed["models"][0]["display_name"] += " refreshed"
    success_get = transport(monkeypatch, json.dumps(refreshed).encode(), etag='"v3"')

    succeeded = await manager.async_update(force=True)

    assert success_get.call_count == 1
    assert succeeded["source"] == "downloaded"
    assert succeeded["update_available"] is True
    assert succeeded["last_error"] is None
    assert manager.catalog == downloaded
    assert manager.available_catalog == refreshed
    assert manager.etag == '"v3"'
    assert manager.store.saved["catalog"] == downloaded
    assert manager.store.saved["available_catalog"] == refreshed
