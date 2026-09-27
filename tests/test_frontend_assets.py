"""Frontend asset manifest validation and registration contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation_responses import frontend_assets


def _write_manifest(tmp_path, payload):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    manifest = dist / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return dist, assets, manifest


def test_manifest_loader_rejects_missing_malformed_and_non_object(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(frontend_assets, "_MANIFEST_PATH", missing)
    with pytest.raises(RuntimeError, match="manifest is unavailable"):
        frontend_assets._load_entry_urls_sync()

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    monkeypatch.setattr(frontend_assets, "_MANIFEST_PATH", malformed)
    with pytest.raises(RuntimeError, match="manifest is unavailable"):
        frontend_assets._load_entry_urls_sync()

    not_object = tmp_path / "array.json"
    not_object.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(frontend_assets, "_MANIFEST_PATH", not_object)
    with pytest.raises(RuntimeError, match="manifest is invalid"):
        frontend_assets._load_entry_urls_sync()


def test_manifest_loader_validates_entry_paths_files_and_management(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist, assets, manifest = _write_manifest(
        tmp_path,
        {
            "ignored": {"isEntry": False, "name": "ignored", "file": "assets/x.js"},
            "bad-shape": {"isEntry": True, "name": 123, "file": "assets/y.js"},
        },
    )
    monkeypatch.setattr(frontend_assets, "_PRODUCTION_DIR", dist)
    monkeypatch.setattr(frontend_assets, "_MANIFEST_PATH", manifest)

    with pytest.raises(RuntimeError, match="management"):
        frontend_assets._load_entry_urls_sync()

    manifest.write_text(
        json.dumps(
            {
                "management": {
                    "isEntry": True,
                    "name": "management",
                    "file": "management.js",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="entry is invalid"):
        frontend_assets._load_entry_urls_sync()

    manifest.write_text(
        json.dumps(
            {
                "management": {
                    "isEntry": True,
                    "name": "management",
                    "file": "assets/missing.js",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="entry is unavailable"):
        frontend_assets._load_entry_urls_sync()

    (assets / "management.js").write_text("export {};", encoding="utf-8")
    (assets / "secondary.js").write_text("export {};", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "management": {
                    "isEntry": True,
                    "name": "management",
                    "file": "assets/management.js",
                },
                "secondary": {
                    "isEntry": True,
                    "name": "secondary",
                    "file": "assets/secondary.js",
                },
            }
        ),
        encoding="utf-8",
    )

    assert frontend_assets._load_entry_urls_sync() == {
        "management": "/extended_openai_conversation_responses/frontend/assets/management.js",
        "secondary": "/extended_openai_conversation_responses/frontend/assets/secondary.js",
    }


def test_frontend_entry_url_requires_registration_and_known_name() -> None:
    hass = SimpleNamespace(data={})
    with pytest.raises(RuntimeError, match="not registered"):
        frontend_assets.frontend_entry_url(hass, "management")

    hass.data[frontend_assets._FRONTEND_ENTRY_URLS] = {
        "management": "/frontend/management.js"
    }
    assert (
        frontend_assets.frontend_entry_url(hass, "management")
        == "/frontend/management.js"
    )
    with pytest.raises(RuntimeError, match="entry is unavailable"):
        frontend_assets.frontend_entry_url(hass, "missing")


@pytest.mark.asyncio
async def test_registration_is_executor_backed_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def async_add_executor_job(func):
        return func()

    http = SimpleNamespace(async_register_static_paths=AsyncMock())
    hass = SimpleNamespace(
        data={},
        async_add_executor_job=async_add_executor_job,
        http=http,
    )
    loader = lambda: {"management": "/frontend/management.js"}
    monkeypatch.setattr(frontend_assets, "_load_entry_urls_sync", loader)

    await frontend_assets.async_register_frontend_assets(hass)
    await frontend_assets.async_register_frontend_assets(hass)

    assert hass.data[frontend_assets._FRONTEND_ENTRY_URLS] == {
        "management": "/frontend/management.js"
    }
    assert hass.data[frontend_assets._FRONTEND_ASSET_SETUP] is True
    http.async_register_static_paths.assert_awaited_once()
