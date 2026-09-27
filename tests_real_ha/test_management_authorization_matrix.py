"""Compact real-HA authorization check for reviewed Management action classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pytest_homeassistant_custom_component.common import MockUser

from homeassistant.core import HomeAssistant
from tests_real_ha.test_management_backend_acceptance import (
    CLIENT_ID,
    _admin_client,
    _entry,
    _management_response,
    _setup_entry,
)

INVENTORY = Path(__file__).resolve().parents[1] / "tests_stress/management_action_inventory.json"

# One representative per admin-controlled section. Handler-specific privacy and
# cross-user ID tests remain in test_user_ownership_privacy.py.
ADMIN_REPRESENTATIVES = {
    "backup": "create",
    "configuration": "get",
    "conversations": "active",
    "diagnostics": "test_agent",
    "function_repair": "get",
    "guest_mode": "save_policy",
    "knowledge": "list",
    "memories": "reassign_legacy",
    "quiet_hours": "get",
    "request_rules": "list",
    "service_catalog": "get",
    "settings": "update",
    "tools": "starter",
    "usage": "runs",
}


@pytest.mark.asyncio
async def test_reviewed_admin_sections_reject_non_admin_at_real_websocket(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    classified = {
        section for section, spec in inventory["sections"].items()
        if "admin_only" in spec["authorization_classes"].values()
    }
    assert set(ADMIN_REPRESENTATIVES) == classified

    entry = _entry("Management authorization matrix")
    await _setup_entry(hass, entry)
    admin = await _admin_client(hass, hass_ws_client)
    restricted = MockUser(id="management-matrix-restricted", is_owner=False)
    restricted.add_to_hass(hass)
    token = await hass.auth.async_create_refresh_token(restricted, CLIENT_ID)
    client = await hass_ws_client(hass, hass.auth.async_create_access_token(token))

    for section, action in ADMIN_REPRESENTATIVES.items():
        assert inventory["sections"][section]["authorization_classes"][action] == "admin_only"
        denied = await _management_response(
            client, entry=entry, section=section, action=action
        )
        assert denied["success"] is False, (section, action, denied)
        assert "Administrator permission is required" in denied["error"]["message"]

    # Safe reads prove that the same registered command remains usable for an
    # owner. Mutating admin actions have their own feature acceptance evidence.
    for section, action in (
        ("backup", "create"),
        ("configuration", "get"),
        ("knowledge", "list"),
        ("quiet_hours", "get"),
        ("request_rules", "list"),
        ("service_catalog", "get"),
        ("tools", "starter"),
        ("usage", "runs"),
    ):
        allowed = await _management_response(
            admin, entry=entry, section=section, action=action
        )
        assert allowed["success"] is True, (section, action, allowed)
        if (section, action) == ("backup", "create"):
            assert "format" in allowed["result"]
            assert "memories" in allowed["result"]

    for section, action in (
        ("overview", "summary"),
        ("guest_mode", "get"),
        ("usage", "summary"),
    ):
        assert inventory["sections"][section]["authorization_classes"][action] == "authenticated_read"
        allowed = await _management_response(
            client, entry=entry, section=section, action=action
        )
        assert allowed["success"] is True, (section, action, allowed)
