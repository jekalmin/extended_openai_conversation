"""Guard reviewed browser payloads at the HA WebSocket validation boundary."""

from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests_stress" / "management_ws_contract.json"
FRONTEND = ROOT / "tests_browser" / "real-ha-backend.spec.mjs"
MANAGEMENT = (
    ROOT
    / "custom_components"
    / "extended_openai_conversation_responses"
    / "management_ui.py"
)
TRANSFER = (
    ROOT
    / "custom_components"
    / "extended_openai_conversation_responses"
    / "backup_transfer.py"
)
EXPECTED_JOURNEYS = {
    "configuration",
    "configuration_extended",
    "memory",
    "request_rules",
    "rule_pack",
    "functions",
    "knowledge",
    "guest_quiet",
    "guest_operations",
    "backup",
}
CRITICAL_ACTIONS = {
    ("request_rules", "groups"),
    ("request_rules", "settings"),
    ("request_rules", "create"),
    ("request_rules", "update"),
    ("request_rules", "delete"),
    ("request_rules", "rule_pack_export"),
    ("request_rules", "rule_pack_review"),
    ("request_rules", "rule_pack_import"),
    ("configuration", "save"),
    ("tools", "save"),
    ("tools", "save_group"),
    ("tools", "set_enabled"),
    ("memories", "add"),
    ("memories", "update"),
    ("memories", "delete"),
    ("knowledge", "create"),
    ("knowledge", "update"),
    ("knowledge", "delete"),
    ("knowledge", "set_enabled"),
    ("guest_mode", "save_policy"),
    ("quiet_hours", "update"),
}


def test_reviewed_browser_payloads_are_accepted_by_websocket_schemas() -> None:
    """New UI keys cannot quietly miss the manually maintained HA allow-list."""
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    actions = document["actions"]
    assert {item["journey"] for item in actions} == EXPECTED_JOURNEYS
    assert {
        (item.get("section"), item["action"]) for item in actions
    } >= CRITICAL_ACTIONS
    assert len(actions) == len(
        {
            (item["journey"], item.get("section", item.get("type")), item["action"])
            for item in actions
        }
    )

    management = MANAGEMENT.read_text(encoding="utf-8")
    management_schema = management.split("async def websocket_management(", 1)[
        0
    ].rsplit("@websocket_api.websocket_command(", 1)[1]
    management_keys = set(
        re.findall(r'vol\.(?:Optional|Required)\("([^"]+)"', management_schema)
    )
    transfer = TRANSFER.read_text(encoding="utf-8")
    transfer_schema = transfer.split("async def websocket_backup_transfer(", 1)[
        0
    ].rsplit("@websocket_api.websocket_command(", 1)[1]
    transfer_keys = set(
        re.findall(r'vol\.(?:Optional|Required)\("([^"]+)"', transfer_schema)
    )
    assert "groups" in management_keys  # Regression for the original browser rejection.
    for item in actions:
        assert item["keys"] and len(item["keys"]) == len(set(item["keys"])), item
        assert set(item.get("equals", {})) <= set(item["keys"]), item
        assert set(item.get("contains", {})) <= set(item["keys"]), item
        schema_keys = transfer_keys if "type" in item else management_keys
        assert set(item["keys"]) <= schema_keys, item
        assert item.get("min_calls", 1) >= 1, item

    frontend = FRONTEND.read_text(encoding="utf-8")
    assert "expectContractCalls" in frontend
    assert {
        match
        for match in re.findall(r'expectContractCalls\(page, "([^"]+)"\)', frontend)
    } == EXPECTED_JOURNEYS
