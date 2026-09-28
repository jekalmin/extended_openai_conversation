"""Guard reviewed browser payloads at the HA WebSocket validation boundary."""

from __future__ import annotations

import json
from pathlib import Path
import re

from tests_stress.test_management_action_inventory import production_actions

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests_stress" / "management_ws_contract.json"
FRONTEND = ROOT / "tests_browser" / "real-ha-backend.spec.mjs"
SEEDED_FRONTEND = ROOT / "tests_browser" / "real-ha-seeded-management.spec.mjs"
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
    "seeded_owners",
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

FRONTEND_DIR = ROOT / "custom_components/extended_openai_conversation_responses/frontend"
# Reviewed literal calls in the shipped source. A new mutation must change this
# snapshot and receive genuine-HA browser evidence before the matrix is complete.
REVIEWED_FRONTEND_MUTATIONS = {
    ("configuration", "duplicate"), ("configuration", "import"),
    ("configuration", "save"), ("conversations", "delete"),
    ("conversations", "end_active"), ("function_repair", "delete_one"),
    ("function_repair", "save"), ("function_repair", "save_one"),
    ("guest_mode", "disable"), ("guest_mode", "update"),
    ("knowledge", "delete"), ("knowledge", "set_enabled"),
    ("memories", "delete"), ("memories", "reassign_legacy"),
    ("memories", "temporary_clear"), ("memories", "temporary_delete"),
    ("memories", "temporary_update"), ("request_rules", "delete"),
    ("request_rules", "duplicate"), ("request_rules", "groups"),
    ("request_rules", "move"), ("request_rules", "rule_pack_import"),
    ("request_rules", "settings"), ("request_rules", "update"),
    ("tools", "delete"), ("tools", "delete_group"),
    ("tools", "ha_add"), ("tools", "save"),
    ("tools", "save_group"), ("tools", "set_enabled"),
    ("usage", "clear_details"),
}
OPEN_FRONTEND_CONTRACT_GAPS = {
    ("function_repair", "delete_one"), ("function_repair", "save"),
    ("function_repair", "save_one"),
    ("tools", "ha_add"),
    ("usage", "clear_details"),
}


def test_shipped_frontend_mutation_actions_are_reviewed() -> None:
    """A newly wired literal mutation must not escape nightly contract review."""
    production = production_actions()
    inventory = json.loads(
        (ROOT / "tests_stress/management_action_inventory.json").read_text(encoding="utf-8")
    )
    mutations = {
        (section, action)
        for section, contract in inventory["sections"].items()
        for action, semantic in contract["semantic_classes"].items()
        if semantic in {
            "durable_mutation", "ephemeral_mutation", "destructive_mutation",
            "transfer/session_operation",
        }
    }
    assert all(action in production[section] for section, action in mutations)
    actual = set()
    for source in FRONTEND_DIR.glob("*.js"):
        actual.update(
            (section, action)
            for section, action in re.findall(
                r'_call\(\s*["\'`]([^"\'`]+)["\'`]\s*,\s*["\'`]([^"\'`]+)["\'`]',
                source.read_text(encoding="utf-8"),
            )
            if (section, action) in mutations
        )
    assert actual == REVIEWED_FRONTEND_MUTATIONS
    covered = {
        (item.get("section"), item["action"])
        for item in json.loads(CONTRACT.read_text(encoding="utf-8"))["actions"]
    }
    assert REVIEWED_FRONTEND_MUTATIONS - covered == OPEN_FRONTEND_CONTRACT_GAPS


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

    frontend = FRONTEND.read_text(encoding="utf-8") + SEEDED_FRONTEND.read_text(encoding="utf-8")
    assert "expectContractCalls" in frontend
    assert {
        match
        for match in re.findall(r'expectContractCalls\(page, "([^"]+)"\)', frontend)
    } == EXPECTED_JOURNEYS
