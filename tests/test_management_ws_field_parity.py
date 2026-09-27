"""Exact HA WebSocket field and value-shape parity for every management action."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import voluptuous as vol

from custom_components.extended_openai_conversation_responses.backup_transfer import (
    websocket_backup_transfer,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    websocket_management,
)
from tests_stress.test_management_action_inventory import production_actions

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests_stress" / "management_ws_field_contract.json"
INVENTORY = ROOT / "tests_stress" / "management_action_inventory.json"
TRANSFER = ROOT / "custom_components" / "extended_openai_conversation_responses" / "backup_transfer.py"


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"kind": "constant", "value": value}
    for primitive, kind in ((str, "string"), (dict, "object"), (list, "array"), (bool, "boolean"), (int, "integer")):
        if value is primitive:
            return {"kind": kind}
    if isinstance(value, vol.Any):
        forms = {None: "null", str: "string", list: "array", dict: "object"}
        return {"kind": "union", "forms": [forms[item] for item in value.validators]}
    if isinstance(value, vol.In):
        return {"kind": "enum", "values": list(value.container)}
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], vol.In):
        return {"kind": "enum_array", "values": list(value[0].container)}
    if isinstance(value, vol.All):
        return {"kind": "nonnegative_integer"}
    raise AssertionError(f"Unreviewed WebSocket validator: {value!r}")


def _actual_fields(handler: Any) -> dict[str, dict[str, Any]]:
    fields = {}
    for marker, validator in handler._ws_schema.schema.items():
        name = marker.schema if isinstance(marker, vol.Marker) else marker
        fields[name] = {"required": isinstance(marker, vol.Required), **_shape(validator)}
        if isinstance(marker, vol.Marker) and marker.default is not vol.UNDEFINED:
            fields[name]["default"] = marker.default()
    return dict(sorted(fields.items()))


def _values(spec: dict[str, Any]) -> list[Any]:
    kind = spec["kind"]
    if kind == "constant":
        return [spec["value"]]
    if kind == "string":
        return ["", "sample"]
    if kind == "object":
        return [{}, {"sample": 0}]
    if kind == "array":
        return [[], ["sample"]]
    if kind == "boolean":
        return [False, True]
    if kind in {"integer", "nonnegative_integer"}:
        return [0, 1]
    if kind == "enum":
        return list(spec["values"])
    if kind == "enum_array":
        return [[], [spec["values"][0]]]
    if kind == "union":
        return [{"null": None, "string": "", "array": [], "object": {}}[form] for form in spec["forms"]]
    raise AssertionError(kind)


def _transfer_actions() -> set[str]:
    tree = ast.parse(TRANSFER.read_text(encoding="utf-8"))
    command = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "async_backup_transfer_command")
    return {
        comparator.value
        for node in ast.walk(command)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "action"
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)
    }


def test_every_management_action_uses_reviewed_exact_websocket_schema() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    assert contract["management"] == _actual_fields(websocket_management)
    assert contract["backup_transfer"] == _actual_fields(websocket_backup_transfer)
    assert set(contract["backup_transfer_actions"]) == _transfer_actions()
    assert {section: set(item["actions"]) for section, item in inventory["sections"].items()} == production_actions()

    for section, item in inventory["sections"].items():
        for action in item["actions"]:
            base = {"id": 0, "type": websocket_management._ws_command, "section": section, "action": action}
            assert websocket_management._ws_schema(base)["action"] == action
            for field, spec in contract["management"].items():
                if field in base:
                    continue
                for value in _values(spec):
                    assert websocket_management._ws_schema({**base, field: value})[field] == value, (section, action, field, value)

    for action in contract["backup_transfer_actions"]:
        base = {"id": 0, "type": websocket_backup_transfer._ws_command, "entry_id": "entry", "subentry_id": "agent", "action": action}
        assert websocket_backup_transfer._ws_schema(base)["action"] == action
        for field, spec in contract["backup_transfer"].items():
            if field in base:
                continue
            for value in _values(spec):
                assert websocket_backup_transfer._ws_schema({**base, field: value})[field] == value


@pytest.mark.parametrize("handler_name", ["management", "backup_transfer"])
def test_boundary_values_missing_fields_and_unknown_keys(handler_name: str) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    handler = websocket_management if handler_name == "management" else websocket_backup_transfer
    fields = contract[handler_name]
    schema = handler._ws_schema
    base = {name: _values(spec)[0] for name, spec in fields.items() if spec["required"]}
    base["type"] = handler._ws_command
    base["action"] = "get"
    assert schema(base)
    for name, spec in fields.items():
        if spec["required"]:
            with pytest.raises(vol.Invalid):
                schema({key: value for key, value in base.items() if key != name})
        else:
            if "default" in spec:
                assert schema(base)[name] == spec["default"]
            else:
                assert name not in schema(base)
        if spec["kind"] not in {"union"} or "null" not in spec["forms"]:
            with pytest.raises(vol.Invalid):
                schema({**base, name: None})
    with pytest.raises(vol.Invalid):
        schema({**base, "unreviewed_payload_field": 0})
    with pytest.raises(vol.Invalid):
        schema({**base, "id": -1})
