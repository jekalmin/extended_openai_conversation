"""Review every literal Management WebSocket section/action before shipping."""

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components/extended_openai_conversation_responses"
INVENTORY = Path(__file__).with_name("management_action_inventory.json")


def _action_operand(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "action"
    if isinstance(node, ast.Subscript):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "action"
    return False


def _actions(function: ast.AsyncFunctionDef) -> set[str]:
    found = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or not _action_operand(node.left):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                found.add(comparator.value)
            elif isinstance(comparator, (ast.Set, ast.Tuple)):
                found.update(
                    value.value for value in comparator.elts
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
    return found


def production_actions() -> dict[str, set[str]]:
    tree = ast.parse((COMPONENT / "management_ui.py").read_text(encoding="utf-8"))
    mapping = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_MANAGEMENT_SECTION_HANDLERS"
    )
    section_dict = mapping.value.args[0]
    assert isinstance(section_dict, ast.Dict)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)}
    sections = {
        key.value: _actions(functions[value.id])
        for key, value in zip(section_dict.keys, section_dict.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Name)
    }
    assert len(sections) == len(section_dict.keys)
    for section, filename, function_name in (
        ("quiet_hours", "management_permissions.py", "async_quiet_hours_command"),
        ("function_repair", "management_function_repair.py", "async_function_repair"),
    ):
        extra = ast.parse((COMPONENT / filename).read_text(encoding="utf-8"))
        function = next(node for node in extra.body if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name)
        sections[section] = _actions(function)
    # These endpoints use request.message["action"] and intentionally have no
    # local action variable. Keep their stable actions explicit.
    assert sections["scopes"] == {"catalog"}
    return sections


def test_management_actions_have_reviewed_evidence() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == 2
    actual = production_actions()
    assert set(inventory["sections"]) == set(actual)
    semantic_classes = {
        "read_only", "durable_mutation", "ephemeral_mutation",
        "validation_or_preview", "destructive_mutation", "transfer/session_operation",
    }
    authorization_classes = {
        "admin_only", "authenticated_user_scoped", "authenticated_read",
        "public_or_not_applicable",
    }
    for section, contract in inventory["sections"].items():
        assert set(contract["actions"]) == actual[section], (
            f"Management {section} actions changed: classify the new action and "
            f"add acceptance evidence in management_action_inventory.json"
        )
        evidence = contract["evidence"]
        assert evidence, section
        for reference in evidence:
            assert (ROOT / reference).is_file(), (section, reference)
        for field, allowed in (
            ("semantic_classes", semantic_classes),
            ("authorization_classes", authorization_classes),
        ):
            assert set(contract[field]) == actual[section], (section, field)
            assert set(contract[field].values()) <= allowed, (section, field)
        assert set(contract["behavioral_evidence"]) == actual[section], section
        for action, reference in contract["behavioral_evidence"].items():
            path, separator, test_name = reference.partition("::")
            assert separator and test_name, reference
            assert test_name.startswith("test_") or path.endswith(".mjs"), reference
            assert path.startswith(("tests/", "tests_real_ha/", "tests_stress/", "tests_browser/")), reference
            assert Path(path).name not in {
                "test_management_action_inventory.py",
                "test_management_ws_contract.py",
                "test_management_ws_field_parity.py",
            }, reference
            source = (ROOT / path).read_text(encoding="utf-8")
            if path.endswith(".py"):
                tree = ast.parse(source)
                tests = [
                    node for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == test_name
                ]
                assert len(tests) == 1, (section, action, reference)
                mentioned = {
                    node.value
                    for node in ast.walk(tests[0])
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                }
                assert action in test_name or action in mentioned, (
                    section, action, reference
                )
                assert any(
                    isinstance(node, ast.Assert)
                    or (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and (
                            node.func.attr == "raises"
                            or node.func.attr.startswith("assert_")
                        )
                    )
                    for node in ast.walk(tests[0])
                ), (section, action, reference)
            else:
                assert re.search(r'test\(["\']' + re.escape(test_name) + r'["\']', source), reference
            # A new CRUD verb cannot be waved through as a read-only action.
            mutating_name = action.startswith((
                "save", "update", "delete", "clear", "restore", "import",
                "duplicate", "set_enabled", "reassign",
            )) or action in {"add", "create", "disable", "end_active", "ha_add", "move", "settings", "groups", "wording_groups"}
            if mutating_name and (section, action) != ("backup", "create"):
                assert contract["semantic_classes"][action] != "read_only", (section, action)


def test_dummy_management_action_requires_review() -> None:
    actual = production_actions()
    assert "nightly_dummy" not in actual["configuration"]
    assert actual["configuration"] | {"nightly_dummy"} != set(
        json.loads(INVENTORY.read_text(encoding="utf-8"))["sections"]["configuration"]["actions"]
    )
