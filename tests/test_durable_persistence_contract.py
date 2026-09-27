"""Review guard for every direct HA Store construction and its durability promise."""

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "custom_components" / "extended_openai_conversation_responses"
CONTRACT = ROOT / "ci" / "durable_persistence_contract.json"


def _store_constructions() -> Counter[tuple[str, str]]:
    result: Counter[tuple[str, str]] = Counter()
    for path in INTEGRATION.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Subscript):
                function = function.value
            if isinstance(function, ast.Name) and function.id.endswith("Store"):
                result[(path.name, function.id)] += 1
    return result


def test_every_store_construction_has_one_reviewed_durability_owner() -> None:
    records = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert len(records) == len({record["owner"] for record in records})
    assert {record["owner"] for record in records} >= {
        "persistent_memory",
        "temporary_memory",
        "knowledge",
        "request_rules",
        "conversation_archive",
        "usage_history",
        "guest_mode",
        "delayed_function_tools",
        "model_catalogue",
        "backup_restore_journal",
        "agent_configuration_and_subentries",
    }
    reviewed: Counter[tuple[str, str]] = Counter()
    for record in records:
        assert record["classification"] in {
            "transactional_durable",
            "best_effort_or_coalesced",
            "ha_owned",
            "ephemeral",
        }
        assert record["guarantee"]
        assert record["evidence"]
        for evidence in record["evidence"]:
            path, test_name = evidence.split("::", 1)
            assert f"def {test_name}(" in (ROOT / path).read_text(encoding="utf-8")
        for site in record["stores"]:
            key = site["module"], site["constructor"]
            reviewed[key] += site["count"]
            if record["classification"] == "transactional_durable":
                assert site["constructor"] != "Store" or record["owner"] == (
                    "backup_restore_journal"
                ), record["owner"]
    assert reviewed == _store_constructions()


def test_transactional_store_subclasses_propagate_writer_failures() -> None:
    for module, name in (
        ("memory.py", "MemoryStore"),
        ("temporary_memory.py", "TemporaryMemoryStore"),
        ("knowledge.py", "KnowledgeStore"),
        ("request_rules.py", "RequestRuleStore"),
    ):
        tree = ast.parse((INTEGRATION / module).read_text(encoding="utf-8"))
        declaration = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        )
        assert any(
            isinstance(base, ast.Name) and base.id == "PropagatingWriteStore"
            for base in declaration.bases
        ), f"{module}:{name} must surface HA writer failures"
