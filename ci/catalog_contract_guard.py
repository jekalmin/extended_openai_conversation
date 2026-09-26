"""Check a semantic catalogue-contract fingerprint without importing Home Assistant.

The fingerprint records schema-defining key shapes and enum literals from the
validator. Formatting and unrelated code changes do not affect it. A new
contract requires a new schema version and a corresponding supported validator.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "extended_openai_conversation_responses"
CATALOG = COMPONENT / "model_catalog.json"
SOURCE = COMPONENT / "model_catalog.py"
FINGERPRINTS = ROOT / "ci" / "catalog_schema_contracts.json"
CONTRACT_FUNCTIONS = {
    "_validate_sampling",
    "_validate_tool_rule",
    "_validate_metadata",
    "validate_catalog",
}
CONTRACT_CONSTANTS = {
    "_EFFORTS",
    "_SUPPORT",
    "_SEND_POLICIES",
    "_STATUSES",
    "_APIS",
    "_METADATA_REQUIRED",
    "_MODEL_WRAPPER_KEYS",
    "_METADATA_OPTIONAL",
}


def _canonical(node: ast.AST) -> object:
    """Normalize schema literals; ignore positions, order of sets and style."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        values = [_canonical(item) for item in node.elts]
        if isinstance(node, ast.Set):
            values.sort(key=lambda value: json.dumps(value, sort_keys=True))
        return values
    if isinstance(node, ast.Dict):
        return sorted(
            [
                (_canonical(key), _canonical(value))
                for key, value in zip(node.keys, node.values, strict=True)
            ],
            key=lambda item: json.dumps(item[0], sort_keys=True),
        )
    if isinstance(node, ast.Name):
        return {"name": node.id}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return {"union": [_canonical(node.left), _canonical(node.right)]}
    return {"expression": ast.dump(node, include_attributes=False)}


def contract() -> dict[str, object]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    functions: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in CONTRACT_CONSTANTS:
                    constants[target.id] = _canonical(node.value)
        if isinstance(node, ast.FunctionDef) and node.name in CONTRACT_FUNCTIONS:
            key_shapes = []
            enum_literals = []
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "_keys"
                ):
                    key_shapes.append([_canonical(arg) for arg in child.args[1:]])
                if isinstance(child, ast.Compare) and any(
                    isinstance(op, (ast.In, ast.NotIn)) for op in child.ops
                ):
                    for comparator in child.comparators:
                        if isinstance(comparator, (ast.Set, ast.Dict)):
                            enum_literals.append(_canonical(comparator))
                if isinstance(child, ast.Assign) and isinstance(child.value, ast.Set):
                    enum_literals.append(_canonical(child.value))
            functions[node.name] = {
                "key_shapes": sorted(
                    key_shapes, key=lambda item: json.dumps(item, sort_keys=True)
                ),
                "enums": sorted(
                    enum_literals, key=lambda item: json.dumps(item, sort_keys=True)
                ),
            }
    if set(constants) != CONTRACT_CONSTANTS or set(functions) != CONTRACT_FUNCTIONS:
        raise SystemExit(
            "Catalogue contract guard cannot find all validator declarations"
        )
    return {"constants": constants, "functions": functions}


def main() -> None:
    catalogue = json.loads(CATALOG.read_text(encoding="utf-8"))
    schema = catalogue["schema_version"]
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    declarations = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "CURRENT_SCHEMA_VERSION",
                    "SUPPORTED_SCHEMA_VERSIONS",
                }:
                    if (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "frozenset"
                    ):
                        declarations[target.id] = frozenset(
                            ast.literal_eval(node.value.args[0])
                        )
                    else:
                        declarations[target.id] = ast.literal_eval(node.value)
    if schema != declarations.get(
        "CURRENT_SCHEMA_VERSION"
    ) or schema not in declarations.get("SUPPORTED_SCHEMA_VERSIONS", ()):
        raise SystemExit(
            f"Catalogue schema {schema} has no declared current validator support"
        )
    digest = hashlib.sha256(
        json.dumps(contract(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registered = json.loads(FINGERPRINTS.read_text(encoding="utf-8"))
    if registered.get(str(schema)) != digest:
        raise SystemExit(
            f"Catalogue schema contract changed without a schema_version bump (schema {schema}). "
            "Increment schema_version, implement consumer support, and register its new contract fingerprint."
        )
    base_sha = os.environ.get("CATALOG_BASE_SHA", "")
    if base_sha:
        if not re.fullmatch(r"[0-9a-f]{40}", base_sha):
            raise SystemExit("Invalid catalogue guard base SHA")

        def previous(path: str) -> bytes | None:
            result = subprocess.run(
                ["git", "show", f"{base_sha}:{path}"],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
            return result.stdout if result.returncode == 0 else None

        old_catalogue = previous(
            "custom_components/extended_openai_conversation_responses/model_catalog.json"
        )
        old_registry = previous("ci/catalog_schema_contracts.json")
        if old_catalogue is None:
            raise SystemExit("Cannot inspect base catalogue for schema guard")
        old_schema = json.loads(old_catalogue)["schema_version"]
        if old_schema == schema and old_registry is not None:
            old_digest = json.loads(old_registry).get(str(schema))
            if old_digest != digest:
                raise SystemExit(
                    f"Catalogue schema contract changed without a schema_version bump "
                    f"relative to the PR base (schema {schema})."
                )
    print(f"Catalogue schema {schema} contract: {digest}")


if __name__ == "__main__":
    main()
