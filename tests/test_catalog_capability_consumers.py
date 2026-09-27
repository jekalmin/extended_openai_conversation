"""Every catalogue capability must have an explicit consumer classification."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "ci" / "catalog_capability_consumers.json"
VALID_CONSUMERS = {"backend", "frontend", "both", "informational"}


def _leaf_paths(value: Any, path: str) -> set[str]:
    if isinstance(value, dict):
        return {
            leaf
            for key, child in value.items()
            for leaf in _leaf_paths(child, f"{path}.{key}")
        }
    if isinstance(value, list) and any(isinstance(child, dict) for child in value):
        return {leaf for child in value for leaf in _leaf_paths(child, f"{path}[]")}
    return {path}


def _supported_tool_condition_dimensions() -> set[str]:
    """Read the validator's literal accepted dimensions, not current JSON examples."""
    source = (
        ROOT
        / "custom_components"
        / "extended_openai_conversation_responses"
        / "model_catalog.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_tool_rule"
    )
    matches = [
        node.right
        for node in ast.walk(validator)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Attribute)
        and node.left.func.attr == "keys"
        and isinstance(node.right, ast.Set)
    ]
    assert len(matches) == 1, "Update the consumer guard for new condition syntax"
    return set(ast.literal_eval(matches[0]))


def test_every_catalogue_field_and_condition_dimension_has_a_consumer() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    fields = registry["fields"]
    observed = _leaf_paths(BUNDLED_CATALOG["defaults"], "metadata")
    for model in BUNDLED_CATALOG["models"]:
        observed |= _leaf_paths(model, "model")
    assert set(fields) == observed, (
        "Catalogue capability fields changed; classify every added or removed "
        "field in ci/catalog_capability_consumers.json",
        sorted(observed - fields.keys()),
        sorted(fields.keys() - observed),
    )
    assert all(value in VALID_CONSUMERS for value in fields.values())
    dimensions = registry["tool_condition_dimensions"]
    assert set(dimensions) == _supported_tool_condition_dimensions(), (
        "Classify new tool-condition dimensions in the consumer registry"
    )
    assert all(value in VALID_CONSUMERS for value in dimensions.values())
