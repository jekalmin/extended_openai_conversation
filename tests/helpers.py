"""Test helper functions."""

from pathlib import Path
from typing import Any

import yaml


def load_function_yaml(filename: str) -> list[dict[str, Any]]:
    """Load function definition from yaml file.

    Args:
        filename: Name of yaml file in tests/fixtures/functions/

    Returns:
        List of function definitions with spec and function keys
    """
    fixtures_dir = Path(__file__).parent / "fixtures" / "functions"
    yaml_path = fixtures_dir / filename

    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_function_from_yaml(filename: str, index: int = 0) -> dict[str, Any]:
    """Get a single function definition from yaml file.

    Args:
        filename: Name of yaml file in tests/fixtures/functions/
        index: Index of function in yaml file (default: 0)

    Returns:
        Function definition with spec and function keys
    """
    functions = load_function_yaml(filename)
    return functions[index]
