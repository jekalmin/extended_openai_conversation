"""Classify PR paths against the management UI's local Python dependency graph.

The browser contract starts at every EOAI WebSocket endpoint used by the panel.
Relative imports are followed transitively, so a new helper imported by a
management handler is covered without editing a workflow path list.
"""

import argparse
import ast
from pathlib import Path

COMPONENT = Path("custom_components/extended_openai_conversation_responses")
ROOTS = {
    "management_ui.py", "backup_transfer.py", "debug_ui.py",
    "provider_credentials.py", "model_catalog_manager.py",
    "intercom_panel.py", "frontend_assets.py",
}
FRONTEND_PREFIXES = (
    "frontend/", "tests_browser/", "tests_stress/frontend_route_inventory.json",
    "playwright", "scripts/generate-management-", "scripts/generate-agent-config-",
)
FRONTEND_EXACT = {
    "requirements_test.txt", "pyproject.toml", ".github/workflows/frontend.yml",
    ".github/workflows/cross-browser-smoke.yml", "ci/classify_frontend_changes.py",
    "ci/browser_fixture_server.py", "ci/test_classify_frontend_changes.py",
    "ci/test_route_acceptance.py", "tests_stress/test_frontend_route_inventory.py",
}


def local_dependencies(path: Path, available: set[Path]) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        parent = path.parent
        for _ in range(node.level - 1):
            parent = parent.parent
        module = Path(*(node.module or "").split(".")) if node.module else Path()
        candidates = [parent / module.with_suffix(".py"), parent / module / "__init__.py"] if node.module else []
        candidates += [parent / module / f"{alias.name}.py" for alias in node.names]
        result.update(candidate for candidate in candidates if candidate in available)
    return result


def management_dependencies(repo: Path) -> set[str]:
    component = repo / COMPONENT
    available = set(component.rglob("*.py"))
    pending = [component / name for name in ROOTS]
    missing = [path for path in pending if path not in available]
    if missing:
        raise RuntimeError(f"Missing management contract roots: {missing}")
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        pending.extend(local_dependencies(path, available) - seen)
    return {path.relative_to(repo).as_posix() for path in seen}


def needs_frontend(path: str, dependencies: set[str]) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in FRONTEND_EXACT or normalized in dependencies:
        return True
    if normalized.startswith("tests_real_ha/test_browser_") or normalized == "tests_real_ha/test_management_backend_acceptance.py":
        return True
    if normalized.startswith("tests/") and normalized.endswith(".test.mjs"):
        return True
    if any(normalized.startswith(prefix) for prefix in FRONTEND_PREFIXES):
        return True
    component_prefix = COMPONENT.as_posix() + "/"
    if normalized.startswith(component_prefix):
        local = normalized[len(component_prefix):]
        if local.startswith("frontend/"):
            return True
        # Management family modules are a declared extension point. This also
        # catches newly added dispatch helpers before they are imported.
        return local == "__init__.py" or local.startswith("management_") or not (Path(normalized).exists())
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("changed_files", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dependencies = management_dependencies(Path.cwd())
    paths = [path.strip() for path in args.changed_files.read_text(encoding="utf-8").splitlines() if path.strip()]
    required = not paths or any(needs_frontend(path, dependencies) for path in paths)
    value = f"frontend={'true' if required else 'false'}\n"
    if args.output:
        with args.output.open("a", encoding="utf-8") as output:
            output.write(value)
    print(value.strip())


if __name__ == "__main__":
    main()
