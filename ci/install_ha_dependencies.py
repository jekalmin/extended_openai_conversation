#!/usr/bin/env python3
"""Install Python requirements for HA integrations referenced by EOAI."""

from __future__ import annotations

import argparse
from importlib.metadata import requires
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import homeassistant


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the EOAI manifest.json to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    components = Path(next(iter(homeassistant.__path__))) / "components"
    requirements = set(manifest.get("requirements", []))
    pending = [
        *manifest.get("dependencies", []),
        *manifest.get("after_dependencies", []),
    ]
    visited: set[str] = set()

    while pending:
        domain = pending.pop()
        if domain in visited:
            continue
        visited.add(domain)

        manifest_path = components / domain / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"Home Assistant dependency manifest not found: {domain}"
            )

        dependency_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        requirements.update(dependency_manifest.get("requirements", []))
        pending.extend(dependency_manifest.get("dependencies", []))
        pending.extend(dependency_manifest.get("after_dependencies", []))

    print("Resolved HA dependency integrations:", ", ".join(sorted(visited)))
    if not requirements:
        return

    # Preserve core dependencies that the SDK might otherwise upgrade, and its
    # OpenAI integration pin. Keep the existing installer behavior for other
    # explicit integration requirements rather than changing their policy here.
    explicit_names = {
        canonicalize_name(Requirement(value).name) for value in requirements
    }
    constraints = []
    for value in requires("homeassistant") or []:
        requirement = Requirement(value)
        if canonicalize_name(requirement.name) in explicit_names:
            continue
        if requirement.marker is None or requirement.marker.evaluate():
            constraints.append(f"{requirement.name}{requirement.specifier}")
    for value in (
        (components.parent / "package_constraints.txt").read_text().splitlines()
    ):
        value = value.split("#", 1)[0].strip()
        if value and canonicalize_name(Requirement(value).name) == "openai":
            constraints.append(value)
    with TemporaryDirectory() as directory:
        constraint_path = Path(directory) / "ha-core-and-openai.txt"
        constraint_path.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--constraint",
                str(constraint_path),
                *sorted(requirements),
            ]
        )


if __name__ == "__main__":
    main()
