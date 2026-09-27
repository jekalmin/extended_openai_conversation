"""Broad production bundle ceilings for major size regressions."""

from __future__ import annotations

import json
from pathlib import Path

DIST = Path("custom_components/extended_openai_conversation_responses/frontend/dist")
# 2026-09-27 baseline: 675,482 JS bytes, 48,540 CSS bytes, 91,651 entry bytes.
# These ceilings allow roughly 25-35% growth before flagging a review.
CEILINGS = {"javascript": 850_000, "css": 65_000, "management_entry": 120_000}


def measure(dist: Path) -> dict[str, int]:
    manifest = json.loads((dist / "manifest.json").read_text(encoding="utf-8"))
    entries = [
        item
        for item in manifest.values()
        if item.get("isEntry") and item.get("name") == "management"
    ]
    if len(entries) != 1:
        raise ValueError("Expected exactly one management entry in the Vite manifest")
    assets = list((dist / "assets").iterdir())
    return {
        "javascript": sum(
            path.stat().st_size for path in assets if path.suffix == ".js"
        ),
        "css": sum(path.stat().st_size for path in assets if path.suffix == ".css"),
        "management_entry": (dist / entries[0]["file"]).stat().st_size,
    }


def main() -> None:
    results = measure(DIST)
    failures = []
    for name, size in results.items():
        ceiling = CEILINGS[name]
        print(f"{name}: {size:,} bytes; ceiling {ceiling:,} bytes")
        if size > ceiling:
            failures.append(
                f"{name} exceeds its production bundle ceiling by {size - ceiling:,} bytes"
            )
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
