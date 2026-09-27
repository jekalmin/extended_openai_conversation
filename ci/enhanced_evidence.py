"""Small, privacy-preserving envelope for enhanced nightly evidence."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

SCHEMA = "eoai-enhanced-evidence/v1"
PRIVATE_KEYS = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|credential|secret|password|authorization|"
    r"private[_-]?content|knowledge[_-]?content|prompt(?:[_-]?(?:content|text))?|"
    r"user[_-]?text|message[_-]?content|raw[_-]?body|access[_-]?token)$",
    re.I,
)
SECRET_VALUES = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b|Bearer\s+[^\s\"']+", re.I)
CANARIES = (
    "sk-certification-canary-credential-8274",
    "PRIVATE-MEMORY-CANARY-8274",
    "PRIVATE-KNOWLEDGE-CANARY-8274",
    "SENSITIVE-PROMPT-CANARY-8274",
)


def final_pytest_outcome(reports: dict[str, Any]) -> str:
    """Classify a test only after pytest has reported its teardown phase."""
    if not {"setup", "teardown"} <= reports.keys():
        return "incomplete"
    if any(report.failed for report in reports.values()):
        return "failed"
    if any(
        report.skipped and hasattr(report, "wasxfail") for report in reports.values()
    ):
        return "xfailed"
    if any(report.skipped for report in reports.values()):
        return "skipped"
    call = reports.get("call")
    if call is None:
        return "incomplete"
    if hasattr(call, "wasxfail"):
        return "xpassed"
    return "passed" if call.passed else "incomplete"


def evidence_filename(nodeid: str) -> str:
    """Bound fixture filenames while distinguishing every parametrized test."""
    test_name = nodeid.split("[", 1)[0]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", test_name).strip("._-")
    digest = hashlib.sha256(nodeid.encode("utf-8")).hexdigest()[:16]
    return f"{readable[:160]}-{digest}.json"


def redact_text(value: str) -> str:
    """Remove explicit canaries and credential-shaped strings from diagnostics."""
    for canary in CANARIES:
        value = value.replace(canary, "[REDACTED]")
    return SECRET_VALUES.sub("[REDACTED]", value)


def safe(value: Any, key: str = "") -> Any:
    """Retain structural diagnostics while suppressing private payload fields."""
    if PRIVATE_KEYS.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)[:2000]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value))[:2000]


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def envelope(**specific: Any) -> dict[str, Any]:
    """Describe exactly the checkout and environment exercised by one job."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except OSError, subprocess.CalledProcessError:
        sha = os.environ.get("GITHUB_SHA", "unknown")
    manifest = Path(
        "custom_components/extended_openai_conversation_responses/manifest.json"
    )
    target_version = (
        json.loads(manifest.read_text(encoding="utf-8")).get("version")
        if manifest.exists()
        else None
    )
    return {
        "schema": SCHEMA,
        "eoai_sha": sha,
        "campaign": os.environ.get("STRESS_CAMPAIGN", "unknown"),
        "intensity": os.environ.get("STRESS_INTENSITY", "normal"),
        "seed": os.environ.get("STRESS_SEED", "unknown"),
        "ha_point": os.environ.get("HA_POINT"),
        "ha_version": _version("homeassistant"),
        "python_version": sys.version.split()[0],
        "source_version": os.environ.get("EOAI_SOURCE_VERSION"),
        "target_version": os.environ.get("EOAI_TARGET_VERSION") or target_version,
        **specific,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(data), indent=2, default=str) + "\n", encoding="utf-8"
    )


def sanitize_log(source: Path, target: Path) -> None:
    """Never expose raw pytest output to the Actions log or artifact upload."""
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(redact_text(line)[:2000] for line in lines[-600:]) + "\n",
        encoding="utf-8",
    )


def sanitize_directory(root: Path) -> None:
    """Normalise existing browser and Python JSON traces before upload."""
    if not root.exists():
        return
    for path in root.rglob("*.json"):
        write_json(path, json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "sanitize-log":
        sanitize_log(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "sanitize-dir":
        sanitize_directory(Path(sys.argv[2]))
    else:
        raise SystemExit(
            "usage: enhanced_evidence.py sanitize-log SOURCE TARGET | sanitize-dir ROOT"
        )
