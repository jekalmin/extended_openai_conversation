"""Separate collection and reproducible diagnostics for enhanced campaigns."""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import sys
from time import monotonic

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ci.enhanced_evidence import (  # noqa: E402
    envelope,
    evidence_filename,
    final_pytest_outcome,
    safe,
    write_json,
)
from tests_real_ha.conftest import real_ha_prerequisites  # noqa: F401,E402


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if "stress_trace" not in item.fixturenames:
        return
    reports = getattr(item, "_enhanced_reports", {})
    reports[report.when] = report
    item._enhanced_reports = reports
    if report.when != "teardown":
        return
    report_dir = Path(os.environ.get("STRESS_ARTIFACT_DIR", "stress-artifacts"))
    failure = next((part for part in reports.values() if part.failed), None)
    write_json(
        report_dir / evidence_filename(item.nodeid),
        {
            **envelope(seed=getattr(item, "_enhanced_seed", None)),
            "test": item.nodeid,
            "outcome": final_pytest_outcome(reports),
            "phase_outcomes": {phase: part.outcome for phase, part in reports.items()},
            "duration_seconds": round(
                monotonic() - getattr(item, "_enhanced_started", monotonic()), 3
            ),
            "failure": str(failure.longrepr).splitlines()[-1] if failure else None,
            "health": getattr(item, "_enhanced_health", None),
            "operations": getattr(item, "_enhanced_trace", []),
        },
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--stress-seed", default=os.environ.get("STRESS_SEED"))
    parser.addoption(
        "--stress-intensity",
        choices=("normal", "heavy"),
        default=os.environ.get("STRESS_INTENSITY", "normal"),
    )


@pytest.fixture(scope="session")
def stress_seed(pytestconfig: pytest.Config) -> int:
    raw = pytestconfig.getoption("--stress-seed")
    seed = int(raw) if raw else secrets.randbits(32)
    print(
        f"\nENHANCED STRESS SEED={seed} INTENSITY={pytestconfig.getoption('--stress-intensity')}",
        flush=True,
    )
    return seed


@pytest.fixture(scope="session")
def stress_scale(pytestconfig: pytest.Config) -> int:
    return 4 if pytestconfig.getoption("--stress-intensity") == "heavy" else 1


@pytest.fixture
def stress_trace(request: pytest.FixtureRequest, stress_seed: int) -> list[dict]:
    trace: list[dict] = []
    request.node._enhanced_trace = trace
    request.node._enhanced_seed = stress_seed
    request.node._enhanced_started = monotonic()
    yield trace
    hass = request.node.funcargs.get("hass")
    if hass is not None:
        request.node._enhanced_health = {
            "ha_state_count": len(hass.states.async_all()),
            "eoai_manager_counts": {
                str(key): len(value)
                for key, value in hass.data.items()
                if isinstance(key, str)
                and key.startswith("extended_openai_conversation_responses.")
                and isinstance(value, dict)
            },
        }
    print(
        f"STRESS TRACE seed={stress_seed} test={request.node.nodeid} operations={len(trace)}",
        flush=True,
    )


def record(trace: list[dict], operation: str, **details: object) -> None:
    trace.append(safe({"number": len(trace) + 1, "operation": operation, **details}))
