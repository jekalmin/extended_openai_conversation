"""The saved stress result must agree with pytest's complete phase reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ci.enhanced_evidence import evidence_filename, final_pytest_outcome
from tests_stress import conftest as stress_plugin


def _report(when: str, outcome: str, *, xfail: bool = False) -> SimpleNamespace:
    report = SimpleNamespace(
        when=when,
        outcome=outcome,
        passed=outcome == "passed",
        failed=outcome == "failed",
        skipped=outcome == "skipped",
        longrepr="phase failure",
    )
    if xfail:
        report.wasxfail = "expected failure"
    return report


@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        (["passed", "passed", "passed"], "passed"),
        (["skipped", "passed"], "skipped"),
        (["failed", "passed"], "failed"),
        (["passed", "passed", "failed"], "failed"),
        (["passed", "passed"], "incomplete"),
    ],
)
def test_final_outcome_never_invents_a_success(phases, expected) -> None:
    names = ["setup", "call", "teardown"] if len(phases) == 3 else ["setup", "teardown"]
    if expected == "incomplete":
        names = ["setup", "teardown"]
    reports = {
        name: _report(name, outcome)
        for name, outcome in zip(names, phases, strict=True)
    }
    assert final_pytest_outcome(reports) == expected


def test_expected_failure_and_unexpected_pass_are_distinguished() -> None:
    reports = {
        "setup": _report("setup", "passed"),
        "call": _report("call", "skipped", xfail=True),
        "teardown": _report("teardown", "passed"),
    }
    assert final_pytest_outcome(reports) == "xfailed"
    reports["call"] = _report("call", "passed", xfail=True)
    assert final_pytest_outcome(reports) == "xpassed"
    reports["call"] = _report("call", "failed", xfail=True)
    assert final_pytest_outcome(reports) == "failed"


def test_saved_trace_waits_for_teardown_and_includes_its_failure(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("STRESS_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(stress_plugin, "envelope", lambda **_kwargs: {"schema": "test"})
    item = SimpleNamespace(
        fixturenames=["stress_trace"],
        nodeid="tests_stress/test_campaign.py::test_case",
        _enhanced_trace=[{"operation": "attempted"}],
    )
    for phase, outcome in (
        ("setup", "passed"),
        ("call", "passed"),
        ("teardown", "failed"),
    ):
        hook = stress_plugin.pytest_runtest_makereport(item, None)
        next(hook)
        with pytest.raises(StopIteration):
            hook.send(
                SimpleNamespace(get_result=lambda p=phase, o=outcome: _report(p, o))
            )
        path = tmp_path / evidence_filename(item.nodeid)
        assert path.exists() is (phase == "teardown")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["outcome"] == "failed"
    assert saved["phase_outcomes"]["teardown"] == "failed"
    assert saved["operations"] == [{"operation": "attempted"}]


@pytest.mark.parametrize(
    ("setup", "expected"), [("skipped", "skipped"), ("failed", "failed")]
)
def test_saved_trace_does_not_invent_a_call_report(
    tmp_path, monkeypatch, setup, expected
) -> None:
    monkeypatch.setenv("STRESS_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(stress_plugin, "envelope", lambda **_kwargs: {"schema": "test"})
    item = SimpleNamespace(
        fixturenames=["stress_trace"],
        nodeid=f"tests_stress/test_campaign.py::{expected}",
    )
    for phase, outcome in (("setup", setup), ("teardown", "passed")):
        hook = stress_plugin.pytest_runtest_makereport(item, None)
        next(hook)
        with pytest.raises(StopIteration):
            hook.send(
                SimpleNamespace(get_result=lambda p=phase, o=outcome: _report(p, o))
            )
    saved = json.loads(
        (tmp_path / evidence_filename(item.nodeid)).read_text(encoding="utf-8")
    )
    assert saved["outcome"] == expected
    assert "call" not in saved["phase_outcomes"]


def test_summary_counts_only_passed_scenario_metrics(tmp_path) -> None:
    for name, outcome in (("passed", "passed"), ("failed", "failed")):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "test": f"test_{name}",
                    "outcome": outcome,
                    "operations": [{"operation": "summary", "provider_requests": 2}],
                }
            ),
            encoding="utf-8",
        )
    script = Path(__file__).resolve().parents[1] / "ci" / "enhanced_summary.py"
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "STRESS_CAMPAIGN": "evidence-test"},
    )
    assert result.returncode == 0, result.stderr
    assert "Outcome: **failed** · Attempted layer" in result.stdout
    assert "Outcome: **passed** · Exercised layer" in result.stdout
    certification = json.loads(
        (tmp_path / "certification.json").read_text(encoding="utf-8")
    )
    assert certification["trace_outcomes"] == {"failed": 1, "passed": 1}
    assert certification["measured_totals"] == {"provider_requests": 2}
