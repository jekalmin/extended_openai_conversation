"""Guard the frontend interaction coverage inventory.

The route inventory answers "does this page have a browser journey?".  This
inventory answers the finer-grained question: "which user-facing interactions
on that page have unit, browser, and nightly evidence, and which gaps are
deliberately scheduled for follow-up work?"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS = ROOT / "tests_stress" / "frontend_interaction_inventory.json"
ROUTES = ROOT / "tests_stress" / "frontend_route_inventory.json"
MANAGEMENT_ACTIONS = ROOT / "tests_stress" / "management_action_inventory.json"
STRESS_CONFIG = ROOT / "playwright.stress.config.mjs"
BASE_PLAYWRIGHT_CONFIG = ROOT / "playwright.config.mjs"
STRESS_WORKFLOW = ROOT / ".github" / "workflows" / "enhanced-stress.yml"

COMPLETED_FRONTEND_ROADMAP_PRS = {6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17}

TIERS = {"unit", "browser", "nightly"}
STATUSES = {"covered", "planned", "exempt"}
RISKS = {"low", "medium", "high"}
KINDS = {
    "action",
    "collection",
    "concurrency",
    "destructive_action",
    "dialog",
    "dynamic_dialog",
    "dynamic_form",
    "error_recovery",
    "form",
    "form_action",
    "navigation",
    "runtime_journey",
    "toggle",
    "transfer",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_path(value: str) -> Path:
    # Evidence may eventually name a test node as "path::node".  The inventory
    # currently stores paths only, but accepting node-qualified evidence keeps
    # the schema useful as coverage becomes more precise.
    return ROOT / value.split("::", 1)[0]


def _management_sections(payload: dict) -> dict:
    return payload.get("sections", payload)


def test_frontend_interaction_inventory_matches_shipped_routes() -> None:
    interactions = _load(INTERACTIONS)
    routes = _load(ROUTES)

    assert interactions["schema_version"] == 1
    assert set(interactions["coverage_tiers"]) == TIERS
    assert set(interactions["routes"]) == set(routes["routes"])


def test_every_frontend_interaction_has_explicit_coverage_state() -> None:
    payload = _load(INTERACTIONS)
    seen: set[tuple[str, str]] = set()

    for route, route_data in payload["routes"].items():
        entries = route_data.get("interactions")
        assert isinstance(entries, list) and entries, f"{route} has no classified interactions"

        for interaction in entries:
            identifier = interaction.get("id")
            assert isinstance(identifier, str) and identifier
            key = (route, identifier)
            assert key not in seen, f"duplicate frontend interaction {route}:{identifier}"
            seen.add(key)

            assert interaction.get("kind") in KINDS, key
            assert interaction.get("risk") in RISKS, key
            assert isinstance(interaction.get("source_markers"), list) and interaction["source_markers"], key
            assert all(isinstance(marker, str) and marker for marker in interaction["source_markers"]), key
            assert isinstance(interaction.get("backend_actions"), list), key

            coverage = interaction.get("coverage")
            assert isinstance(coverage, dict), key
            assert set(coverage) == TIERS, f"{key} must classify unit, browser and nightly coverage"

            for tier, classification in coverage.items():
                status = classification.get("status")
                assert status in STATUSES, f"{key}:{tier} has invalid status {status!r}"
                if status == "covered":
                    evidence = classification.get("evidence")
                    assert isinstance(evidence, list) and evidence, f"{key}:{tier} needs evidence"
                    missing = [item for item in evidence if not _evidence_path(item).is_file()]
                    assert not missing, f"{key}:{tier} references missing evidence: {missing}"
                    assert "roadmap_pr" not in classification, f"{key}:{tier} is covered but still scheduled"
                    assert "reason" not in classification, f"{key}:{tier} is covered but also exempt"
                elif status == "planned":
                    pr = classification.get("roadmap_pr")
                    assert isinstance(pr, int) and 2 <= pr <= 19, (
                        f"{key}:{tier} planned coverage must point to a later roadmap PR"
                    )
                    assert "evidence" not in classification, f"{key}:{tier} planned coverage cannot claim evidence"
                    assert "reason" not in classification, f"{key}:{tier} planned coverage cannot be exempt"
                else:
                    reason = classification.get("reason")
                    assert isinstance(reason, str) and reason.strip(), f"{key}:{tier} exemption needs a reason"
                    assert "evidence" not in classification, f"{key}:{tier} exempt coverage cannot claim evidence"
                    assert "roadmap_pr" not in classification, f"{key}:{tier} exempt coverage cannot be scheduled"


def test_high_risk_interactions_are_never_browser_exempt() -> None:
    payload = _load(INTERACTIONS)
    for route, route_data in payload["routes"].items():
        for interaction in route_data["interactions"]:
            if interaction["risk"] != "high":
                continue
            browser = interaction["coverage"]["browser"]
            if browser["status"] == "exempt":
                reason = browser.get("reason", "").lower()
                nightly = interaction["coverage"]["nightly"]
                assert "nightly-only" in reason and nightly["status"] == "covered", (
                    f"{route}:{interaction['id']} high-risk browser exemption must explain the nightly-only scope"
                )


def test_declared_backend_actions_exist_in_management_inventory() -> None:
    payload = _load(INTERACTIONS)
    management = _management_sections(_load(MANAGEMENT_ACTIONS))

    for route, route_data in payload["routes"].items():
        for interaction in route_data["interactions"]:
            for reference in interaction["backend_actions"]:
                assert "/" in reference, f"{route}:{interaction['id']} has malformed backend action {reference!r}"
                section, action = reference.split("/", 1)
                assert section in management, (
                    f"{route}:{interaction['id']} references unknown management section {section!r}"
                )
                actions = set(management[section].get("actions", []))
                assert action in actions, (
                    f"{route}:{interaction['id']} references unknown management action "
                    f"{section}/{action}; update the interaction inventory when the frontend contract changes"
                )


def test_covered_browser_and_nightly_evidence_uses_appropriate_test_surfaces() -> None:
    payload = _load(INTERACTIONS)
    for route, route_data in payload["routes"].items():
        for interaction in route_data["interactions"]:
            key = f"{route}:{interaction['id']}"
            for tier in ("browser", "nightly"):
                classification = interaction["coverage"][tier]
                if classification["status"] != "covered":
                    continue
                for item in classification["evidence"]:
                    path = item.split("::", 1)[0]
                    assert path.startswith("tests_browser/"), (
                        f"{key}:{tier} evidence should exercise the real frontend through Playwright: {path}"
                    )
                    if tier == "nightly":
                        assert path.endswith(".stress.mjs") or path == "tests_browser/nightly-diagnostics-probe.mjs", (
                            f"{key}:nightly evidence is not in the nightly stress collection: {path}"
                        )


def test_completed_frontend_roadmap_entries_are_not_left_planned() -> None:
    payload = _load(INTERACTIONS)
    for route, route_data in payload["routes"].items():
        for interaction in route_data["interactions"]:
            for tier, coverage in interaction["coverage"].items():
                if coverage.get("status") == "planned":
                    assert coverage.get("roadmap_pr") not in COMPLETED_FRONTEND_ROADMAP_PRS, (
                        f"{route}:{interaction['id']}:{tier} still points at completed roadmap PR "
                        f"{coverage.get('roadmap_pr')}"
                    )


def test_enhanced_nightly_workflow_discovers_every_stress_browser_suite() -> None:
    config = STRESS_CONFIG.read_text(encoding="utf-8")
    base_config = BASE_PLAYWRIGHT_CONFIG.read_text(encoding="utf-8")
    workflow = STRESS_WORKFLOW.read_text(encoding="utf-8")
    assert 'testMatch: /.*\\.stress\\.mjs$/' in config
    assert re.search(r'testDir:\s*["\']\.?/?tests_browser["\']', base_config)
    assert "npx playwright test --config=playwright.stress.config.mjs" in workflow
    suites = sorted((ROOT / "tests_browser").glob("*.stress.mjs"))
    assert suites, "nightly Playwright stress discovery must contain frontend suites"
    assert all(path.is_file() for path in suites)
