"""Turn deterministic enhanced-test traces into a useful GitHub Step Summary."""

from collections import Counter
import json
import os
from pathlib import Path
import sys

from enhanced_evidence import envelope, safe, write_json

COUNT_METRICS = {
    "ai_task_turns",
    "ai_task_concurrent",
    "ai_task_agents",
    "ai_task_provider_failures",
    "voice_household_turns",
    "voice_household_users",
    "voice_household_satellites",
    "voice_mapping_changes",
    "voice_user_deletions",
    "voice_private_context_probes",
    "voice_temporary_context_probes",
    "voice_guest_context_probes",
    "intercom_broadcasts",
    "intercom_satellites",
    "intercom_deliveries",
    "intercom_failures",
    "intercom_expired",
    "archive_turns_written",
    "archive_scopes",
    "archive_privacy_transitions",
    "archive_searches",
    "local_intent_turns",
    "local_intent_provider_fallbacks",
    "local_intent_policy_reloads",
    "skill_lifecycle_operations",
    "skill_publishes",
    "skill_removals",
    "skill_scans",
    "skill_blocked_mutations",
    "public_turns",
    "provider_requests",
    "embedding_provider_requests",
    "actual_tool_executions",
    "actual_function_executions",
    "local_function_executions",
    "native_function_executions",
    "template_function_executions",
    "script_function_executions",
    "rest_function_executions",
    "scrape_function_executions",
    "composite_function_executions",
    "sqlite_function_executions",
    "bash_function_executions",
    "read_file_function_executions",
    "write_file_function_executions",
    "edit_file_function_executions",
    "provider_wire_function_errors",
    "delayed_tool_mutation_cases",
    "delayed_tool_due_executions",
    "ha_service_calls",
    "guest_end_to_end_combinations",
    "private_context_probes",
    "rollback_phases",
    "backup_chunks_transferred",
    "transfer_sessions",
    "concurrent_import_sessions",
    "transfer_previews",
    "stale_apply_rejections",
    "expired_import_sessions",
    "cancelled_import_sessions",
    "historical_fixtures",
    "browser_creates",
    "browser_edits",
    "browser_deletes",
    "multi_tab_conflicts",
    "stale_responses",
    "reconnect_cycles",
    "reconnect_mutation_cycles",
    "accessibility_layout_pages",
    "quiet_hours_transitions",
    "quiet_ownership_cases",
    "quiet_heterogeneous_devices",
    "quiet_active_policy_mutations",
    "quiet_transient_service_failures",
    "quiet_time_boundary_cases",
    "quiet_dst_cases",
    "chaos_operations",
    "process_terminations",
    "setup_flows",
}


def layer_for(test: str, operations: list[dict]) -> str:
    """Classify evidence by its deepest exercised boundary, not by test count."""
    explicit = next(
        (item.get("layer") for item in operations if item.get("layer")), None
    )
    if explicit:
        return str(explicit)
    if "browser" in test or test.endswith(".stress.mjs"):
        return "browser"
    if "function_groups_state_machine" in test or "request_rules_matrix" in test:
        return "model-level"
    return "real-ha"


def main() -> None:
    folder = Path(sys.argv[1])
    files = sorted(folder.glob("*.json")) if folder.exists() else []
    campaign = os.environ.get("STRESS_CAMPAIGN", "unknown")
    seed = os.environ.get("STRESS_SEED", "unknown")
    intensity = os.environ.get("STRESS_INTENSITY", "normal")
    status = os.environ.get("ENHANCED_JOB_STATUS", "unknown")
    metadata = envelope(status=status)
    lines = [
        f"### Enhanced acceptance: {campaign}",
        "",
        f"SHA: `{metadata['eoai_sha']}` · Seed: `{seed}` · Intensity: `{intensity}` · Status: **{status}**",
        f"HA: `{metadata['ha_version'] or 'not applicable'}` · Python: `{metadata['python_version']}`",
        "",
    ]
    totals: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    if not files:
        lines += [
            "Selected Real HA tests report their assertions in the pytest log; no enhanced operation trace was produced.",
            "",
        ]
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("test") and not path.stem.startswith("browser-"):
            continue
        operations = safe(data.get("operations", []))
        outcome = data.get("outcome", "unreported")
        outcomes[outcome] += 1
        counts = Counter(item.get("operation", "unknown") for item in operations)
        lines += [
            f"**{data.get('test', path.stem)}**",
            "",
            f"Outcome: **{outcome}** · {'Exercised' if outcome == 'passed' else 'Attempted'} layer: **{'browser' if path.stem.startswith('browser-') else layer_for(data.get('test', path.stem), operations)}** · Trace events: {len(operations)}",
            "",
        ]
        if counts:
            lines += ["| Operation | Count |", "| --- | ---: |"]
            lines += [f"| {name} | {count} |" for name, count in sorted(counts.items())]
            lines.append("")
        for item in operations:
            if item.get("operation") == "summary" and outcome == "passed":
                for key in COUNT_METRICS:
                    value = item.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        totals[key] += value
                details = ", ".join(
                    f"{key}={safe(value, key)}"
                    for key, value in item.items()
                    if key not in {"operation", "number"}
                )
                lines += [f"Measured: {details}", ""]
        if "maxNodes" in data:
            totals["browser_creates"] += int(data.get("creates", 0))
            totals["browser_edits"] += int(data.get("edits", 0))
            totals["browser_deletes"] += int(data.get("deletes", 0))
            lines += [
                f"Browser transitions: {data['count']}; creates: {data.get('creates', 0)}; edits: {data.get('edits', 0)}; deletes: {data.get('deletes', 0)}; maximum observed panel DOM nodes: {data['maxNodes']}",
                "",
            ]
        if "cycles" in data:
            totals["reconnect_cycles"] += int(data["cycles"])
            totals["reconnect_mutation_cycles"] += int(data.get("mutationCycles", 0))
            lines += [
                f"Reconnect cycles: {data['cycles']}; mutation cycles: {data.get('mutationCycles', 0)}; baseline backend calls per forced read: {data.get('baselineCalls')}",
                "",
            ]
        if "staleResponses" in data:
            totals["stale_responses"] += int(data["staleResponses"])
            lines += [
                f"Injected stale responses by surface: {data.get('staleBySurface', {})}",
                "",
            ]
        if "accessibilityLayoutPages" in data:
            totals["accessibility_layout_pages"] += int(
                data["accessibilityLayoutPages"]
            )
            lines += [
                f"Accessibility/layout page-width combinations checked: {data['accessibilityLayoutPages']}",
                "",
            ]
    if totals:
        lines += ["**Measured totals**", "", "| Metric | Count |", "| --- | ---: |"]
        lines += [f"| {key} | {value} |" for key, value in sorted(totals.items())]
        lines.append("")
    if outcomes:
        lines += ["**Trace outcomes**", "", "| Outcome | Count |", "| --- | ---: |"]
        lines += [f"| {name} | {value} |" for name, value in sorted(outcomes.items())]
        lines.append("")
    summary = "\n".join(lines)
    write_json(
        folder / "certification.json",
        {
            **metadata,
            "tests_traced": sum(
                1 for path in files if path.name != "certification.json"
            ),
            "measured_totals": dict(totals),
            "trace_outcomes": dict(outcomes),
            "artifact_files": [
                path.name for path in files if path.name != "certification.json"
            ],
        },
    )
    print(summary)
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(summary + "\n")


if __name__ == "__main__":
    main()
