"""Certify the exact source commit before the release workflow publishes it."""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

REQUIRED_WORKFLOWS = (
    "ci.yml",
    "frontend.yml",
    "cross-browser-smoke.yml",
    "real-ha.yml",
    "release-smoke.yml",
    "enhanced-stress.yml",
)
HEAVY_CAMPAIGNS = (
    "runtime",
    "lifecycle",
    "ai-task",
    "voice-intercom",
    "archive",
    "feature-crossroads",
    "setup",
    "backup",
    "backup-transfer",
    "persistence",
    "request-rules",
    "guest-security",
    "quiet-hours",
    "functions",
    "provider-resilience",
    "memory-knowledge",
    "large-installation",
    "chaos",
    "process-chaos",
)
HEAVY_JOBS = frozenset(
    {"prepare", "Consolidated nightly certification"}
    | {f"{campaign} / heavy" for campaign in HEAVY_CAMPAIGNS}
    | {
        f"{browser} / heavy"
        for browser in ("browser", "browser-firefox", "browser-webkit")
    }
    | {
        f"HA {point} / shared lifecycle contract"
        for point in ("oldest", "stable", "dev")
    }
)


def qualifying_run(run: dict, *, workflow_id: int, source_sha: str) -> bool:
    """Reject ancestors, other workflows, unfinished and non-successful runs."""
    return (
        run.get("workflow_id") == workflow_id
        and run.get("head_sha") == source_sha
        and run.get("event") in {"push", "workflow_dispatch"}
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
    )


def complete_heavy_nightly(run: dict, jobs: list[dict]) -> bool:
    """Require the full heavy campaign, browsers, HA matrix, and final gate."""
    if run.get("event") != "workflow_dispatch":
        return False
    successful = {job.get("name") for job in jobs if job.get("conclusion") == "success"}
    return successful >= HEAVY_JOBS


class GitHubActions:
    """Small, read-only GitHub Actions REST client."""

    def __init__(self, repository: str, token: str) -> None:
        self.repository = repository
        self.token = token

    def get(self, path: str, **query: object) -> dict:
        url = f"https://api.github.com/repos/{self.repository}/{path}"
        if query:
            url += "?" + urlencode(query)
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def workflow_runs(self, filename: str, source_sha: str) -> tuple[int, list[dict]]:
        workflow_id = self.get(f"actions/workflows/{quote(filename)}")["id"]
        runs = []
        for page in range(1, 11):
            batch = self.get(
                f"actions/workflows/{workflow_id}/runs",
                head_sha=source_sha,
                per_page=100,
                page=page,
            ).get("workflow_runs", [])
            runs.extend(batch)
            if len(batch) < 100:
                break
        else:
            raise RuntimeError(
                f"Too many {filename} runs for {source_sha}; cannot certify"
            )
        return workflow_id, runs

    def run_jobs(self, run_id: int) -> list[dict]:
        result = self.get(f"actions/runs/{run_id}/jobs", per_page=100)
        if result.get("total_count", 0) > 100:
            raise RuntimeError(f"Nightly run {run_id} has too many jobs to certify")
        return result.get("jobs", [])


def certify(actions: GitHubActions, source_sha: str) -> dict[str, str]:
    """Return evidence URLs or fail closed with the missing check classes."""
    evidence = {}
    missing = []
    for filename in REQUIRED_WORKFLOWS:
        try:
            workflow_id, runs = actions.workflow_runs(filename, source_sha)
        except Exception as exc:
            missing.append(
                f"{filename}: expected successful workflow run; metadata could not be verified ({exc})"
            )
            continue
        candidates = [
            run
            for run in runs
            if qualifying_run(run, workflow_id=workflow_id, source_sha=source_sha)
        ]
        if filename == "enhanced-stress.yml":
            complete = []
            job_errors = []
            for run in candidates:
                try:
                    if complete_heavy_nightly(run, actions.run_jobs(run["id"])):
                        complete.append(run)
                except Exception as exc:
                    job_errors.append(f"run {run['id']}: {exc}")
            candidates = complete
        if candidates:
            evidence[filename] = candidates[0]["html_url"]
        else:
            detail = (
                "complete heavy Enhanced nightly"
                if filename == "enhanced-stress.yml"
                else "successful workflow run"
            )
            suffix = (
                f"; job metadata errors: {', '.join(job_errors)}"
                if filename == "enhanced-stress.yml" and job_errors
                else ""
            )
            missing.append(f"{filename}: expected {detail} on this exact SHA{suffix}")
    if missing:
        raise RuntimeError(
            f"Release source {source_sha} is not certified:\n- "
            + "\n- ".join(missing)
            + "\nA passing ancestor or another branch's latest run is insufficient."
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("GH_TOKEN or GITHUB_TOKEN is required")
    try:
        evidence = certify(GitHubActions(args.repository, token), args.sha)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Release source {args.sha} has exact-SHA certification:")
    for filename, url in evidence.items():
        print(f"- {filename}: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
