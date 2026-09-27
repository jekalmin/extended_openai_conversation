"""Release selection must be bound to one exact, completely certified SHA."""

import pytest

from ci.release_certification import HEAVY_JOBS, REQUIRED_WORKFLOWS, certify

SOURCE = "b" * 40
PARENT = "a" * 40


class FakeActions:
    def __init__(self):
        self.runs = {}
        self.jobs = {}
        for index, filename in enumerate(REQUIRED_WORKFLOWS, 1):
            self.runs[filename] = [
                {
                    "id": index,
                    "workflow_id": index,
                    "head_sha": SOURCE,
                    "event": "workflow_dispatch"
                    if filename == "enhanced-stress.yml"
                    else "push",
                    "status": "completed",
                    "conclusion": "success",
                    "html_url": f"https://example.test/actions/runs/{index}",
                }
            ]
            self.jobs[index] = [
                {"name": name, "conclusion": "success"} for name in HEAVY_JOBS
            ]

    def workflow_runs(self, filename, source_sha):
        assert source_sha == SOURCE
        return REQUIRED_WORKFLOWS.index(filename) + 1, self.runs[filename]

    def run_jobs(self, run_id):
        return self.jobs[run_id]


def test_exact_sha_with_all_certifications_passes():
    actions = FakeActions()
    assert set(certify(actions, SOURCE)) == set(REQUIRED_WORKFLOWS)


def test_passing_parent_sha_cannot_certify_source():
    actions = FakeActions()
    for runs in actions.runs.values():
        runs[0]["head_sha"] = PARENT
    with pytest.raises(RuntimeError, match=f"Release source {SOURCE} is not certified"):
        certify(actions, SOURCE)


@pytest.mark.parametrize(
    "nightly_change",
    [
        "missing",
        "normal",
        "failed",
        "diagnostics",
        "cancelled_job",
    ],
)
def test_ordinary_ci_does_not_replace_complete_heavy_nightly(nightly_change):
    actions = FakeActions()
    nightly = actions.runs["enhanced-stress.yml"]
    if nightly_change == "missing":
        nightly.clear()
    elif nightly_change == "normal":
        actions.jobs[6] = [
            {"name": name.replace(" / heavy", " / normal"), "conclusion": "success"}
            for name in HEAVY_JOBS
        ]
    elif nightly_change == "failed":
        nightly[0]["conclusion"] = "failure"
    elif nightly_change == "diagnostics":
        actions.jobs[6] = [{"name": "diagnostics / heavy", "conclusion": "success"}]
    else:
        actions.jobs[6][0]["conclusion"] = "cancelled"
    with pytest.raises(RuntimeError, match=r"enhanced-stress\.yml"):
        certify(actions, SOURCE)


def test_unrelated_workflow_and_cancelled_or_skipped_runs_do_not_count():
    actions = FakeActions()
    actions.runs["ci.yml"][0]["workflow_id"] = 999
    actions.runs["ci.yml"].append(
        {**actions.runs["ci.yml"][0], "workflow_id": 1, "conclusion": "cancelled"}
    )
    actions.runs["ci.yml"].append(
        {**actions.runs["ci.yml"][0], "workflow_id": 1, "conclusion": "skipped"}
    )
    with pytest.raises(RuntimeError, match=r"ci\.yml"):
        certify(actions, SOURCE)


def test_any_complete_successful_heavy_run_is_sufficient():
    actions = FakeActions()
    failed = {**actions.runs["enhanced-stress.yml"][0], "conclusion": "failure"}
    actions.runs["enhanced-stress.yml"].insert(0, failed)
    assert "enhanced-stress.yml" in certify(actions, SOURCE)


def test_unavailable_workflow_metadata_fails_with_source_and_expected_check():
    actions = FakeActions()

    def unavailable(filename, source_sha):
        if filename == "frontend.yml":
            raise OSError("Actions API unavailable")
        return FakeActions.workflow_runs(actions, filename, source_sha)

    actions.workflow_runs = unavailable
    with pytest.raises(RuntimeError, match=f"Release source {SOURCE} is not certified") as error:
        certify(actions, SOURCE)
    assert "frontend.yml: expected successful workflow run" in str(error.value)
