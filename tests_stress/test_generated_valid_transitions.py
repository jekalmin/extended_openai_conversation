"""Transition planner obligations and deterministic production validity."""

from __future__ import annotations

import json

from tests_stress.generated_valid_states import DIMENSIONS, _valid
from tests_stress.generated_valid_transitions import generate_transitions


def test_transition_paths_cover_every_enterable_value_with_valid_aba_returns() -> None:
    suite = generate_transitions(2983703646)
    assert 25 <= len(suite.paths) < 100
    assert len({path.path_id for path in suite.paths}) == len(suite.paths)
    covered = set().union(*(path.obligations for path in suite.paths))
    assert suite.obligations <= covered
    assert suite.obligations == {
        f"enter:{key}:{json.dumps(value, sort_keys=True)}"
        for key, values in DIMENSIONS.items()
        for value in values
    }
    assert {path.family for path in suite.paths[:8]} == {
        "provider-api",
        "model-capability",
        "memory-knowledge",
        "function-loading",
        "guest-security",
        "local-voice",
        "archive-retention",
        "ui-speech",
    }
    for path in suite.paths:
        assert all(_valid(state)[0] for state in path.states), path.path_id
    assert all(path.states[0] == path.states[-1] for path in suite.paths[8:])


def test_transition_seed_reproduces_and_changes_equivalent_paths() -> None:
    first = generate_transitions(2983703646)
    assert first == generate_transitions(2983703646)
    alternative = generate_transitions(2983703647)
    assert first.obligations == alternative.obligations
    assert first.paths != alternative.paths
