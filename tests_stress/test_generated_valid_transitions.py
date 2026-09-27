"""Transition planner obligations and deterministic production validity."""

from __future__ import annotations

import json
from pathlib import Path

from tests_stress.generated_valid_states import DIMENSIONS, _valid
from tests_stress.generated_valid_transitions import (
    JOURNEYS,
    _named_paths,
    generate_transitions,
)


def test_major_feature_contract_has_static_transition_and_journey_evidence() -> None:
    contract = json.loads(
        Path(__file__)
        .with_name("agent_field_contract.json")
        .read_text(encoding="utf-8")
    )
    features = contract["feature_evidence"]
    assert set().union(
        *(set(feature["dimensions"]) for feature in features.values())
    ) == set(DIMENSIONS)
    assert {feature["journey"] for feature in features.values()} <= set(JOURNEYS)
    named = {path.family: path for path in _named_paths()}
    for feature in features.values():
        assert feature["transition"] == "generated_valid_transitions"
        assert feature["dimensions"]
        path = named[JOURNEYS[feature["journey"]]]
        assert any(
            any(a[key] != b[key] for key in feature["dimensions"])
            for a, b in zip(path.states, path.states[1:], strict=False)
        )


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
