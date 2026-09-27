"""Completeness and deterministic covering-set proofs for valid agent states."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom_components.extended_openai_conversation_responses import agent_config, const
from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)
from tests_stress.generated_valid_states import (
    DIMENSIONS,
    TRIPLE_GROUPS,
    generate,
    normalized_state,
    obligations,
)

CONTRACT = Path(__file__).with_name("agent_field_contract.json")


def _check_contract(fields: dict, dimension_values: dict) -> None:
    actual = {
        name
        for name in dir(agent_config)
        if name.startswith("CONF_")
        and getattr(agent_config, name) in agent_config.AGENT_CONFIG_FIELDS
    }
    assert set(fields) == actual
    assert dimension_values == {key: list(values) for key, values in DIMENSIONS.items()}
    for name, classification in fields.items():
        coverage = classification["coverage"]
        if coverage["kind"] == "combinatorial_dimension":
            assert coverage["dimension"] in DIMENSIONS, name
        else:
            assert coverage.get("reason"), name


def test_every_persistent_field_and_enum_is_classified() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _check_contract(contract["fields"], contract["dimension_values"])
    fake = contract["fields"] | {"CONF_UNCLASSIFIED": {}}
    with pytest.raises(AssertionError):
        _check_contract(fake, contract["dimension_values"])
    assert DIMENSIONS["api_mode"] == tuple(
        item["key"] for item in const.API_MODE_OPTIONS
    )
    assert DIMENSIONS["memory_mode"] == tuple(const.MEMORY_MODES)
    assert DIMENSIONS["temporary_memory"] == tuple(const.TEMPORARY_MEMORY_OPTIONS)
    assert DIMENSIONS["conversation_continuity"] == tuple(
        const.CONVERSATION_CONTINUITY_OPTIONS
    )
    assert DIMENSIONS["voice_scope_policy"] == tuple(const.VOICE_POLICIES)
    assert DIMENSIONS["chat_model"] == tuple(sorted(BUNDLED_CATALOG.resolved))


def test_covering_generator_is_valid_complete_bounded_and_reproducible() -> None:
    first = generate(2983703646, heavy=True)
    second = generate(2983703646, heavy=True)
    assert first == second
    assert first.cases
    assert len(first.cases) < 500
    assert first.exploratory_count <= 15
    actual = set()
    for case in first.cases:
        normalized, api = normalized_state(case)
        assert normalized["chat_model"] == case["chat_model"]
        assert api in {"responses", "chat_completions"}
        actual.update(obligations(case))
    assert first.obligations <= actual
    assert all(len(item) == 2 for item in first.obligations if len(item) == 2)
    assert {
        tuple(key for key, _ in item) for item in first.obligations if len(item) == 3
    } == set(TRIPLE_GROUPS)
    with pytest.raises(AssertionError, match="budget"):
        generate(2983703646, heavy=False, budget=1)


def test_seed_changes_equivalent_covering_choice() -> None:
    one = generate(2983703646, heavy=False)
    other = generate(2983703647, heavy=False)
    assert one.obligations == other.obligations
    assert one.cases != other.cases
