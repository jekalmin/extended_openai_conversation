"""Completeness and deterministic covering-set proofs for valid agent states."""

from __future__ import annotations

from hashlib import sha256
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
REVIEWED_CHOICES = (
    "SERVICE_TIER_OPTIONS",
    "MEMORY_RETRIEVAL_MODES",
    "SHARED_MEMORY_MODES",
    "GUEST_ACCESS_POLICIES",
    "GUEST_SHARED_MEMORY_POLICIES",
    "WEB_SEARCH_CONTEXT_OPTIONS",
    "FUNCTION_GROUP_LOADING_MODES",
    "CONTINUE_CONVERSATION_OPTIONS",
    "ARCHIVE_RETENTION_OPTIONS",
    "USAGE_RETENTION_OPTIONS",
    "CONVERSATION_TIMEOUT_OPTIONS",
)
CAPABILITY_KEYS = (
    "api",
    "reasoning",
    "temperature",
    "top_p",
    "service_tiers",
    "streaming",
    "tools",
    "responses_web_search",
    "recommended_profile",
    "output_tokens",
)


def _catalogue_digest() -> str:
    reviewed = {
        model: {key: metadata[key] for key in CAPABILITY_KEYS}
        for model, metadata in sorted(BUNDLED_CATALOG.resolved.items())
    }
    return sha256(json.dumps(reviewed, sort_keys=True).encode()).hexdigest()


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
    by_value = {getattr(agent_config, name): name for name in fields}
    for dimension in DIMENSIONS:
        if dimension in agent_config.AGENT_CONFIG_FIELDS:
            coverage = fields[by_value[dimension]]["coverage"]
            assert coverage == {
                "kind": "combinatorial_dimension",
                "dimension": dimension,
            }
    for field, dimension in {
        "reasoning_effort": "reasoning_profile",
        "functions": "function_tools",
        "temperature": "sampling",
        "top_p": "sampling",
        "archive_enabled": "archive",
        "shared_archive_enabled": "archive",
    }.items():
        assert fields[by_value[field]]["coverage"] == {
            "kind": "combinatorial_dimension",
            "dimension": dimension,
        }


def test_every_persistent_field_and_enum_is_classified() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    _check_contract(contract["fields"], contract["dimension_values"])
    fake = contract["fields"] | {"CONF_UNCLASSIFIED": {}}
    with pytest.raises(AssertionError):
        _check_contract(fake, contract["dimension_values"])
    assert set(DIMENSIONS["api_mode"]) == {
        item["key"] for item in const.API_MODE_OPTIONS
    }
    assert set(DIMENSIONS["memory_mode"]) == set(const.MEMORY_MODES)
    assert set(DIMENSIONS["temporary_memory"]) == set(const.TEMPORARY_MEMORY_OPTIONS)
    assert set(DIMENSIONS["conversation_continuity"]) == set(
        const.CONVERSATION_CONTINUITY_OPTIONS
    )
    assert set(DIMENSIONS["voice_scope_policy"]) == set(const.VOICE_POLICIES)
    assert set(DIMENSIONS["chat_model"]) == set(BUNDLED_CATALOG.resolved)
    assert contract["reviewed_choice_values"] == {
        name: list(getattr(const, name)) for name in REVIEWED_CHOICES
    }
    assert contract["catalogue_capability_digest"] == _catalogue_digest()


def test_covering_generator_is_valid_complete_bounded_and_reproducible() -> None:
    first = generate(2983703646, heavy=True)
    second = generate(2983703646, heavy=True)
    assert first == second
    assert first.cases
    assert len(first.cases) < 500
    assert first.exploratory_count <= 15
    actual = set()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    participating = {
        getattr(agent_config, name): set()
        for name, item in contract["fields"].items()
        if item["coverage"]["kind"] == "combinatorial_dimension"
    }
    for case in first.cases:
        normalized, api = normalized_state(case)
        assert normalized["chat_model"] == case["chat_model"]
        assert api in {"responses", "chat_completions"}
        actual.update(obligations(case))
        for field, seen in participating.items():
            seen.add(json.dumps(normalized.get(field), sort_keys=True))
    assert first.obligations <= actual
    assert all(len(seen) > 1 for seen in participating.values()), {
        field: len(seen) for field, seen in participating.items() if len(seen) <= 1
    }
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
