"""Seeded valid configuration paths built from the PR F covering states."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Any

from tests_stress.generated_valid_states import DIMENSIONS, _valid, generate


def fingerprint(state: dict[str, Any]) -> str:
    return sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()[:12]


def entered(source: dict[str, Any], target: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        f"enter:{key}:{json.dumps(target[key], sort_keys=True)}"
        for key in DIMENSIONS
        if source[key] != target[key]
    )


@dataclass(frozen=True)
class TransitionPath:
    path_id: str
    family: str
    states: tuple[dict[str, Any], ...]

    @property
    def obligations(self) -> frozenset[str]:
        return frozenset().union(
            *(entered(a, b) for a, b in zip(self.states, self.states[1:], strict=False))
        )


@dataclass(frozen=True)
class TransitionSuite:
    paths: tuple[TransitionPath, ...]
    obligations: frozenset[str]
    generated_count: int

    def evidence(self, seed: int) -> dict[str, Any]:
        return {
            "seed": seed,
            "obligations_total": len(self.obligations),
            "obligations_covered": len(self.obligations),
            "paths": len(self.paths),
            "generated_paths": self.generated_count,
            "journey_like_paths": len(self.paths) - self.generated_count,
            "steps": sum(len(path.states) - 1 for path in self.paths),
            "path_ids": [path.path_id for path in self.paths],
        }


def _path(family: str, states: list[dict[str, Any]]) -> TransitionPath:
    for state in states:
        valid, reason = _valid(state)
        if not valid:
            raise AssertionError(f"Invalid {family} path: {reason}")
    digest = sha256(json.dumps(states, sort_keys=True).encode()).hexdigest()[:10]
    return TransitionPath(f"{family}-{digest}", family, tuple(states))


def _named_paths() -> list[TransitionPath]:
    base = {key: values[0] for key, values in DIMENSIONS.items()}
    base.update(chat_model="gpt-5.6", reasoning_profile="none")

    def states(*updates: dict[str, Any]) -> list[dict[str, Any]]:
        return [base | update for update in updates]

    return [
        _path(
            "provider-api",
            states(
                {"api_mode": "auto"},
                {"api_mode": "responses"},
                {"api_mode": "chat_completions"},
                {"api_mode": "auto"},
            ),
        ),
        _path(
            "model-capability",
            states(
                {
                    "chat_model": "gpt-5.6",
                    "reasoning_profile": "none",
                    "web_search": True,
                },
                {
                    "chat_model": "gpt-4.1",
                    "reasoning_profile": "recommended",
                    "web_search": False,
                },
                {
                    "chat_model": "gpt-5.6",
                    "reasoning_profile": "none",
                    "web_search": True,
                },
            ),
        ),
        _path(
            "memory-knowledge",
            states(
                {},
                {"memory_mode": "manual", "knowledge_enabled": True},
                {
                    "memory_mode": "automatic",
                    "knowledge_enabled": True,
                    "temporary_memory": "balanced",
                },
                {},
            ),
        ),
        _path(
            "function-loading",
            states(
                {},
                {"function_tools": "direct"},
                {"function_tools": "direct", "function_groups": "always"},
                {"function_tools": "direct", "function_groups": "on_demand"},
                {"function_tools": "disabled", "function_groups": "none"},
                {},
            ),
        ),
        _path(
            "guest-security",
            states(
                {},
                {"guest_mode_enabled": True},
                {
                    "guest_mode_enabled": True,
                    "guest_function_policy": "custom",
                    "guest_knowledge_policy": "custom",
                    "guest_shared_memory_policy": "read_only",
                },
                {},
            ),
        ),
        _path(
            "local-voice",
            states(
                {},
                {"local_intents_enabled": True, "local_intent_exclusions": "turn_on"},
                {
                    "local_intents_enabled": True,
                    "local_intent_delayed_commands_to_ai": True,
                    "conversation_continuity": "user",
                    "voice_scope_policy": "default_user",
                },
                {
                    "conversation_continuity": "device",
                    "voice_scope_policy": "device_mapping",
                    "voice_device_mappings": "mapped",
                },
                {},
            ),
        ),
        _path(
            "archive-retention",
            states(
                {},
                {"archive": "private"},
                {"archive": "shared", "retention": "short"},
                {},
            ),
        ),
        _path(
            "ui-speech",
            states(
                {},
                {
                    "exposed_entities_enabled": True,
                    "speech_processing_enabled": True,
                    "speech_behavior": "regex",
                },
                {
                    "exposed_entities_enabled": True,
                    "speech_processing_enabled": True,
                    "advanced_options": True,
                },
                {},
            ),
        ),
    ]


def generate_transitions(seed: int, *, budget: int | None = None) -> TransitionSuite:
    """Cover every supported dimension value entered from a different value.

    First choose explicit cross-feature paths. For uncovered enum entries, find
    single-dimension valid neighbors in F's production-validated states and use
    seeded greedy set cover. Each generated path returns A→B→A.
    """
    rng = random.Random(seed ^ 0xC0DEC0DE)
    covering = generate(seed, heavy=True)
    cases = list(covering.cases)
    named = _named_paths()
    required = frozenset(
        f"enter:{key}:{json.dumps(value, sort_keys=True)}"
        for key, values in DIMENSIONS.items()
        for value in values
        if any(case[key] == value for case in cases)
    )
    covered = set().union(*(path.obligations for path in named))
    candidates: dict[tuple[str, str], TransitionPath] = {}
    for key, values in DIMENSIONS.items():
        by_value = {
            value: [case for case in cases if case[key] == value] for value in values
        }
        for target_value in values:
            shuffled = list(by_value[target_value])
            rng.shuffle(shuffled)
            for target in shuffled:
                other_values = [value for value in values if value != target_value]
                rng.shuffle(other_values)
                for source_value in other_values:
                    source = target | {key: source_value}
                    if _valid(source)[0]:
                        path = _path(key, [source, target, source])
                        candidates[
                            (
                                min(fingerprint(source), fingerprint(target)),
                                max(fingerprint(source), fingerprint(target)),
                            )
                        ] = path
                        break
                else:
                    continue
                break
    pool = list(candidates.values())
    rng.shuffle(pool)
    selected: list[TransitionPath] = []
    missing = set(required - covered)
    while missing:
        if not pool:
            raise AssertionError(f"No valid transition covers {sorted(missing)[:5]}")
        best = max(pool, key=lambda path: len(path.obligations & missing))
        gain = best.obligations & missing
        if not gain:
            raise AssertionError(f"No valid transition covers {sorted(missing)[:5]}")
        selected.append(best)
        missing.difference_update(gain)
        pool.remove(best)
        if budget is not None and len(selected) > budget:
            raise AssertionError(
                f"Transition budget {budget} exhausted with {len(missing)} obligations left"
            )
    paths = tuple((*named, *selected))
    assert required <= frozenset().union(*(path.obligations for path in paths))
    return TransitionSuite(paths, required, len(selected))
