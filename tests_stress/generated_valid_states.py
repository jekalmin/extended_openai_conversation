"""Deterministic covering states, checked against EOAI's actual validators.

The dimensions are intentionally semantic choices.  Scalar payload boundaries and
large collections have their own tests and are classified in the field contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, product
import json
import random
from typing import Any

from custom_components.extended_openai_conversation_responses import agent_config, const
from custom_components.extended_openai_conversation_responses.model_capabilities import (
    ModelCapabilityError,
    select_api_path,
)
from custom_components.extended_openai_conversation_responses.model_catalog import (
    BUNDLED_CATALOG,
)
from custom_components.extended_openai_conversation_responses.request import (
    build_provider_request_snapshot,
)
from homeassistant.exceptions import HomeAssistantError

TOOL = {
    "spec": {
        "name": "coverage_marker",
        "description": "Return a marker",
        "parameters": {"type": "object", "properties": {}},
    },
    "function": {"type": "native", "name": "execute_service"},
    "enabled": True,
}
DISABLED_TOOL = {**TOOL, "enabled": False}
GROUP = {
    "id": "coverage-group",
    "name": "Coverage group",
    "description": "Coverage marker",
    "loading_mode": "always",
    "functions": ["coverage_marker"],
    "enabled": True,
}

# The full resolved bundle is the model dimension. The field contract snapshots
# these IDs so adding or removing a model requires an explicit matrix review.
MODEL_IDS = tuple(sorted(BUNDLED_CATALOG.resolved))

DIMENSIONS: dict[str, tuple[Any, ...]] = {
    "chat_model": MODEL_IDS,
    "api_mode": tuple(item["key"] for item in const.API_MODE_OPTIONS),
    "reasoning_profile": ("recommended", "none", "low", "high"),
    "sampling": ("default", "temperature", "top_p"),
    "service_tier": ("default", "flex", "priority"),
    "web_search": (False, True),
    "function_tools": ("none", "direct", "disabled"),
    "function_groups": ("none", "always", "on_demand"),
    "function_tool_error_recovery": (False, True),
    "shorten_tool_call_id": (False, True),
    "memory_mode": ("off", "manual", "automatic"),
    "memory_retrieval_mode": ("lexical", "hybrid"),
    "memory_auto_retrieve_limit": (0, 3),
    "knowledge_enabled": (False, True),
    "archive": ("off", "private", "shared"),
    "temporary_memory": ("off", "balanced", "eager"),
    "guest_mode_enabled": (False, True),
    "guest_function_policy": ("off", "custom"),
    "guest_knowledge_policy": ("off", "custom"),
    "shared_memory_mode": ("disabled", "explicit"),
    "local_intents_enabled": (False, True),
    "local_intent_exclusions": ("none", "turn_on"),
    "local_intent_delayed_commands_to_ai": (False, True),
    "conversation_continuity": ("ha_default", "user", "device"),
    "voice_scope_policy": ("unretained", "shared", "default_user", "device_mapping"),
    "voice_device_mappings": ("none", "mapped"),
    "exposed_entities_enabled": (False, True),
    "speech_processing_enabled": (False, True),
    "speech_behavior": ("plain", "strip", "regex"),
    "retention": ("default", "short"),
    "template_context": ("default", "custom"),
    "advanced_options": (False, True),
}

TRIPLE_GROUPS = (
    ("chat_model", "api_mode", "reasoning_profile"),
    ("chat_model", "api_mode", "web_search"),
    ("api_mode", "reasoning_profile", "function_tools"),
    ("memory_mode", "knowledge_enabled", "guest_mode_enabled"),
    ("function_tools", "function_groups", "guest_mode_enabled"),
    ("conversation_continuity", "voice_scope_policy", "voice_device_mappings"),
)
CAPABILITY_KEYS = (
    "chat_model",
    "api_mode",
    "reasoning_profile",
    "web_search",
    "function_tools",
    "function_groups",
)


def _config(state: dict[str, Any]) -> dict[str, Any]:
    derived = {
        "function_groups",
        "service_tier",
        "local_intent_exclusions",
        "voice_device_mappings",
    }
    config = {
        k: v
        for k, v in state.items()
        if k in agent_config.AGENT_CONFIG_FIELDS and k not in derived
    }
    model = state["chat_model"]
    recommended = BUNDLED_CATALOG.resolved[model]["recommended_profile"][
        "reasoning_effort"
    ]
    profile = state["reasoning_profile"]
    if profile == "recommended":
        if recommended is not None:
            config["reasoning_effort"] = recommended
    elif profile == "none":
        config["reasoning_effort"] = "none"
    else:
        config["reasoning_effort"] = profile
    sampling = state["sampling"]
    if sampling == "temperature":
        config["temperature"] = 0.3
    elif sampling == "top_p":
        config["top_p"] = 0.8
    if state["service_tier"] != "default":
        config["service_tier"] = state["service_tier"]
    tools = state["function_tools"]
    config["functions"] = []
    if tools != "none":
        config["functions"] = [TOOL if tools == "direct" else DISABLED_TOOL]
    group = state["function_groups"]
    config["function_groups"] = []
    if group != "none":
        config["function_groups"] = [{**GROUP, "loading_mode": group}]
    archive = state["archive"]
    config["archive_enabled"] = archive != "off"
    config["shared_archive_enabled"] = archive == "shared"
    config["local_intent_exclusions"] = (
        ["HassTurnOn"] if state["local_intent_exclusions"] == "turn_on" else []
    )
    config["voice_device_mappings"] = (
        {"coverage-device": "user:coverage-owner"}
        if state["voice_device_mappings"] == "mapped"
        else {}
    )
    if state["voice_scope_policy"] == "default_user":
        config["voice_default_user_id"] = "coverage-owner"
    if state["speech_behavior"] == "strip":
        config.update(speech_strip_markdown=True, speech_strip_urls=True)
    elif state["speech_behavior"] == "regex":
        config["speech_regex_replacements"] = [
            {"pattern": "alpha", "replacement": "beta"}
        ]
    if state["retention"] == "short":
        config.update(
            archive_retention_days=7,
            usage_request_retention_days=7,
            usage_run_retention_days=7,
        )
    if state["template_context"] == "custom":
        config.update(
            current_datetime_enabled=True,
            current_datetime_template="Coverage date: {{ now().year }}",
            exposed_entities_template="Coverage entities: {{ exposed_entities | length }}",
        )
    return config


def normalized_state(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Use production normalization and request-path selection as the oracle."""
    normalized = agent_config.normalize_agent_config(_config(state))
    request = build_provider_request_snapshot(normalized, {})
    return normalized, request.api_mode


def _valid(state: dict[str, Any]) -> tuple[bool, str]:
    try:
        # Most exclusions are capability constraints.  Resolve these through the
        # catalogue consumer before paying for YAML/template normalization.
        model = state["chat_model"]
        profile = state["reasoning_profile"]
        effort = (
            BUNDLED_CATALOG.resolved[model]["recommended_profile"]["reasoning_effort"]
            if profile == "recommended"
            else profile
        )
        select_api_path(
            model,
            state["api_mode"],
            state["function_tools"] == "direct"
            or state["function_groups"] != "none"
            or state["memory_mode"] != "off"
            or state["knowledge_enabled"]
            or state["archive"] != "off"
            or state["guest_mode_enabled"],
            effort,
            state["web_search"],
        )
        normalized_state(state)
    except (
        agent_config.AgentConfigError,
        ModelCapabilityError,
        HomeAssistantError,
    ) as error:
        return False, f"{type(error).__name__}:{str(error).split(':', 1)[0][:80]}"
    return True, ""


def obligations(state: dict[str, Any]) -> frozenset[tuple[tuple[str, Any], ...]]:
    pairs = [tuple((k, state[k]) for k in keys) for keys in combinations(DIMENSIONS, 2)]
    triples = [tuple((k, state[k]) for k in keys) for keys in TRIPLE_GROUPS]
    return frozenset((*pairs, *triples))


@dataclass(frozen=True)
class CoveringSuite:
    cases: tuple[dict[str, Any], ...]
    obligations: frozenset[tuple[tuple[str, Any], ...]]
    excluded: dict[str, int]
    exploratory_count: int

    def evidence(self, seed: int) -> dict[str, Any]:
        pair_count = sum(len(item) == 2 for item in self.obligations)
        triple_count = len(self.obligations) - pair_count
        return {
            "seed": seed,
            "dimensions": len(DIMENSIONS),
            "cases": len(self.cases),
            "pair_obligations_total": pair_count,
            "pair_obligations_covered": pair_count,
            "triple_obligations_total": triple_count,
            "triple_obligations_covered": triple_count,
            "excluded": self.excluded,
            "exploratory_count": self.exploratory_count,
            "models": sorted({case["chat_model"] for case in self.cases}),
            "apis": sorted({case["api_mode"] for case in self.cases}),
            "case_ids": [
                sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:12]
                for case in self.cases
            ],
        }


def generate(seed: int, *, heavy: bool, budget: int | None = None) -> CoveringSuite:
    """Construct one valid witness per feasible obligation, then greedily cover.

    Feasibility is decided by EOAI normalization and request construction. The
    capability-interaction core is exhausted for every pair/triple obligation;
    infeasible partials are counted by production error, never a mirrored rule.
    """
    rng = random.Random(seed)
    default = {key: values[0] for key, values in DIMENSIONS.items()}
    candidates: dict[str, dict[str, Any]] = {}
    required: set[tuple[tuple[str, Any], ...]] = set()
    excluded: dict[str, int] = {}
    keys_to_cover = (*combinations(DIMENSIONS, 2), *TRIPLE_GROUPS)
    # Exhaust the small capability-interaction core through production validation.
    # Every remaining field has an independent normalized representation, so this
    # supplies a witness for each feasible pair/triple without heuristic retries.
    capability_states = []
    capability_rejected = []
    for values in product(*(DIMENSIONS[key] for key in CAPABILITY_KEYS)):
        cap = dict(zip(CAPABILITY_KEYS, values, strict=True))
        state = default | cap
        valid, reason = _valid(state)
        if valid:
            capability_states.append(cap)
        else:
            capability_rejected.append((cap, reason))
    rng.shuffle(capability_states)
    for keys in keys_to_cover:
        for values in product(*(DIMENSIONS[key] for key in keys)):
            obligation = tuple(zip(keys, values, strict=True))
            if obligation in required:
                continue
            fixed = dict(obligation)
            witness = None
            rejected_reason = None
            for cap in capability_states:
                if any(fixed[key] != cap[key] for key in keys if key in cap):
                    continue
                state = default | cap | fixed
                valid, reason = _valid(state)
                if valid:
                    witness = state
                    break
                rejected_reason = reason
            if witness is not None:
                required.add(obligation)
                candidates[json.dumps(witness, sort_keys=True)] = witness
            else:
                if rejected_reason is None:
                    rejected_reason = next(
                        (
                            reason
                            for cap, reason in capability_rejected
                            if all(fixed[key] == cap[key] for key in keys if key in cap)
                        ),
                        "no production-valid capability completion",
                    )
                reason = rejected_reason
                excluded[reason] = excluded.get(reason, 0) + 1
            # An obligation with no capability witness is genuinely excluded by
            # production config/request validation (within these dimensions).
    # Diverse valid candidates make the cover small without removing a single
    # obligation.  These are candidate solutions, not padded executed cases.
    diverse = 0
    for _ in range(6000):
        if diverse >= 1200:
            break
        state = {key: rng.choice(values) for key, values in DIMENSIONS.items()}
        if state["function_groups"] != "none" and state["function_tools"] == "none":
            state["function_tools"] = "direct"
        if not _valid(state)[0]:
            continue
        fingerprint = json.dumps(state, sort_keys=True)
        if fingerprint not in candidates:
            candidates[fingerprint] = state
            diverse += 1
    # A valid witness may cover obligations not used in its construction.
    required.update(*(obligations(case) for case in candidates.values()))
    uncovered = set(required)
    selected: list[dict[str, Any]] = []
    pool = [(case, obligations(case)) for case in candidates.values()]
    rng.shuffle(pool)
    while uncovered:
        best, best_cover = max(pool, key=lambda item: len(item[1] & uncovered))
        gain = best_cover & uncovered
        if not gain:
            raise AssertionError(f"Uncovered generated obligations: {len(uncovered)}")
        selected.append(best)
        uncovered.difference_update(gain)
        pool.remove((best, best_cover))
        if budget is not None and len(selected) > budget:
            raise AssertionError(
                f"Covering budget {budget} exhausted with {len(uncovered)} obligations left"
            )
    exploratory_count = 0
    if heavy:
        for _ in range(8):
            for _attempt in range(100):
                state = {key: rng.choice(values) for key, values in DIMENSIONS.items()}
                if _valid(state)[0] and state not in selected:
                    selected.append(state)
                    exploratory_count += 1
                    break
    rng.shuffle(selected)
    return CoveringSuite(
        tuple(selected), frozenset(required), excluded, exploratory_count
    )
