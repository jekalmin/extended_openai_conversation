"""Request Rules management, persistence, migration, and mutation-safety tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from custom_components.extended_openai_conversation_responses import request_rules as rr


class MemoryStore:
    """Minimal in-memory Store stand-in for persistence-focused tests."""

    def __init__(self, data=None):
        self.data = deepcopy(data)
        self.saves = 0

    async def async_load(self):
        return deepcopy(self.data)

    async def async_save(self, data):
        self.data = deepcopy(data)
        self.saves += 1


def local_rule(*, rule_id: str = "good-night", name: str = "Good night"):
    return {
        "id": rule_id,
        "name": name,
        "enabled": True,
        "phrases": ["good night"],
        "match_type": "equals",
        "action_type": "local_action",
        "action": {
            "actions": [
                {
                    "domain": "script",
                    "service": "turn_on",
                    "target": {"entity_id": ["script.goodnight"]},
                    "data": {},
                }
            ],
            "success_response": "Done",
            "failure_response": "Failed safely",
        },
        "matching_behavior": "defaults",
        "matching": dict(rr.DEFAULT_MATCHING),
        "order": 0,
    }


def routing_rule(*, rule_id: str = "route"):
    return {
        "id": rule_id,
        "name": "Route",
        "enabled": True,
        "phrases": ["quiet mode"],
        "match_type": "equals",
        "action_type": "model_routing",
        "action": {"reset": True, "scope": "request"},
        "matching_behavior": "defaults",
        "matching": dict(rr.DEFAULT_MATCHING),
        "order": 0,
    }


async def test_initialize_repairs_malformed_container_and_persists_defaults() -> None:
    store = MemoryStore(["not", "an", "object"])
    rules = rr.RequestRules(store)

    await rules.async_initialize()

    snapshot = rules.snapshot()
    assert snapshot["rules"] == []
    assert snapshot["defaults"] == rr.DEFAULT_MATCHING
    assert snapshot["wording_groups"] == list(rr.DEFAULT_WORDING_GROUPS)
    assert store.saves == 1


async def test_initialize_repairs_invalid_sections_and_duplicate_rule_ids() -> None:
    first = local_rule()
    duplicate = local_rule(name="Duplicate name")
    store = MemoryStore(
        {
            "defaults": {"fuzzy_threshold": 101},
            "wording_groups": "not-a-list",
            "rules": [first, duplicate],
        }
    )
    rules = rr.RequestRules(store)

    await rules.async_initialize()

    snapshot = rules.snapshot()
    assert snapshot["defaults"] == rr.DEFAULT_MATCHING
    assert snapshot["wording_groups"] == list(rr.DEFAULT_WORDING_GROUPS)
    assert [rule["id"] for rule in snapshot["rules"]] == ["good-night"]
    assert store.saves == 1


async def test_initialize_migrates_complete_legacy_request_route_once() -> None:
    store = MemoryStore(
        {
            "defaults": dict(rr.DEFAULT_MATCHING),
            "wording_groups": list(rr.DEFAULT_WORDING_GROUPS),
            "rules": [routing_rule()],
        }
    )
    rules = rr.RequestRules(store)

    await rules.async_initialize()

    action = rules.snapshot()["rules"][0]["action"]
    assert action["scope"] == "conversation"
    assert action["continue_to_ai"] is False
    assert action["reset"] is True
    assert store.saves == 1

    reloaded = rr.RequestRules(store)
    await reloaded.async_initialize()
    assert reloaded.snapshot()["rules"][0]["action"] == action
    assert store.saves == 1


async def test_revision_guard_rejects_bad_or_stale_writer_without_saving() -> None:
    store = MemoryStore()
    rules = rr.RequestRules(store)
    await rules.async_initialize()
    revision = rules.revision()

    with pytest.raises(ValueError, match="revision must be a string"):
        await rules.async_set_defaults(rr.DEFAULT_MATCHING, expected_revision=123)
    with pytest.raises(ValueError, match="changed in another tab"):
        await rules.async_set_defaults(rr.DEFAULT_MATCHING, expected_revision="stale")

    assert rules.revision() == revision
    assert store.saves == 0
    assert "revision" not in await rules.async_backup_data()


async def test_revision_guard_accepts_current_revision_and_changes_after_save() -> None:
    store = MemoryStore()
    rules = rr.RequestRules(store)
    await rules.async_initialize()
    revision = rules.revision()

    updated = await rules.async_set_defaults(
        {**rr.DEFAULT_MATCHING, "fuzzy": True}, expected_revision=revision
    )

    assert updated["fuzzy"] is True
    assert rules.revision() != revision
    assert store.saves == 1


async def test_revision_guard_rejects_aba_after_committed_changes() -> None:
    """A writer suspended at A must not overwrite intervening B and A saves."""
    store = MemoryStore()
    rules = rr.RequestRules(store)
    await rules.async_initialize()
    original = rules.revision()

    await rules.async_set_defaults({**rr.DEFAULT_MATCHING, "fuzzy": True})
    await rules.async_set_defaults(rr.DEFAULT_MATCHING)
    assert rules.snapshot()["defaults"] == rr.DEFAULT_MATCHING
    assert rules.revision() != original

    with pytest.raises(ValueError, match="changed in another tab"):
        await rules.async_set_defaults(
            {**rr.DEFAULT_MATCHING, "fuzzy_threshold": 91},
            expected_revision=original,
        )
    assert store.saves == 2


async def test_request_rule_settings_save_atomically_once() -> None:
    store = MemoryStore()
    rules = rr.RequestRules(store)
    await rules.async_initialize()
    revision = rules.revision()
    groups = [{"canonical": "turn on", "alternatives": ["switch on"]}]

    result = await rules.async_set_settings(
        {**rr.DEFAULT_MATCHING, "fuzzy": True},
        groups,
        expected_revision=revision,
    )

    assert result["defaults"]["fuzzy"] is True
    assert result["wording_groups"] == groups
    assert result["revision"] == rules.revision()
    assert store.saves == 1

    saved = rules.snapshot()
    with pytest.raises(ValueError, match="changed in another tab"):
        await rules.async_set_settings(
            {**rr.DEFAULT_MATCHING, "fuzzy_threshold": 91},
            [{"canonical": "off", "alternatives": ["disable"]}],
            expected_revision=revision,
        )
    assert rules.snapshot() == saved
    assert store.saves == 1


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "request_rules must be an object"),
        ({"unexpected": True}, "unknown request_rules fields"),
        ({"rules": "bad"}, "request_rules.rules must be a list"),
    ],
)
def test_backup_validation_rejects_invalid_container_shapes(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        rr.RequestRules.validate_backup_data(value)


def test_backup_validation_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate Request Rule id"):
        rr.RequestRules.validate_backup_data(
            {"rules": [local_rule(), local_rule(name="Duplicate")]}
        )


async def test_move_validates_direction_and_boundary_is_a_noop() -> None:
    store = MemoryStore({"rules": [local_rule()]})
    rules = rr.RequestRules(store)
    await rules.async_initialize()
    before = rules.snapshot()["rules"]
    saves = store.saves

    with pytest.raises(
        ValueError, match="direction must be up, down, top, bottom, before or after"
    ):
        await rules.async_move("good-night", "sideways")
    result = await rules.async_move("good-night", "up")

    assert result == before[0]
    assert rules.snapshot()["rules"] == before
    assert store.saves == saves


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "matching settings must be an object"),
        ({"mystery": True}, "unknown matching settings"),
        ({"fuzzy": 1}, "fuzzy must be true or false"),
        ({"fuzzy_threshold": True}, "fuzzy_threshold must be an integer"),
        ({"fuzzy_threshold": 69}, "fuzzy_threshold must be an integer"),
    ],
)
def test_matching_settings_reject_invalid_values(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        rr.validate_matching_settings(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-list", "wording_groups must be a list"),
        ([{"canonical": "on"}], "needs canonical and alternatives"),
        (
            [{"canonical": "on", "alternatives": "switch on"}],
            "wording alternatives must be a list",
        ),
        (
            [{"canonical": "on", "alternatives": []}],
            "wording alternatives must contain 1 to 25 items",
        ),
        (
            [{"canonical": "!!!", "alternatives": ["switch on"]}],
            "searchable text",
        ),
    ],
)
def test_wording_group_validation_rejects_malformed_catalogs(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        rr.validate_wording_groups(value)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rule: rule.update(extra=True), "unknown rule fields"),
        (lambda rule: rule.update(phrases="good night"), "phrases must be a list"),
        (lambda rule: rule.update(phrases=[]), "phrases must contain"),
        (lambda rule: rule.update(match_type="regex"), "unsupported match type"),
        (lambda rule: rule.update(action_type="remote"), "unsupported action type"),
        (
            lambda rule: rule.update(matching_behavior="sometimes"),
            "matching_behavior must be defaults or custom",
        ),
        (lambda rule: rule.update(order=True), "order must be a non-negative integer"),
        (lambda rule: rule.update(enabled="yes"), "enabled must be true or false"),
    ],
)
def test_rule_validation_rejects_invalid_contract_fields(mutate, message) -> None:
    rule = local_rule()
    mutate(rule)
    with pytest.raises(ValueError, match=message):
        rr.validate_rule(rule)


def test_non_pattern_rule_rejects_captured_value_reference() -> None:
    rule = local_rule()
    rule["phrases"] = ["turn on {room}"]

    with pytest.raises(ValueError, match="require Sentence pattern matching"):
        rr.validate_rule(rule)


def test_request_only_complete_routing_rule_is_rejected() -> None:
    rule = routing_rule()
    rule["action"] = {
        "reset": True,
        "scope": "request",
        "continue_to_ai": False,
    }

    with pytest.raises(
        ValueError, match="Request-only routing requires Continue to AI"
    ):
        rr.validate_rule(rule)


def test_routing_reset_discards_supplied_model_and_effort() -> None:
    rule = routing_rule()
    rule["action"] = {
        "reset": True,
        "scope": "conversation",
        "continue_to_ai": False,
        "model": "ignored-model",
        "reasoning_effort": "ignored-effort",
    }

    validated = rr.validate_rule(rule)

    assert validated["action"]["model"] is None
    assert validated["action"]["reasoning_effort"] is None
    assert validated["action"]["success_response"] == "Using the configured defaults"


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (
            {"domain": "Light", "service": "turn_on", "target": {}, "data": {}},
            "lowercase slugs",
        ),
        (
            {"domain": "light", "service": "turn_on", "target": [], "data": {}},
            "target and data must be objects",
        ),
        (
            {
                "type": "function",
                "function": "remember",
                "arguments": [],
            },
            "function arguments must be an object",
        ),
        (
            {
                "type": "function",
                "function": "remember",
                "arguments": {"bad-name": "value"},
            },
            "argument names must be simple identifiers",
        ),
        (
            {
                "type": "function",
                "function": "remember",
                "arguments": {"fact": {"source": "other", "value": "x"}},
            },
            "source must be fixed or slot",
        ),
        (
            {
                "type": "function",
                "function": "remember",
                "arguments": {
                    "fact": {"source": "slot", "slot": "fact", "extra": True}
                },
            },
            "slot arguments need only source and slot",
        ),
    ],
)
def test_local_action_migration_rejects_malformed_legacy_actions(
    action, message
) -> None:
    rule = local_rule()
    rule["action"]["actions"] = [action]

    with pytest.raises(ValueError, match=message):
        rr.validate_rule(rule)


def test_runtime_expires_old_overrides_and_refreshes_timeout(monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(rr, "monotonic", lambda: clock["now"])
    runtime = rr.RequestRuleRuntime()

    runtime.set("first", {"model": "one"}, timeout_minutes=1)
    clock["now"] = 30.0
    assert runtime.get("first", timeout_minutes=2) == {"model": "one"}

    clock["now"] = 151.0
    assert runtime.get("first") == {}


def test_runtime_reset_request_ignores_conversation_override() -> None:
    runtime = rr.RequestRuleRuntime()
    runtime.set("session", {"model": "conversation-model"})

    effective = runtime.effective_options(
        {"model": "configured-model", "reasoning_effort": "low"},
        "session",
        {
            rr._REQUEST_RESET_SENTINEL: "1",
            "reasoning_effort": "high",
        },
    )

    assert effective == {"model": "configured-model", "reasoning_effort": "high"}


def test_runtime_reset_removes_only_requested_session() -> None:
    runtime = rr.RequestRuleRuntime()
    runtime.set("one", {"model": "first"})
    runtime.set("two", {"model": "second"})

    runtime.reset("one")

    assert runtime.get("one") == {}
    assert runtime.get("two") == {"model": "second"}


# Determinism and canonical-management regressions formerly split by PR provenance.

from copy import deepcopy
from types import SimpleNamespace

import pytest

from custom_components.extended_openai_conversation_responses.const import (
    CONF_CHAT_MODEL,
    CONF_REASONING_EFFORT,
    DOMAIN,
    SERVICE_CALL_FUNCTION,
)
from custom_components.extended_openai_conversation_responses.management_ui import (
    async_management_command,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
    RequestRuleRuntime,
    RequestRules,
    async_evaluate_rule,
    validate_rule,
    validate_wording_groups,
)
from homeassistant.exceptions import HomeAssistantError


class DeterminismMemoryStore:
    """Minimal in-memory Store seam."""

    def __init__(self, data=None):
        self.data = deepcopy(data)
        self.saves = 0

    async def async_load(self):
        return deepcopy(self.data)

    async def async_save(self, data):
        self.data = deepcopy(data)
        self.saves += 1


def determinism_local_rule(
    name: str = "Good night",
    *,
    rule_id: str = "good-night",
    order: int = 0,
    phrases=None,
    match_type: str = "equals",
):
    return {
        "id": rule_id,
        "name": name,
        "enabled": True,
        "phrases": phrases or ["good night"],
        "match_type": match_type,
        "action_type": "local_action",
        "action": {
            "actions": [
                {
                    "domain": "script",
                    "service": "turn_on",
                    "target": {"entity_id": ["script.goodnight"]},
                    "data": {},
                }
            ],
            "success_response": "Done",
            "failure_response": "Failed safely",
        },
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": order,
    }


def determinism_routing_rule(
    *,
    rule_id: str = "routing",
    phrase: str = "think carefully",
    match_type: str = "starts_with",
    scope: str = "conversation",
    model: str | None = "gpt-5",
    effort: str | None = None,
    reset: bool = False,
):
    return {
        "id": rule_id,
        "name": "Routing",
        "enabled": True,
        "phrases": [phrase],
        "match_type": match_type,
        "action_type": "model_routing",
        "action": {
            "model": model,
            "reasoning_effort": effort,
            "scope": scope,
            "reset": reset,
            "success_response": "Updated",
        },
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": 0,
    }


async def determinism_manager(*rules):
    result = RequestRules(DeterminismMemoryStore({"rules": list(rules)}))
    await result.async_initialize()
    return result


@pytest.mark.parametrize(
    "stored",
    [
        ["not", "an", "object"],
        {"rules": {"not": "a list"}},
        {"rules": 123},
    ],
)
async def test_malformed_store_container_self_heals(stored) -> None:
    store = DeterminismMemoryStore(stored)
    rules = RequestRules(store)
    await rules.async_initialize()

    assert rules.snapshot()["rules"] == []
    assert store.saves == 1
    assert store.data["rules"] == []


async def test_duplicate_orders_are_reindexed_after_mutations() -> None:
    store = DeterminismMemoryStore(
        {
            "rules": [
                determinism_local_rule("Zulu", rule_id="z", order=3),
                determinism_local_rule("Alpha", rule_id="a", order=3),
            ]
        }
    )
    rules = RequestRules(store)
    await rules.async_initialize()
    assert [rule["order"] for rule in rules.snapshot()["rules"]] == [0, 1]
    assert store.saves == 1

    created = await rules.async_create(
        determinism_local_rule("Middle", rule_id="m", order=0)
    )
    assert created["id"] == "m"
    assert [rule["order"] for rule in rules.snapshot()["rules"]] == [0, 1, 2]

    await rules.async_update(
        "z", {**determinism_local_rule("Zulu", rule_id="wrong"), "order": 0}
    )
    assert [rule["order"] for rule in rules.snapshot()["rules"]] == [0, 1, 2]
    assert (
        next(rule for rule in rules.snapshot()["rules"] if rule["id"] == "z")["id"]
        == "z"
    )

    await rules.async_delete("m")
    assert [rule["order"] for rule in rules.snapshot()["rules"]] == [0, 1]


def test_wording_groups_reject_phrases_that_normalize_empty() -> None:
    with pytest.raises(ValueError, match="searchable text"):
        validate_wording_groups([{"canonical": "!!!", "alternatives": ["activate"]}])
    with pytest.raises(ValueError, match="searchable text"):
        validate_wording_groups([{"canonical": "activate", "alternatives": ["---"]}])


async def test_hassil_nested_group_with_wildcard_slot_uses_hassil_matcher() -> None:
    rules = await determinism_manager(
        determinism_local_rule(
            phrases=["((turn|switch) on|activate) {room} lights"],
            match_type="sentence_pattern",
        )
    )

    first = rules.match("switch on kitchen lights")
    second = rules.match("activate upstairs guest room lights")
    assert first is not None and first.slots == {"room": "kitchen"}
    assert second is not None and second.slots == {"room": "upstairs guest room"}


async def test_request_reset_does_not_clear_conversation_override() -> None:
    runtime = RequestRuleRuntime()
    runtime.set(
        "session",
        {CONF_CHAT_MODEL: "gpt-5", CONF_REASONING_EFFORT: "high"},
    )
    rule = determinism_routing_rule(
        phrase="use defaults",
        scope="request",
        model=None,
        effort=None,
        reset=True,
    )
    result = await async_evaluate_rule(
        SimpleNamespace(),
        await determinism_manager(rule),
        runtime,
        "use defaults for this request",
        "session",
    )
    assert result is not None
    assert runtime.get("session") == {
        CONF_CHAT_MODEL: "gpt-5",
        CONF_REASONING_EFFORT: "high",
    }
    defaults = {CONF_CHAT_MODEL: "gpt-4o", CONF_REASONING_EFFORT: "low"}
    assert (
        runtime.effective_options(defaults, "session", result.request_override)
        == defaults
    )


async def test_conversation_reset_still_clears_conversation_override() -> None:
    runtime = RequestRuleRuntime()
    runtime.set("session", {CONF_CHAT_MODEL: "gpt-5"})
    rule = determinism_routing_rule(
        phrase="use defaults",
        match_type="equals",
        scope="conversation",
        model=None,
        effort=None,
        reset=True,
    )
    result = await async_evaluate_rule(
        SimpleNamespace(),
        await determinism_manager(rule),
        runtime,
        "use defaults",
        "session",
    )
    assert result is not None and result.consume
    assert runtime.get("session") == {}


async def test_duplicate_names_are_bounded_and_unique() -> None:
    source_name = "x" * 120
    rules = await determinism_manager(
        determinism_local_rule(source_name, rule_id="source")
    )
    first = await rules.async_duplicate("source")
    second = await rules.async_duplicate("source")

    assert len(first["name"]) <= 120
    assert len(second["name"]) <= 120
    assert first["name"].endswith(" copy")
    assert second["name"].endswith(" copy 2")
    assert first["name"] != second["name"]
    assert [rule["order"] for rule in rules.snapshot()["rules"]] == [0, 1, 2]


async def test_captured_routing_values_are_resolved_and_validated() -> None:
    runtime = RequestRuleRuntime()
    model_rule = determinism_routing_rule(
        rule_id="model",
        phrase="use {model_name}",
        match_type="sentence_pattern",
        model="{model_name}",
    )
    model_rule["action"]["success_response"] = "Using {model_name}"
    result = await async_evaluate_rule(
        SimpleNamespace(),
        await determinism_manager(model_rule),
        runtime,
        "use gpt-5",
        "session",
    )
    assert result is not None
    assert result.response == "Using gpt-5"
    assert runtime.get("session")[CONF_CHAT_MODEL] == "gpt-5"

    effort_rule = determinism_routing_rule(
        rule_id="effort",
        phrase="reason {effort}",
        match_type="sentence_pattern",
        model="gpt-5",
        effort="{effort}",
    )
    await async_evaluate_rule(
        SimpleNamespace(),
        await determinism_manager(effort_rule),
        runtime,
        "reason high",
        "session",
    )
    assert runtime.get("session")[CONF_REASONING_EFFORT] == "high"

    with pytest.raises(
        HomeAssistantError, match="Unsupported captured reasoning effort"
    ):
        await async_evaluate_rule(
            SimpleNamespace(),
            await determinism_manager(effort_rule),
            runtime,
            "reason extreme",
            "session",
        )


def test_routing_captures_reject_jinja_and_partial_effort_templates() -> None:
    rule = determinism_routing_rule(
        phrase="use {model_name}",
        match_type="sentence_pattern",
        model="{{ model_name }}",
    )
    with pytest.raises(ValueError, match="simple .* references"):
        validate_rule(rule)

    rule = determinism_routing_rule(
        phrase="reason {effort}",
        match_type="sentence_pattern",
        model="gpt-5",
        effort="level-{effort}",
    )
    with pytest.raises(ValueError, match="single .* reference"):
        validate_rule(rule)


async def test_management_create_assigns_id_and_validates_canonical_function_reference(
    hass,
    monkeypatch,
) -> None:
    store = DeterminismMemoryStore()
    rules = RequestRules(store)
    await rules.async_initialize()
    configured_tool = {
        "spec": {
            "name": "remember",
            "description": "Remember a fact",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
        "function": {"type": "template", "value_template": "{{ fact }}"},
        "enabled": True,
    }
    subentry = SimpleNamespace(
        subentry_id="agent",
        subentry_type="conversation",
        data={"functions": [configured_tool]},
    )
    entry = SimpleNamespace(
        domain=DOMAIN,
        subentries={"agent": subentry},
    )
    hass.config_entries.async_get_entry.return_value = entry

    async def get_rules(*_args):
        return rules

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.management_ui.async_get_request_rules",
        get_rules,
    )
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation_responses.management_ui.configured_function_tools_from_data",
        lambda _data: [configured_tool],
    )

    rule = determinism_local_rule()
    rule.pop("id")
    rule["action"]["actions"] = [
        {
            "type": "function",
            "function": "remember",
            "arguments": {"fact": {"source": "fixed", "value": "bin day"}},
        }
    ]
    base = {
        "section": "request_rules",
        "entry_id": "entry",
        "subentry_id": "agent",
    }
    created = await async_management_command(
        hass,
        "admin",
        True,
        {**base, "action": "create", "rule": rule},
    )
    assert created["rule"]["id"]
    assert created["rule"]["action"]["actions"] == [
        {
            "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
            "data": {"function": "remember", "arguments": {"fact": "bin day"}},
        }
    ]

    bad = deepcopy(rule)
    bad["action"]["actions"][0]["function"] = "missing_tool"
    with pytest.raises(HomeAssistantError, match="unavailable or disabled"):
        await async_management_command(
            hass,
            "admin",
            True,
            {**base, "action": "create", "rule": bad},
        )

    missing_input = deepcopy(rule)
    missing_input["action"]["actions"][0]["arguments"] = {}
    with pytest.raises(
        HomeAssistantError, match="Missing required function input: fact"
    ):
        await async_management_command(
            hass,
            "admin",
            True,
            {**base, "action": "create", "rule": missing_input},
        )

    updated_payload = deepcopy(created["rule"])
    updated_payload["id"] = "wrong-id"
    updated_payload["enabled"] = False
    updated = await async_management_command(
        hass,
        "admin",
        True,
        {
            **base,
            "action": "update",
            "rule_id": created["rule"]["id"],
            "rule": updated_payload,
        },
    )
    assert updated["rule"]["id"] == created["rule"]["id"]
    assert updated["rule"]["enabled"] is False


# Mutation-order and stale-writer regressions formerly split by roadmap provenance.

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation_responses.const import DOMAIN
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
    RequestRules,
    validate_rule,
)


def _routing_rule(
    rule_id: str,
    name: str,
    *,
    phrase: str = "use the careful model",
    match_type: str = "contains",
    order: int = 0,
) -> dict:
    return validate_rule(
        {
            "id": rule_id,
            "name": name,
            "enabled": True,
            "phrases": [phrase],
            "match_type": match_type,
            "action_type": "model_routing",
            "action": {
                "model": "gpt-5-mini",
                "reasoning_effort": None,
                "scope": "request"
                if match_type not in {"equals", "sentence_pattern"}
                else "conversation",
                "reset": False,
                "success_response": "Updated",
            },
            "matching_behavior": "defaults",
            "matching": DEFAULT_MATCHING,
            "order": order,
        }
    )


def _function_rule(function_name: str) -> dict:
    return validate_rule(
        {
            "id": "function-rule",
            "name": "Run configured function",
            "enabled": True,
            "phrases": ["do the thing"],
            "match_type": "equals",
            "action_type": "local_action",
            "action": {
                "actions": [
                    {
                        "action": f"{DOMAIN}.call_function",
                        "data": {"function": function_name, "arguments": {}},
                    }
                ],
                "success_response": "Done",
                "failure_response": "Failed",
            },
            "matching_behavior": "defaults",
            "matching": DEFAULT_MATCHING,
            "order": 0,
        }
    )


def _manager(rules: list[dict]) -> tuple[RequestRules, SimpleNamespace]:
    store = SimpleNamespace(async_save=AsyncMock())
    manager = RequestRules(store)
    manager._initialized = True
    manager._rules = rules
    manager._sort_and_compile()
    return manager, store


async def test_move_changes_only_final_order_tiebreaker() -> None:
    first = _routing_rule("first", "First", order=0)
    second = _routing_rule("second", "Second", order=1)
    manager, store = _manager([first, second])

    assert manager.match("please use the careful model now").rule["id"] == "first"
    revision = manager.revision()

    moved = await manager.async_move("second", "up", expected_revision=revision)

    assert moved["id"] == "second"
    assert [rule["id"] for rule in manager.snapshot()["rules"]] == ["second", "first"]
    assert manager.match("please use the careful model now").rule["id"] == "second"
    store.async_save.assert_awaited_once()


async def test_move_overrides_match_type_specificity_via_list_order() -> None:
    broad = _routing_rule(
        "broad",
        "Broad",
        phrase="careful model",
        match_type="contains",
        order=0,
    )
    exact = _routing_rule(
        "exact",
        "Exact",
        phrase="use the careful model",
        match_type="equals",
        order=1,
    )
    manager, _store = _manager([broad, exact])

    assert manager.match("use the careful model").rule["id"] == "broad"

    await manager.async_move("exact", "up", expected_revision=manager.revision())

    assert manager.match("use the careful model").rule["id"] == "exact"


async def test_move_rejects_stale_revision_without_changing_rules() -> None:
    manager, store = _manager(
        [
            _routing_rule("first", "First", order=0),
            _routing_rule("second", "Second", order=1),
        ]
    )
    stale_revision = manager.revision()
    await manager.async_update(
        "first",
        _routing_rule("first", "Updated first", order=0),
        expected_revision=stale_revision,
    )
    store.async_save.reset_mock()

    with pytest.raises(ValueError, match="changed in another tab"):
        await manager.async_move("second", "up", expected_revision=stale_revision)

    assert [rule["id"] for rule in manager.snapshot()["rules"]] == ["first", "second"]
    store.async_save.assert_not_awaited()


async def test_function_reference_rename_rejects_stale_revision() -> None:
    manager, store = _manager([_function_rule("old_tool")])
    stale_revision = manager.revision()
    await manager.async_set_defaults(
        {**DEFAULT_MATCHING, "fuzzy_threshold": 91},
        expected_revision=stale_revision,
    )
    store.async_save.reset_mock()

    with pytest.raises(ValueError, match="changed in another tab"):
        await manager.async_rename_function_reference(
            "old_tool",
            "new_tool",
            expected_revision=stale_revision,
        )

    assert manager.function_references("old_tool") == [
        {"id": "function-rule", "name": "Run configured function"}
    ]
    assert manager.function_references("new_tool") == []
    store.async_save.assert_not_awaited()
