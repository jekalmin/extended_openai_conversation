"""Fast local Request Rules for conversation routing."""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Mapping,
    Sequence,
)
from contextlib import suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import sha256
import json
import logging
import re
from time import monotonic
from typing import Any, cast
import unicodedata
from uuid import uuid4

from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import condition as ha_condition, config_validation as cv
from homeassistant.helpers.script import Script, async_validate_actions_config
from homeassistant.helpers.storage import Store

from .const import (
    CONF_CHAT_MODEL,
    CONF_REASONING_EFFORT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    DOMAIN,
    SERVICE_CALL_FUNCTION,
)
from .guest_mode import (
    GUEST_MODE_UNAVAILABLE,
    GuestCapabilityPolicy,
    GuestModeDenied,
    guest_arguments_allowed_runtime,
)
from .helpers import get_model_config, get_reasoning_effort_options
from .model_catalog import all_reasoning_efforts
from .persistence_hardening import (
    _async_repair_private_store_mode,
    _async_settle_transactional_save,
)
from .request_rule_patterns import (
    MAX_AGENT_PATTERN_STATES,
    CompiledSentencePattern,
    MatchBudget,
    PreparedSentenceText,
    SentenceMatchLimitError,
    SentencePatternError,
    compile_sentence_pattern,
    prepare_match_text,
    sentence_capture_names,
    validate_match_input,
)

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 6
STORAGE_KEY_PREFIX = "extended_openai_conversation_responses.request_rules"
MAX_RULES = 500
MAX_PHRASES = 25
MAX_ACTIONS = 20
MAX_SCRIPT_NODES = 500
MAX_SCRIPT_DEPTH = 12
MAX_RULE_NAME_LENGTH = 120
MATCH_TYPES = ("equals", "starts_with", "ends_with", "contains", "sentence_pattern")
ACTION_TYPES = ("local_action", "model_routing")
SLOT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
SLOT_REFERENCE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]{0,63})\}(?!\})")
RESULT_REFERENCE = re.compile(
    r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)(\.[A-Za-z_0-9][A-Za-z0-9_]*)+\}(?!\})"
)
RESERVED_RESULT_ALIASES = {
    "request",
    "conversation",
    "system",
    "trigger",
    "this",
    "repeat",
    "wait",
}
MAX_RESULT_BYTES = 16384
MAX_RESULT_DEPTH = 8
JINJA_SLOT_REFERENCE = re.compile(
    r"\{\{\s*(?:request\.slots\.)?([A-Za-z_][A-Za-z0-9_]{0,63})\s*\}\}"
)
ROUTING_SCOPES = ("request", "conversation")
_REQUEST_RESET_SENTINEL = "__request_rule_reset__"
DEFAULT_MATCHING = {
    "word_forms": True,
    "wording_alternatives": True,
    "fuzzy": False,
    "fuzzy_threshold": 90,
}

# Phrase mappings are deliberately small and directional. Both sides normalize to
# the same canonical wording, which keeps matching predictable and extensible.
DEFAULT_WORDING_GROUPS: tuple[dict[str, Any], ...] = (
    {"canonical": "turn on", "alternatives": ["switch on"]},
    {"canonical": "turn off", "alternatives": ["switch off"]},
    {"canonical": "close", "alternatives": ["shut"]},
    {"canonical": "tv", "alternatives": ["television"]},
    {"canonical": "increase", "alternatives": ["raise", "turn up"]},
    {"canonical": "decrease", "alternatives": ["lower", "turn down"]},
)
SENSITIVE_DOMAINS = {"lock", "alarm_control_panel"}
RequestRuleFunctionExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
_ACTIVE_FUNCTION_EXECUTOR: ContextVar[RequestRuleFunctionExecutor | None] = ContextVar(
    "request_rule_function_executor", default=None
)
_ACTIVE_FUNCTION_RESULTS: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_rule_function_results", default=None
)


@dataclass(frozen=True, slots=True)
class RuleMatch:
    """The single winning rule and how it matched."""

    rule: dict[str, Any]
    phrase: str
    fuzzy: bool
    score: float
    slots: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompiledPhrase:
    """One normalized phrase or compiled ExtendedOpenAI sentence pattern."""

    original: str
    normalized: str | None = None
    sentence_pattern: CompiledSentencePattern | None = None


_MATCH_RANK = {
    "equals": 5,
    "sentence_pattern": 4,
    "starts_with": 3,
    "ends_with": 2,
    "contains": 1,
}


@dataclass(frozen=True, slots=True)
class _MatchingSnapshot:
    """One privately owned generation, published only after compilation finishes."""

    phrases: tuple[tuple[dict[str, Any], dict[str, Any], CompiledPhrase], ...]
    wording_groups: tuple[dict[str, Any], ...]
    deterministic: tuple[
        tuple[dict[str, Any], dict[str, Any], CompiledPhrase], ...
    ] = ()
    fuzzy: tuple[tuple[dict[str, Any], dict[str, Any], CompiledPhrase], ...] = ()


def _match_compiled_sentence(
    compiled: CompiledPhrase,
    prepared: PreparedSentenceText,
    budget: MatchBudget,
) -> dict[str, str] | None:
    """Match one compiled sentence pattern with the shared request budget."""
    if compiled.sentence_pattern is None:
        return None
    result = compiled.sentence_pattern.match_prepared(prepared, budget)
    return None if result is None else dict(result.captures)


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """Result of evaluating one utterance."""

    match: RuleMatch
    consume: bool
    response: str | None = None
    request_override: dict[str, str] | None = None
    successful: bool = True
    provider_input: str | None = None


class _MatchCursor:
    """Continue one immutable matcher snapshot without revisiting earlier phrases."""

    def __init__(self, snapshot: _MatchingSnapshot, text: str) -> None:
        validate_match_input(text)
        self.snapshot = snapshot
        self.text = text
        self.position = 0
        self.normalized: dict[tuple[bool, bool], str] = {}
        self.sentence_text: PreparedSentenceText | None = None
        self.budget = MatchBudget()
        self.seen: set[str] = set()
        self.last_matched_order: int | None = None
        self.fuzzy_matches: list[RuleMatch] | None = None

    def _candidate(self, settings: dict[str, Any]) -> str:
        key = (
            bool(settings.get("word_forms")),
            bool(settings.get("wording_alternatives")),
        )
        if key not in self.normalized:
            self.normalized[key] = normalize_text(
                self.text, settings, self.snapshot.wording_groups
            )
        return self.normalized[key]

    def next_match(self) -> RuleMatch | None:
        """Return the next strict candidate, then ranked fuzzy fallback candidates."""
        phrases = self.snapshot.deterministic
        while self.position < len(phrases):
            rule, settings, compiled = phrases[self.position]
            self.position += 1
            if rule["id"] in self.seen:
                continue
            if compiled.sentence_pattern is not None:
                if self.sentence_text is None:
                    self.sentence_text = prepare_match_text(self.text)
                slots = _match_compiled_sentence(
                    compiled, self.sentence_text, self.budget
                )
                if slots is not None:
                    self.seen.add(rule["id"])
                    self.last_matched_order = rule["order"]
                    return RuleMatch(rule, compiled.original, False, 100.0, slots)
            elif _deterministic_match(
                self._candidate(settings),
                cast(str, compiled.normalized),
                rule["match_type"],
            ):
                self.seen.add(rule["id"])
                self.last_matched_order = rule["order"]
                return RuleMatch(rule, compiled.original, False, 100.0)
        if self.fuzzy_matches is None:
            ranked: dict[str, tuple[tuple[float, int, int], RuleMatch]] = {}
            for rule, settings, compiled in self.snapshot.fuzzy:
                if rule["id"] in self.seen:
                    continue
                score = _fuzzy_score(
                    self._candidate(settings),
                    cast(str, compiled.normalized),
                    rule["match_type"],
                )
                if score < settings["fuzzy_threshold"]:
                    continue
                rank = (score, _MATCH_RANK[rule["match_type"]], -rule["order"])
                previous = ranked.get(rule["id"])
                if previous is None or rank > previous[0]:
                    ranked[rule["id"]] = (
                        rank,
                        RuleMatch(rule, compiled.original, True, score),
                    )
            self.fuzzy_matches = [
                match
                for _, match in sorted(
                    ranked.values(), key=lambda item: item[0], reverse=True
                )
            ]
        if not self.fuzzy_matches:
            return None
        if self.last_matched_order is None:
            result = self.fuzzy_matches.pop(0)
        else:
            later = (
                match
                for match in self.fuzzy_matches
                if match.rule["order"] > self.last_matched_order
            )
            later_result = min(
                later, key=lambda match: match.rule["order"], default=None
            )
            if later_result is None:
                return None
            result = later_result
            self.fuzzy_matches.remove(result)
        self.last_matched_order = result.rule["order"]
        return result


class RequestRuleStore(Store[dict[str, Any]]):
    """Versioned private Home Assistant storage."""

    def __init__(self, hass: HomeAssistant, version: int, key: str) -> None:
        """Initialize private, atomic Request Rule storage."""
        super().__init__(hass, version, key, private=True, atomic_writes=True)

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Migrate additive Request Rule storage changes."""
        if old_major_version == 1:
            return {
                **old_data,
                "wording_groups": _copy_wording_groups(DEFAULT_WORDING_GROUPS),
            }
        if old_major_version in {2, 3, 4, 5}:
            return old_data
        raise NotImplementedError


def _normalize_legacy_consumed_request_scope(value: Any) -> tuple[Any, bool]:
    """Preserve complete routing commands saved with now-meaningless request scope."""
    if not isinstance(value, Mapping):
        return value, False
    if value.get("action_type", "local_action") != "model_routing":
        return value, False
    if value.get("match_type", "equals") not in {"equals", "sentence_pattern"}:
        return value, False
    action = value.get("action")
    if (
        not isinstance(action, Mapping)
        or "continue_to_ai" in action
        or action.get("scope", "request") != "request"
    ):
        return value, False
    normalized = deepcopy(dict(value))
    normalized["action"] = {**dict(action), "scope": "conversation"}
    return normalized, True


class RequestRules:
    """Concurrency-safe persisted rules with precomputed matcher state."""

    def __init__(self, store: RequestRuleStore) -> None:
        self._store = store
        self._rules: list[dict[str, Any]] = []
        self._opaque_fields: dict[str, Any] = {}
        self._committed_opaque_fields: dict[str, Any] = {}
        self._defaults = dict(DEFAULT_MATCHING)
        self._wording_groups = _copy_wording_groups(DEFAULT_WORDING_GROUPS)
        self._groups: list[dict[str, str]] = []
        self._matching_snapshot = _MatchingSnapshot((), ())
        self._has_continuation = False
        self._condition_checkers: dict[str, tuple[list[Any], tuple[Any, ...]]] = {}
        self._diagnostics: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._committed_state: dict[str, Any] | None = None
        self._generation = 0

    async def async_initialize(self) -> None:
        """Load stored rules while preserving newly unsupported patterns for repair."""
        async with self._lock:
            if self._initialized:
                return
            try:
                await _async_repair_private_store_mode(self._store)
                stored = await self._store.async_load()
                migrated = False
                raw_rules: Sequence[Any] = ()
                if stored is None:
                    pass
                elif not isinstance(stored, Mapping):
                    _LOGGER.warning(
                        "Resetting malformed stored Request Rules container"
                    )
                    migrated = True
                else:
                    self._opaque_fields = {
                        key: deepcopy(value)
                        for key, value in stored.items()
                        if key not in {"defaults", "wording_groups", "groups", "rules"}
                    }
                    try:
                        self._groups = validate_rule_groups(stored.get("groups", []))
                    except ValueError:
                        _LOGGER.warning("Ignoring invalid stored Request Rule groups")
                        migrated = True
                    try:
                        self._defaults = validate_matching_settings(
                            stored.get("defaults", DEFAULT_MATCHING)
                        )
                    except ValueError:
                        _LOGGER.warning("Ignoring invalid stored Request Rule defaults")
                        migrated = True
                    try:
                        self._wording_groups = validate_wording_groups(
                            stored.get("wording_groups", DEFAULT_WORDING_GROUPS)
                        )
                    except ValueError:
                        _LOGGER.warning(
                            "Ignoring invalid stored Request Rule wording groups"
                        )
                        migrated = True
                    stored_rules = stored.get("rules", [])
                    if not isinstance(stored_rules, Sequence) or isinstance(
                        stored_rules, (str, bytes)
                    ):
                        _LOGGER.warning("Resetting malformed stored Request Rules list")
                        migrated = True
                    else:
                        if len(stored_rules) > MAX_RULES:
                            _LOGGER.warning(
                                "Stored Request Rules exceed the supported limit; "
                                "keeping the first %d",
                                MAX_RULES,
                            )
                            migrated = True
                        raw_rules = stored_rules[:MAX_RULES]

                seen_ids: set[str] = set()
                for raw in raw_rules:
                    try:
                        candidate, scope_migrated = (
                            _normalize_legacy_consumed_request_scope(raw)
                        )
                        candidate = _assign_missing_result_step_ids(candidate)
                        if scope_migrated:
                            _LOGGER.warning(
                                "Migrating stored complete Request Rule %s from request "
                                "scope to conversation scope",
                                raw.get("id", "<unknown>")
                                if isinstance(raw, Mapping)
                                else "<unknown>",
                            )
                            migrated = True
                        validated = validate_rule(
                            candidate, validate_sentence_pattern=False
                        )
                        if validated["id"] in seen_ids:
                            _LOGGER.warning(
                                "Ignoring duplicate stored Request Rule id: %s",
                                validated["id"],
                            )
                            migrated = True
                            continue
                        seen_ids.add(validated["id"])
                        self._rules.append(validated)
                        migrated = migrated or validated != raw
                    except ValueError as err:
                        _LOGGER.warning("Ignoring invalid stored Request Rule: %s", err)
                        migrated = True
                migrated = self._sort_and_compile() or migrated
                if migrated:
                    await self._async_save_locked()
                self._remember_committed_state()
                self._initialized = True
            except BaseException:
                self._defaults = dict(DEFAULT_MATCHING)
                self._wording_groups = deepcopy(list(DEFAULT_WORDING_GROUPS))
                self._groups = []
                self._rules = []
                self._opaque_fields = {}
                self._sort_and_compile()
                self._initialized = False
                self._committed_state = None
                raise

    def revision(self) -> str:
        """Return a revision that also detects a committed A -> B -> A cycle."""
        payload = json.dumps(
            {
                "generation": self._generation,
                "defaults": self._defaults,
                "wording_groups": self._wording_groups,
                "groups": self._groups,
                "rules": self._rules,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _require_revision_locked(self, expected_revision: str | None) -> None:
        """Reject a stale management writer while the mutation lock is held."""
        if expected_revision is None:
            return
        if not isinstance(expected_revision, str):
            raise ValueError("revision must be a string")
        if expected_revision != self.revision():
            raise ValueError(
                "Request Rules changed in another tab. Reload the latest rules before saving."
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a copy suitable for the management API."""
        return {
            "storage_version": STORAGE_VERSION,
            "revision": self.revision(),
            "defaults": dict(self._defaults),
            "wording_groups": _copy_wording_groups(self._wording_groups),
            "groups": deepcopy(self._groups),
            "rules": [dict(rule) for rule in self._rules],
            "diagnostics": dict(self._diagnostics),
        }

    def function_references(self, function_name: str) -> list[dict[str, str]]:
        """Return references throughout the native script tree."""
        from .function_dependency_integrity import recursive_function_references

        return recursive_function_references(self, function_name)

    async def async_rename_function_reference(
        self, old_name: str, new_name: str, *, expected_revision: str | None = None
    ) -> int:
        """Persist recursive reference changes with rollback on failure."""
        from .function_dependency_integrity import (
            async_rename_function_reference_recursive,
        )

        return await async_rename_function_reference_recursive(
            self, old_name, new_name, expected_revision=expected_revision
        )

    async def async_backup_data(self) -> dict[str, Any]:
        """Return durable Request Rule state without management-only fields."""
        snapshot = self.snapshot()
        snapshot.pop("revision", None)
        snapshot.pop("diagnostics", None)
        return snapshot

    @staticmethod
    def validate_backup_data(value: Any) -> dict[str, Any]:
        """Validate backup state without mutating the live manager."""
        if not isinstance(value, Mapping):
            raise ValueError("request_rules must be an object")
        unknown = set(value) - {
            "storage_version",
            "defaults",
            "wording_groups",
            "groups",
            "rules",
        }
        if unknown:
            raise ValueError("unknown request_rules fields")
        defaults = validate_matching_settings(value.get("defaults", DEFAULT_MATCHING))
        wording_groups = validate_wording_groups(
            value.get("wording_groups", DEFAULT_WORDING_GROUPS)
        )
        groups = validate_rule_groups(value.get("groups", []))
        raw_rules = value.get("rules", [])
        if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, str):
            raise ValueError("request_rules.rules must be a list")
        if len(raw_rules) > MAX_RULES:
            raise ValueError("Request Rule limit reached")
        rules = []
        for item in raw_rules:
            candidate, _ = _normalize_legacy_consumed_request_scope(item)
            rules.append(validate_rule(candidate, validate_sentence_pattern=False))
        if len({rule["id"] for rule in rules}) != len(rules):
            raise ValueError("duplicate Request Rule id")
        if any(
            rule["group_id"]
            and rule["group_id"] not in {group["id"] for group in groups}
            for rule in rules
        ):
            raise ValueError("Request Rule references an unknown group")
        return {
            "defaults": defaults,
            "wording_groups": wording_groups,
            "groups": groups,
            "rules": rules,
        }

    async def async_replace_backup(self, value: Any) -> None:
        """Replace all durable state from a fully validated backup."""
        prepared = self.validate_backup_data(value)
        async with self._lock:
            self._defaults = prepared["defaults"]
            self._wording_groups = prepared["wording_groups"]
            self._groups = prepared["groups"]
            self._rules = [
                validate_rule(
                    _assign_missing_result_step_ids(rule),
                    validate_sentence_pattern=False,
                )
                for rule in prepared["rules"]
            ]
            self._condition_checkers.clear()
            self._sort_and_compile()
            self._initialized = True
            await self._async_save_locked()

    async def async_set_defaults(
        self, value: Any, *, expected_revision: str | None = None
    ) -> dict[str, Any]:
        """Replace global matching defaults."""
        defaults = validate_matching_settings(value)
        async with self._lock:
            self._require_revision_locked(expected_revision)
            self._defaults = defaults
            self._sort_and_compile()
            await self._async_save_locked()
        return dict(defaults)

    async def async_set_wording_groups(
        self, value: Any, *, expected_revision: str | None = None
    ) -> list[dict[str, Any]]:
        """Replace the persisted wording synonym groups."""
        groups = validate_wording_groups(value)
        async with self._lock:
            self._require_revision_locked(expected_revision)
            self._wording_groups = groups
            self._sort_and_compile()
            await self._async_save_locked()
        return _copy_wording_groups(groups)

    async def async_set_settings(
        self,
        defaults_value: Any,
        wording_groups_value: Any,
        *,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Replace Request Rule page settings atomically under one revision."""
        defaults = validate_matching_settings(defaults_value)
        groups = validate_wording_groups(wording_groups_value)
        async with self._lock:
            self._require_revision_locked(expected_revision)
            self._defaults = defaults
            self._wording_groups = groups
            self._sort_and_compile()
            await self._async_save_locked()
            revision = self.revision()
        return {
            "defaults": dict(defaults),
            "wording_groups": _copy_wording_groups(groups),
            "revision": revision,
        }

    async def async_create(
        self, value: Any, *, expected_revision: str | None = None
    ) -> dict[str, Any]:
        """Create one rule."""
        if not isinstance(value, Mapping):
            raise ValueError("rule must be an object")
        async with self._lock:
            self._require_revision_locked(expected_revision)
            if len(self._rules) >= MAX_RULES:
                raise ValueError("Request Rule limit reached")
            raw = dict(value)
            raw.setdefault("id", uuid4().hex)
            raw.setdefault("order", len(self._rules))
            rule = validate_rule(_assign_missing_result_step_ids(raw))
            self._require_group(rule)
            if any(item["id"] == rule["id"] for item in self._rules):
                raise ValueError("rule id already exists")
            _validate_total_pattern_states(
                [*self._rules, rule], inactive_rule_ids=self._diagnostics
            )
            self._rules.append(rule)
            self._sort_and_compile()
            await self._async_save_locked()
        return dict(rule)

    async def async_update(
        self,
        rule_id: str,
        value: Any,
        *,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Replace one rule while preserving its id."""
        if not isinstance(value, Mapping):
            raise ValueError("rule must be an object")
        async with self._lock:
            self._require_revision_locked(expected_revision)
            index = self._index(rule_id)
            raw = dict(value)
            raw["id"] = rule_id
            raw.setdefault("order", self._rules[index]["order"])
            previous = self._rules[index]
            preserve_inactive = (
                rule_id in self._diagnostics
                and raw.get("phrases") == previous["phrases"]
                and raw.get("match_type", "equals") == previous["match_type"]
                and not raw.get("enabled", True)
            )
            rule = validate_rule(
                _assign_missing_result_step_ids(raw),
                validate_sentence_pattern=not preserve_inactive,
            )
            self._require_group(rule)
            prospective = [*self._rules]
            prospective[index] = rule
            _validate_total_pattern_states(
                prospective, inactive_rule_ids=set(self._diagnostics) - {rule_id}
            )
            self._rules[index] = rule
            self._condition_checkers.pop(rule_id, None)
            match_fields = (
                "enabled",
                "phrases",
                "match_type",
                "matching_behavior",
                "matching",
                "order",
            )
            if all(previous[key] == rule[key] for key in match_fields):
                self._refresh_snapshot_rule(rule_id, rule)
            else:
                self._sort_and_compile()
            await self._async_save_locked()
        return dict(rule)

    async def async_delete(
        self, rule_id: str, *, expected_revision: str | None = None
    ) -> bool:
        """Delete one rule."""
        async with self._lock:
            self._require_revision_locked(expected_revision)
            index = self._index(rule_id)
            del self._rules[index]
            self._condition_checkers.pop(rule_id, None)
            self._sort_and_compile()
            await self._async_save_locked()
        return True

    async def async_duplicate(
        self, rule_id: str, *, expected_revision: str | None = None
    ) -> dict[str, Any]:
        """Duplicate one rule immediately after its source."""
        async with self._lock:
            self._require_revision_locked(expected_revision)
            if len(self._rules) >= MAX_RULES:
                raise ValueError("Request Rule limit reached")
            source_index = self._index(rule_id)
            source = deepcopy(self._rules[source_index])
            source.update(
                id=uuid4().hex,
                name=_duplicate_rule_name(source["name"], self._rules),
                order=source_index + 1,
            )
            rule = validate_rule(source)
            prospective = [*self._rules]
            prospective.insert(source_index + 1, rule)
            for order, item in enumerate(prospective):
                item["order"] = order
            _validate_total_pattern_states(
                prospective, inactive_rule_ids=self._diagnostics
            )
            self._rules = prospective
            self._sort_and_compile()
            await self._async_save_locked()
        return dict(rule)

    async def async_move(
        self,
        rule_id: str,
        direction: str,
        *,
        target_rule_id: str | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Move one rule by one position and update matching priority."""
        if direction not in {"up", "down", "top", "bottom", "before", "after"}:
            raise ValueError("direction must be up, down, top, bottom, before or after")
        async with self._lock:
            self._require_revision_locked(expected_revision)
            index = self._index(rule_id)
            if direction in {"before", "after"}:
                if not target_rule_id:
                    raise ValueError("target rule id is required")
                target = self._index(target_rule_id)
            else:
                target = {
                    "up": index - 1,
                    "down": index + 1,
                    "top": 0,
                    "bottom": len(self._rules) - 1,
                }[direction]
            if target < 0 or target >= len(self._rules):
                return dict(self._rules[index])
            if target == index:
                return dict(self._rules[index])
            moved = self._rules.pop(index)
            if direction in {"before", "after"}:
                if index < target:
                    target -= 1
                if direction == "after":
                    target += 1
            self._rules.insert(target, moved)
            for order, rule in enumerate(self._rules):
                rule["order"] = order
            self._reorder_matching_snapshot()
            await self._async_save_locked()
            return dict(self._rules[target])

    def match(
        self, text: str, excluded_ids: frozenset[str] = frozenset()
    ) -> RuleMatch | None:
        """Use list-order deterministic precedence, with fuzzy only as fallback."""
        snapshot = self._matching_snapshot
        validate_match_input(text)
        normalized_candidates: dict[tuple[bool, bool], str] = {}
        sentence_text: PreparedSentenceText | None = None
        sentence_budget = MatchBudget()

        def candidate(settings: dict[str, Any]) -> str:
            key = (
                bool(settings.get("word_forms")),
                bool(settings.get("wording_alternatives")),
            )
            if key not in normalized_candidates:
                normalized_candidates[key] = normalize_text(
                    text, settings, snapshot.wording_groups
                )
            return normalized_candidates[key]

        for rule, settings, compiled in snapshot.deterministic:
            if rule["id"] in excluded_ids:
                continue
            if compiled.sentence_pattern is not None:
                if sentence_text is None:
                    sentence_text = prepare_match_text(text)
                slots = _match_compiled_sentence(
                    compiled, sentence_text, sentence_budget
                )
                if slots is not None:
                    return RuleMatch(rule, compiled.original, False, 100.0, slots)
            elif _deterministic_match(
                candidate(settings), cast(str, compiled.normalized), rule["match_type"]
            ):
                return RuleMatch(rule, compiled.original, False, 100.0)

        fuzzy: list[tuple[tuple[float, int, int], RuleMatch]] = []
        for rule, settings, compiled in snapshot.fuzzy:
            if rule["id"] in excluded_ids:
                continue
            score = _fuzzy_score(
                candidate(settings), cast(str, compiled.normalized), rule["match_type"]
            )
            if score >= settings["fuzzy_threshold"]:
                result = RuleMatch(rule, compiled.original, True, score)
                fuzzy.append(
                    ((score, _MATCH_RANK[rule["match_type"]], -rule["order"]), result)
                )
        return max(fuzzy, key=lambda item: item[0])[1] if fuzzy else None

    async def async_match(self, hass: HomeAssistant, text: str) -> RuleMatch | None:
        match, _ = await self.async_match_with_skipped(hass, text)
        return match

    async def async_match_with_skipped(
        self, hass: HomeAssistant, text: str
    ) -> tuple[RuleMatch | None, list[dict[str, str]]]:
        """Run matching off-loop only when the compiled snapshot has work."""
        if getattr(self.match, "__func__", None) is RequestRules.match:
            skipped: list[dict[str, str]] = []
            async for eligible_match in self.async_eligible_matches(
                hass, text, skipped
            ):
                return eligible_match, skipped
            return None, skipped
        # Preserve the public match seam used by lightweight instrumentation.
        snapshot = self._matching_snapshot
        if not snapshot.deterministic:
            validate_match_input(text)
            return None, []
        executor = getattr(hass, "async_add_executor_job", None)
        excluded: set[str] = set()
        skipped = []
        while True:
            args = (text, frozenset(excluded)) if excluded else (text,)
            match = (
                cast(RuleMatch | None, await executor(self.match, *args))
                if callable(executor)
                else await asyncio.to_thread(self.match, *args)
            )
            if match is None:
                return None, skipped
            if not isinstance(match, RuleMatch):
                # Preserve lightweight matcher instrumentation/test doubles.
                return match, skipped
            if await self._async_conditions_pass(hass, match):
                return match, skipped
            excluded.add(match.rule["id"])
            skipped.append(
                {
                    "id": match.rule["id"],
                    "name": match.rule["name"],
                    "reason": "conditions_false",
                }
            )

    async def _async_conditions_pass(
        self, hass: HomeAssistant, match: RuleMatch
    ) -> bool:
        """Check a text-matched rule, caching native condition checkers."""
        conditions = match.rule.get("conditions", [])
        if not conditions:
            return True
        try:
            cached = self._condition_checkers.get(match.rule["id"])
            if cached is None or cached[0] != conditions:
                checkers = []
                for config in conditions:
                    checked = await ha_condition.async_validate_condition_config(
                        hass, deepcopy(config)
                    )
                    checkers.append(await ha_condition.async_from_config(hass, checked))
                cached = (deepcopy(conditions), tuple(checkers))
                self._condition_checkers[match.rule["id"]] = cached
            for checker in cached[1]:
                outcome = checker.async_check(
                    variables={"request": {"slots": match.slots}, **match.slots}
                )
                if outcome is None:
                    raise HomeAssistantError(
                        "Request Rule condition returned no result"
                    )
                if outcome is False:
                    return False
            return True
        except Exception as err:
            raise HomeAssistantError(
                "Request Rule condition could not be evaluated"
            ) from err

    async def async_eligible_matches(
        self, hass: HomeAssistant, text: str, skipped: list[dict[str, str]]
    ) -> AsyncIterator[RuleMatch]:
        """Stream matches from one snapshot and a single matcher work budget."""
        cursor = _MatchCursor(self._matching_snapshot, text)
        if not cursor.snapshot.deterministic:
            return
        executor = getattr(hass, "async_add_executor_job", None)
        while True:
            match = (
                cast(RuleMatch | None, await executor(cursor.next_match))
                if callable(executor)
                else await asyncio.to_thread(cursor.next_match)
            )
            if match is None:
                return
            if await self._async_conditions_pass(hass, match):
                yield match
            else:
                skipped.append(
                    {
                        "id": match.rule["id"],
                        "name": match.rule["name"],
                        "reason": "conditions_false",
                    }
                )

    def _require_group(self, rule: Mapping[str, Any]) -> None:
        if rule["group_id"] and rule["group_id"] not in {
            group["id"] for group in self._groups
        }:
            raise ValueError("Request Rule group does not exist")

    def _refresh_snapshot_rule(self, rule_id: str, replacement: dict[str, Any]) -> None:
        """Publish a metadata-only change without recompiling every phrase."""
        self._refresh_snapshot_rules({rule_id: replacement})
        self._has_continuation = any(
            rule.get("continue_matching", False)
            for rule, _, _ in self._matching_snapshot.deterministic
        )

    def _refresh_snapshot_rules(
        self, replacements: Mapping[str, dict[str, Any]]
    ) -> None:
        """Update changed rule metadata in one pass over compiled phrases."""
        snapshot = self._matching_snapshot
        published = {rule_id: deepcopy(rule) for rule_id, rule in replacements.items()}

        def replace(
            items: tuple[tuple[dict[str, Any], dict[str, Any], CompiledPhrase], ...],
        ):
            return tuple(
                (
                    published.get(rule["id"], rule),
                    settings,
                    phrase,
                )
                for rule, settings, phrase in items
            )

        self._matching_snapshot = _MatchingSnapshot(
            replace(snapshot.phrases),
            snapshot.wording_groups,
            replace(snapshot.deterministic),
            replace(snapshot.fuzzy),
        )

    def _reorder_matching_snapshot(self) -> None:
        """Publish the new priority order while retaining compiled phrases."""
        snapshot = self._matching_snapshot
        current = {rule["id"]: rule for rule in self._rules}
        published = {rule_id: deepcopy(rule) for rule_id, rule in current.items()}

        def ordered(items):
            return tuple(
                sorted(
                    (
                        (published[rule["id"]], settings, phrase)
                        for rule, settings, phrase in items
                    ),
                    key=lambda item: item[0]["order"],
                )
            )

        phrases = ordered(snapshot.phrases)
        fuzzy = tuple(
            item
            for item in phrases
            if item[2].sentence_pattern is None and item[1]["fuzzy"]
        )
        self._matching_snapshot = _MatchingSnapshot(
            phrases, snapshot.wording_groups, phrases, fuzzy
        )

    async def async_set_groups(
        self, value: Any, *, expected_revision: str | None = None
    ) -> dict[str, Any]:
        """Update organization without changing the global rule order."""
        if not isinstance(value, list):
            raise ValueError("groups must be a list")
        async with self._lock:
            self._require_revision_locked(expected_revision)
            groups = validate_rule_groups(
                [
                    {**item, "id": item.get("id") or uuid4().hex}
                    if isinstance(item, Mapping)
                    else item
                    for item in value
                ]
            )
            valid_ids = {group["id"] for group in groups}
            changed: dict[str, dict[str, Any]] = {}
            for rule in self._rules:
                if rule["group_id"] and rule["group_id"] not in valid_ids:
                    rule["group_id"] = None
                    changed[rule["id"]] = rule
            self._groups = groups
            if changed:
                self._refresh_snapshot_rules(changed)
            await self._async_save_locked()
            return {
                "groups": deepcopy(groups),
                "rules": deepcopy(self._rules),
                "revision": self.revision(),
            }

    def _index(self, rule_id: str) -> int:
        for index, rule in enumerate(self._rules):
            if rule["id"] == rule_id:
                return index
        raise ValueError("Request Rule not found")

    def _sort_and_compile(self) -> bool:
        """Sort, reindex, compile safe rules, and retain diagnostics for unsafe ones."""
        self._rules.sort(
            key=lambda item: (
                item["order"],
                item["name"].casefold(),
                item["id"],
            )
        )
        order_changed = False
        for index, rule in enumerate(self._rules):
            if rule["order"] != index:
                rule["order"] = index
                order_changed = True

        compiled_rules: list[tuple[dict[str, Any], dict[str, Any], CompiledPhrase]] = []
        diagnostics: dict[str, str] = {}
        total_pattern_states = 0
        for stored_rule in self._rules:
            rule = deepcopy(stored_rule)
            settings = (
                self._defaults
                if rule["matching_behavior"] == "defaults"
                else rule["matching"]
            )
            if rule["match_type"] == "sentence_pattern":
                try:
                    phrases = [
                        _compile_sentence_pattern(item) for item in rule["phrases"]
                    ]
                    phrase_slots = [
                        set(
                            cast(
                                CompiledSentencePattern, item.sentence_pattern
                            ).capture_names
                        )
                        for item in phrases
                    ]
                    if any(names != phrase_slots[0] for names in phrase_slots[1:]):
                        raise ValueError(
                            "all sentence variants must capture the same slots"
                        )
                    state_count = sum(
                        cast(CompiledSentencePattern, item.sentence_pattern).state_count
                        for item in phrases
                    )
                    if rule["enabled"] and (
                        total_pattern_states + state_count > MAX_AGENT_PATTERN_STATES
                    ):
                        raise ValueError(
                            "enabled sentence patterns exceed the per-agent compiled "
                            f"state limit of {MAX_AGENT_PATTERN_STATES}"
                        )
                except ValueError as err:
                    diagnostic = f"Sentence pattern is inactive: {err}"
                    diagnostics[rule["id"]] = diagnostic
                    _LOGGER.warning(
                        "Request Rule '%s' is inactive and needs attention in Extended "
                        "OpenAI > Request Rules: %s",
                        rule.get("name") or rule["id"],
                        err,
                    )
                    continue
                if rule["enabled"]:
                    total_pattern_states += state_count
            else:
                phrases = [
                    CompiledPhrase(
                        item, normalize_text(item, settings, self._wording_groups)
                    )
                    for item in rule["phrases"]
                ]

            if rule["enabled"]:
                compiled_rules.extend(
                    (rule, dict(settings), phrase) for phrase in phrases
                )
        fuzzy_rules = tuple(
            (rule, settings, phrase)
            for rule, settings, phrase in compiled_rules
            if phrase.sentence_pattern is None and settings["fuzzy"]
        )
        self._matching_snapshot = _MatchingSnapshot(
            tuple(compiled_rules),
            tuple(_copy_wording_groups(self._wording_groups)),
            tuple(compiled_rules),
            fuzzy_rules,
        )
        self._has_continuation = any(
            rule.get("continue_matching", False) for rule, _, _ in compiled_rules
        )
        self._diagnostics = diagnostics
        return order_changed

    async def _async_save_locked(self) -> None:
        """Settle each Store write before propagating cancellation or rolling back."""
        await _async_settle_transactional_save(
            self._store.async_save(
                {
                    **self._opaque_fields,
                    "defaults": self._defaults,
                    "wording_groups": self._wording_groups,
                    "groups": self._groups,
                    "rules": self._rules,
                }
            ),
            self._restore_committed_state,
            self._remember_committed_state,
        )

    def _remember_committed_state(self) -> None:
        """Capture the exact last committed Request Rule configuration."""
        if self._initialized:
            self._generation += 1
        self._committed_state = {
            "defaults": deepcopy(self._defaults),
            "wording_groups": deepcopy(self._wording_groups),
            "groups": deepcopy(self._groups),
            "rules": deepcopy(self._rules),
        }
        self._committed_opaque_fields = deepcopy(self._opaque_fields)

    def _restore_committed_state(self) -> None:
        snapshot = self._committed_state
        if snapshot is None:
            return
        self._defaults = deepcopy(snapshot["defaults"])
        self._opaque_fields = deepcopy(self._committed_opaque_fields)
        self._wording_groups = deepcopy(snapshot["wording_groups"])
        self._groups = deepcopy(snapshot["groups"])
        self._rules = deepcopy(snapshot["rules"])
        self._sort_and_compile()


class RequestRuleRuntime:
    """Per-agent, in-memory conversation routing overrides."""

    def __init__(self) -> None:
        self._conversation_overrides: dict[str, tuple[dict[str, str], float, int]] = {}

    def get(
        self,
        session_id: str,
        timeout_minutes: int = DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    ) -> dict[str, str]:
        now = monotonic()
        for key, (_, last_used, stored_timeout) in list(
            self._conversation_overrides.items()
        ):
            if now - last_used >= max(1, stored_timeout) * 60:
                self._conversation_overrides.pop(key, None)
        entry = self._conversation_overrides.get(session_id)
        if entry is None:
            return {}
        values, _, _ = entry
        self._conversation_overrides[session_id] = (
            values,
            now,
            max(1, timeout_minutes),
        )
        return dict(values)

    def set(
        self,
        session_id: str,
        override: Mapping[str, str],
        timeout_minutes: int = DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    ) -> None:
        values = {**self.get(session_id, timeout_minutes), **dict(override)}
        self._conversation_overrides[session_id] = (
            values,
            monotonic(),
            max(1, timeout_minutes),
        )

    def reset(self, session_id: str) -> None:
        self._conversation_overrides.pop(session_id, None)

    def effective_options(
        self,
        defaults: Mapping[str, Any],
        session_id: str,
        request_override: Mapping[str, str] | None = None,
        timeout_minutes: int = DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    ) -> dict[str, Any]:
        """Apply documented request > conversation > configured precedence."""
        request_values = dict(request_override or {})
        reset_request = request_values.pop(_REQUEST_RESET_SENTINEL, None) == "1"
        if reset_request:
            return {**defaults, **request_values}
        return {
            **defaults,
            **self.get(session_id, timeout_minutes),
            **request_values,
        }


def request_rule_session_id(continuity_key: str | None, conversation_id: str) -> str:
    """Use the resolved continuity identity, or Core's actual ChatLog identity."""
    return (
        f"continuity:{continuity_key}"
        if continuity_key
        else f"conversation:{conversation_id}"
    )


def validate_matching_settings(value: Any) -> dict[str, Any]:
    """Validate global or custom lightweight matching settings."""
    if not isinstance(value, Mapping):
        raise ValueError("matching settings must be an object")
    unknown = set(value) - set(DEFAULT_MATCHING)
    if unknown:
        raise ValueError("unknown matching settings: " + ", ".join(sorted(unknown)))
    result = {**DEFAULT_MATCHING, **value}
    for key in ("word_forms", "wording_alternatives", "fuzzy"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be true or false")
    threshold = result["fuzzy_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int)
        or not 70 <= threshold <= 100
    ):
        raise ValueError("fuzzy_threshold must be an integer from 70 to 100")
    return result


def validate_wording_groups(value: Any) -> list[dict[str, Any]]:
    """Validate an unambiguous, bounded synonym-group catalog."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("wording_groups must be a list")
    if len(value) > 100:
        raise ValueError("wording_groups may contain at most 100 groups")
    result: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"canonical", "alternatives"}:
            raise ValueError("each wording group needs canonical and alternatives")
        canonical = _clean(item["canonical"], 100, "canonical wording")
        raw_alternatives = item["alternatives"]
        if not isinstance(raw_alternatives, Sequence) or isinstance(
            raw_alternatives, (str, bytes)
        ):
            raise ValueError("wording alternatives must be a list")
        if not raw_alternatives or len(raw_alternatives) > 25:
            raise ValueError("wording alternatives must contain 1 to 25 items")
        alternatives = list(
            dict.fromkeys(
                _clean(item, 100, "alternative wording") for item in raw_alternatives
            )
        )
        terms = [canonical, *alternatives]
        normalized = [_basic_normalize(term) for term in terms]
        if any(not term for term in normalized):
            raise ValueError("wording phrases must contain searchable text")
        if len(set(normalized)) != len(normalized) or claimed.intersection(normalized):
            raise ValueError("wording groups contain an ambiguous duplicate phrase")
        claimed.update(normalized)
        result.append({"canonical": canonical, "alternatives": alternatives})
    return result


def _copy_wording_groups(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "canonical": str(group["canonical"]),
            "alternatives": list(group["alternatives"]),
        }
        for group in value
    ]


def validate_rule_groups(value: Any) -> list[dict[str, str]]:
    """Validate organization metadata independently of precedence."""
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("groups must be a list of at most 100 items")
    groups: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"id", "name"}:
            raise ValueError("each group needs an id and name")
        groups.append(
            {
                "id": _clean(item["id"], 64, "group id"),
                "name": _clean(item["name"], 100, "group name"),
            }
        )
    if len({group["id"] for group in groups}) != len(groups):
        raise ValueError("group ids must be unique")
    return groups


def validate_rule(
    value: Any, *, validate_sentence_pattern: bool = True
) -> dict[str, Any]:
    """Validate and normalize the persisted rule contract."""
    if not isinstance(value, Mapping):
        raise ValueError("rule must be an object")
    allowed = {
        "id",
        "name",
        "enabled",
        "phrases",
        "match_type",
        "action_type",
        "action",
        "matching_behavior",
        "matching",
        "order",
        "slots",
        "conditions",
        "group_id",
        "continue_matching",
        "ai_input_mode",
        "ai_input_capture",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("unknown rule fields: " + ", ".join(sorted(unknown)))
    rule_id = _clean(value.get("id"), 64, "id")
    name = _clean(value.get("name"), MAX_RULE_NAME_LENGTH, "name")
    phrases_value = value.get("phrases")
    if not isinstance(phrases_value, Sequence) or isinstance(phrases_value, str):
        raise ValueError("phrases must be a list")
    phrases = list(dict.fromkeys(_clean(item, 200, "phrase") for item in phrases_value))
    if not phrases or len(phrases) > MAX_PHRASES:
        raise ValueError(f"phrases must contain 1 to {MAX_PHRASES} items")
    match_type = value.get("match_type", "equals")
    if match_type not in MATCH_TYPES:
        raise ValueError("unsupported match type")

    sentence_valid = True
    if match_type == "sentence_pattern":
        if validate_sentence_pattern:
            compiled_phrases = [_compile_sentence_pattern(phrase) for phrase in phrases]
            phrase_slots = [
                set(cast(CompiledSentencePattern, item.sentence_pattern).capture_names)
                for item in compiled_phrases
            ]
        else:
            try:
                phrase_slots = [
                    set(sentence_capture_names(phrase)) for phrase in phrases
                ]
            except SentencePatternError:
                sentence_valid = False
                phrase_slots = []
        if phrase_slots and any(names != phrase_slots[0] for names in phrase_slots[1:]):
            if validate_sentence_pattern:
                raise ValueError("all sentence variants must capture the same slots")
            sentence_valid = False
        if sentence_valid and phrase_slots:
            slot_names = sorted(phrase_slots[0])
        else:
            slot_names = _stored_slot_names(value)
    else:
        slot_names = []
        if any(SLOT_REFERENCE.search(phrase) for phrase in phrases):
            raise ValueError("variable values require Sentence pattern matching")

    action_type = value.get("action_type", "local_action")
    if action_type not in ACTION_TYPES:
        raise ValueError("unsupported action type")
    raw_action = value.get("action", {})
    if (
        action_type == "model_routing"
        and isinstance(raw_action, Mapping)
        and "continue_to_ai" not in raw_action
    ):
        # Preserve historical routing behaviour once for legacy rules. New and
        # edited rules store this choice explicitly, so matching no longer decides
        # whether the provider is called.
        raw_action = {
            **raw_action,
            "continue_to_ai": match_type not in {"equals", "sentence_pattern"},
        }
    action = _validate_action(action_type, raw_action)
    continue_matching = value.get("continue_matching", False)
    if not isinstance(continue_matching, bool):
        raise ValueError("continue_matching must be true or false")
    ai_input_mode = value.get("ai_input_mode", "original")
    if ai_input_mode not in {"original", "capture"}:
        raise ValueError("AI input must be Original request or Captured value")
    ai_input_capture = value.get("ai_input_capture")
    if ai_input_mode == "capture":
        if match_type != "sentence_pattern" or not isinstance(ai_input_capture, str):
            raise ValueError("Captured AI input requires a Sentence Pattern capture")
        if not sentence_valid or ai_input_capture not in slot_names:
            raise ValueError("Captured AI input must exist in every trigger")
        required_phrases = (
            compiled_phrases
            if validate_sentence_pattern
            else [_compile_sentence_pattern(phrase) for phrase in phrases]
        )
        if any(
            ai_input_capture
            not in cast(
                CompiledSentencePattern, item.sentence_pattern
            ).required_capture_names
            for item in required_phrases
        ):
            raise ValueError("Captured AI input must be present on every match")
        if not action["continue_to_ai"]:
            raise ValueError("Captured AI input requires Continue to AI")
    elif ai_input_capture is not None:
        raise ValueError("Original AI input cannot select a capture")
    conditions = value.get("conditions", [])
    if not isinstance(conditions, list) or len(conditions) > MAX_ACTIONS:
        raise ValueError("Only when conditions must be a list of at most 20 conditions")
    _validate_script_complexity(conditions)
    try:
        cv.CONDITIONS_SCHEMA(deepcopy(conditions))
    except Exception as err:
        raise ValueError(f"Invalid Only when condition: {err}") from err
    group_id = value.get("group_id")
    if group_id is not None:
        group_id = _clean(group_id, 64, "group id")
    referenced_slots = _referenced_slots(action) | _legacy_action_slots(raw_action)
    if action_type == "local_action":
        _validate_result_dependencies(action, set(slot_names))
        aliases = {
            step.get("data", {}).get("result_alias")
            for step in action["actions"]
            if isinstance(step, Mapping) and isinstance(step.get("data"), Mapping)
        }
        referenced_slots -= aliases
    unknown_slots = referenced_slots - set(slot_names)
    if unknown_slots and (validate_sentence_pattern or sentence_valid):
        raise ValueError("unknown captured value: " + ", ".join(sorted(unknown_slots)))
    if (
        action_type == "model_routing"
        and not action["continue_to_ai"]
        and not continue_matching
        and action["scope"] == "request"
    ):
        raise ValueError(
            "Request-only routing requires Continue to AI; enable it or use "
            "the rest of the conversation scope"
        )
    behavior = value.get("matching_behavior", "defaults")
    if behavior not in {"defaults", "custom"}:
        raise ValueError("matching_behavior must be defaults or custom")
    matching = validate_matching_settings(value.get("matching", DEFAULT_MATCHING))
    order = value.get("order", 0)
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise ValueError("order must be a non-negative integer")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false")
    return {
        "id": rule_id,
        "name": name,
        "enabled": enabled,
        "phrases": phrases,
        "match_type": match_type,
        "action_type": action_type,
        "action": action,
        "matching_behavior": behavior,
        "matching": matching,
        "order": order,
        "slots": [{"name": item} for item in slot_names],
        "conditions": deepcopy(conditions),
        "group_id": group_id,
        "continue_matching": continue_matching,
        "ai_input_mode": ai_input_mode,
        "ai_input_capture": ai_input_capture,
    }


def _stored_slot_names(value: Mapping[str, Any]) -> list[str]:
    """Recover prior capture metadata when a stored legacy pattern cannot parse."""
    result: list[str] = []
    raw_slots = value.get("slots", [])
    if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
        return result
    for slot in raw_slots:
        if (
            isinstance(slot, Mapping)
            and isinstance(slot.get("name"), str)
            and SLOT_NAME.fullmatch(str(slot["name"]))
        ):
            result.append(str(slot["name"]))
    return sorted(set(result))


def _validate_total_pattern_states(
    rules: Sequence[Mapping[str, Any]], *, inactive_rule_ids: Collection[str] = ()
) -> None:
    """Account in loader order, retaining unchanged inactive rules for repair."""
    total = 0
    for rule in sorted(
        rules, key=lambda item: (item["order"], item["name"].casefold(), item["id"])
    ):
        if not rule.get("enabled") or rule.get("match_type") != "sentence_pattern":
            continue
        phrases = rule.get("phrases", [])
        if not isinstance(phrases, Sequence) or isinstance(phrases, str):
            continue
        try:
            compiled = [compile_sentence_pattern(str(phrase)) for phrase in phrases]
            if any(
                set(item.capture_names) != set(compiled[0].capture_names)
                for item in compiled[1:]
            ):
                raise ValueError("all sentence variants must capture the same slots")
            state_count = sum(item.state_count for item in compiled)
            if total + state_count > MAX_AGENT_PATTERN_STATES:
                raise ValueError(
                    "enabled sentence patterns exceed the per-agent compiled "
                    f"state limit of {MAX_AGENT_PATTERN_STATES}"
                )
        except ValueError:
            if rule.get("id") in inactive_rule_ids:
                continue
            raise
        total += state_count


def _validate_action(action_type: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("action must be an object")
    if action_type == "local_action":
        allowed = {
            "actions",
            "success_response",
            "failure_response",
            "canonical_signature",
            "continue_to_ai",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "unknown local action fields: " + ", ".join(sorted(unknown))
            )
        actions_value = value.get("actions")
        if not isinstance(actions_value, Sequence) or isinstance(actions_value, str):
            raise ValueError("actions must be a list")
        if not actions_value or len(actions_value) > MAX_ACTIONS:
            raise ValueError(f"actions must contain 1 to {MAX_ACTIONS} items")
        actions = _validate_script_sequence(actions_value)
        continue_to_ai = value.get("continue_to_ai", False)
        if not isinstance(continue_to_ai, bool):
            raise ValueError("continue_to_ai must be true or false")
        return {
            "actions": actions,
            "success_response": _clean(
                value.get("success_response", "Done"), 500, "success_response"
            ),
            "failure_response": _clean(
                value.get("failure_response", "Sorry, that did not work"),
                500,
                "failure_response",
            ),
            "canonical_signature": canonical_action_signature(actions),
            "continue_to_ai": continue_to_ai,
        }
    allowed = {
        "model",
        "reasoning_effort",
        "scope",
        "reset",
        "success_response",
        "continue_to_ai",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("unknown model routing fields: " + ", ".join(sorted(unknown)))
    reset = value.get("reset", False)
    if not isinstance(reset, bool):
        raise ValueError("reset must be true or false")
    continue_to_ai = value.get("continue_to_ai", True)
    if not isinstance(continue_to_ai, bool):
        raise ValueError("continue_to_ai must be true or false")
    scope = value.get("scope", "request")
    if scope not in ROUTING_SCOPES:
        raise ValueError("unsupported routing scope")
    model = str(value.get("model") or "").strip()
    effort = str(value.get("reasoning_effort") or "").strip()
    if reset:
        model = ""
        effort = ""
    if not reset and not model and not effort:
        raise ValueError("model routing must set a model or reasoning effort")
    for routing_value in (model, effort):
        if any(marker in routing_value for marker in ("{{", "{%", "{#")):
            raise ValueError(
                "model routing captured values must use simple {name} references"
            )
    model_dynamic = bool(SLOT_REFERENCE.search(model))
    effort_dynamic = bool(SLOT_REFERENCE.search(effort))
    if effort_dynamic and SLOT_REFERENCE.fullmatch(effort) is None:
        raise ValueError("captured reasoning effort must be a single {name} reference")
    if effort and not effort_dynamic:
        if model and not model_dynamic:
            if not get_model_config(model).get("supports_reasoning_effort"):
                raise ValueError(f"model {model} does not support reasoning effort")
            if effort not in get_reasoning_effort_options(model):
                raise ValueError(
                    f"reasoning effort {effort} is not supported by model {model}"
                )
        elif effort not in all_reasoning_efforts():
            # The effective model may come from the configured/conversation route.
            # Accept every currently supported value here; runtime validates it
            # against that effective model before publishing any route change.
            raise ValueError("unsupported reasoning effort")
    return {
        "model": model or None,
        "reasoning_effort": effort or None,
        "scope": scope,
        "reset": reset,
        "continue_to_ai": continue_to_ai,
        "success_response": _clean(
            value.get(
                "success_response",
                "Using the configured defaults" if reset else "Updated",
            ),
            500,
            "success_response",
        ),
    }


def _validate_ha_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("each Home Assistant action must be an object")
    unknown = set(value) - {"domain", "service", "target", "data"}
    if unknown:
        raise ValueError(
            "unknown Home Assistant action fields: " + ", ".join(sorted(unknown))
        )
    domain = _clean(value.get("domain"), 64, "domain")
    service = _clean(value.get("service"), 64, "service")
    result = {
        "action": f"{domain}.{service}",
        "target": _migrate_slot_templates(dict(value.get("target") or {})),
        "data": _migrate_slot_templates(dict(value.get("data") or {})),
    }
    if not re.fullmatch(r"[a-z0-9_]+", domain) or not re.fullmatch(
        r"[a-z0-9_]+", service
    ):
        raise ValueError("action domain and service must use lowercase slugs")
    if not isinstance(value.get("target", {}), Mapping) or not isinstance(
        value.get("data", {}), Mapping
    ):
        raise ValueError("action target and data must be objects")
    return result


def _validate_local_action(value: Any) -> dict[str, Any]:
    """Migrate one legacy HA or configured-function action to native syntax."""
    if isinstance(value, Mapping) and value.get("type") == "function":
        unknown = set(value) - {
            "type",
            "function",
            "arguments",
            "result_alias",
            "step_id",
        }
        if unknown:
            raise ValueError(
                "unknown function action fields: " + ", ".join(sorted(unknown))
            )
        function_name = _clean(value.get("function"), 120, "function")
        arguments = value.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("function arguments must be an object")
        normalized_arguments: dict[str, Any] = {}
        for name, binding in arguments.items():
            if not isinstance(name, str) or not SLOT_NAME.fullmatch(name):
                raise ValueError("function argument names must be simple identifiers")
            if not isinstance(binding, Mapping):
                normalized_arguments[name] = _migrate_slot_templates(binding)
                continue
            source = binding.get("source", "fixed")
            if source == "slot":
                if set(binding) != {"source", "slot"}:
                    raise ValueError("slot arguments need only source and slot")
                slot = binding.get("slot")
                if not isinstance(slot, str) or not SLOT_NAME.fullmatch(slot):
                    raise ValueError("slot argument must name a captured value")
                normalized_arguments[name] = f"{{{{ {slot} }}}}"
            elif source == "fixed":
                if set(binding) - {"source", "value"}:
                    raise ValueError("fixed arguments need only source and value")
                normalized_arguments[name] = _migrate_slot_templates(
                    binding.get("value")
                )
            else:
                raise ValueError("function argument source must be fixed or slot")
        return {
            "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
            "data": {
                "function": function_name,
                "arguments": normalized_arguments,
                **(
                    {"result_alias": value["result_alias"]}
                    if "result_alias" in value
                    else {}
                ),
                **({"step_id": value["step_id"]} if "step_id" in value else {}),
            },
        }
    if isinstance(value, Mapping) and value.get("type") == "home_assistant":
        value = {key: item for key, item in value.items() if key != "type"}
    return _validate_ha_action(value)


def _migrate_slot_templates(value: Any) -> Any:
    """Translate legacy braces into Home Assistant script templates."""
    if isinstance(value, str):
        return SLOT_REFERENCE.sub(lambda match: f"{{{{ {match.group(1)} }}}}", value)
    if isinstance(value, Mapping):
        if set(value) == {"value_from", "slot"} and value.get("value_from") == "slot":
            return f"{{{{ {value['slot']} }}}}"
        return {key: _migrate_slot_templates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_migrate_slot_templates(item) for item in value]
    return value


def _legacy_action_slots(value: Any) -> set[str]:
    """Collect slot references only from recognizably legacy local actions."""
    if not isinstance(value, Mapping):
        return set()
    actions = value.get("actions", [])
    if not isinstance(actions, Sequence) or isinstance(actions, str):
        return set()
    result: set[str] = set()
    for action in actions:
        if not isinstance(action, Mapping) or not (
            "domain" in action or action.get("type") in {"function", "home_assistant"}
        ):
            continue
        result.update(_referenced_slots(action))
    return result


def _validate_script_sequence(value: Sequence[Any]) -> list[dict[str, Any]]:
    """Validate native HA script syntax and enforce conservative size bounds."""
    migrated: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each Home Assistant action must be an object")
        if "domain" in item or item.get("type") in {"function", "home_assistant"}:
            migrated.append(_validate_local_action(item))
        else:
            migrated.append(dict(deepcopy(item)))
    _validate_script_complexity(migrated)
    try:
        cv.SCRIPT_SCHEMA(_mask_script_templates(migrated))
    except Exception as err:
        raise ValueError(f"invalid Home Assistant action sequence: {err}") from err
    return migrated


def _assign_missing_result_step_ids(value: Any) -> Any:
    """Give newly saved result-producing steps identities once at a write boundary."""
    if not isinstance(value, Mapping):
        return value
    rule = deepcopy(dict(value))
    action = rule.get("action")
    if not isinstance(action, dict) or not isinstance(action.get("actions"), list):
        return rule
    for step in action["actions"]:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "function":
            if step.get("result_alias") and not step.get("step_id"):
                step["step_id"] = uuid4().hex
            continue
        data = step.get("data")
        if (
            step.get("action") == f"{DOMAIN}.{SERVICE_CALL_FUNCTION}"
            and isinstance(data, dict)
            and data.get("result_alias")
            and not data.get("step_id")
        ):
            data["step_id"] = uuid4().hex
    return rule


def _result_references(value: Any) -> set[str]:
    if isinstance(value, str):
        return {match.group(1) for match in RESULT_REFERENCE.finditer(value)}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for key, item in value.items():
            if key not in {"result_alias", "step_id"}:
                result.update(_result_references(item))
        return result
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(_result_references(item))
        return result
    return set()


def _validate_result_dependencies(action: Mapping[str, Any], slots: set[str]) -> None:
    produced: set[str] = set()
    step_ids: set[str] = set()
    top_level_ids = {id(step) for step in action["actions"]}
    for nested in _iter_script_actions(action["actions"]):
        data = nested.get("data")
        if (
            isinstance(data, Mapping)
            and data.get("result_alias")
            and id(nested) not in top_level_ids
        ):
            raise ValueError("Function results can be captured only by top-level steps")
    all_aliases = {
        step.get("data", {}).get("result_alias")
        for step in action["actions"]
        if isinstance(step, Mapping) and isinstance(step.get("data"), Mapping)
    }
    for step in action["actions"]:
        if not isinstance(step, Mapping):
            continue
        missing = (
            _result_references(step) | (_referenced_slots(step) & all_aliases)
        ) - produced
        if missing:
            raise ValueError(
                "Function result must be produced by an earlier step: "
                + ", ".join(sorted(missing))
            )
        data = step.get("data", {})
        if step.get("action") != f"{DOMAIN}.{SERVICE_CALL_FUNCTION}" or not isinstance(
            data, Mapping
        ):
            continue
        alias = data.get("result_alias")
        if alias is None:
            continue
        if (
            not isinstance(alias, str)
            or not SLOT_NAME.fullmatch(alias)
            or alias in RESERVED_RESULT_ALIASES
            or alias in slots
        ):
            raise ValueError(
                "Function result alias must be a distinct simple identifier"
            )
        if alias in produced:
            raise ValueError("Function result aliases must be unique within a rule")
        step_id = data.get("step_id")
        if step_id is not None and (
            not isinstance(step_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", step_id)
            or step_id in step_ids
        ):
            raise ValueError("Function result step IDs must be unique")
        if step_id is not None:
            step_ids.add(step_id)
        produced.add(alias)
    missing = (
        _result_references(action["success_response"])
        | (_referenced_slots(action["success_response"]) & all_aliases)
    ) - produced
    if missing:
        raise ValueError(
            "Success response references a missing Function result: "
            + ", ".join(sorted(missing))
        )
    if _result_references(action["failure_response"]) or (
        _referenced_slots(action["failure_response"]) & all_aliases
    ):
        raise ValueError("Failure response cannot reference Function results")


def _mask_script_templates(value: Any, *, key: str | None = None) -> Any:
    """Permit context-free schema validation while preserving stored templates."""
    if isinstance(value, str) and ("{{" in value or "{%" in value or "{#" in value):
        return (
            "homeassistant.update_entity"
            if key in {"action", "service"}
            else "request_rule_template"
        )
    if isinstance(value, Mapping):
        return {
            item_key: _mask_script_templates(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_mask_script_templates(item) for item in value]
    return value


def _validate_script_complexity(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_SCRIPT_DEPTH:
        raise ValueError(f"action sequence exceeds maximum depth {MAX_SCRIPT_DEPTH}")
    if isinstance(value, Mapping):
        total = 1 + sum(
            _validate_script_complexity(item, depth=depth + 1)
            for item in value.values()
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        total = 1 + sum(
            _validate_script_complexity(item, depth=depth + 1) for item in value
        )
    else:
        total = 1
    if total > MAX_SCRIPT_NODES:
        raise ValueError(f"action sequence exceeds {MAX_SCRIPT_NODES} nodes")
    return total


def _referenced_slots(value: Any) -> set[str]:
    """Collect deterministic slot references from a persisted rule value."""
    if isinstance(value, str):
        return set(SLOT_REFERENCE.findall(value))
    if isinstance(value, Mapping):
        if value.get("source") == "slot" and isinstance(value.get("slot"), str):
            return {str(value["slot"])}
        if value.get("value_from") == "slot" and isinstance(value.get("slot"), str):
            return {str(value["slot"])}
        result: set[str] = set()
        for item in value.values():
            result.update(_referenced_slots(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = set()
        for item in value:
            result.update(_referenced_slots(item))
        return result
    return set()


def resolve_slot_values(value: Any, slots: Mapping[str, str]) -> Any:
    """Resolve safe slot references recursively without evaluating templates."""
    if isinstance(value, str):
        return SLOT_REFERENCE.sub(lambda match: slots[match.group(1)], value)
    if isinstance(value, Mapping):
        if set(value) == {"value_from", "slot"} and value.get("value_from") == "slot":
            return slots[str(value["slot"])]
        return {key: resolve_slot_values(item, slots) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_slot_values(item, slots) for item in value]
    return value


def resolve_function_arguments(
    arguments: Mapping[str, Mapping[str, Any]], slots: Mapping[str, str]
) -> dict[str, Any]:
    """Resolve fixed and request-backed function arguments."""
    return {
        name: (
            slots[str(binding["slot"])]
            if binding.get("source") == "slot"
            else binding.get("value")
        )
        for name, binding in arguments.items()
    }


def canonical_action_signature(actions: Sequence[Mapping[str, Any]]) -> str:
    """Stable action identity for future Suggested Local Commands comparisons."""
    import json

    return json.dumps(actions, sort_keys=True, separators=(",", ":"))


def _iter_script_actions(
    sequence: Sequence[Mapping[str, Any]],
    *,
    depth: int = 0,
    budget: list[int] | None = None,
):
    """Yield executable action mappings from native nested script branches."""
    if depth > MAX_SCRIPT_DEPTH:
        raise ValueError(f"action sequence exceeds maximum depth {MAX_SCRIPT_DEPTH}")
    if budget is None:
        budget = [0]

    def nested_sequence(value: Any):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
        if not all(isinstance(child, Mapping) for child in value):
            return None
        return cast(Sequence[Mapping[str, Any]], value)

    for item in sequence:
        budget[0] += 1
        if budget[0] > MAX_SCRIPT_NODES:
            raise ValueError(f"action sequence exceeds {MAX_SCRIPT_NODES} nodes")
        yield item

        for key in ("sequence", "then", "else", "default", "parallel"):
            nested = nested_sequence(item.get(key))
            if nested is not None:
                yield from _iter_script_actions(nested, depth=depth + 1, budget=budget)

        choose = item.get("choose")
        if isinstance(choose, Sequence) and not isinstance(choose, (str, bytes)):
            for branch in choose:
                if not isinstance(branch, Mapping):
                    continue
                nested = nested_sequence(branch.get("sequence"))
                if nested is not None:
                    yield from _iter_script_actions(
                        nested, depth=depth + 1, budget=budget
                    )

        repeat = item.get("repeat")
        if isinstance(repeat, Mapping):
            nested = nested_sequence(repeat.get("sequence"))
            if nested is not None:
                yield from _iter_script_actions(nested, depth=depth + 1, budget=budget)


def rule_has_sensitive_actions(rule: Mapping[str, Any]) -> bool:
    """Flag tolerant matching for sensitive actions anywhere in a script tree."""
    if rule.get("action_type") != "local_action":
        return False
    actions = cast(Mapping[str, Any], rule.get("action", {})).get("actions", [])
    for action in _iter_script_actions(cast(Sequence[Mapping[str, Any]], actions)):
        service_name = str(action.get("action", action.get("service", "")))
        domain, _, service = service_name.casefold().partition(".")
        if domain in SENSITIVE_DOMAINS or (
            domain == "cover" and any(term in service for term in ("open", "close"))
        ):
            return True
    return False


async def async_call_active_function(
    function: str, arguments: Any, result_alias: str | None = None
) -> Any:
    """Execute an integration function in the active Request Rule context."""
    executor = _ACTIVE_FUNCTION_EXECUTOR.get()
    if executor is None:
        raise HomeAssistantError(
            "This action is only available while a Request Rule is running"
        )
    if not isinstance(arguments, Mapping):
        raise HomeAssistantError("Function arguments must be an object")
    result = await executor(function, dict(arguments))
    if result_alias is not None:
        results = _ACTIVE_FUNCTION_RESULTS.get()
        if results is None:
            raise HomeAssistantError("Function results require an active Request Rule")
        from .ha_tool_result_compat import tool_result_data

        payload = tool_result_data(result, default=result)
        if isinstance(payload, Mapping) and "result" in payload:
            payload = payload["result"]
        if isinstance(payload, str):
            with suppress(json.JSONDecodeError):
                payload = json.loads(payload)
        if isinstance(payload, Mapping) and payload.get("status") in {
            "error",
            "denied",
            "unavailable",
        }:
            raise HomeAssistantError("Function Tool returned a failure")
        results[result_alias] = _bounded_function_result(payload)
    return result


def _bounded_function_result(
    value: Any, depth: int = 0, budget: list[int] | None = None
) -> Any:
    """Retain only bounded JSON values; false, zero, empty text and null are valid."""
    if budget is None:
        budget = [0, 0]
    budget[0] += 1
    if budget[0] > MAX_SCRIPT_NODES:
        raise HomeAssistantError("Function result contains too many values")
    if depth > MAX_RESULT_DEPTH:
        raise HomeAssistantError("Function result is too deeply nested")
    result: Any
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HomeAssistantError("Function result object keys must be text")
            budget[1] += len(key.encode("utf-8"))
            if budget[1] > MAX_RESULT_BYTES:
                raise HomeAssistantError("Function result is too large")
            result[key] = _bounded_function_result(item, depth + 1, budget)
    elif isinstance(value, list):
        result = [_bounded_function_result(item, depth + 1, budget) for item in value]
    elif value is None or isinstance(value, (str, int, float, bool)):
        result = value
        budget[1] += len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if budget[1] > MAX_RESULT_BYTES:
            raise HomeAssistantError("Function result is too large")
    else:
        raise HomeAssistantError("Function result must contain JSON values")
    if (
        depth == 0
        and len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        > MAX_RESULT_BYTES
    ):
        raise HomeAssistantError("Function result is too large")
    return result


def resolve_result_values(
    value: Any, slots: Mapping[str, str], results: Mapping[str, Any]
) -> Any:
    """Resolve exact result paths, keeping captures in their own namespace."""

    def lookup(token: str) -> Any:
        alias, *path = token.split(".")
        if alias not in results:
            raise ValueError(f"Function result {alias} is unavailable")
        current = results[alias]
        for part in path:
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            elif (
                isinstance(current, list)
                and part.isdigit()
                and int(part) < len(current)
            ):
                current = current[int(part)]
            else:
                raise ValueError(f"Function result path {token} is unavailable")
        return current

    if isinstance(value, str):
        exact = re.fullmatch(
            r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_0-9][A-Za-z0-9_]*)*)\}", value
        )
        if exact and exact.group(1).split(".")[0] in results:
            return lookup(exact.group(1))
        rendered = RESULT_REFERENCE.sub(
            lambda match: str(lookup(match.group(0)[1:-1])), value
        )
        return SLOT_REFERENCE.sub(
            lambda match: (
                str(results[match.group(1)])
                if match.group(1) in results
                else slots[match.group(1)]
            ),
            rendered,
        )
    if isinstance(value, Mapping):
        return {
            key: resolve_result_values(item, slots, results)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_result_values(item, slots, results) for item in value]
    return value


def _resolve_guest_slot_templates(value: Any, slots: Mapping[str, str]) -> Any:
    """Resolve only deterministic captured-value templates for Guest preflight."""

    def replace_slot(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in slots:
            raise GuestModeDenied(GUEST_MODE_UNAVAILABLE)
        return slots[name]

    if isinstance(value, str):
        rendered = JINJA_SLOT_REFERENCE.sub(replace_slot, value)
        if "{{" in rendered or "{%" in rendered or "{#" in rendered:
            raise GuestModeDenied(GUEST_MODE_UNAVAILABLE)
        return rendered
    if isinstance(value, Mapping):
        return {
            key: _resolve_guest_slot_templates(item, slots)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_guest_slot_templates(item, slots) for item in value]
    return value


def _guest_script_allowed(
    hass: HomeAssistant,
    sequence: Sequence[Mapping[str, Any]],
    policy: GuestCapabilityPolicy,
) -> bool:
    """Preauthorize every executable action before a Guest script can start."""
    for item in _iter_script_actions(sequence):
        action_name = item.get("action", item.get("service"))
        if action_name is not None:
            if not isinstance(action_name, str):
                return False
            if action_name == f"{DOMAIN}.{SERVICE_CALL_FUNCTION}":
                data = item.get("data", {})
                if not isinstance(data, Mapping):
                    return False
                function_name = data.get("function")
                if not isinstance(
                    function_name, str
                ) or not policy.allows_configured_tool(function_name):
                    return False
                continue
            if not guest_arguments_allowed_runtime(
                hass,
                item,
                policy,
                control=True,
                require_entity_selector=True,
            ):
                return False
        elif any(key in item for key in ("device_id", "event", "event_type")):
            return False
    return True


def _resolved_routing_value(value: str, slots: Mapping[str, str], field: str) -> str:
    """Resolve one deterministic captured routing value and reject empty results."""
    resolved = resolve_slot_values(value, slots)
    if not isinstance(resolved, str) or not resolved.strip():
        raise HomeAssistantError(f"Captured routing {field} is empty")
    return resolved.strip()


def _validate_effective_reasoning(
    model: str, effort: str, *, captured: bool = False
) -> None:
    """Validate a resolved reasoning value against the model that will receive it."""
    if not get_model_config(model).get("supports_reasoning_effort"):
        raise HomeAssistantError(f"Model {model} does not support reasoning effort")
    if effort not in get_reasoning_effort_options(model):
        if captured:
            raise HomeAssistantError(f"Unsupported captured reasoning effort: {effort}")
        raise HomeAssistantError(
            f"Reasoning effort {effort} is not supported by model {model}"
        )


def rule_stops_matching(rule: Mapping[str, Any]) -> bool:
    """A provider handoff or an ordinary match is terminal for this utterance."""
    return not rule.get("continue_matching", False) or bool(
        rule["action"].get("continue_to_ai", False)
    )


def rule_provider_input(match: RuleMatch, original_text: str) -> str | None:
    """Resolve only an explicitly selected, validated capture at handoff."""
    rule = match.rule
    if rule.get("ai_input_mode", "original") != "capture":
        return None
    capture = rule.get("ai_input_capture")
    if not isinstance(capture, str):
        raise HomeAssistantError("Captured AI input is unavailable for this request")
    value = match.slots.get(capture)
    if not isinstance(value, str) or not value.strip():
        raise HomeAssistantError("Captured AI input is unavailable for this request")
    # Sentence-ending punctuation is optional matcher syntax, so the capture
    # omits it. Keep it in the provider text when the capture ends the request.
    tail = original_text.rstrip()
    end = len(tail)
    while end and tail[end - 1] in ".!?。؟":
        end -= 1
    if end < len(tail) and tail[:end].endswith(value):
        return value + tail[end:]
    return value


async def async_evaluate_rule(
    hass: HomeAssistant,
    rules: RequestRules,
    runtime: RequestRuleRuntime,
    text: str,
    session_id: str,
    configured_model: str = DEFAULT_CHAT_MODEL,
    guest_policy: GuestCapabilityPolicy | None = None,
    timeout_minutes: int = DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    function_executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
    context: Context | None = None,
) -> RuleEvaluation | None:
    """Apply each eligible rule once, stopping on handoff or failure."""
    if not isinstance(rules, RequestRules) or not rules._has_continuation:
        try:
            match = await rules.async_match(hass, text)
        except SentenceMatchLimitError as err:
            _LOGGER.warning(
                "Skipping Request Rules for bounded matching failure: %s", err
            )
            return None
        if match is None:
            return None
        return await _async_evaluate_matched_rule(
            hass,
            match,
            text,
            runtime,
            session_id,
            configured_model,
            guest_policy,
            timeout_minutes,
            function_executor,
            context,
        )

    last: RuleEvaluation | None = None
    request_override: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    try:
        async for match in rules.async_eligible_matches(hass, text, skipped):
            evaluation = await _async_evaluate_matched_rule(
                hass,
                match,
                text,
                runtime,
                session_id,
                configured_model,
                guest_policy,
                timeout_minutes,
                function_executor,
                context,
                request_override,
            )
            action = match.rule["action"]
            if match.rule["action_type"] == "model_routing":
                if action["reset"]:
                    if action["scope"] == "conversation":
                        request_override.clear()
                    else:
                        request_override = dict(evaluation.request_override or {})
                elif action["scope"] == "conversation":
                    applied = runtime.get(session_id, timeout_minutes)
                    for key, value in (
                        (CONF_CHAT_MODEL, action["model"]),
                        (CONF_REASONING_EFFORT, action["reasoning_effort"]),
                    ):
                        if value:
                            if _REQUEST_RESET_SENTINEL in request_override:
                                request_override[key] = applied[key]
                            else:
                                request_override.pop(key, None)
            if evaluation.request_override:
                request_override.update(evaluation.request_override)
            last = RuleEvaluation(
                evaluation.match,
                evaluation.consume,
                evaluation.response,
                dict(request_override) or None,
                evaluation.successful,
                evaluation.provider_input,
            )
            if not evaluation.successful or rule_stops_matching(match.rule):
                return last
    except SentenceMatchLimitError as err:
        if last is None:
            _LOGGER.warning(
                "Skipping Request Rules for bounded matching failure: %s", err
            )
            return None
        raise HomeAssistantError(
            "Request Rule matching could not safely continue"
        ) from err
    return last


async def _async_evaluate_matched_rule(
    hass: HomeAssistant,
    match: RuleMatch,
    original_text: str,
    runtime: RequestRuleRuntime,
    session_id: str,
    configured_model: str,
    guest_policy: GuestCapabilityPolicy | None,
    timeout_minutes: int,
    function_executor: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
    context: Context | None,
    prior_request_override: Mapping[str, str] | None = None,
) -> RuleEvaluation:
    """Execute one already matched and condition-eligible rule."""
    rule = match.rule
    action = rule["action"]
    if rule["action_type"] == "local_action":
        policy = guest_policy or GuestCapabilityPolicy.unrestricted()
        executable_actions = action["actions"]
        if policy.guest_active:
            try:
                executable_actions = _resolve_guest_slot_templates(
                    executable_actions, match.slots
                )
                allowed = _guest_script_allowed(
                    hass,
                    cast(Sequence[Mapping[str, Any]], executable_actions),
                    policy,
                )
            except GuestModeDenied:
                allowed = False
            except Exception:
                _LOGGER.exception(
                    "Guest authorization failed for Request Rule %s", rule["id"]
                )
                allowed = False
            if not allowed:
                return RuleEvaluation(
                    match, True, GUEST_MODE_UNAVAILABLE, successful=False
                )
        try:
            captures_results = any(
                isinstance(step, Mapping)
                and isinstance(step.get("data"), Mapping)
                and step["data"].get("result_alias")
                for step in executable_actions
            )
            script = None
            if not captures_results:
                schema_actions = cv.SCRIPT_SCHEMA(executable_actions)
                validated_actions = await async_validate_actions_config(
                    hass, schema_actions
                )
                script = Script(
                    hass,
                    validated_actions,
                    f"Request Rule {rule['id']}",
                    DOMAIN,
                    log_exceptions=False,
                )
            token = _ACTIVE_FUNCTION_EXECUTOR.set(function_executor)
            result_values: dict[str, Any] = {}
            result_token = _ACTIVE_FUNCTION_RESULTS.set(result_values)
            try:
                if captures_results:
                    for step in executable_actions:
                        resolved = resolve_result_values(
                            step, match.slots, result_values
                        )
                        one = Script(
                            hass,
                            await async_validate_actions_config(
                                hass, cv.SCRIPT_SCHEMA([resolved])
                            ),
                            f"Request Rule {rule['id']}",
                            DOMAIN,
                            log_exceptions=False,
                        )
                        try:
                            await one.async_run(
                                {
                                    **match.slots,
                                    "request": {"slots": dict(match.slots)},
                                },
                                context,
                            )
                        finally:
                            await one.async_unload()
                else:
                    assert script is not None
                    await script.async_run(
                        {**match.slots, "request": {"slots": dict(match.slots)}},
                        context,
                    )
            finally:
                _ACTIVE_FUNCTION_RESULTS.reset(result_token)
                _ACTIVE_FUNCTION_EXECUTOR.reset(token)
                if script is not None:
                    await script.async_unload()
        except GuestModeDenied:
            return RuleEvaluation(match, True, GUEST_MODE_UNAVAILABLE, successful=False)
        except Exception:
            _LOGGER.exception(
                "Request Rule '%s' failed while running its local Home Assistant "
                "action. Review the rule's actions and referenced entities/services "
                "in Extended OpenAI > Request Rules",
                rule.get("name") or rule["id"],
            )
            return RuleEvaluation(
                match,
                True,
                resolve_slot_values(action["failure_response"], match.slots),
                successful=False,
            )
        if action["continue_to_ai"]:
            return RuleEvaluation(
                match, False, provider_input=rule_provider_input(match, original_text)
            )
        try:
            response = resolve_result_values(
                action["success_response"], match.slots, result_values
            )
        except ValueError, KeyError:
            return RuleEvaluation(
                match,
                True,
                resolve_slot_values(action["failure_response"], match.slots),
                successful=False,
            )
        return RuleEvaluation(match, True, str(response))

    if action["reset"]:
        if action["scope"] == "conversation":
            runtime.reset(session_id)
            request_override = None
        else:
            request_override = {_REQUEST_RESET_SENTINEL: "1"}
        return RuleEvaluation(
            match,
            not action["continue_to_ai"],
            resolve_slot_values(action["success_response"], match.slots),
            request_override,
            provider_input=(
                rule_provider_input(match, original_text)
                if action["continue_to_ai"]
                else None
            ),
        )

    model = (
        _resolved_routing_value(action["model"], match.slots, "model")
        if action["model"]
        else None
    )
    effort = (
        _resolved_routing_value(
            action["reasoning_effort"], match.slots, "reasoning effort"
        )
        if action["reasoning_effort"]
        else None
    )
    prior = dict(prior_request_override or {})
    conversation_override = (
        {}
        if _REQUEST_RESET_SENTINEL in prior
        else runtime.get(session_id, timeout_minutes)
    )
    conversation_override.update(prior)
    selected_model = (
        model or conversation_override.get(CONF_CHAT_MODEL) or configured_model
    )
    if effort:
        captured_effort = bool(
            action["reasoning_effort"]
            and SLOT_REFERENCE.fullmatch(action["reasoning_effort"])
        )
        captured_model = bool(
            action["model"] and SLOT_REFERENCE.search(action["model"])
        )
        _validate_effective_reasoning(
            selected_model,
            effort,
            captured=captured_effort and not captured_model,
        )

    override = {}
    if model:
        override[CONF_CHAT_MODEL] = model
    if effort:
        override[CONF_REASONING_EFFORT] = effort
    combined_override = {**conversation_override, **override}
    combined_model = combined_override.get(CONF_CHAT_MODEL, configured_model)
    combined_effort = combined_override.get(CONF_REASONING_EFFORT)
    if combined_effort and (not effort or combined_model != selected_model):
        _validate_effective_reasoning(combined_model, combined_effort)
    if action["scope"] == "conversation":
        runtime.set(session_id, override, timeout_minutes)
        request_override = None
    else:
        request_override = override
    consume = not action["continue_to_ai"]
    return RuleEvaluation(
        match,
        consume,
        (
            resolve_slot_values(action["success_response"], match.slots)
            if consume
            else None
        ),
        request_override,
        provider_input=(
            rule_provider_input(match, original_text) if not consume else None
        ),
    )


def _basic_normalize(text: str) -> str:
    """Normalize punctuation and spacing without semantic transformations."""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = value.replace("\u2019", "'").replace("-", " ")
    value = re.sub(r"[^\w\s']+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_text(
    text: str,
    settings: Mapping[str, Any],
    wording_groups: Sequence[Mapping[str, Any]] = DEFAULT_WORDING_GROUPS,
) -> str:
    """Apply deterministic, conservative speech-text normalization."""
    value = _basic_normalize(text)
    if settings.get("wording_alternatives"):
        padded = f" {value} "
        replacements = sorted(
            (
                (_basic_normalize(alternative), _basic_normalize(group["canonical"]))
                for group in wording_groups
                for alternative in group["alternatives"]
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for alternative, canonical in replacements:
            padded = padded.replace(f" {alternative} ", f" {canonical} ")
        value = padded.strip()
    if settings.get("word_forms"):
        value = " ".join(_singularize(token) for token in value.split())
    return value


def _compile_sentence_pattern(pattern: str) -> CompiledPhrase:
    """Compile the documented ExtendedOpenAI sentence-pattern syntax."""
    try:
        sentence_pattern = compile_sentence_pattern(pattern)
    except SentencePatternError as err:
        raise ValueError(f"invalid sentence pattern: {err}") from err
    return CompiledPhrase(pattern, sentence_pattern=sentence_pattern)


def _singularize(token: str) -> str:
    if (
        token in {"news", "series", "species"}
        or len(token) <= 3
        or token.endswith(("ss", "us", "is"))
    ):
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes", "ses")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def _deterministic_match(text: str, phrase: str, match_type: str) -> bool:
    if match_type == "equals":
        return text == phrase
    if match_type == "starts_with":
        return text == phrase or text.startswith(phrase + " ")
    if match_type == "ends_with":
        return text == phrase or text.endswith(" " + phrase)
    return text == phrase or f" {phrase} " in f" {text} "


def _fuzzy_score(text: str, phrase: str, match_type: str) -> float:
    if match_type == "equals":
        candidates = [text]
    else:
        words = text.split()
        size = max(1, len(phrase.split()))
        if match_type == "starts_with":
            candidates = [" ".join(words[:size])]
        elif match_type == "ends_with":
            candidates = [" ".join(words[-size:])]
        else:
            candidates = [
                " ".join(words[index : index + size])
                for index in range(max(1, len(words) - size + 1))
            ]
    return max(
        (
            SequenceMatcher(None, phrase, candidate).ratio() * 100
            for candidate in candidates
        ),
        default=0.0,
    )


def _duplicate_rule_name(source_name: str, rules: Sequence[Mapping[str, Any]]) -> str:
    """Return a unique duplicate name without exceeding the persisted limit."""
    existing = {str(rule.get("name", "")).casefold() for rule in rules}
    number = 1
    while True:
        suffix = " copy" if number == 1 else f" copy {number}"
        base = source_name[: MAX_RULE_NAME_LENGTH - len(suffix)].rstrip()
        candidate = f"{base}{suffix}"
        if candidate.casefold() not in existing:
            return candidate
        number += 1


def _clean(value: Any, limit: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    cleaned = value.strip()
    if len(cleaned) > limit:
        raise ValueError(f"{field} is too long")
    return cleaned


_MANAGERS = "extended_openai_conversation_responses.request_rule_managers"
_RUNTIMES = "extended_openai_conversation_responses.request_rule_runtimes"


async def async_get_request_rules(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> RequestRules:
    """Return the shared initialized per-agent rule store."""
    managers = hass.data.setdefault(_MANAGERS, {})
    key = (entry_id, subentry_id)
    if key not in managers:
        managers[key] = RequestRules(
            RequestRuleStore(
                hass,
                STORAGE_VERSION,
                f"{STORAGE_KEY_PREFIX}.{entry_id}.{subentry_id}",
            )
        )
    manager = cast(RequestRules, managers[key])
    await manager.async_initialize()
    return manager


def get_request_rule_runtime(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> RequestRuleRuntime:
    """Return per-agent transient conversation overrides."""
    runtimes = hass.data.setdefault(_RUNTIMES, {})
    return cast(
        RequestRuleRuntime,
        runtimes.setdefault((entry_id, subentry_id), RequestRuleRuntime()),
    )
