"""Tests for unified portable export and selective import/restore planning."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from custom_components.extended_openai_conversation_responses import backup, transfer
from custom_components.extended_openai_conversation_responses.agent_config import (
    agent_config_defaults,
    agent_config_snapshot,
)
from custom_components.extended_openai_conversation_responses.const import (
    AGENT_CONFIG_EXPORT_VERSION,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_PROMPT,
    DOMAIN,
    SERVICE_CALL_FUNCTION,
)
from custom_components.extended_openai_conversation_responses.ha_llm_tools import (
    new_reference_tool,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
    DEFAULT_WORDING_GROUPS,
    RequestRules,
    validate_rule,
)
from custom_components.extended_openai_conversation_responses.secret_redaction import (
    REDACTED_SECRET_SENTINEL,
)
from tests.test_backup import _document


def _rules_backup(rules: list[dict] | None = None) -> dict:
    return {
        "storage_version": 1,
        "defaults": dict(DEFAULT_MATCHING),
        "wording_groups": [deepcopy(group) for group in DEFAULT_WORDING_GROUPS],
        "rules": rules or [],
    }


def _function_rule(
    function_name: str = "remember",
    *,
    arguments: dict | None = None,
    nested: bool = False,
) -> dict:
    function_action = {
        "action": f"{DOMAIN}.{SERVICE_CALL_FUNCTION}",
        "data": {
            "function": function_name,
            "arguments": (
                arguments if arguments is not None else {"fact": "hello"}
            ),
        },
    }
    raw = {
        "id": "remember-rule",
        "name": "Remember this",
        "enabled": True,
        "phrases": ["remember this"],
        "match_type": "equals",
        "action_type": "local_action",
        "action": {
            "actions": (
                [{"parallel": [function_action]}] if nested else [function_action]
            ),
            "success_response": "Done",
            "failure_response": "Failed safely",
        },
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": 0,
    }
    return validate_rule(raw, validate_sentence_pattern=False)


def _remember_tool() -> dict:
    return {
        "enabled": True,
        "spec": {
            "name": "remember",
            "description": "Remember a fact",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
                "additionalProperties": False,
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    }


def _entry_and_subentry(config: dict | None = None):
    return (
        SimpleNamespace(entry_id="entry-1"),
        SimpleNamespace(
            subentry_id="agent-1",
            title="Jarvis",
            data=config or agent_config_defaults(),
        ),
    )


async def test_shareable_setup_reads_only_setup_state_and_round_trips_ha_llm_reference(
    hass,
    monkeypatch,
) -> None:
    reference = {
        "type": "ha_llm",
        "source_type": "platform",
        "source_id": "homeassistant",
        "api_id": "assist",
        "tool_name": "GetLiveContext",
    }
    config = agent_config_defaults()
    config[CONF_FUNCTION_TOOLS] = [new_reference_tool(reference, set())]
    entry, subentry = _entry_and_subentry(config)
    rules = SimpleNamespace(async_backup_data=AsyncMock(return_value=_rules_backup()))

    async def request_rules(_hass, _entry_id, _subentry_id):
        return rules

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("shareable setup accessed private durable state")

    monkeypatch.setattr(transfer, "async_get_request_rules", request_rules)
    for name in (
        "async_get_memory",
        "async_get_temporary_memory",
        "async_get_knowledge",
        "async_get_archive",
        "async_get_usage",
        "async_get_guest_mode",
    ):
        monkeypatch.setattr(transfer, name, forbidden)

    result = await transfer.async_create_setup_export(
        hass, entry, subentry
    )
    document = result["document"]
    assert set(document["sections"]) == transfer.SETUP_SECTIONS

    prepared = transfer.inspect_transfer(document, "target-agent")
    tools = yaml.safe_load(prepared.config[CONF_FUNCTION_TOOLS])
    assert tools[0]["function"] == reference
    assert prepared.request_rules == RequestRules.validate_backup_data(_rules_backup())


async def test_custom_export_reads_only_selected_sections(hass, monkeypatch) -> None:
    entry, subentry = _entry_and_subentry()
    memory = SimpleNamespace(async_backup_data=AsyncMock(return_value={"memories": []}))

    async def get_memory(_hass, _entry_id, _subentry_id):
        return memory

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("unselected manager was accessed")

    monkeypatch.setattr(transfer, "async_get_memory", get_memory)
    for name in (
        "async_get_request_rules",
        "async_get_temporary_memory",
        "async_get_knowledge",
        "async_get_archive",
        "async_get_usage",
        "async_get_guest_mode",
    ):
        monkeypatch.setattr(transfer, name, forbidden)

    document = await transfer.async_collect_transfer_snapshot(
        hass,
        entry,
        subentry,
        mode="custom",
        sections=[transfer.SECTION_PERSISTENT_MEMORY],
    )
    assert set(document["sections"]) == {transfer.SECTION_PERSISTENT_MEMORY}
    memory.async_backup_data.assert_awaited_once()


def test_new_transfer_normalizes_all_detected_secret_markers() -> None:
    document = {
        "format": transfer.TRANSFER_FORMAT,
        "version": transfer.TRANSFER_VERSION,
        "mode": "setup",
        "created_at": "2026-09-08T20:00:00+00:00",
        "integration_version": "5.0.0",
        "agent": {
            "title": "Jarvis",
            "source_entry_id": "entry",
            "source_subentry_id": "agent",
        },
        "sections": {
            transfer.SECTION_CONFIGURATION: {
                "token": "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
                "api_key": "literal-secret",
            },
            transfer.SECTION_REQUEST_RULES: {
                "rules": [],
                "authorization": "Bearer abcdefghijklmnopqrstuvwxyz123456",
            },
        },
    }
    redacted = transfer.redact_transfer_document(document)
    serialized = yaml.safe_dump(redacted)
    assert "literal-secret" not in serialized
    assert "sk-proj-" not in serialized
    assert "Bearer abc" not in serialized
    assert "[redacted]" not in serialized
    assert "__extended_openai_redacted_secret__" in serialized


@pytest.mark.parametrize("placeholder", [REDACTED_SECRET_SENTINEL, "[redacted]"])
def test_secret_placeholder_preserves_destination_value(placeholder) -> None:
    restored, preserved, missing = transfer._restore_section_secrets(
        {"prompt": placeholder}, {"prompt": "destination secret"}
    )
    assert restored == {"prompt": "destination secret"}
    assert preserved == ("prompt",)
    assert missing == ()


def test_missing_secret_placeholder_is_reported_and_not_restored_literally() -> None:
    restored, preserved, missing = transfer._restore_section_secrets(
        {"api_key": REDACTED_SECRET_SENTINEL}, {}
    )
    assert restored == {}
    assert preserved == ()
    assert missing == ("api_key",)


def test_legacy_setup_export_is_still_classified() -> None:
    config = agent_config_defaults()
    prepared = transfer.inspect_transfer(
        {
            "schema": transfer.LEGACY_AGENT_SCHEMA,
            "version": AGENT_CONFIG_EXPORT_VERSION,
            "title": "Legacy Jarvis",
            "config": config,
        },
        "target-agent",
    )
    assert prepared.source_kind == "legacy_setup"
    assert prepared.title == "Legacy Jarvis"
    assert prepared.available_sections == frozenset({transfer.SECTION_CONFIGURATION})


def test_current_full_backup_is_classified_without_inventing_absent_sections() -> None:
    document = _document()
    prepared = transfer.inspect_transfer(document, "target-agent")
    assert prepared.source_kind == "full_backup"
    assert transfer.SECTION_CONFIGURATION in prepared.available_sections
    assert transfer.SECTION_REQUEST_RULES in prepared.available_sections
    assert transfer.SECTION_GUEST_MODE not in prepared.available_sections
    assert transfer.SECTION_PERSISTENT_MEMORY in prepared.available_sections


@pytest.mark.parametrize("snapshot_config", [False, True])
async def test_selective_restore_replaces_only_selected_section(
    monkeypatch, snapshot_config
) -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    expected_config = deepcopy(current.config)
    if snapshot_config:
        current = replace(current, config=agent_config_snapshot(current.config))
    imported_memory = deepcopy(current.memories)
    imported_memory[0] = replace(imported_memory[0], content="Imported memory")
    imported = transfer.PreparedTransfer(
        source_kind="custom_backup",
        mode="custom",
        title="Imported title",
        available_sections=frozenset({transfer.SECTION_PERSISTENT_MEMORY}),
        created_at="2026-09-08T20:00:00+00:00",
        integration_version="5.0.0",
        memories=imported_memory,
    )
    monkeypatch.setattr(transfer, "_current_snapshot", AsyncMock(return_value=current))
    entry, subentry = _entry_and_subentry()

    target, preview = await transfer.async_materialize_restore(
        SimpleNamespace(),
        entry,
        subentry,
        imported,
        sections=[transfer.SECTION_PERSISTENT_MEMORY],
    )
    assert target.memories[0].content == "Imported memory"
    assert target.title == current.title
    assert target.config == expected_config
    assert isinstance(target.config[CONF_FUNCTION_TOOLS], str)
    assert target.knowledge == current.knowledge
    assert preview["selected_sections"] == [transfer.SECTION_PERSISTENT_MEMORY]


async def test_request_rule_dependency_uses_combined_target_state(
    hass, monkeypatch
) -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    rules = RequestRules.validate_backup_data(_rules_backup([_function_rule()]))
    imported = transfer.PreparedTransfer(
        source_kind="custom_backup",
        mode="custom",
        title="Rules",
        available_sections=frozenset({transfer.SECTION_REQUEST_RULES}),
        created_at="2026-09-08T20:00:00+00:00",
        integration_version="5.0.0",
        request_rules=rules,
    )
    monkeypatch.setattr(transfer, "_current_snapshot", AsyncMock(return_value=current))
    entry, subentry = _entry_and_subentry()

    with pytest.raises(
        backup.BackupError, match="unavailable or disabled: remember"
    ):
        await transfer.async_materialize_restore(hass, entry, subentry, imported)

    current.config[CONF_FUNCTION_TOOLS] = yaml.safe_dump(
        [_remember_tool()], sort_keys=False
    )
    current.config[CONF_FUNCTION_GROUPS] = []
    target, _preview = await transfer.async_materialize_restore(
        hass, entry, subentry, imported
    )
    assert target.request_rules["rules"][0]["id"] == "remember-rule"


async def test_request_rule_dependency_checks_nested_actions(hass, monkeypatch) -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    rules = RequestRules.validate_backup_data(
        _rules_backup([_function_rule(nested=True)])
    )
    imported = transfer.PreparedTransfer(
        source_kind="custom_backup",
        mode="custom",
        title="Nested rules",
        available_sections=frozenset({transfer.SECTION_REQUEST_RULES}),
        created_at="2026-09-08T20:00:00+00:00",
        integration_version="5.0.0",
        request_rules=rules,
    )
    monkeypatch.setattr(transfer, "_current_snapshot", AsyncMock(return_value=current))
    entry, subentry = _entry_and_subentry()

    with pytest.raises(
        backup.BackupError, match="unavailable or disabled: remember"
    ):
        await transfer.async_materialize_restore(hass, entry, subentry, imported)


async def test_request_rule_dependency_validates_nested_static_arguments(
    hass, monkeypatch
) -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    current.config[CONF_FUNCTION_TOOLS] = yaml.safe_dump(
        [_remember_tool()], sort_keys=False
    )
    current.config[CONF_FUNCTION_GROUPS] = []
    rules = RequestRules.validate_backup_data(
        _rules_backup([_function_rule(arguments={"fact": 42}, nested=True)])
    )
    imported = transfer.PreparedTransfer(
        source_kind="custom_backup",
        mode="custom",
        title="Nested rules",
        available_sections=frozenset({transfer.SECTION_REQUEST_RULES}),
        created_at="2026-09-08T20:00:00+00:00",
        integration_version="5.0.0",
        request_rules=rules,
    )
    monkeypatch.setattr(transfer, "_current_snapshot", AsyncMock(return_value=current))
    entry, subentry = _entry_and_subentry()

    with pytest.raises(backup.BackupError, match="Function input `fact` must be string"):
        await transfer.async_materialize_restore(hass, entry, subentry, imported)


async def test_request_rule_dependency_accepts_tool_imported_with_rule(
    hass, monkeypatch
) -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    imported_config = deepcopy(current.config)
    imported_config[CONF_FUNCTION_TOOLS] = yaml.safe_dump(
        [_remember_tool()], sort_keys=False
    )
    rules = RequestRules.validate_backup_data(
        _rules_backup([_function_rule(nested=True)])
    )
    imported = transfer.PreparedTransfer(
        source_kind="custom_backup",
        mode="custom",
        title="Imported setup",
        available_sections=frozenset(
            {transfer.SECTION_CONFIGURATION, transfer.SECTION_REQUEST_RULES}
        ),
        created_at="2026-09-08T20:00:00+00:00",
        integration_version="5.0.0",
        config=imported_config,
        request_rules=rules,
    )
    monkeypatch.setattr(transfer, "_current_snapshot", AsyncMock(return_value=current))
    entry, subentry = _entry_and_subentry()

    target, _preview = await transfer.async_materialize_restore(
        hass, entry, subentry, imported
    )

    assert yaml.safe_load(target.config[CONF_FUNCTION_TOOLS])[0]["spec"]["name"] == "remember"
    assert target.request_rules["rules"][0]["id"] == "remember-rule"


async def test_configuration_secret_placeholder_uses_current_destination(
    monkeypatch,
) -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    current.config[CONF_PROMPT] = "Bearer sk-destination-secret-value-1234567890"
    raw = deepcopy(current.config)
    raw[CONF_PROMPT] = REDACTED_SECRET_SENTINEL
    imported = transfer.PreparedTransfer(
        source_kind="portable_transfer",
        mode="setup",
        title="Imported title",
        available_sections=frozenset({transfer.SECTION_CONFIGURATION}),
        created_at="2026-09-08T20:00:00+00:00",
        integration_version="5.0.0",
        config=deepcopy(current.config),
        raw_configuration=raw,
        redacted_sensitive_fields=(f"configuration.{CONF_PROMPT}",),
    )
    monkeypatch.setattr(transfer, "_current_snapshot", AsyncMock(return_value=current))
    entry, subentry = _entry_and_subentry()

    target, preview = await transfer.async_materialize_restore(
        SimpleNamespace(), entry, subentry, imported
    )
    assert target.config[CONF_PROMPT] == current.config[CONF_PROMPT]
    assert preview["preserved_sensitive_field_count"] == 1
    assert preview["missing_sensitive_field_count"] == 0


def test_section_selection_rejects_empty_and_unknown_values() -> None:
    with pytest.raises(backup.BackupError, match="Select at least one"):
        transfer.validate_section_selection([])
    with pytest.raises(backup.BackupError, match="Unknown transfer section"):
        transfer.validate_section_selection(["not-a-section"])


def _portable_document(*, mode: str = "custom") -> dict:
    source = _document()
    sections = {
        transfer.SECTION_CONFIGURATION: source["agent"]["config"],
        transfer.SECTION_REQUEST_RULES: source["request_rules"],
        transfer.SECTION_PERSISTENT_MEMORY: source["memories"],
        transfer.SECTION_TEMPORARY_MEMORY: source["temporary_memories"],
        transfer.SECTION_KNOWLEDGE: source["knowledge"],
        transfer.SECTION_CONVERSATION_ARCHIVE: source["archive"],
        transfer.SECTION_USAGE: source["usage"],
        transfer.SECTION_GUEST_MODE: {"schedule": None},
    }
    if mode == "setup":
        sections = {
            key: sections[key]
            for key in (transfer.SECTION_CONFIGURATION, transfer.SECTION_REQUEST_RULES)
        }
    return {
        "format": transfer.TRANSFER_FORMAT,
        "version": transfer.TRANSFER_VERSION,
        "mode": mode,
        "created_at": "2026-09-13T12:00:00+00:00",
        "integration_version": "5.0.0",
        "agent": {
            "title": "Coverage agent",
            "source_entry_id": "entry-source",
            "source_subentry_id": "agent-source",
        },
        "sections": sections,
    }


def test_selective_request_rules_transfer_keeps_captured_ai_input() -> None:
    document = _portable_document(mode="setup")
    rule = {
        "id": "ask-rule",
        "name": "Ask",
        "enabled": True,
        "phrases": ["ask {question}"],
        "match_type": "sentence_pattern",
        "action_type": "model_routing",
        "action": {
            "model": "gpt-5",
            "reasoning_effort": "",
            "scope": "request",
            "reset": False,
            "continue_to_ai": True,
        },
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": 0,
        "continue_matching": True,
        "ai_input_mode": "capture",
        "ai_input_capture": "question",
    }
    document["sections"][transfer.SECTION_REQUEST_RULES] = _rules_backup([rule])
    prepared = transfer.inspect_transfer(document, "target-agent")
    imported = prepared.request_rules["rules"][0]
    assert imported["ai_input_mode"] == "capture"
    assert imported["ai_input_capture"] == "question"
    assert imported["continue_matching"] is True


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ("configuration", "must be a list"),
        (42, "must be a list"),
        ([transfer.SECTION_CONFIGURATION, 42], "contain names only"),
    ],
)
def test_section_selection_rejects_ambiguous_types(sections, message) -> None:
    with pytest.raises(backup.BackupError, match=message):
        transfer.validate_section_selection(sections)


async def test_custom_snapshot_collects_every_selected_manager(hass, monkeypatch) -> None:
    entry, subentry = _entry_and_subentry()
    monkeypatch.setattr(transfer, "agent_config_snapshot", lambda value: dict(value))
    payloads = {
        "async_get_request_rules": _rules_backup(),
        "async_get_memory": {"memories": []},
        "async_get_temporary_memory": {"records": []},
        "async_get_knowledge": {"sources": []},
        "async_get_archive": {"sessions": [], "turns": []},
        "async_get_usage": {
            "totals": {},
            "daily": {},
            "requests": [],
            "runs": [],
        },
        "async_get_guest_mode": {"schedule": None},
    }
    managers = {}
    for getter_name, payload in payloads.items():
        manager = SimpleNamespace(async_backup_data=AsyncMock(return_value=payload))
        managers[getter_name] = manager

        async def getter(_hass, _entry_id, _subentry_id, *, _manager=manager):
            return _manager

        monkeypatch.setattr(transfer, getter_name, getter)

    result = await transfer.async_collect_transfer_snapshot(
        hass,
        entry,
        subentry,
        mode="custom",
        sections=transfer.ALL_SECTIONS,
    )

    assert set(result["sections"]) == transfer.ALL_SECTIONS
    assert result["sections"][transfer.SECTION_CONFIGURATION]
    for manager in managers.values():
        manager.async_backup_data.assert_awaited_once()


async def test_snapshot_rejects_unknown_mode() -> None:
    entry, subentry = _entry_and_subentry()
    with pytest.raises(backup.BackupError, match="mode must be setup or custom"):
        await transfer.async_collect_transfer_snapshot(
            SimpleNamespace(), entry, subentry, mode="full"
        )


def test_setup_export_enforces_serialized_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(backup, "MAX_LEGACY_EXPORT_BYTES", 1)
    with pytest.raises(backup.BackupError, match="16 MB safety limit"):
        transfer.finalize_setup_export(_portable_document(mode="setup"))


def test_secret_restore_matches_reordered_items_by_identity() -> None:
    imported = [
        {"id": "first", "token": REDACTED_SECRET_SENTINEL},
        {"id": "second", "token": REDACTED_SECRET_SENTINEL},
    ]
    destination = [
        {"id": "second", "token": "second-secret"},
        {"id": "first", "token": "first-secret"},
    ]

    restored, preserved, missing = transfer._restore_section_secrets(
        imported, destination
    )

    assert restored == [
        {"id": "first", "token": "first-secret"},
        {"id": "second", "token": "second-secret"},
    ]
    assert preserved == ("[0].token", "[1].token")
    assert missing == ()


def test_secret_restore_rejects_incompatible_or_duplicate_list_context() -> None:
    imported = [{"id": "duplicate", "token": REDACTED_SECRET_SENTINEL}]
    destination = [
        {"id": "duplicate", "token": "one"},
        {"id": "duplicate", "token": "two"},
    ]

    restored, preserved, missing = transfer._restore_section_secrets(
        imported, destination
    )

    assert restored == [{"id": "duplicate"}]
    assert preserved == ()
    assert missing == ("[0].token",)
    assert transfer._collect_secret_paths(REDACTED_SECRET_SENTINEL) == ["value"]
    assert not transfer._compatible_secret_context([], {})


def test_secret_restore_drops_unmatched_list_marker() -> None:
    restored, preserved, missing = transfer._restore_section_secrets(
        [REDACTED_SECRET_SENTINEL], []
    )

    assert restored == []
    assert preserved == ()
    assert missing == ("[0]",)


def test_function_tools_match_secrets_by_stable_spec_name() -> None:
    imported = [{"spec": {"name": "weather"}, "token": REDACTED_SECRET_SENTINEL}]
    destination = [{"spec": {"name": "weather"}, "token": "destination"}]

    restored, preserved, missing = transfer._restore_section_secrets(
        imported, destination
    )

    assert restored[0]["token"] == "destination"
    assert preserved == ("[0].token",)
    assert missing == ()
    assert transfer._fallback_list_index({"id": ""}, [{"id": ""}]) is None


def test_custom_transfer_validates_every_supported_section() -> None:
    prepared = transfer.inspect_transfer(_portable_document(), "target-agent")

    assert prepared.available_sections == transfer.ALL_SECTIONS
    assert prepared.memories and prepared.memories[0].memory_id == "memory-1"
    assert prepared.temporary_memories
    assert prepared.knowledge and prepared.knowledge[0].source_id == "source-1"
    assert prepared.archive_sessions == []
    assert prepared.usage_requests[0].agent_subentry_id == "target-agent"
    assert prepared.guest_mode_schedule is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("created_at"), "incomplete or corrupted"),
        (lambda value: value.update(created_at="not-a-date"), "metadata is invalid"),
        (lambda value: value.update(integration_version=""), "metadata is invalid"),
        (lambda value: value.update(agent=[]), "agent metadata is invalid"),
        (
            lambda value: value["agent"].update(extra="field"),
            "agent metadata is invalid",
        ),
        (lambda value: value["agent"].update(title=""), "agent name is invalid"),
        (
            lambda value: value["agent"].update(source_entry_id=3),
            "agent identity is invalid",
        ),
        (lambda value: value.update(version=0), "unsupported format version"),
        (lambda value: value.update(mode="full"), "transfer mode is invalid"),
        (lambda value: value.update(sections=[]), "transfer sections are invalid"),
        (
            lambda value: value["sections"].pop(transfer.SECTION_REQUEST_RULES),
            "setup sections are incomplete",
        ),
        (
            lambda value: value["sections"].update(
                {transfer.SECTION_CONFIGURATION: []}
            ),
            "configuration must be an object",
        ),
    ],
)
def test_portable_transfer_rejects_corrupted_boundaries(mutation, message) -> None:
    document = _portable_document(mode="setup")
    mutation(document)
    with pytest.raises(backup.BackupError, match=message):
        transfer.inspect_transfer(document, "target-agent")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("{", "not valid JSON"),
        ([], "not an Extended OpenAI Conversation transfer"),
        ({"unrelated": True}, "not a recognised"),
    ],
)
def test_inspection_rejects_unrecognised_inputs(value, message) -> None:
    with pytest.raises(backup.BackupError, match=message):
        transfer.inspect_transfer(value, "target-agent")


def test_inspection_enforces_raw_json_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(backup, "MAX_BACKUP_BYTES", 1)
    with pytest.raises(backup.BackupError, match="128 MB safety limit"):
        transfer.inspect_transfer(json.dumps({}), "target-agent")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(version=0), "unsupported version"),
        (lambda value: value.update(extra=True), "unknown fields"),
        (lambda value: value.update(config=[]), "setup export is invalid"),
    ],
)
def test_legacy_setup_rejects_corrupted_boundaries(mutation, message) -> None:
    document = {
        "schema": transfer.LEGACY_AGENT_SCHEMA,
        "version": transfer.AGENT_CONFIG_EXPORT_VERSION,
        "title": "Legacy",
        "config": agent_config_defaults(),
    }
    mutation(document)
    with pytest.raises(backup.BackupError, match=message):
        transfer.inspect_transfer(document, "target-agent")


def test_full_backup_classifies_optional_guest_mode() -> None:
    document = _document()
    document["guest_mode"] = {"schedule": None}

    prepared = transfer.inspect_transfer(document, "target-agent")

    assert transfer.SECTION_GUEST_MODE in prepared.available_sections


def test_older_full_backup_does_not_invent_newer_sections() -> None:
    document = _document()
    document["version"] = 2
    document.pop("request_rules")

    prepared = transfer.inspect_transfer(document, "target-agent")

    assert transfer.SECTION_REQUEST_RULES not in prepared.available_sections
    assert prepared.request_rules is None


@pytest.mark.parametrize("section", ["configuration", "request_rules"])
def test_selected_setup_section_must_have_materialized_payload(section) -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    imported = transfer.PreparedTransfer(
        source_kind="portable_transfer",
        mode="setup",
        title="Imported",
        available_sections=frozenset({section}),
        created_at=None,
        integration_version=None,
    )

    with pytest.raises(backup.BackupError, match="unavailable"):
        transfer._prepared_restore_from_selection(
            current, imported, frozenset({section})
        )


def test_selected_raw_configuration_must_restore_to_mapping() -> None:
    current = backup.inspect_backup(_document(), "target-agent")
    imported = transfer.PreparedTransfer(
        source_kind="portable_transfer",
        mode="setup",
        title="Imported",
        available_sections=frozenset({transfer.SECTION_CONFIGURATION}),
        created_at=None,
        integration_version=None,
        raw_configuration=[],
    )

    with pytest.raises(backup.BackupError, match="configuration is invalid"):
        transfer._prepared_restore_from_selection(
            current, imported, frozenset({transfer.SECTION_CONFIGURATION})
        )


async def test_dependency_validation_skips_non_rule_records(monkeypatch) -> None:
    validate = AsyncMock()
    monkeypatch.setattr(transfer, "async_validate_request_rule_functions", validate)
    monkeypatch.setattr(
        transfer, "configured_function_tools_from_data", lambda _data: []
    )
    rule = {"id": "valid-shape"}

    await transfer._async_validate_request_rule_function_dependencies(
        SimpleNamespace(), {"rules": [None, rule]}, agent_config_defaults()
    )

    validate.assert_awaited_once()
    assert validate.await_args.args[1] is rule


def test_frontend_inspection_reports_create_agent_eligibility() -> None:
    prepared = transfer.inspect_transfer(_portable_document(mode="setup"), "target")

    result = transfer.inspection_for_frontend(prepared)

    assert result["valid"] is True
    assert result["can_create_new_agent"] is True
    assert result["available_sections"] == list(transfer.SECTION_ORDER[:2])


def test_literal_text_context_requires_exact_destination_value() -> None:
    literal = {transfer.LITERAL_TEXT_KEY: "keep this literal"}

    assert transfer._compatible_secret_context(literal, "keep this literal")
    assert not transfer._compatible_secret_context(literal, "different")
    restored, preserved, missing = transfer._restore_section_secrets(
        {CONF_PROMPT: literal}, {CONF_PROMPT: "different"}
    )
    assert restored == {CONF_PROMPT: "keep this literal"}
    assert preserved == missing == ()
    assert transfer._normalize_redaction_placeholders(literal) == literal
    assert transfer._collect_secret_paths(literal) == []


def test_redaction_handles_partial_or_non_mapping_sections() -> None:
    document = _portable_document(mode="setup")
    document["sections"] = "not-a-mapping"
    assert transfer.redact_transfer_document(document)["sections"] == "not-a-mapping"

    document["sections"] = {transfer.SECTION_CONFIGURATION: {"api_key": "ordinary"}}
    redacted = transfer.redact_transfer_document(document)["sections"]
    assert redacted[transfer.SECTION_CONFIGURATION]["api_key"] == dict(
        REDACTED_SECRET_SENTINEL
    )

    document["sections"] = {transfer.SECTION_REQUEST_RULES: {"rules": []}}
    assert (
        transfer.redact_transfer_document(document)["sections"] == document["sections"]
    )


def test_redaction_placeholder_detection_accepts_structured_and_legacy_forms() -> None:
    assert transfer._is_redacted_placeholder(REDACTED_SECRET_SENTINEL)
    assert transfer._is_redacted_placeholder("[redacted]")
    assert not transfer._is_redacted_placeholder({"ordinary": "mapping"})
    assert not transfer._is_redacted_placeholder("ordinary value")


def test_future_transfer_version_is_rejected_cleanly() -> None:
    document = {
        "format": transfer.TRANSFER_FORMAT,
        "version": transfer.TRANSFER_VERSION + 1,
        "mode": "setup",
        "created_at": "2026-09-08T20:00:00+00:00",
        "integration_version": "99.0.0",
        "agent": {
            "title": "Future agent",
            "source_entry_id": "entry",
            "source_subentry_id": "agent",
        },
        "sections": {
            transfer.SECTION_CONFIGURATION: {},
            transfer.SECTION_REQUEST_RULES: {},
        },
    }

    with pytest.raises(backup.BackupError, match="newer unsupported format"):
        transfer.inspect_transfer(document, "target-agent")


async def test_setup_export_reads_manifest_outside_event_loop(
    hass, monkeypatch
) -> None:
    import asyncio
    from pathlib import Path
    import threading

    hass.async_add_executor_job.side_effect = asyncio.to_thread
    event_loop_thread = threading.get_ident()
    original_read = Path.read_text
    reads = []

    def read_text(path, *args, **kwargs):
        if path.name == "manifest.json":
            assert threading.get_ident() != event_loop_thread
            reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(
        transfer,
        "async_get_request_rules",
        AsyncMock(
            return_value=SimpleNamespace(
                async_backup_data=AsyncMock(return_value=_rules_backup())
            )
        ),
    )
    entry, subentry = _entry_and_subentry()
    result = await transfer.async_create_setup_export(hass, entry, subentry)
    assert len(reads) == 1
    assert (
        result["document"]["integration_version"]
        == json.loads(original_read(reads[0]))["version"]
    )
    assert set(result["document"]["sections"]) == transfer.SETUP_SECTIONS
