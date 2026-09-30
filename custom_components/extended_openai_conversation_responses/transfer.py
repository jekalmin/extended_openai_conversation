"""Unified portable export, custom backup, and import/restore planning."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import yaml

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from . import backup
from .agent_config import (
    agent_config_snapshot,
    configured_function_tools_from_data,
    validate_agent_title,
)
from .const import AGENT_CONFIG_EXPORT_VERSION
from .conversation_archive import (
    ArchiveSession,
    ArchiveTurn,
    ConversationArchive,
    async_get_archive,
)
from .function_dependency_integrity import async_validate_request_rule_functions
from .guest_mode import GuestModeManager, GuestModeSchedule, async_get_guest_mode
from .knowledge import KnowledgeLibrary, KnowledgeSource, async_get_knowledge
from .memory import MemoryRecord, PersistentMemory, async_get_memory
from .request_rules import RequestRules, async_get_request_rules
from .secret_redaction import (
    LITERAL_TEXT_KEY,
    REDACTED_SECRET_SENTINEL,
    is_literal_text,
    is_redacted_secret,
    redact_secrets,
    restore_redacted_secrets,
)
from .temporary_memory import (
    TemporaryMemory,
    TemporaryMemoryRecord,
    async_get_temporary_memory,
)
from .usage import UsageManager, UsageRequest, UsageRun, UsageTotals, async_get_usage

TRANSFER_FORMAT = "extended_openai_conversation_transfer"
TRANSFER_VERSION = 1
LEGACY_AGENT_SCHEMA = "extended_openai_conversation.agent"

SECTION_CONFIGURATION = "configuration"
SECTION_REQUEST_RULES = "request_rules"
SECTION_PERSISTENT_MEMORY = "persistent_memory"
SECTION_TEMPORARY_MEMORY = "temporary_memory"
SECTION_KNOWLEDGE = "knowledge"
SECTION_CONVERSATION_ARCHIVE = "conversation_archive"
SECTION_USAGE = "usage"
SECTION_GUEST_MODE = "guest_mode"

SETUP_SECTIONS = frozenset({SECTION_CONFIGURATION, SECTION_REQUEST_RULES})
ALL_SECTIONS = frozenset(
    {
        SECTION_CONFIGURATION,
        SECTION_REQUEST_RULES,
        SECTION_PERSISTENT_MEMORY,
        SECTION_TEMPORARY_MEMORY,
        SECTION_KNOWLEDGE,
        SECTION_CONVERSATION_ARCHIVE,
        SECTION_USAGE,
        SECTION_GUEST_MODE,
    }
)
SECTION_ORDER = (
    SECTION_CONFIGURATION,
    SECTION_REQUEST_RULES,
    SECTION_PERSISTENT_MEMORY,
    SECTION_TEMPORARY_MEMORY,
    SECTION_KNOWLEDGE,
    SECTION_CONVERSATION_ARCHIVE,
    SECTION_USAGE,
    SECTION_GUEST_MODE,
)
SECTION_LABELS = {
    SECTION_CONFIGURATION: "Agent configuration and Function Tools",
    SECTION_REQUEST_RULES: "Request Rules",
    SECTION_PERSISTENT_MEMORY: "Persistent memories",
    SECTION_TEMPORARY_MEMORY: "Active temporary memories",
    SECTION_KNOWLEDGE: "Knowledge sources",
    SECTION_CONVERSATION_ARCHIVE: "Conversation archive",
    SECTION_USAGE: "Usage history",
    SECTION_GUEST_MODE: "Guest Mode schedule",
}

_ABSENT = object()
_DROP = object()


@dataclass(slots=True)
class PreparedTransfer:
    """Validated import payload before a target-specific restore plan is applied."""

    source_kind: str
    mode: str
    title: str
    available_sections: frozenset[str]
    created_at: str | None
    integration_version: str | None
    config: dict[str, Any] | None = None
    request_rules: dict[str, Any] | None = None
    memories: list[MemoryRecord] | None = None
    temporary_memories: list[TemporaryMemoryRecord] | None = None
    knowledge: list[KnowledgeSource] | None = None
    archive_sessions: list[ArchiveSession] | None = None
    archive_turns: list[ArchiveTurn] | None = None
    usage_totals: UsageTotals | None = None
    usage_daily: dict[str, dict[str, Any]] | None = None
    usage_requests: list[UsageRequest] | None = None
    usage_runs: list[UsageRun] | None = None
    guest_mode_schedule: GuestModeSchedule | None = None
    raw_configuration: Any = None
    raw_request_rules: Any = None
    redacted_sensitive_fields: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """Return a bounded, frontend-friendly description of available content."""
        sections = [
            section for section in SECTION_ORDER if section in self.available_sections
        ]
        return {
            "source_kind": self.source_kind,
            "mode": self.mode,
            "sections": sections,
            "section_labels": {
                section: SECTION_LABELS[section] for section in sections
            },
            "configuration": SECTION_CONFIGURATION in self.available_sections,
            "request_rules": (
                len(self.request_rules.get("rules", []))
                if self.request_rules is not None
                else 0
            ),
            "persistent_memories": len(self.memories or []),
            "temporary_memories": len(self.temporary_memories or []),
            "knowledge_sources": len(self.knowledge or []),
            "archive_sessions": len(self.archive_sessions or []),
            "archive_turns": len(self.archive_turns or []),
            "usage_requests": len(self.usage_requests or []),
            "usage_runs": len(self.usage_runs or []),
            "guest_mode_scheduled": (
                SECTION_GUEST_MODE in self.available_sections
                and self.guest_mode_schedule is not None
            ),
            "created_at": self.created_at,
            "integration_version": self.integration_version,
            "redacted_sensitive_fields": list(self.redacted_sensitive_fields[:50]),
            "redacted_sensitive_field_count": len(self.redacted_sensitive_fields),
        }


def _integration_version() -> str:
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text("utf-8"))
    return str(manifest["version"])


def validate_section_selection(
    sections: Iterable[str] | None,
    *,
    allowed: Iterable[str] = ALL_SECTIONS,
    default: Iterable[str] | None = None,
) -> frozenset[str]:
    """Validate a non-empty bounded section selection."""
    allowed_set = frozenset(allowed)
    if sections is None:
        selected = frozenset(default if default is not None else allowed_set)
    else:
        if isinstance(sections, (str, bytes)):
            raise backup.BackupError("Transfer sections must be a list")
        try:
            values = tuple(sections)
        except TypeError as err:
            raise backup.BackupError("Transfer sections must be a list") from err
        if not all(isinstance(item, str) for item in values):
            raise backup.BackupError("Transfer sections must contain names only")
        selected = frozenset(values)
    if not selected:
        raise backup.BackupError("Select at least one section")
    unknown = selected - allowed_set
    if unknown:
        raise backup.BackupError(
            "Unknown transfer section: " + ", ".join(sorted(unknown))
        )
    return selected


def _safe_title(title: str) -> str:
    return (
        re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-") or "conversation-agent"
    )


def _new_transfer_document(entry: Any, subentry: Any, mode: str) -> dict[str, Any]:
    return {
        "format": TRANSFER_FORMAT,
        "version": TRANSFER_VERSION,
        "mode": mode,
        "created_at": dt_util.utcnow().isoformat(),
        "integration_version": _integration_version(),
        "agent": {
            "title": subentry.title,
            "source_entry_id": entry.entry_id,
            "source_subentry_id": subentry.subentry_id,
        },
        "sections": {},
    }


async def async_collect_transfer_snapshot(
    hass: HomeAssistant,
    entry: Any,
    subentry: Any,
    *,
    mode: str,
    sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Collect only the sections requested for a portable/custom transfer."""
    if mode == "setup":
        selected = SETUP_SECTIONS
    elif mode == "custom":
        selected = validate_section_selection(sections)
    else:
        raise backup.BackupError("Transfer mode must be setup or custom")

    document = await hass.async_add_executor_job(
        _new_transfer_document, entry, subentry, mode
    )
    payload = document["sections"]
    if SECTION_CONFIGURATION in selected:
        payload[SECTION_CONFIGURATION] = backup.export_configuration_snapshot(
            subentry.data
        )
    if SECTION_REQUEST_RULES in selected:
        request_rules_manager = await async_get_request_rules(
            hass, entry.entry_id, subentry.subentry_id
        )
        payload[SECTION_REQUEST_RULES] = await request_rules_manager.async_backup_data()
    if SECTION_PERSISTENT_MEMORY in selected:
        memory_manager = await async_get_memory(
            hass, entry.entry_id, subentry.subentry_id
        )
        payload[SECTION_PERSISTENT_MEMORY] = await memory_manager.async_backup_data()
    if SECTION_TEMPORARY_MEMORY in selected:
        temporary_memory_manager = await async_get_temporary_memory(
            hass, entry.entry_id, subentry.subentry_id
        )
        payload[
            SECTION_TEMPORARY_MEMORY
        ] = await temporary_memory_manager.async_backup_data()
    if SECTION_KNOWLEDGE in selected:
        knowledge_manager = await async_get_knowledge(
            hass, entry.entry_id, subentry.subentry_id
        )
        payload[SECTION_KNOWLEDGE] = await knowledge_manager.async_backup_data()
    if SECTION_CONVERSATION_ARCHIVE in selected:
        archive_manager = await async_get_archive(
            hass, entry.entry_id, subentry.subentry_id
        )
        payload[
            SECTION_CONVERSATION_ARCHIVE
        ] = await archive_manager.async_backup_data()
    if SECTION_USAGE in selected:
        usage_manager = await async_get_usage(
            hass, entry.entry_id, subentry.subentry_id
        )
        payload[SECTION_USAGE] = await usage_manager.async_backup_data()
    if SECTION_GUEST_MODE in selected:
        guest_mode_manager = await async_get_guest_mode(
            hass, entry.entry_id, subentry.subentry_id
        )
        payload[SECTION_GUEST_MODE] = await guest_mode_manager.async_backup_data()
    return document


def _is_redacted_placeholder(value: Any) -> bool:
    """Return whether a value is one of our explicit redaction placeholders."""
    return is_redacted_secret(value)


def _normalize_redaction_placeholders(value: Any) -> Any:
    """Use one explicit placeholder in newly created transfer documents."""
    if is_literal_text(value):
        return deepcopy(value)
    if _is_redacted_placeholder(value):
        return dict(REDACTED_SECRET_SENTINEL)
    if isinstance(value, list):
        return [_normalize_redaction_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_redaction_placeholders(item) for key, item in value.items()
        }
    return value


def redact_transfer_document(document: dict[str, Any]) -> dict[str, Any]:
    """Redact secret-bearing setup sections without altering private state content."""
    result = deepcopy(document)
    sections = result.get("sections")
    if isinstance(sections, dict):
        if SECTION_CONFIGURATION in sections:
            sections[SECTION_CONFIGURATION] = _normalize_redaction_placeholders(
                redact_secrets(sections[SECTION_CONFIGURATION])
            )
        if SECTION_REQUEST_RULES in sections:
            sections[SECTION_REQUEST_RULES] = _normalize_redaction_placeholders(
                redact_secrets(sections[SECTION_REQUEST_RULES])
            )
    return result


def finalize_setup_export(document: dict[str, Any]) -> dict[str, Any]:
    """Serialize a lightweight shareable setup document."""
    redacted = redact_transfer_document(document)
    serialized = json.dumps(redacted, indent=2, ensure_ascii=False)
    if len(serialized.encode("utf-8")) > backup.MAX_LEGACY_EXPORT_BYTES:
        raise backup.BackupError(
            "This setup export exceeds the 16 MB safety limit; use a custom backup instead"
        )
    title = validate_agent_title(redacted["agent"]["title"])
    date = str(redacted["created_at"])[:10]
    return {
        "document": redacted,
        "json": serialized,
        "filename": f"{_safe_title(title)}-shareable-setup-{date}.json",
        "mode": "setup",
        "sections": list(SECTION_ORDER[:2]),
    }


async def async_create_setup_export(
    hass: HomeAssistant, entry: Any, subentry: Any
) -> dict[str, Any]:
    """Build a lightweight, redacted setup-sharing document."""
    document = await async_collect_transfer_snapshot(
        hass, entry, subentry, mode="setup"
    )
    return finalize_setup_export(document)


def _secret_path(path: tuple[Any, ...]) -> str:
    if not path:
        return "value"
    text = ""
    for item in path:
        if isinstance(item, int):
            text += f"[{item}]"
        else:
            text += ("." if text else "") + str(item)
    return text


def _collect_secret_paths(value: Any, path: tuple[Any, ...] = ()) -> list[str]:
    if is_literal_text(value):
        return []
    if _is_redacted_placeholder(value):
        return [_secret_path(path)]
    if isinstance(value, list):
        result: list[str] = []
        for index, item in enumerate(value):
            result.extend(_collect_secret_paths(item, (*path, index)))
        return result
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            result.extend(_collect_secret_paths(item, (*path, key)))
        return result
    return []


def _list_item_identity(item: Any) -> Any:
    """Use explicit IDs, or the stable spec name of a Function Tool."""
    if isinstance(item, Mapping):
        if "id" in item:
            return ("id", item["id"])
        spec = item.get("spec")
        if isinstance(spec, Mapping) and "name" in spec:
            return ("spec.name", spec["name"])
    return None


def _compatible_secret_context(value: Any, fallback: Any) -> bool:
    """Require identical visible structure; only secret scalar leaves may differ.

    A marker cannot hide a container (and therefore a destination). Nested lists
    may reorder only when their items have unique, one-to-one safe matches.
    """
    if is_literal_text(value):
        return bool(value[LITERAL_TEXT_KEY] == fallback)
    if _is_redacted_placeholder(value):
        return (
            fallback is not _ABSENT
            and not isinstance(fallback, (Mapping, list))
            and not _is_redacted_placeholder(fallback)
        )
    if isinstance(value, dict):
        return (
            isinstance(fallback, Mapping)
            and value.keys() == fallback.keys()
            and all(
                _compatible_secret_context(item, fallback[key])
                for key, item in value.items()
            )
        )
    if isinstance(value, list):
        if not isinstance(fallback, list) or len(value) != len(fallback):
            return False
        matches = [_fallback_list_index(item, fallback) for item in value]
        return None not in matches and len(set(matches)) == len(matches)
    return type(value) is type(fallback) and bool(value == fallback)


def _fallback_list_index(item: Any, fallback: list[Any]) -> int | None:
    """Select a unique compatible item, never infer authority from its position."""
    identity = _list_item_identity(item)
    if identity is not None and (not isinstance(identity[1], str) or not identity[1]):
        return None
    candidates = [
        index
        for index, candidate in enumerate(fallback)
        if _list_item_identity(candidate) == identity
    ]
    # Duplicate stable identities are invalid even if their public fields differ.
    if identity is not None and len(candidates) != 1:
        return None
    matches = [
        index
        for index in candidates
        if _compatible_secret_context(item, fallback[index])
    ]
    return matches[0] if len(matches) == 1 else None


def _restore_with_fallback(
    value: Any,
    fallback: Any,
    *,
    path: tuple[Any, ...] = (),
    preserved: list[str],
    missing: list[str],
) -> Any:
    if is_literal_text(value):
        return value[LITERAL_TEXT_KEY]
    if _is_redacted_placeholder(value):
        label = _secret_path(path)
        if _compatible_secret_context(value, fallback):
            preserved.append(label)
            return deepcopy(fallback)
        missing.append(label)
        return _DROP
    if isinstance(value, list):
        restored: list[Any] = []
        local_items = fallback if isinstance(fallback, list) else []
        matches = [_fallback_list_index(item, local_items) for item in value]
        for index, item in enumerate(value):
            match = matches[index]
            candidate = (
                local_items[match]
                if match is not None and matches.count(match) == 1
                else _ABSENT
            )
            child = _restore_with_fallback(
                item,
                candidate,
                path=(*path, index),
                preserved=preserved,
                missing=missing,
            )
            if child is not _DROP:
                restored.append(child)
        return restored
    if isinstance(value, dict):
        restored_mapping: dict[Any, Any] = {}
        fallback_mapping = fallback if isinstance(fallback, Mapping) else {}
        for key, item in value.items():
            candidate = fallback_mapping.get(key, _ABSENT)
            child = _restore_with_fallback(
                item,
                candidate,
                path=(*path, key),
                preserved=preserved,
                missing=missing,
            )
            if child is not _DROP:
                restored_mapping[key] = child
        return restored_mapping
    return deepcopy(value)


def _restore_section_secrets(
    value: Any, fallback: Any
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    preserved: list[str] = []
    missing: list[str] = []
    restored = _restore_with_fallback(
        value,
        fallback,
        preserved=preserved,
        missing=missing,
    )
    return restored, tuple(preserved), tuple(missing)


def _validate_metadata(value: Mapping[str, Any]) -> tuple[str, str, str]:
    created_at = value.get("created_at")
    integration_version = value.get("integration_version")
    if (
        not isinstance(created_at, str)
        or dt_util.parse_datetime(created_at) is None
        or not isinstance(integration_version, str)
        or not integration_version
    ):
        raise backup.BackupError("The transfer metadata is invalid")
    agent = value.get("agent")
    if not isinstance(agent, Mapping) or set(agent) != {
        "title",
        "source_entry_id",
        "source_subentry_id",
    }:
        raise backup.BackupError("The transfer agent metadata is invalid")
    try:
        title = validate_agent_title(agent.get("title"))
    except HomeAssistantError as err:
        raise backup.BackupError("The transferred agent name is invalid") from err
    if not isinstance(agent.get("source_entry_id"), str) or not isinstance(
        agent.get("source_subentry_id"), str
    ):
        raise backup.BackupError("The transfer agent identity is invalid")
    return title, created_at, integration_version


def _validate_transfer_document(
    value: Mapping[str, Any], target_agent_id: str
) -> PreparedTransfer:
    expected = {
        "format",
        "version",
        "mode",
        "created_at",
        "integration_version",
        "agent",
        "sections",
    }
    if set(value) != expected:
        raise backup.BackupError("The transfer file is incomplete or corrupted")
    if value.get("version") != TRANSFER_VERSION:
        version = value.get("version")
        if isinstance(version, int) and version > TRANSFER_VERSION:
            raise backup.BackupError(
                "This transfer was created using a newer unsupported format"
            )
        raise backup.BackupError("This transfer uses an unsupported format version")
    mode = value.get("mode")
    if mode not in {"setup", "custom"}:
        raise backup.BackupError("The transfer mode is invalid")
    title, created_at, integration_version = _validate_metadata(value)
    raw_sections = value.get("sections")
    if not isinstance(raw_sections, Mapping):
        raise backup.BackupError("The transfer sections are invalid")
    available = validate_section_selection(raw_sections.keys())
    if mode == "setup" and available != SETUP_SECTIONS:
        raise backup.BackupError("The shareable setup sections are incomplete")

    prepared = PreparedTransfer(
        source_kind="portable_transfer" if mode == "setup" else "custom_backup",
        mode=mode,
        title=title,
        available_sections=available,
        created_at=created_at,
        integration_version=integration_version,
    )
    redacted: list[str] = []
    try:
        if SECTION_CONFIGURATION in available:
            raw = raw_sections[SECTION_CONFIGURATION]
            prepared.raw_configuration = deepcopy(raw)
            redacted.extend(
                f"configuration.{path}" for path in _collect_secret_paths(raw)
            )
            restored = restore_redacted_secrets(raw)
            if not isinstance(restored, dict):
                raise ValueError("configuration must be an object")
            prepared.config = backup.recoverable_configuration_snapshot(restored)
        if SECTION_REQUEST_RULES in available:
            raw = raw_sections[SECTION_REQUEST_RULES]
            prepared.raw_request_rules = deepcopy(raw)
            redacted.extend(
                f"request_rules.{path}" for path in _collect_secret_paths(raw)
            )
            prepared.request_rules = RequestRules.validate_backup_data(
                restore_redacted_secrets(raw)
            )
        if SECTION_PERSISTENT_MEMORY in available:
            prepared.memories = PersistentMemory.validate_backup_data(
                raw_sections[SECTION_PERSISTENT_MEMORY]
            )
        if SECTION_TEMPORARY_MEMORY in available:
            prepared.temporary_memories = TemporaryMemory.validate_backup_data(
                raw_sections[SECTION_TEMPORARY_MEMORY]
            )
        if SECTION_KNOWLEDGE in available:
            prepared.knowledge = KnowledgeLibrary.validate_backup_data(
                raw_sections[SECTION_KNOWLEDGE]
            )
        if SECTION_CONVERSATION_ARCHIVE in available:
            prepared.archive_sessions, prepared.archive_turns = (
                ConversationArchive.validate_backup_data(
                    raw_sections[SECTION_CONVERSATION_ARCHIVE], target_agent_id
                )
            )
        if SECTION_USAGE in available:
            (
                prepared.usage_totals,
                prepared.usage_daily,
                prepared.usage_requests,
                prepared.usage_runs,
            ) = UsageManager.validate_backup_data(
                raw_sections[SECTION_USAGE], target_agent_id
            )
        if SECTION_GUEST_MODE in available:
            prepared.guest_mode_schedule = GuestModeManager.validate_backup_data(
                raw_sections[SECTION_GUEST_MODE]
            )
    except (HomeAssistantError, TypeError, ValueError) as err:
        raise backup.BackupError(
            f"The transfer is incomplete or corrupted: {err}"
        ) from err
    prepared.redacted_sensitive_fields = tuple(redacted)
    return prepared


def _inspect_legacy_setup(value: Mapping[str, Any]) -> PreparedTransfer:
    if value.get("version") != AGENT_CONFIG_EXPORT_VERSION:
        raise backup.BackupError("This setup export uses an unsupported version")
    unknown = set(value) - {"schema", "version", "title", "config"}
    if unknown:
        raise backup.BackupError(
            "The setup export contains unknown fields: " + ", ".join(sorted(unknown))
        )
    raw_config = value.get("config")
    redacted = tuple(
        f"configuration.{path}" for path in _collect_secret_paths(raw_config)
    )
    try:
        restored = restore_redacted_secrets(raw_config)
        if not isinstance(restored, dict):
            raise ValueError("configuration must be an object")
        config = backup.recoverable_configuration_snapshot(restored)
        title = validate_agent_title(
            value.get("title"), default="Imported conversation agent"
        )
    except (HomeAssistantError, TypeError, ValueError) as err:
        raise backup.BackupError(f"The setup export is invalid: {err}") from err
    return PreparedTransfer(
        source_kind="legacy_setup",
        mode="setup",
        title=title,
        available_sections=frozenset({SECTION_CONFIGURATION}),
        created_at=None,
        integration_version=None,
        config=config,
        raw_configuration=deepcopy(raw_config),
        redacted_sensitive_fields=redacted,
    )


def _inspect_full_backup(
    value: Mapping[str, Any], target_agent_id: str
) -> PreparedTransfer:
    prepared = backup.inspect_backup(
        value, target_agent_id, max_bytes=backup.MAX_BACKUP_BYTES
    )
    version = value.get("version")
    available = {
        SECTION_CONFIGURATION,
        SECTION_PERSISTENT_MEMORY,
        SECTION_TEMPORARY_MEMORY,
        SECTION_KNOWLEDGE,
        SECTION_CONVERSATION_ARCHIVE,
        SECTION_USAGE,
    }
    if isinstance(version, int) and version >= 2 and "guest_mode" in value:
        available.add(SECTION_GUEST_MODE)
    if isinstance(version, int) and version >= 3 and "request_rules" in value:
        available.add(SECTION_REQUEST_RULES)
    raw_config = (
        value.get("agent", {}).get("config")
        if isinstance(value.get("agent"), Mapping)
        else None
    )
    raw_rules = value.get("request_rules")
    redacted = [f"configuration.{path}" for path in _collect_secret_paths(raw_config)]
    if raw_rules is not None:
        redacted.extend(
            f"request_rules.{path}" for path in _collect_secret_paths(raw_rules)
        )
    return PreparedTransfer(
        source_kind="full_backup",
        mode="full",
        title=prepared.title,
        available_sections=frozenset(available),
        created_at=prepared.created_at,
        integration_version=prepared.integration_version,
        config=prepared.config,
        request_rules=prepared.request_rules
        if SECTION_REQUEST_RULES in available
        else None,
        memories=prepared.memories,
        temporary_memories=prepared.temporary_memories,
        knowledge=prepared.knowledge,
        archive_sessions=prepared.archive_sessions,
        archive_turns=prepared.archive_turns,
        usage_totals=prepared.usage_totals,
        usage_daily=prepared.usage_daily,
        usage_requests=prepared.usage_requests,
        usage_runs=prepared.usage_runs,
        guest_mode_schedule=(
            prepared.guest_mode_schedule if SECTION_GUEST_MODE in available else None
        ),
        raw_configuration=deepcopy(raw_config),
        raw_request_rules=deepcopy(raw_rules),
        redacted_sensitive_fields=tuple(redacted),
    )


def inspect_transfer(value: Any, target_agent_id: str) -> PreparedTransfer:
    """Classify and fully validate portable, custom, legacy, or full backup input."""
    if isinstance(value, PreparedTransfer):
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > backup.MAX_BACKUP_BYTES:
            raise backup.BackupError("The transfer exceeds the 128 MB safety limit")
        try:
            value = json.loads(value)
        except json.JSONDecodeError as err:
            raise backup.BackupError("The transfer is not valid JSON") from err
    if not isinstance(value, Mapping):
        raise backup.BackupError(
            "This file is not an Extended OpenAI Conversation transfer"
        )
    if value.get("format") == TRANSFER_FORMAT:
        return _validate_transfer_document(value, target_agent_id)
    if value.get("format") == backup.BACKUP_FORMAT:
        return _inspect_full_backup(value, target_agent_id)
    if value.get("schema") == LEGACY_AGENT_SCHEMA:
        return _inspect_legacy_setup(value)
    raise backup.BackupError(
        "This file is not a recognised Extended OpenAI Conversation export or backup"
    )


async def _async_validate_request_rule_function_dependencies(
    hass: HomeAssistant,
    request_rules: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Apply canonical Request Rule Function validation to one combined target."""
    quarantined_names: set[str] = set()
    try:
        tools = configured_function_tools_from_data(config)
    except HomeAssistantError, yaml.YAMLError, TypeError, ValueError:
        from .management_function_repair import (
            function_tools_issue,
            isolated_function_tools,
        )

        tools, issue = function_tools_issue(dict(config))
        if issue is None:
            raise
        _valid, invalid, _isolated_issue = isolated_function_tools(dict(config))
        quarantined_names = {
            str(item["name"])
            for item in invalid
            if isinstance(item.get("name"), str) and item["name"]
        }
    for rule in request_rules.get("rules", []):
        if not isinstance(rule, Mapping):
            continue
        try:
            await async_validate_request_rule_functions(
                hass, rule, tools, quarantined_names=quarantined_names
            )
        except (HomeAssistantError, ValueError) as err:
            label = rule.get("name", rule.get("id", "unnamed"))
            raise backup.BackupError(
                f"Request Rule `{label}` has an invalid Function Tool action: {err}"
            ) from err


def _prepared_restore_from_selection(
    current: backup.PreparedRestore,
    imported: PreparedTransfer,
    selected: frozenset[str],
) -> tuple[backup.PreparedRestore, tuple[str, ...], tuple[str, ...]]:
    """Overlay selected replacement sections on one complete current snapshot."""
    preserved: list[str] = []
    missing: list[str] = []

    title = current.title
    config = current.config
    if SECTION_CONFIGURATION in selected:
        title = imported.title
        if imported.raw_configuration is not None:
            raw, kept, absent = _restore_section_secrets(
                imported.raw_configuration, agent_config_snapshot(current.config)
            )
            preserved.extend(f"configuration.{path}" for path in kept)
            missing.extend(f"configuration.{path}" for path in absent)
            if not isinstance(raw, dict):
                raise backup.BackupError("Transferred configuration is invalid")
            config = backup.recoverable_configuration_snapshot(raw)
        elif imported.config is not None:
            config = deepcopy(imported.config)
        else:
            raise backup.BackupError("Transferred configuration is unavailable")
    else:
        # Durable snapshots contain frontend-shaped, parsed Function Tools.
        # Retained configuration must use the same persisted YAML representation
        # as imported configuration before dependency validation and restore.
        config = backup.recoverable_configuration_snapshot(current.config)

    rules = current.request_rules
    if SECTION_REQUEST_RULES in selected:
        if imported.raw_request_rules is not None:
            raw, kept, absent = _restore_section_secrets(
                imported.raw_request_rules, current.request_rules
            )
            preserved.extend(f"request_rules.{path}" for path in kept)
            missing.extend(f"request_rules.{path}" for path in absent)
            rules = RequestRules.validate_backup_data(raw)
        elif imported.request_rules is not None:
            rules = deepcopy(imported.request_rules)
        else:
            raise backup.BackupError("Transferred Request Rules are unavailable")

    target = backup.PreparedRestore(
        title=title,
        config=config,
        memories=(
            deepcopy(imported.memories)
            if SECTION_PERSISTENT_MEMORY in selected and imported.memories is not None
            else current.memories
        ),
        temporary_memories=(
            deepcopy(imported.temporary_memories)
            if SECTION_TEMPORARY_MEMORY in selected
            and imported.temporary_memories is not None
            else current.temporary_memories
        ),
        knowledge=(
            deepcopy(imported.knowledge)
            if SECTION_KNOWLEDGE in selected and imported.knowledge is not None
            else current.knowledge
        ),
        archive_sessions=(
            deepcopy(imported.archive_sessions)
            if SECTION_CONVERSATION_ARCHIVE in selected
            and imported.archive_sessions is not None
            else current.archive_sessions
        ),
        archive_turns=(
            deepcopy(imported.archive_turns)
            if SECTION_CONVERSATION_ARCHIVE in selected
            and imported.archive_turns is not None
            else current.archive_turns
        ),
        usage_totals=(
            deepcopy(imported.usage_totals)
            if SECTION_USAGE in selected and imported.usage_totals is not None
            else current.usage_totals
        ),
        usage_daily=(
            deepcopy(imported.usage_daily)
            if SECTION_USAGE in selected and imported.usage_daily is not None
            else current.usage_daily
        ),
        usage_requests=(
            deepcopy(imported.usage_requests)
            if SECTION_USAGE in selected and imported.usage_requests is not None
            else current.usage_requests
        ),
        usage_runs=(
            deepcopy(imported.usage_runs)
            if SECTION_USAGE in selected and imported.usage_runs is not None
            else current.usage_runs
        ),
        guest_mode_schedule=(
            deepcopy(imported.guest_mode_schedule)
            if SECTION_GUEST_MODE in selected
            else current.guest_mode_schedule
        ),
        request_rules=rules,
        created_at=imported.created_at or current.created_at,
        integration_version=imported.integration_version or current.integration_version,
    )
    return target, tuple(preserved), tuple(missing)


async def _current_snapshot(
    hass: HomeAssistant, entry: Any, subentry: Any
) -> backup.PreparedRestore:
    """Read durable current state for selective planning without mutating it."""
    try:
        from .restore_recovery import _durable_managers

        managers = await _durable_managers(hass, entry.entry_id, subentry.subentry_id)
    except ImportError:
        managers = await backup._managers(hass, entry.entry_id, subentry.subentry_id)
    return await backup._snapshot_for_restore(managers, subentry)


async def async_materialize_restore(
    hass: HomeAssistant,
    entry: Any,
    subentry: Any,
    imported: PreparedTransfer | Any,
    *,
    sections: Iterable[str] | None = None,
    current_snapshot: backup.PreparedRestore | None = None,
) -> tuple[backup.PreparedRestore, dict[str, Any]]:
    """Build and validate a complete restore target from a selective transfer."""
    prepared = inspect_transfer(imported, subentry.subentry_id)
    selected = validate_section_selection(
        sections,
        allowed=prepared.available_sections,
        default=prepared.available_sections,
    )
    current = (
        current_snapshot
        if current_snapshot is not None
        else await _current_snapshot(hass, entry, subentry)
    )
    target, preserved, missing = _prepared_restore_from_selection(
        current, prepared, selected
    )
    if missing:
        raise backup.BackupError(
            "Cannot safely restore unavailable secrets: " + ", ".join(missing[:50])
        )
    await _async_validate_request_rule_function_dependencies(
        hass, target.request_rules, target.config
    )
    preview = prepared.summary()
    preview.update(
        {
            "selected_sections": [
                section for section in SECTION_ORDER if section in selected
            ],
            "preserved_sensitive_fields": list(preserved[:50]),
            "preserved_sensitive_field_count": len(preserved),
            "missing_sensitive_fields": list(missing[:50]),
            "missing_sensitive_field_count": len(missing),
            "target_title": target.title,
        }
    )
    return target, preview


async def async_restore_transfer(
    hass: HomeAssistant,
    entry: Any,
    subentry: Any,
    imported: PreparedTransfer | Any,
    *,
    sections: Iterable[str] | None = None,
    precondition: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Preserve unselected state and commit under one shielded exclusive lease."""
    from .agent_maintenance import (
        _async_run_exclusive_operation,
        get_agent_maintenance_gate,
    )
    from .restore_recovery import async_restore_backup_recoverably

    async def restore_exclusively_owned() -> dict[str, Any]:
        if precondition is not None:
            await precondition()
        target, preview = await async_materialize_restore(
            hass, entry, subentry, imported, sections=sections
        )
        # Exclusivity already covers the destination snapshot. Call the journaled
        # inner restore directly: the public backup wrapper would reacquire this
        # non-reentrant gate. Lock order remains maintenance gate -> backup lock.
        result = await async_restore_backup_recoverably(hass, entry, subentry, target)
        return {**result, "transfer": preview}

    gate = get_agent_maintenance_gate(hass, entry.entry_id, subentry.subentry_id)
    return await _async_run_exclusive_operation(gate, restore_exclusively_owned)


def inspection_for_frontend(prepared: PreparedTransfer) -> dict[str, Any]:
    """Return stable inspection metadata before a target-specific preview."""
    return {
        "valid": True,
        "title": prepared.title,
        "source_kind": prepared.source_kind,
        "mode": prepared.mode,
        "available_sections": [
            section
            for section in SECTION_ORDER
            if section in prepared.available_sections
        ],
        "summary": prepared.summary(),
        "can_create_new_agent": prepared.available_sections.issubset(SETUP_SECTIONS)
        and SECTION_CONFIGURATION in prepared.available_sections,
    }
