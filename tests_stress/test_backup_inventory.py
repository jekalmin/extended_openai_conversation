"""Explicit nightly inventory of every persisted agent configuration field."""

import json
from pathlib import Path

from custom_components.extended_openai_conversation_responses import (
    agent_config,
    backup,
)

# Frozen names, deliberately reviewed rather than generated at test time. A new
# field must be classified here before its backup semantics can go unnoticed.
BACKED_UP_AGENT_FIELDS = {
    "CONF_ADVANCED_OPTIONS",
    "CONF_API_MODE",
    "CONF_ARCHIVE_ENABLED",
    "CONF_ARCHIVE_MODEL_SEARCH_ENABLED",
    "CONF_ARCHIVE_RETENTION_DAYS",
    "CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES",
    "CONF_CHAT_MODEL",
    "CONF_CONTEXT_THRESHOLD",
    "CONF_CONTEXT_TRUNCATE_STRATEGY",
    "CONF_CONTINUE_CONVERSATION",
    "CONF_CONVERSATION_CONTINUITY",
    "CONF_CONVERSATION_TIMEOUT_MINUTES",
    "CONF_CURRENT_DATETIME_ENABLED",
    "CONF_CURRENT_DATETIME_TEMPLATE",
    "CONF_EXPOSED_ENTITIES_ENABLED",
    "CONF_EXPOSED_ENTITIES_TEMPLATE",
    "CONF_EXPOSED_ENTITY_ATTRIBUTES",
    "CONF_FUNCTION_GROUPS",
    "CONF_FUNCTION_TOOLS",
    "CONF_FUNCTION_TOOL_ERROR_RECOVERY",
    "CONF_GUEST_ALLOWED_FUNCTION_NAMES",
    "CONF_GUEST_ALLOWED_GROUP_IDS",
    "CONF_GUEST_CONTROLLABLE_AREAS",
    "CONF_GUEST_CONTROLLABLE_DOMAINS",
    "CONF_GUEST_CONTROLLABLE_ENTITIES",
    "CONF_GUEST_CONTROLLABLE_LABELS",
    "CONF_GUEST_CONTROL_EXCLUDED_AREAS",
    "CONF_GUEST_CONTROL_EXCLUDED_DOMAINS",
    "CONF_GUEST_CONTROL_EXCLUDED_ENTITIES",
    "CONF_GUEST_CONTROL_EXCLUDED_LABELS",
    "CONF_GUEST_EXCLUDED_AREAS",
    "CONF_GUEST_EXCLUDED_DOMAINS",
    "CONF_GUEST_EXCLUDED_ENTITIES",
    "CONF_GUEST_EXCLUDED_LABELS",
    "CONF_GUEST_FUNCTION_POLICY",
    "CONF_GUEST_KNOWLEDGE_ENABLED",
    "CONF_GUEST_KNOWLEDGE_POLICY",
    "CONF_GUEST_KNOWLEDGE_SOURCE_IDS",
    "CONF_GUEST_MODE_ENABLED",
    "CONF_GUEST_POLICY_VERSION",
    "CONF_GUEST_READABLE_AREAS",
    "CONF_GUEST_READABLE_DOMAINS",
    "CONF_GUEST_READABLE_ENTITIES",
    "CONF_GUEST_READABLE_LABELS",
    "CONF_GUEST_SEPARATE_CONTROL_RESTRICTIONS",
    "CONF_GUEST_SHARED_MEMORY_POLICY",
    "CONF_GUEST_SHARED_MEMORY_READ",
    "CONF_GUEST_SHARED_MEMORY_WRITE",
    "CONF_GUEST_WEB_SEARCH",
    "CONF_KNOWLEDGE_ENABLED",
    "CONF_LOCAL_INTENTS_ENABLED",
    "CONF_LOCAL_INTENT_DELAYED_COMMANDS_TO_AI",
    "CONF_LOCAL_INTENT_EXCLUSIONS",
    "CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION",
    "CONF_MAX_TOKENS",
    "CONF_MEMORY_AUTO_CREATE",
    "CONF_MEMORY_AUTO_RETRIEVE_LIMIT",
    "CONF_MEMORY_EMBEDDING_MODEL",
    "CONF_MEMORY_ENABLED",
    "CONF_MEMORY_MODE",
    "CONF_MEMORY_RETRIEVAL_MODE",
    "CONF_PROMPT",
    "CONF_REASONING_EFFORT",
    "CONF_SERVICE_TIER",
    "CONF_SHARED_ARCHIVE_ENABLED",
    "CONF_SHARED_MEMORY_MODE",
    "CONF_SHORTEN_TOOL_CALL_ID",
    "CONF_SKILLS",
    "CONF_SPEECH_PROCESSING_ENABLED",
    "CONF_SPEECH_REGEX_REPLACEMENTS",
    "CONF_SPEECH_STRIP_MARKDOWN",
    "CONF_SPEECH_STRIP_URLS",
    "CONF_TEMPERATURE",
    "CONF_TEMPORARY_MEMORY",
    "CONF_TOP_P",
    "CONF_USAGE_REQUEST_RETENTION_DAYS",
    "CONF_USAGE_RUN_RETENTION_DAYS",
    "CONF_VOICE_DEFAULT_USER_ID",
    "CONF_VOICE_DEVICE_MAPPINGS",
    "CONF_VOICE_SCOPE_POLICY",
    "CONF_VOICE_UNMAPPED_POLICY",
    "CONF_WEB_SEARCH",
    "CONF_WEB_SEARCH_CONTEXT",
}

BACKED_UP_SUBSYSTEMS = {
    "agent",
    "memories",
    "temporary_memories",
    "knowledge",
    "archive",
    "usage",
    "guest_mode",
    "request_rules",
}


def test_every_persisted_agent_field_has_backup_classification() -> None:
    classified = {getattr(agent_config, name) for name in BACKED_UP_AGENT_FIELDS}
    assert classified == set(agent_config.AGENT_CONFIG_FIELDS), (
        "A persistent agent field was added or removed. Review its backup semantics "
        "and update BACKED_UP_AGENT_FIELDS explicitly."
    )
    exported = backup.export_configuration_snapshot(
        agent_config.agent_config_defaults()
    )
    assert set(exported) == classified


def test_every_agent_field_has_lifecycle_contract() -> None:
    path = Path(__file__).with_name("agent_field_contract.json")
    contracts = json.loads(path.read_text(encoding="utf-8"))
    assert contracts["schema_version"] == 2
    fields = contracts["fields"]
    assert set(fields) == BACKED_UP_AGENT_FIELDS, (
        "Agent configuration changed: classify each field's save, persistence, "
        "backup, and migration semantics in agent_field_contract.json"
    )
    required = {"save_reload", "persistence", "backup", "migration", "coverage"}
    for name, contract in fields.items():
        assert set(contract) == required, name
        assert contract["save_reload"] == "config_subentry", name
        assert contract["persistence"] == "ha_config_entries", name
        assert contract["backup"] == "full_and_setup", name
        assert contract["migration"] in {"standard", "special"}, name
        assert contract["coverage"]["kind"] in {
            "combinatorial_dimension", "derived", "bounded_payload",
            "covered_separately", "intentionally_excluded",
        }, name
        assert contract["coverage"].get("dimension") or contract["coverage"].get("reason"), name


def test_new_agent_field_fails_lifecycle_contract() -> None:
    """Keep the inventory's negative case executable without modifying production."""
    path = Path(__file__).with_name("agent_field_contract.json")
    fields = json.loads(path.read_text(encoding="utf-8"))["fields"]
    assert set(fields) != BACKED_UP_AGENT_FIELDS | {"CONF_NIGHTLY_DUMMY"}
