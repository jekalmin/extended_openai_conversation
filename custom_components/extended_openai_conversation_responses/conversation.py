"""Extended OpenAI Conversation (Responses) agent entity."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import replace
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Literal, cast

import httpx
from openai import OpenAIError

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    async_get_chat_log,
)
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval

from . import ExtendedOpenAIConfigEntry
from .agent_config import function_tool_enabled
from .agent_configuration import (
    _archive_runtime_required,
    async_ensure_optional_manager,
    async_reconcile_runtime_configuration,
    sync_memory_embedding_provider,
)
from .agent_maintenance import conversation_request_lease
from .const import (
    CONF_ARCHIVE_ENABLED,
    CONF_ARCHIVE_MODEL_SEARCH_ENABLED,
    CONF_ARCHIVE_RETENTION_DAYS,
    CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    CONF_CHAT_MODEL,
    CONF_CONTINUE_CONVERSATION,
    CONF_CONVERSATION_CONTINUITY,
    CONF_CONVERSATION_TIMEOUT_MINUTES,
    CONF_FUNCTION_GROUPS,
    CONF_GUEST_MODE_ENABLED,
    CONF_KNOWLEDGE_ENABLED,
    CONF_MEMORY_AUTO_RETRIEVE_LIMIT,
    CONF_MEMORY_EMBEDDING_MODEL,
    CONF_MEMORY_RETRIEVAL_MODE,
    CONF_SHARED_ARCHIVE_ENABLED,
    CONF_SHARED_MEMORY_MODE,
    CONF_SKILLS,
    CONF_TEMPORARY_MEMORY,
    CONF_USAGE_REQUEST_RETENTION_DAYS,
    CONF_USAGE_RUN_RETENTION_DAYS,
    CONTINUE_CONVERSATION_ALWAYS,
    CONTINUE_CONVERSATION_CONDITIONAL,
    DEFAULT_ARCHIVE_ENABLED,
    DEFAULT_ARCHIVE_MODEL_SEARCH_ENABLED,
    DEFAULT_ARCHIVE_RETENTION_DAYS,
    DEFAULT_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_CONVERSATION_CONTINUITY,
    DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
    DEFAULT_FUNCTION_GROUPS,
    DEFAULT_KNOWLEDGE_ENABLED,
    DEFAULT_MEMORY_AUTO_RETRIEVE_LIMIT,
    DEFAULT_MEMORY_EMBEDDING_MODEL,
    DEFAULT_MEMORY_RETRIEVAL_MODE,
    DEFAULT_SHARED_ARCHIVE_ENABLED,
    DEFAULT_SHARED_MEMORY_MODE,
    DEFAULT_TEMPORARY_MEMORY,
    DEFAULT_USAGE_REQUEST_RETENTION_DAYS,
    DEFAULT_USAGE_RUN_RETENTION_DAYS,
    DEFAULT_WORKING_DIRECTORY,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
    MEMORY_RETRIEVAL_HYBRID,
    SHARED_MEMORY_AUTOMATIC,
    SHARED_MEMORY_DISABLED,
    SUBSYSTEM_STATUS_KEY,
    TEMPORARY_MEMORY_OFF,
)
from .continuity import (
    GUEST_CONTINUITY_NAMESPACE,
    ConversationContinuity,
    async_get_continuity,
)
from .conversation_archive import ArchiveSession, ConversationArchive, async_get_archive
from .conversation_id_ownership import (
    claim_conversation_id,
    register_conversation_id_cleanup,
)
from .conversation_lifecycle import (
    async_reset_conversation_context,
    begin_conversation_lifecycle,
    end_conversation_lifecycle,
    request_fresh_conversation,
    requested_conversation_reset,
)
from .debug import (
    conversation_debug_trace,
    current_debug_trace,
    model_path_timing,
    record_memory_retrieval,
    record_system_prompt,
)
from .entity import ExtendedOpenAIBaseLLMEntity
from .exceptions import FunctionLoadFailed, FunctionNotFound, InvalidFunction
from .function_groups import (
    FunctionGroupRuntime,
    FunctionGroupSession,
    assemble_function_tools,
    load_function_groups,
    remove_function_group_runtime,
    reset_function_group_runtime,
)
from .function_tool_quarantine import (
    _runtime_configured_function_tools as configured_function_tools_from_data,
    _runtime_validate_function_groups as validate_function_groups,
)
from .function_tool_resolution import (
    configured_function_tool_for_execution,
    latest_function_tool_for_execution,
)
from .functions.security import (
    FunctionSecurity,
    classify_tool,
    contains_indirect_service_call,
)
from .guest_mode import (
    EXECUTION_FAILED,
    GUEST_MODE_UNAVAILABLE,
    GuestCapabilityPolicy,
    GuestModeDenied,
    GuestModeManager,
    async_get_guest_mode,
    can_reuse_request_policy,
    execution_failure_result,
    guest_arguments_allowed_runtime,
    guest_mode_denial_result,
    resolve_guest_policy,
)
from .ha_llm_tools import (
    ToolSnapshot,
    async_discover,
    current_snapshot,
    is_ha_tool,
    tool_snapshot_scope,
)
from .ha_permissions import bind_active_ha_context
from .ha_tool_result_compat import make_tool_result_content, tool_result_data
from .helpers import get_exposed_entities
from .knowledge import KnowledgeLibrary, async_get_knowledge, search_result_as_dict
from .local_intents import LocalIntentResult, async_try_handle_local_intent
from .memory import (
    MemoryRecord,
    PersistentMemory,
    async_get_memory,
    automatic_memory_enabled,
    memory_enabled,
    memory_user_id,
)
from .model_search_hardening import _require_nonblank_query
from .model_tool_results import (
    _compact_json_result_content,
    _compact_memory_result,
    knowledge_search_payload,
    model_memory_as_dict,
    omit_null_paging_cursor,
)
from .prompt import render_effective_prompt
from .prompt_cache import _PROMPT_CACHE_CONTEXT, prompt_cache_context
from .regex_execution import async_process_speech_text
from .request import assemble_integration_function_tools
from .request_diagnostics import (
    record_exposed_entities,
    record_prompt_render,
    record_tool_assembly,
    record_tool_execution,
)
from .request_rules import (
    RequestRuleRuntime,
    RequestRules,
    async_evaluate_rule,
    async_get_request_rules,
    get_request_rule_runtime,
    request_rule_session_id,
)
from .request_static_cache import formatted_tool_cache, tools_for_available_skills
from .runtime_failure_hardening import _conversation_error_result
from .runtime_hardening import bounded_tool_result_text
from .scope import (
    SHARED_HOUSEHOLD_SCOPE_ID,
    ResolvedDataScope,
    memory_scope_id,
    resolve_data_scope,
)
from .skill_runtime_availability import effective_tool_runtime_scope
from .skills import Skill, SkillManager
from .speech import has_custom_speech_replacements, process_speech_text
from .temporary_memory import (
    _ACTIVE_OWNER_SCOPE_ID,
    TemporaryMemory,
    TemporaryMemoryRecord,
    _owner_from_resolved_scope,
    async_get_temporary_memory,
    temporary_memory_as_dict,
)
from .usage import async_get_usage
from .voice_identity_runtime import voice_identity_scope

_TEMPORARY_MEMORY_PREFETCH: ContextVar[asyncio.Task[Any] | None] = ContextVar(
    "extended_openai_temporary_memory_prefetch", default=None
)

_LOGGER = logging.getLogger(__name__)

_ACTIVE_SCOPE: ContextVar[ResolvedDataScope | None] = ContextVar(
    "extended_openai_active_scope", default=None
)
_ACTIVE_ARCHIVE: ContextVar[tuple[str, str] | None] = ContextVar(
    "extended_openai_active_archive", default=None
)
_ACTIVE_TEMPORARY_SCOPE: ContextVar[str | None] = ContextVar(
    "extended_openai_active_temporary_scope", default=None
)
_ACTIVE_FUNCTION_GROUP_SESSION: ContextVar[FunctionGroupSession | None] = ContextVar(
    "extended_openai_active_function_group_session", default=None
)
_ACTIVE_MEMORY_SESSION: ContextVar[tuple[str, int] | None] = ContextVar(
    "extended_openai_active_memory_session", default=None
)
_ACTIVE_GUEST_POLICY: ContextVar[GuestCapabilityPolicy | None] = ContextVar(
    "extended_openai_active_guest_policy", default=None
)
_ACTIVE_FUNCTION_CONFIG: ContextVar[
    tuple[list[dict[str, Any]], list[dict[str, Any]] | None] | None
] = ContextVar("extended_openai_active_function_config", default=None)
_ACTIVE_LLM_CONTEXT: ContextVar[Any | None] = ContextVar(
    "extended_openai_active_llm_context", default=None
)
_ACTIVE_RUNTIME_RECONCILED: ContextVar[bool] = ContextVar(
    "extended_openai_active_runtime_reconciled", default=False
)
_ACTIVE_REQUEST_CONFIG_DATA: ContextVar[Any | None] = ContextVar(
    "extended_openai_active_request_config_data", default=None
)
_PROCESS_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "extended_openai_process_metadata", default=None
)


def _request_llm_context(user_input: ConversationInput) -> Any:
    """Reuse the request-scoped LLM context, with a direct-call fallback."""
    return _ACTIVE_LLM_CONTEXT.get() or user_input.as_llm_context(DOMAIN)


def _request_function_groups(
    entity: Any, configured_tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate Function Groups once for the request-stable config revision."""
    request_config = _ACTIVE_FUNCTION_CONFIG.get()
    if request_config is not None and request_config[0] is configured_tools:
        cached_groups = request_config[1]
        if cached_groups is not None:
            return cached_groups
    groups = validate_function_groups(
        entity.subentry.data.get(CONF_FUNCTION_GROUPS, list(DEFAULT_FUNCTION_GROUPS)),
        configured_tools,
    )
    if request_config is not None and request_config[0] is configured_tools:
        _ACTIVE_FUNCTION_CONFIG.set((configured_tools, groups))
    return groups


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ExtendedOpenAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the OpenAI Conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [ExtendedOpenAIAgentEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class ExtendedOpenAIAgentEntity(
    ConversationEntity,
    conversation.AbstractConversationAgent,
    ExtendedOpenAIBaseLLMEntity,
):
    """Extended OpenAI conversation agent."""

    _attr_supports_streaming = True
    _attr_supported_features = ConversationEntityFeature.CONTROL
    skill_manager: SkillManager
    _memory: PersistentMemory | None = None
    _temporary_memory: TemporaryMemory | None = None
    _continuity: ConversationContinuity | None = None
    _knowledge: KnowledgeLibrary | None = None
    _archive: ConversationArchive | None = None
    _function_groups_runtime: FunctionGroupRuntime | None = None
    _guest_mode: GuestModeManager | None = None
    _request_rules: RequestRules | None = None
    _request_rule_runtime: RequestRuleRuntime | None = None

    def __init__(self, entry: ExtendedOpenAIConfigEntry, subentry: Any) -> None:
        """Initialize the conversation agent and its streaming capability."""
        super().__init__(entry, subentry)
        # Arbitrary regex can depend on future text. Home Assistant selects whether
        # to attach its progressive listener from this per-agent capability flag.
        self._attr_supports_streaming = not has_custom_speech_replacements(
            subentry.data
        )
        if not self._attr_supports_streaming:
            _LOGGER.debug(
                "Progressive TTS disabled because custom speech regex rules are configured"
            )

    @property
    def supports_streaming(self) -> bool:
        """Expose live capability before HA attaches its progressive listener."""
        return not has_custom_speech_replacements(self.subentry.data)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @property
    def skills(self) -> list[str]:
        """Get the enabled skills list for this entity."""
        return self.subentry.data.get(CONF_SKILLS, []) or []

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        await self._async_initialize_agent_state()
        await self._async_initialize_optional_managers()
        # Shared managers can retain a provider bound to the previous entity.
        sync_memory_embedding_provider(self)
        self._schedule_archive_retention()

    async def _async_initialize_agent_state(self) -> None:
        """Load required shared state and reset per-registration request state."""
        # Calculate skills directory based on working directory
        working_dir = DEFAULT_WORKING_DIRECTORY
        if Path(working_dir).is_absolute():
            skills_dir = Path(working_dir) / "skills"
        else:
            skills_dir = Path(self.hass.config.config_dir) / working_dir / "skills"

        self.skill_manager = await SkillManager.async_get_instance(
            self.hass, user_skills_dir=str(skills_dir)
        )

        self._usage = await async_get_usage(
            self.hass, self.entry.entry_id, self.subentry.subentry_id
        )
        self._guest_mode = await async_get_guest_mode(
            self.hass, self.entry.entry_id, self.subentry.subentry_id
        )
        self._usage.request_retention_days = int(
            self.subentry.data.get(
                CONF_USAGE_REQUEST_RETENTION_DAYS,
                DEFAULT_USAGE_REQUEST_RETENTION_DAYS,
            )
        )
        self._usage.run_retention_days = int(
            self.subentry.data.get(
                CONF_USAGE_RUN_RETENTION_DAYS, DEFAULT_USAGE_RUN_RETENTION_DAYS
            )
        )
        await self._usage.async_prune_details()
        self._continuity = async_get_continuity(
            self.hass, self.entry.entry_id, self.subentry.subentry_id
        )
        self._function_groups_runtime = reset_function_group_runtime(
            self.hass, self.entry.entry_id, self.subentry.subentry_id
        )
        self._request_rules = await async_get_request_rules(
            self.hass, self.entry.entry_id, self.subentry.subentry_id
        )
        self._request_rule_runtime = get_request_rule_runtime(
            self.hass, self.entry.entry_id, self.subentry.subentry_id
        )

    async def _async_initialize_optional_managers(self) -> None:
        """Load independent optional capabilities concurrently."""
        temporary_configured = (
            self.subentry.data.get(CONF_TEMPORARY_MEMORY, DEFAULT_TEMPORARY_MEMORY)
            != TEMPORARY_MEMORY_OFF
        )
        archive_configured = bool(
            self.subentry.data.get(CONF_ARCHIVE_ENABLED, DEFAULT_ARCHIVE_ENABLED)
        )
        knowledge_configured = bool(
            self.subentry.data.get(CONF_KNOWLEDGE_ENABLED, DEFAULT_KNOWLEDGE_ENABLED)
        )
        memory_configured = memory_enabled(self.subentry.data)

        self._set_subsystem_status("temporary_memory", temporary_configured)
        self._set_subsystem_status("knowledge", knowledge_configured)
        self._set_subsystem_status("persistent_memory", memory_configured)

        async def initialize_temporary_memory() -> None:
            if not temporary_configured:
                return
            await async_ensure_optional_manager(
                self,
                attribute="_temporary_memory",
                subsystem="temporary_memory",
                loader=async_get_temporary_memory,
                failure_message="Unable to initialize temporary memory",
            )

        async def initialize_knowledge() -> None:
            try:
                self._knowledge = await async_get_knowledge(
                    self.hass, self.entry.entry_id, self.subentry.subentry_id
                )
            except Exception as err:
                self._set_subsystem_status("knowledge", knowledge_configured, err)
                _LOGGER.exception("Unable to initialize Knowledge Library")
            else:
                self._set_subsystem_status(
                    "knowledge", knowledge_configured, healthy=True
                )

        async def initialize_persistent_memory() -> None:
            if not memory_configured:
                return
            initialized = await async_ensure_optional_manager(
                self,
                attribute="_memory",
                subsystem="persistent_memory",
                loader=async_get_memory,
                failure_message="Unable to initialize persistent memory",
            )
            if (
                initialized
                and self._memory is not None
                and (
                    self.subentry.data.get(
                        CONF_MEMORY_RETRIEVAL_MODE, DEFAULT_MEMORY_RETRIEVAL_MODE
                    )
                    == MEMORY_RETRIEVAL_HYBRID
                )
            ):
                self._memory.set_embedding_provider(
                    self._async_create_embeddings,
                    str(
                        self.subentry.data.get(
                            CONF_MEMORY_EMBEDDING_MODEL,
                            DEFAULT_MEMORY_EMBEDDING_MODEL,
                        )
                    ),
                )

        await asyncio.gather(
            initialize_temporary_memory(),
            self._async_initialize_archive(archive_configured),
            initialize_knowledge(),
            initialize_persistent_memory(),
        )

    def _schedule_archive_retention(self) -> None:
        """Tie daily retention to this entity's HA registration lifetime."""
        self.async_on_remove(
            async_track_time_interval(
                self.hass, self._async_prune_archive_retention, timedelta(days=1)
            )
        )

    async def _async_prune_archive_retention(
        self, _now: datetime | None = None
    ) -> None:
        """Apply live retention settings independently of conversation traffic."""
        if self._archive is None:
            return
        try:
            await self._archive.async_prune(
                int(
                    self.subentry.data.get(
                        CONF_ARCHIVE_RETENTION_DAYS, DEFAULT_ARCHIVE_RETENTION_DAYS
                    )
                )
            )
        except Exception:
            _LOGGER.exception(
                "Background conversation archive retention maintenance failed"
            )

    async def _async_initialize_archive(self, configured: bool) -> None:
        """Initialize archive storage without taking down the conversation agent."""
        if not configured and not self.subentry.data.get(
            CONF_ARCHIVE_MODEL_SEARCH_ENABLED, DEFAULT_ARCHIVE_MODEL_SEARCH_ENABLED
        ):
            self._archive = None
            self._set_subsystem_status("archive", False)
            return
        self._set_subsystem_status("archive", configured)
        try:
            self._archive = await async_get_archive(
                self.hass, self.entry.entry_id, self.subentry.subentry_id
            )
            await self._archive.async_prune(
                int(
                    self.subentry.data.get(
                        CONF_ARCHIVE_RETENTION_DAYS, DEFAULT_ARCHIVE_RETENTION_DAYS
                    )
                )
            )
        except Exception as err:
            self._archive = None
            self._set_subsystem_status("archive", configured, err)
            _LOGGER.exception(
                "Unable to initialize conversation archive; archive features are unavailable"
            )
        else:
            self._set_subsystem_status("archive", configured, healthy=True)

    def _set_subsystem_status(
        self,
        subsystem: str,
        configured: bool,
        error: Exception | None = None,
        *,
        healthy: bool = False,
    ) -> None:
        """Publish a non-sensitive runtime status for diagnostics."""
        statuses = self.hass.data.setdefault(SUBSYSTEM_STATUS_KEY, {})
        agent = statuses.setdefault(
            (self.entry.entry_id, self.subentry.subentry_id), {}
        )
        state = "disabled" if not configured else "healthy" if healthy else "failed"
        agent[subsystem] = {
            "configured": configured,
            "status": state,
            **({"error_type": type(error).__name__} if error is not None else {}),
        }

    async def _async_begin_archive_session(
        self,
        session_key: str,
        scope: ResolvedDataScope,
        home_assistant_conversation_id: str | None,
        request_policy: GuestCapabilityPolicy,
    ) -> ArchiveSession | None:
        """Begin optional archive retention without taking down the conversation."""
        if self._archive is None or not request_policy.archive_retention:
            return None
        configured = bool(
            self.subentry.data.get(CONF_ARCHIVE_ENABLED, DEFAULT_ARCHIVE_ENABLED)
        )
        try:
            session = await self._archive.async_begin_session(
                session_key,
                scope,
                home_assistant_conversation_id,
                archive_enabled=configured,
                shared_archive_enabled=bool(
                    self.subentry.data.get(
                        CONF_SHARED_ARCHIVE_ENABLED, DEFAULT_SHARED_ARCHIVE_ENABLED
                    )
                ),
                inactivity_minutes=int(
                    self.subentry.data.get(
                        CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
                        DEFAULT_ARCHIVE_SESSION_TIMEOUT_MINUTES,
                    )
                ),
            )
        except Exception as err:
            self._set_subsystem_status("archive", configured, err)
            _LOGGER.exception(
                "Unable to begin conversation archive session; continuing without "
                "archiving this turn"
            )
            return None
        self._set_subsystem_status("archive", configured, healthy=True)
        return session

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        remove_function_group_runtime(
            self.hass, self.entry.entry_id, self.subentry.subentry_id
        )
        await super().async_will_remove_from_hass()

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        return await self._async_process(user_input)

    async def async_process_direct(
        self, user_input: ConversationInput
    ) -> tuple[ConversationResult, dict[str, Any]]:
        """Process through the identical pipeline while collecting safe metadata."""
        metadata: dict[str, Any] = {"handled_locally": False}
        token = _PROCESS_METADATA.set(metadata)
        try:
            return await self._async_process(user_input), dict(metadata)
        finally:
            _PROCESS_METADATA.reset(token)

    async def _async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Own the complete request boundary shared by Assist and direct processing.

        Order is intentional: authenticated HA context and maintenance admission
        surround debug capture, request caches, Voice Identity and reconciliation.
        No installer may replace this method or any of its request-entry owners.
        """
        with bind_active_ha_context(user_input.context):
            async with conversation_request_lease(self):
                with conversation_debug_trace(self, user_input) as trace:
                    with formatted_tool_cache():
                        config_token = _ACTIVE_REQUEST_CONFIG_DATA.set(
                            self.subentry.data
                        )
                        try:
                            async with voice_identity_scope(self, user_input):
                                await async_reconcile_runtime_configuration(self)
                                reconciled_token = _ACTIVE_RUNTIME_RECONCILED.set(True)
                                try:
                                    result = await self._async_process_with_continuity(
                                        user_input
                                    )
                                finally:
                                    _ACTIVE_RUNTIME_RECONCILED.reset(reconciled_token)
                        finally:
                            _ACTIVE_REQUEST_CONFIG_DATA.reset(config_token)
                    if trace is not None:
                        trace.result = result
                    return result

    async def _async_process_with_continuity(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Resolve the request policy/scope and release any claimed continuity turn."""
        cache_token = _PROMPT_CACHE_CONTEXT.set(None)
        try:
            llm_context = user_input.as_llm_context(DOMAIN)
            llm_context_token = _ACTIVE_LLM_CONTEXT.set(llm_context)
            request_policy = self._resolve_live_guest_policy()
            guest_policy_token = _ACTIVE_GUEST_POLICY.set(request_policy)
            try:
                source_device_id = user_input.satellite_id or user_input.device_id
                scope = resolve_data_scope(
                    SimpleNamespace(
                        context=llm_context.context,
                        device_id=source_device_id,
                    ),
                    self.subentry.data,
                )
                continuity_mode = self.subentry.data.get(
                    CONF_CONVERSATION_CONTINUITY, DEFAULT_CONVERSATION_CONTINUITY
                )
                timeout_minutes = int(
                    self.subentry.data.get(
                        CONF_CONVERSATION_TIMEOUT_MINUTES,
                        DEFAULT_CONVERSATION_TIMEOUT_MINUTES,
                    )
                )
                source_device_id = source_device_id or scope.device_id
                assert self._continuity is not None
                resolution = await self._continuity.async_resolve(
                    continuity_mode,
                    scope,
                    source_device_id,
                    user_input.conversation_id,
                    timeout_minutes,
                    namespace=(
                        GUEST_CONTINUITY_NAMESPACE
                        if request_policy.guest_active
                        else None
                    ),
                )
                resolved_conversation_id = getattr(
                    resolution, "conversation_id", user_input.conversation_id
                )
                claimed_conversation_id = claim_conversation_id(
                    getattr(self, "hass", None),
                    getattr(getattr(self, "subentry", None), "subentry_id", None),
                    scope,
                    resolved_conversation_id,
                    guest_active=request_policy.guest_active,
                )
                if claimed_conversation_id != resolved_conversation_id:
                    resolution = replace(
                        resolution, conversation_id=claimed_conversation_id
                    )
                try:
                    return await self._async_process_claimed(
                        user_input,
                        llm_context,
                        request_policy,
                        scope,
                        source_device_id,
                        timeout_minutes,
                        resolution,
                    )
                finally:
                    await asyncio.shield(
                        self._continuity.async_release(
                            resolution.key, resolution.claim_token
                        )
                    )
            finally:
                _ACTIVE_GUEST_POLICY.reset(guest_policy_token)
                _ACTIVE_LLM_CONTEXT.reset(llm_context_token)
        finally:
            _PROMPT_CACHE_CONTEXT.reset(cache_token)

    async def _async_process_claimed(
        self,
        user_input: ConversationInput,
        llm_context: Any,
        request_policy: GuestCapabilityPolicy,
        scope: ResolvedDataScope,
        source_device_id: str | None,
        timeout_minutes: int,
        resolution: Any,
    ) -> ConversationResult:
        """Process a request after continuity ownership has been claimed."""
        continuity = self._continuity
        assert continuity is not None
        user_input.conversation_id = resolution.conversation_id
        context_id = getattr(getattr(llm_context, "context", None), "id", None)
        session_key = (
            f"continuity:{resolution.key}"
            if resolution.key
            else f"conversation:{user_input.conversation_id}"
            if user_input.conversation_id
            else f"context:{context_id or scope.device_id or 'unidentified'}"
        )
        archive_session = await self._async_begin_archive_session(
            session_key,
            scope,
            user_input.conversation_id,
            request_policy,
        )
        scope_token = _ACTIVE_SCOPE.set(scope)
        archive_token = _ACTIVE_ARCHIVE.set(
            (session_key, archive_session.session_id) if archive_session else None
        )
        memory_session_token = _ACTIVE_MEMORY_SESSION.set(
            (session_key, timeout_minutes if resolution.key else 5)
        )
        with (
            async_get_chat_session(self.hass, resolution.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            subentry_id = getattr(getattr(self, "subentry", None), "subentry_id", None)
            claimed_id = resolution.conversation_id
            if claimed_id is not None and isinstance(subentry_id, str):
                claim_owner = (
                    subentry_id,
                    "guest" if request_policy.guest_active else scope.scope_id,
                )
                register_conversation_id_cleanup(
                    session,
                    getattr(self, "hass", None),
                    claimed_id,
                    claim_owner,
                )
            rule_session_key = request_rule_session_id(
                resolution.key, chat_log.conversation_id
            )
            temporary_scope = (
                None
                if request_policy.guest_active or not scope.allows_retention
                else resolution.key or f"conversation:{chat_log.conversation_id}"
            )
            temporary_token = _ACTIVE_TEMPORARY_SCOPE.set(temporary_scope)
            function_group_session = (
                self._function_groups_runtime.begin(
                    (
                        f"continuity:{resolution.key}"
                        if resolution.key
                        else f"conversation:{chat_log.conversation_id}"
                    ),
                    timeout_minutes if resolution.key else 5,
                )
                if self._function_groups_runtime is not None
                else None
            )
            function_group_token = _ACTIVE_FUNCTION_GROUP_SESSION.set(
                function_group_session
            )
            lifecycle_token = begin_conversation_lifecycle()
            if resolution.history and len(chat_log.content) <= 2:
                # Core chat logs expire independently after five minutes. Restore the
                # integration-owned bounded model history when Core recreates the log.
                current_user = chat_log.content[-1]
                chat_log.content[:] = [*resolution.history, current_user]
            try:
                try:
                    evaluation = (
                        await async_evaluate_rule(
                            self.hass,
                            self._request_rules,
                            self._request_rule_runtime,
                            user_input.text,
                            rule_session_key,
                            str(
                                self.subentry.data.get(
                                    CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL
                                )
                            ),
                            request_policy,
                            timeout_minutes,
                            lambda name, arguments: (
                                self._async_execute_request_rule_function(
                                    name, arguments, llm_context
                                )
                            ),
                            context=user_input.context,
                        )
                        if self._request_rules is not None
                        and self._request_rule_runtime is not None
                        else None
                    )
                except HomeAssistantError as err:
                    _LOGGER.warning("Request Rule routing rejected: %s", err)
                    return self._local_rule_result(
                        user_input,
                        chat_log,
                        f"Sorry, this Request Rule cannot be used: {err}",
                        successful=False,
                    )
                metadata = _PROCESS_METADATA.get()
                if metadata is not None and evaluation is not None:
                    metadata["matched_rule"] = {
                        "id": evaluation.match.rule["id"],
                        "name": evaluation.match.rule["name"],
                    }
                    metadata["captured_values"] = dict(evaluation.match.slots)
                if evaluation is not None and evaluation.consume:
                    return await self._async_complete_local_rule(
                        user_input,
                        chat_log,
                        evaluation.response or "Done",
                        archive_session,
                        resolution.key,
                        resolution.claim_token,
                        source_device_id,
                        successful=evaluation.successful,
                    )
                if evaluation is not None and evaluation.provider_input is not None:
                    current_user = chat_log.content[-1]
                    if not isinstance(current_user, conversation.UserContent):
                        raise HomeAssistantError(
                            "Current request is unavailable for AI handoff"
                        )
                    chat_log.content[-1] = replace(
                        current_user, content=evaluation.provider_input
                    )
                request_options = (
                    self._request_rule_runtime.effective_options(
                        self.subentry.data,
                        rule_session_key,
                        evaluation.request_override if evaluation else None,
                        timeout_minutes,
                    )
                    if self._request_rule_runtime is not None
                    else dict(self.subentry.data)
                )
                local_intent = (
                    await async_try_handle_local_intent(
                        self.hass,
                        user_input,
                        chat_log,
                        request_options,
                        guest_active=request_policy.guest_active,
                    )
                    if evaluation is None
                    else None
                )
                if self._usage is None:
                    result = (
                        self._local_intent_result(user_input, chat_log, local_intent)
                        if local_intent is not None
                        else await self._async_handle_message_with_ha_tools(
                            user_input, chat_log, request_options
                        )
                    )
                    if chat_log.content and isinstance(
                        chat_log.content[-1], conversation.AssistantContent
                    ):
                        await continuity.async_record_success(
                            resolution.key, resolution.claim_token, chat_log.content
                        )
                    return result
                async with self._usage.async_run(
                    home_assistant_conversation_id=user_input.conversation_id,
                    source_device_id=source_device_id,
                ) as run:
                    result = (
                        self._local_intent_result(user_input, chat_log, local_intent)
                        if local_intent is not None
                        else await self._async_handle_message_with_ha_tools(
                            user_input, chat_log, request_options
                        )
                    )
                    await self._async_archive_turn(
                        archive_session,
                        run.run_id,
                        user_input,
                        chat_log,
                        successful=run.successful,
                    )
                    if run.successful:
                        await continuity.async_record_success(
                            resolution.key, resolution.claim_token, chat_log.content
                        )
                    return result
            finally:
                reset_request = requested_conversation_reset()
                try:
                    if reset_request is not None:
                        await asyncio.shield(
                            async_reset_conversation_context(
                                self.hass,
                                continuity,
                                self.entry.entry_id,
                                self.subentry.subentry_id,
                                continuity_key=resolution.key,
                                state_session_id=reset_request.state_session_id,
                                memory_session_id=reset_request.memory_session_id,
                            )
                        )
                finally:
                    end_conversation_lifecycle(lifecycle_token)
                    _ACTIVE_FUNCTION_GROUP_SESSION.reset(function_group_token)
                    _ACTIVE_TEMPORARY_SCOPE.reset(temporary_token)
                    _ACTIVE_ARCHIVE.reset(archive_token)
                    _ACTIVE_MEMORY_SESSION.reset(memory_session_token)
                    _ACTIVE_SCOPE.reset(scope_token)

    async def _async_handle_message_with_ha_tools(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
        request_options: Mapping[str, Any] | None = None,
    ) -> ConversationResult:
        """Resolve HA references and cache one validated config revision per request."""
        llm_context = _request_llm_context(user_input)
        configured = self._configured_function_tools_from_data(self.subentry.data)
        references = [
            tool["function"]
            for tool in configured
            if is_ha_tool(tool) and function_tool_enabled(tool)
        ]
        snapshot = ToolSnapshot()
        if references and not self._effective_guest_policy().guest_active:
            snapshot = await async_discover(self.hass, llm_context, references)
        with tool_snapshot_scope(snapshot):
            function_config_token = _ACTIVE_FUNCTION_CONFIG.set(
                (snapshot.project(configured), None)
            )
            try:
                return await self._async_handle_message(
                    user_input, chat_log, request_options
                )
            finally:
                _ACTIVE_FUNCTION_CONFIG.reset(function_config_token)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
        request_options: Mapping[str, Any] | None = None,
    ) -> ConversationResult:
        """Own model timings and completed-response isolated speech processing."""
        try:
            deferred_speech: list[tuple[str, Mapping[str, Any]]] = []
            with model_path_timing(self):
                result = await ExtendedOpenAIAgentEntity._async_generate_message(
                    self,
                    user_input,
                    chat_log,
                    request_options,
                    deferred_speech=deferred_speech,
                )
            if deferred_speech:
                result.response.async_set_speech(
                    await async_process_speech_text(self.hass, *deferred_speech[0])
                )
            return result
        except (OpenAIError, HomeAssistantError, httpx.RequestError) as err:
            return _conversation_error_result(self, user_input, chat_log, err)

    async def _async_generate_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
        request_options: Mapping[str, Any] | None = None,
        *,
        deferred_speech: list[tuple[str, Mapping[str, Any]]],
    ) -> ConversationResult:
        """Call the API."""
        llm_context = _request_llm_context(user_input)

        # Get exposed entities for function tools
        exposed_entities = self._get_exposed_entities()

        retrieved_memories = await self._async_retrieve_memories(
            llm_context,
            user_input.text,
            runtime_reconciled=_ACTIVE_RUNTIME_RECONCILED.get(),
        )
        temporary_memories = await self._async_retrieve_temporary_memories()

        # Build custom prompt with exposed entities
        system_prompt = self._build_system_prompt(
            exposed_entities,
            llm_context,
            user_input,
            retrieved_memories,
            temporary_memories,
        )

        # Set system prompt in chat log
        chat_log.content[0] = conversation.SystemContent(content=system_prompt)

        # Call the LLM

        try:
            check_aba = getattr(self, "_assert_no_aba_configuration", None)
            if callable(check_aba):
                check_aba()
            continue_mode = _get_continue_conversation_mode(self.subentry.data)
            conditional_decision = await self._async_handle_chat_log(
                chat_log,
                function_tools=[],
                exposed_entities=exposed_entities,
                llm_context=llm_context,
                conditional_continue=(
                    continue_mode == CONTINUE_CONVERSATION_CONDITIONAL
                ),
                function_tools_factory=self._get_function_tools,
                function_group_loader=(
                    self._load_function_groups
                    if _ACTIVE_FUNCTION_GROUP_SESSION.get() is not None
                    else None
                ),
                request_options=request_options,
            )
        except (OpenAIError, HomeAssistantError, httpx.RequestError) as err:
            return _conversation_error_result(
                self,
                user_input,
                chat_log,
                err,
                logger=_LOGGER,
                provider_log_message="OpenAI conversation request failed",
                conversation_id=(
                    None
                    if requested_conversation_reset() is not None
                    else user_input.conversation_id
                ),
            )

        # Fire conversation finished event
        self._fire_conversation_finished(user_input, chat_log, status="success")

        # Build response from chat log
        intent_response = intent.IntentResponse(language=user_input.language)

        # Get last assistant message
        last_content = chat_log.content[-1]
        if isinstance(last_content, conversation.AssistantContent):
            original_text = last_content.content or ""
            if has_custom_speech_replacements(self.subentry.data):
                deferred_speech.append((original_text, self.subentry.data))
                speech_text = original_text
            else:
                speech_text = process_speech_text(original_text, self.subentry.data)
            intent_response.async_set_speech(speech_text)
        else:
            intent_response.async_set_speech("")

        return ConversationResult(
            response=intent_response,
            conversation_id=(
                None
                if requested_conversation_reset() is not None
                else chat_log.conversation_id
            ),
            continue_conversation=_resolve_continue_conversation(
                continue_mode,
                chat_log.continue_conversation,
                conditional_decision,
            ),
        )

    def _local_rule_result(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
        response: str,
        *,
        successful: bool,
    ) -> ConversationResult:
        """Return a local rule response without invoking the provider."""
        metadata = _PROCESS_METADATA.get()
        if metadata is not None:
            metadata["handled_locally"] = True
        chat_log.content.append(
            conversation.AssistantContent(agent_id=self.entity_id, content=response)
        )
        self._fire_conversation_finished(
            user_input, chat_log, status="local", handled_locally=True
        )
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response)
        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=successful
            and _resolve_continue_conversation(
                _get_continue_conversation_mode(self.subentry.data), False, None
            ),
        )

    def _local_intent_result(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
        local_intent: LocalIntentResult,
    ) -> ConversationResult:
        """Return Home Assistant's original local intent response without a provider call."""
        metadata = _PROCESS_METADATA.get()
        if metadata is not None:
            metadata["handled_locally"] = True
            metadata["matched_intent"] = local_intent.intent_name
        speech = local_intent.response.speech.get("plain", {}).get("speech", "")
        chat_log.content.append(
            conversation.AssistantContent(
                agent_id=conversation.HOME_ASSISTANT_AGENT,
                content=speech,
            )
        )
        self._fire_conversation_finished(
            user_input, chat_log, status="local", handled_locally=True
        )
        return ConversationResult(
            response=local_intent.response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=local_intent.response.error_code is None
            and _resolve_continue_conversation(
                _get_continue_conversation_mode(self.subentry.data), False, None
            ),
        )

    def _fire_conversation_finished(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
        *,
        status: str,
        handled_locally: bool = False,
        error_type: str | None = None,
    ) -> None:
        """Emit exactly one completion event with backward-compatible fields."""
        payload: dict[str, Any] = {
            "user_input": user_input,
            "messages": [content.as_dict() for content in chat_log.content],
            "agent_id": self.subentry.subentry_id,
            "status": status,
        }
        if handled_locally:
            payload["handled_locally"] = True
        if error_type is not None:
            payload["error_type"] = error_type
        self.hass.bus.async_fire(EVENT_CONVERSATION_FINISHED, payload)

    async def _async_archive_turn(
        self,
        archive_session: ArchiveSession | None,
        run_id: str | None,
        user_input: ConversationInput,
        chat_log: ChatLog,
        *,
        successful: bool,
    ) -> None:
        """Archive a completed turn without changing its user-facing outcome."""
        if (
            self._archive is None
            or archive_session is None
            or not self._effective_guest_policy().archive_retention
        ):
            return
        assistant_text = ""
        if chat_log.content and isinstance(
            chat_log.content[-1], conversation.AssistantContent
        ):
            assistant_text = chat_log.content[-1].content or ""
        try:
            await self._archive.async_record_turn(
                archive_session.session_id,
                run_id=run_id,
                user_text=user_input.text,
                assistant_text=assistant_text,
                successful=successful,
            )
        except Exception:
            _LOGGER.exception(
                "Unable to persist completed conversation turn to the archive"
            )

    async def _async_complete_local_rule(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
        response: str,
        archive_session: ArchiveSession | None,
        continuity_key: str | None,
        continuity_claim_token: str | None,
        source_device_id: str | None,
        *,
        successful: bool = True,
    ) -> ConversationResult:
        """Finalize a locally consumed rule as a zero-provider-request run."""
        if self._usage is not None:
            async with self._usage.async_run(
                home_assistant_conversation_id=user_input.conversation_id,
                source_device_id=source_device_id,
            ) as run:
                result = self._local_rule_result(
                    user_input, chat_log, response, successful=successful
                )
                if not successful:
                    self._usage.mark_current_run_failed("RequestRuleExecutionFailed")
                await self._async_archive_turn(
                    archive_session,
                    run.run_id,
                    user_input,
                    chat_log,
                    successful=successful,
                )
        else:
            result = self._local_rule_result(
                user_input, chat_log, response, successful=successful
            )
            await self._async_archive_turn(
                archive_session,
                None,
                user_input,
                chat_log,
                successful=successful,
            )
        assert self._continuity is not None
        if successful:
            await self._continuity.async_record_success(
                continuity_key, continuity_claim_token, chat_log.content
            )
        return result

    async def _async_execute_request_rule_function(
        self,
        function_name: str,
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
    ) -> Any:
        """Execute a configured function through the model-tool security seam."""
        function_tool = configured_function_tool_for_execution(self, function_name)
        result = await self._execute_function_tool(
            function_tool,
            llm.ToolInput(
                id="request_rule",
                tool_name=function_name,
                tool_args=arguments,
                external=True,
            ),
            llm_context,
            self._get_exposed_entities(),
        )
        result_value = tool_result_data(result).get("result", "")
        try:
            outcome = json.loads(result_value) if isinstance(result_value, str) else {}
        except json.JSONDecodeError:
            outcome = {}
        if outcome.get("status") == "denied" and outcome.get("reason") == "guest_mode":
            raise GuestModeDenied(GUEST_MODE_UNAVAILABLE)
        if (
            outcome.get("status") == "error"
            and outcome.get("reason") == "execution_failed"
        ):
            raise HomeAssistantError(EXECUTION_FAILED)
        return result

    def _build_system_prompt(
        self,
        exposed_entities: list[dict],
        llm_context: llm.LLMContext,
        user_input: ConversationInput,
        memories: list[MemoryRecord] | None = None,
        temporary_memories: list[TemporaryMemoryRecord] | None = None,
    ) -> str:
        """Build system prompt with exposed entities and skills."""
        started = time.monotonic()
        policy = self._effective_guest_policy()
        render_started = time.monotonic()
        effective = render_effective_prompt(
            self.hass,
            self.subentry.data,
            exposed_entities=exposed_entities,
            current_device_id=None if policy.guest_active else llm_context.device_id,
            user_input=user_input,
            skills=self._get_enabled_skills(),
            memories=memories,
            temporary_memories=temporary_memories,
            knowledge_available=self._knowledge_available,
            guest_policy=policy,
        )
        record_prompt_render(effective, render_started)
        _PROMPT_CACHE_CONTEXT.set(prompt_cache_context(effective, self.subentry.data))
        record_system_prompt(effective.text, started)
        return effective.text

    async def _async_retrieve_memories(
        self,
        llm_context: llm.LLMContext,
        query: str,
        *,
        runtime_reconciled: bool = False,
    ) -> list[MemoryRecord]:
        """Retrieve live-enabled memory while overlapping temporary retrieval."""
        existing = _TEMPORARY_MEMORY_PREFETCH.get()
        task = existing
        if task is None and self._temporary_memory is not None:
            task = asyncio.create_task(self._async_load_temporary_memories())
            _TEMPORARY_MEMORY_PREFETCH.set(task)
        try:
            if not memory_enabled(self.subentry.data):
                return []
            if not runtime_reconciled:
                sync_memory_embedding_provider(self)
            started = time.monotonic()
            records = await self._async_select_memories(llm_context, query)
            record_memory_retrieval("persistent", started, records)
            return records
        except BaseException:
            if existing is None and task is not None:
                task.cancel()
                with suppress(BaseException):
                    await task
                _TEMPORARY_MEMORY_PREFETCH.set(None)
            raise

    async def _async_select_memories(
        self, llm_context: llm.LLMContext, query: str
    ) -> list[MemoryRecord]:
        """Select once, then resolve the same bundle without automatic reranking."""
        if self._memory is None:
            return []
        readable_scope_ids = self._current_readable_memory_scope_ids(llm_context)
        if not readable_scope_ids:
            return []
        try:
            retrieve_limit = int(
                self.subentry.data.get(
                    CONF_MEMORY_AUTO_RETRIEVE_LIMIT,
                    DEFAULT_MEMORY_AUTO_RETRIEVE_LIMIT,
                )
            )
            if retrieve_limit <= 0:
                return []
            active_session = _ACTIVE_MEMORY_SESSION.get()
            if active_session is None or self._continuity is None:
                return await self._async_rank_memories(
                    readable_scope_ids, query, retrieve_limit
                )
            session_key, timeout_minutes = active_session
            references = await self._continuity.async_get_memory_bundle(
                session_key, timeout_minutes
            )
            if references is None:
                selected = await self._async_rank_memories(
                    readable_scope_ids, query, retrieve_limit
                )
                references = await self._continuity.async_set_memory_bundle(
                    session_key,
                    [(memory.user_id, memory.memory_id) for memory in selected],
                    timeout_minutes,
                )
            return await self._memory.async_get_many(references, readable_scope_ids)
        except Exception:
            _LOGGER.exception("Automatic memory retrieval failed; continuing")
            return []

    async def _async_rank_memories(
        self, readable_scope_ids: list[str], query: str, limit: int
    ) -> list[MemoryRecord]:
        if not query.strip():
            return []
        return await self._async_search_memories(readable_scope_ids, query, limit)

    async def _async_search_memories(
        self,
        readable_scope_ids: list[str],
        query: str,
        limit: int,
        category: str | None = None,
    ) -> list[MemoryRecord]:
        """Search using the configured lexical or Hybrid retrieval semantics."""
        assert self._memory is not None
        if not readable_scope_ids:
            return []
        hybrid = (
            self.subentry.data.get(
                CONF_MEMORY_RETRIEVAL_MODE, DEFAULT_MEMORY_RETRIEVAL_MODE
            )
            == MEMORY_RETRIEVAL_HYBRID
        )
        query_embedding = (
            await self._memory.async_prepare_hybrid(readable_scope_ids, query)
            if hybrid
            else None
        )
        search_scope: str | list[str] = (
            readable_scope_ids[0]
            if not hybrid and len(readable_scope_ids) == 1
            else readable_scope_ids
        )
        return await self._memory.async_search(
            search_scope,
            query,
            category,
            limit,
            query_embedding=query_embedding,
            hybrid=hybrid and query_embedding is not None,
        )

    async def _async_create_embeddings(self, inputs: list[str]) -> list[list[float]]:
        """Create embedding vectors without an LLM/classifier retrieval call."""
        response = await self._client.embeddings.create(
            model=self.subentry.data.get(
                CONF_MEMORY_EMBEDDING_MODEL, DEFAULT_MEMORY_EMBEDDING_MODEL
            ),
            input=inputs,
        )
        return [list(item.embedding) for item in response.data]

    async def _async_retrieve_temporary_memories(self) -> list[TemporaryMemoryRecord]:
        """Consume one prefetch, checking live capability before exposing records."""
        task = _TEMPORARY_MEMORY_PREFETCH.get()
        _TEMPORARY_MEMORY_PREFETCH.set(None)
        try:
            if (
                self.subentry.data.get(CONF_TEMPORARY_MEMORY, DEFAULT_TEMPORARY_MEMORY)
                == TEMPORARY_MEMORY_OFF
                or _ACTIVE_TEMPORARY_SCOPE.get() is None
                or _owner_from_resolved_scope(_ACTIVE_SCOPE.get()) is None
                or not self._effective_guest_policy().temporary_memory
            ):
                if task is not None:
                    task.cancel()
                    # A cancelled child is expected; cancellation of this request is not.
                    await asyncio.gather(task, return_exceptions=True)
                return []
            if task is not None:
                return cast(list[TemporaryMemoryRecord], await task)
            return await self._async_load_temporary_memories()
        except BaseException:
            if task is not None:
                task.cancel()
                with suppress(BaseException):
                    await task
            raise

    async def _async_load_temporary_memories(self) -> list[TemporaryMemoryRecord]:
        """Load active facts under the resolved retained owner, including prefetch."""
        if (
            self.subentry.data.get(CONF_TEMPORARY_MEMORY, DEFAULT_TEMPORARY_MEMORY)
            == TEMPORARY_MEMORY_OFF
            or not self._effective_guest_policy().temporary_memory
        ):
            return []
        scope_id = _ACTIVE_TEMPORARY_SCOPE.get()
        owner = _owner_from_resolved_scope(_ACTIVE_SCOPE.get())
        if self._temporary_memory is None or scope_id is None or owner is None:
            return []
        token = _ACTIVE_OWNER_SCOPE_ID.set(owner)
        started = time.monotonic()
        try:
            try:
                records = await self._temporary_memory.async_active(scope_id)
            except Exception:
                _LOGGER.exception("Temporary-memory retrieval failed; continuing")
                records = []
            record_memory_retrieval("temporary", started, records)
            return records
        finally:
            _ACTIVE_OWNER_SCOPE_ID.reset(token)

    def _get_enabled_skills(self) -> list[Skill]:
        """Get enabled skills as list for template rendering."""
        if not self._effective_guest_policy().skills:
            return []
        enabled_skill_names = self.skills
        if not enabled_skill_names:
            return []
        all_skills = self.skill_manager.get_all_skills()

        return [s for s in all_skills if s.name in enabled_skill_names]

    def _get_exposed_entities(self) -> list[dict[str, Any]]:
        started = time.monotonic()
        result = self._filter_guest_entities(get_exposed_entities(self.hass))
        record_exposed_entities(result, started)
        return result

    def _resolve_live_guest_policy(self) -> GuestCapabilityPolicy:
        """Resolve policy from current state without expanding a request policy."""
        try:
            configured = self._get_configured_function_tools()
        except Exception:
            configured = []
        return resolve_guest_policy(
            self.hass, self.subentry.data, self._guest_mode, configured
        )

    def _effective_guest_policy(self) -> GuestCapabilityPolicy:
        """Hold request permissions stable while allowing mid-request tightening."""
        request_policy = _ACTIVE_GUEST_POLICY.get()
        if request_policy is not None and can_reuse_request_policy(
            request_policy, getattr(self, "_guest_mode", None)
        ):
            return request_policy
        if request_policy is not None and request_policy.guest_active:
            return request_policy
        live_policy = self._resolve_live_guest_policy()
        if live_policy.guest_active:
            # Pin a mid-request activation so a later trusted disable cannot
            # expand this in-flight request. The next user turn resolves afresh.
            _ACTIVE_GUEST_POLICY.set(live_policy)
            return live_policy
        return request_policy or GuestCapabilityPolicy.unrestricted()

    def _filter_guest_entities(
        self, entities: list[dict[str, Any]], *, control: bool = False
    ) -> list[dict[str, Any]]:
        policy = self._effective_guest_policy()
        if not policy.guest_active:
            return entities
        allows = policy.allows_entity_control if control else policy.allows_entity_read
        return [
            entity
            for entity in entities
            if isinstance(entity.get("entity_id"), str)
            and allows(str(entity["entity_id"]))
        ]

    def _get_function_tools(self) -> list[dict[str, Any]]:
        """Get the effective configured and integration-owned function tools."""
        assembly_started = time.monotonic()
        try:
            configured_tools = self._get_configured_function_tools()
            policy = self._effective_guest_policy()
            groups = _request_function_groups(self, configured_tools)
            manager = getattr(self, "skill_manager", None)
            if not isinstance(manager, SkillManager):
                manager = SkillManager.get_loaded_instance()
            with effective_tool_runtime_scope(
                self.subentry.data, configured_tools, manager, groups
            ):
                configured_tools, groups = self._filter_guest_tools_and_groups(
                    configured_tools, groups, policy
                )
                session = _ACTIVE_FUNCTION_GROUP_SESSION.get()
                assembly = assemble_function_tools(
                    tools_for_available_skills(
                        configured_tools,
                        policy.skills,
                    ),
                    groups,
                    session.loaded_group_ids if session is not None else set(),
                )
                if self._function_groups_runtime is not None:
                    self._function_groups_runtime.record_request(assembly)
                result = list(assembly.tools)
                configured_names = {
                    tool.get("spec", {}).get("name")
                    for tool in configured_tools
                    if isinstance(tool, dict)
                }
                result.extend(
                    assemble_integration_function_tools(
                        self.subentry.data,
                        configured_names,
                        memory_scope_available=self._current_memory_scope_id()
                        is not None,
                        temporary_scope_available=(
                            self._temporary_memory is not None
                            and _ACTIVE_TEMPORARY_SCOPE.get() is not None
                        ),
                        knowledge_available=self._knowledge_available,
                        archive_available=self._archive is not None,
                        guest_policy=policy,
                    )
                )
                record_tool_assembly(self, result, assembly_started)
                return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except Exception as e:
            raise FunctionLoadFailed() from e

    def _get_configured_function_tools(self) -> list[dict[str, Any]]:
        """Return this request's projected tools, or build them for direct callers."""
        request_config = _ACTIVE_FUNCTION_CONFIG.get()
        if request_config is not None:
            return request_config[0]
        return current_snapshot().project(
            self._configured_function_tools_from_data(self.subentry.data)
        )

    def _configured_function_tools_from_data(self, data: Any) -> list[dict[str, Any]]:
        """Parse configured tools from one current or persisted data mapping."""
        return configured_function_tools_from_data(data)

    def _load_function_groups(self, requested: Any) -> dict[str, Any]:
        """Apply an integration-owned loader call to the active conversation."""
        session = _ACTIVE_FUNCTION_GROUP_SESSION.get()
        if session is None:
            return {
                "status": "error",
                "error": "No active conversation is available for group loading",
            }
        configured_tools = self._get_configured_function_tools()
        policy = self._effective_guest_policy()
        groups = _request_function_groups(self, configured_tools)
        manager = getattr(self, "skill_manager", None)
        if not isinstance(manager, SkillManager):
            manager = SkillManager.get_loaded_instance()
        with effective_tool_runtime_scope(
            self.subentry.data, configured_tools, manager, groups
        ):
            configured_tools, groups = self._filter_guest_tools_and_groups(
                configured_tools, groups, policy
            )
            return load_function_groups(
                session,
                requested,
                groups,
                tools_for_available_skills(
                    configured_tools,
                    policy.skills,
                ),
            )

    @staticmethod
    def _filter_guest_tools_and_groups(
        configured_tools: list[dict[str, Any]],
        groups: list[dict[str, Any]],
        policy: GuestCapabilityPolicy,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Apply the resolved Guest function policy without orphaning members."""
        if not policy.guest_active:
            return configured_tools, groups
        membership = {
            name: group for group in groups for name in group.get("functions", [])
        }
        configured_tools = [
            tool
            for tool in configured_tools
            if not is_ha_tool(tool)
            and policy.allows_configured_tool(tool["spec"]["name"])
            and (
                not policy.legacy_function_flags
                or (
                    tool["spec"]["name"] not in membership
                    or membership[tool["spec"]["name"]].get("guest_allowed") is True
                )
            )
        ]
        allowed_names = {tool["spec"]["name"] for tool in configured_tools}
        guest_groups = [
            {
                **group,
                "functions": [
                    name for name in group["functions"] if name in allowed_names
                ],
            }
            for group in groups
            if (not policy.legacy_function_flags or group.get("guest_allowed") is True)
            and any(name in allowed_names for name in group.get("functions", []))
        ]
        return configured_tools, guest_groups

    async def _execute_function_tool(
        self,
        function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> conversation.ToolResultContent:
        """Own tool dispatch, bounded model projection, and attempt diagnostics."""
        trace = current_debug_trace()
        started = time.monotonic()
        successful = False
        content: Any = None
        try:
            # A provider response may arrive after HA exposure changes. Native
            # actions must use the current Assist-visible set at dispatch.
            if function_tool.get("function", {}).get("type") == "native" and isinstance(
                getattr(self, "hass", None), HomeAssistant
            ):
                exposed_entities = self._get_exposed_entities()
            content = await self._async_dispatch_function_tool(
                function_tool, tool_input, llm_context, exposed_entities
            )
            payload = tool_result_data(content)
            if isinstance(payload, dict) and isinstance(payload.get("result"), str):
                payload["result"] = bounded_tool_result_text(payload["result"])
            content = _compact_json_result_content(content)
            successful = True
            return cast(conversation.ToolResultContent, content)
        finally:
            record_tool_execution(
                trace, function_tool, tool_input, started, successful, content
            )

    def _assert_no_aba_configuration(self) -> None:
        """Reject a request whose HA config mapping left and returned to A."""
        started_with = _ACTIVE_REQUEST_CONFIG_DATA.get()
        if started_with is None:
            return
        current = self.subentry.data
        if current is not started_with and current == started_with:
            raise HomeAssistantError(
                "Agent configuration changed during this request; please retry"
            )

    async def _async_dispatch_function_tool(
        self,
        function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> conversation.ToolResultContent:
        """Execute an integration-owned tool or a configured tool."""
        check_aba = getattr(self, "_assert_no_aba_configuration", None)
        if callable(check_aba):
            check_aba()
        function_type = function_tool.get("function", {}).get("type")
        policy = self._effective_guest_policy()
        if function_type == "ha_llm" and policy.guest_active:
            return self._tool_result(tool_input, guest_mode_denial_result())
        if function_type == "guest_mode":
            try:
                result = await self._async_execute_guest_mode_tool(tool_input.tool_args)
            except (RuntimeError, ValueError) as err:
                result = {"status": "error", "error": str(err)}
            return self._tool_result(tool_input, result)
        if function_type not in {
            "memory",
            "temporary_memory",
            "knowledge",
            "archive",
            "conversation_lifecycle",
        }:
            tool_name = function_tool.get("spec", {}).get("name")
            try:
                resolved_tool = latest_function_tool_for_execution(self, function_tool)
            except FunctionNotFound:
                if policy.guest_active:
                    return self._tool_result(
                        tool_input,
                        guest_mode_denial_result(),
                    )
                raise
            guest_entities = exposed_entities
            if policy.guest_active:
                latest_entry = self.hass.config_entries.async_get_entry(
                    self.entry.entry_id
                )
                latest_subentry = (
                    latest_entry.subentries.get(self.subentry.subentry_id)
                    if latest_entry is not None
                    else None
                )
                latest_data = (
                    latest_subentry.data
                    if latest_subentry is not None
                    else self.subentry.data
                )
                current_configured = self._configured_function_tools_from_data(
                    latest_data
                )
                current_tool = next(
                    (
                        tool
                        for tool in current_configured
                        if tool.get("spec", {}).get("name") == tool_name
                    ),
                    None,
                )
                if current_tool is None:
                    return self._tool_result(tool_input, guest_mode_denial_result())
                if (
                    (
                        policy.legacy_function_flags
                        and current_tool.get("guest_allowed") is not True
                    )
                    or not policy.allows_configured_tool(str(tool_name))
                    or self._is_guest_unscopable_tool(current_tool)
                ):
                    return self._tool_result(tool_input, guest_mode_denial_result())
                current_groups = validate_function_groups(
                    latest_data.get(
                        CONF_FUNCTION_GROUPS, list(DEFAULT_FUNCTION_GROUPS)
                    ),
                    current_configured,
                )
                allowed_tools, _allowed_groups = self._filter_guest_tools_and_groups(
                    current_configured, current_groups, policy
                )
                if str(tool_name) not in {
                    tool["spec"]["name"] for tool in allowed_tools
                }:
                    return self._tool_result(tool_input, guest_mode_denial_result())
                control = self._is_control_tool(current_tool)
                if control and contains_indirect_service_call(tool_input.tool_args):
                    return self._tool_result(tool_input, guest_mode_denial_result())
                if not self._guest_arguments_allowed_runtime(
                    tool_input.tool_args, policy, control=control
                ):
                    return self._tool_result(tool_input, guest_mode_denial_result())
                if control and not self._guest_arguments_allowed_runtime(
                    current_tool.get("function", {}), policy, control=True
                ):
                    return self._tool_result(tool_input, guest_mode_denial_result())
                guest_entities = self._filter_guest_entities(
                    get_exposed_entities(self.hass), control=control
                )
            try:
                return await super()._execute_function_tool(
                    resolved_tool, tool_input, llm_context, guest_entities
                )
            except Exception:
                if policy.guest_active:
                    _LOGGER.warning("Guest-permitted tool execution failed")
                    return self._tool_result(
                        tool_input,
                        execution_failure_result(),
                    )
                raise

        try:
            if policy.guest_active and not self._guest_integration_allowed(
                function_type, function_tool.get("function", {}).get("operation", "")
            ):
                raise RuntimeError(GUEST_MODE_UNAVAILABLE)
            if function_type == "conversation_lifecycle":
                result = self._execute_conversation_lifecycle_tool(
                    function_tool["function"]["operation"]
                )
            elif function_type == "archive":
                result = await self._async_execute_archive_tool(
                    function_tool["function"]["operation"], tool_input.tool_args
                )
            elif function_type == "knowledge":
                result = await self._async_execute_knowledge_tool(
                    function_tool["function"]["operation"], tool_input.tool_args
                )
            elif function_type == "memory":
                result = await self._async_execute_memory_tool(
                    function_tool["function"]["operation"],
                    tool_input.tool_args,
                    llm_context,
                )
            else:
                result = await self._async_execute_temporary_memory_tool(
                    function_tool["function"]["operation"], tool_input.tool_args
                )
        except (RuntimeError, ValueError) as err:
            result = {"status": "error", "error": str(err)}
        except Exception:
            if function_type in {"memory", "temporary_memory"}:
                _LOGGER.exception("Memory tool failed")
                result = {
                    "status": "unavailable",
                    "error": "Memory is temporarily unavailable",
                }
            elif function_type == "archive":
                _LOGGER.exception("Conversation Archive tool failed")
                result = {
                    "status": "unavailable",
                    "error": "Conversation Archive is temporarily unavailable",
                }
            else:
                _LOGGER.exception("Knowledge Library tool failed")
                result = {
                    "status": "unavailable",
                    "error": "Knowledge Library is temporarily unavailable",
                }

        return self._tool_result(tool_input, result)

    def _tool_result(
        self, tool_input: llm.ToolInput, result: dict[str, Any]
    ) -> conversation.ToolResultContent:
        return make_tool_result_content(
            agent_id=self.entity_id,
            tool_call_id=tool_input.id,
            tool_name=tool_input.tool_name,
            tool_result={"result": json.dumps(result, ensure_ascii=False)},
        )

    def _execute_conversation_lifecycle_tool(self, operation: str) -> dict[str, Any]:
        """Schedule a lifecycle change for the exact active conversation."""
        if operation != "start_fresh":
            raise ValueError("unknown conversation lifecycle operation")
        state_session = _ACTIVE_FUNCTION_GROUP_SESSION.get()
        memory_session = _ACTIVE_MEMORY_SESSION.get()
        request_fresh_conversation(
            state_session.session_key if state_session is not None else None,
            memory_session[0] if memory_session is not None else None,
        )
        return {
            "status": "scheduled",
            "message": "The next Assist request will start with fresh conversation context.",
        }

    async def _async_execute_guest_mode_tool(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if (
            self._guest_mode is None
            or self.subentry.data.get(CONF_GUEST_MODE_ENABLED, False) is not True
        ):
            raise RuntimeError("Guest Mode is unavailable")
        return await self._guest_mode.async_restrict(
            active_from=arguments.get("active_from"),
            active_until=arguments.get("active_until"),
            make_indefinite=arguments.get("make_indefinite", False) is True,
        )

    def _guest_integration_allowed(self, function_type: str, operation: str) -> bool:
        policy = self._effective_guest_policy()
        if function_type == "conversation_lifecycle":
            return operation == "start_fresh"
        if function_type == "memory":
            return (
                policy.shared_memory_read
                if operation in {"search", "list"}
                else policy.shared_memory_write
            )
        if function_type == "knowledge":
            return policy.knowledge_access
        if function_type == "archive":
            return policy.archive_access
        if function_type == "temporary_memory":
            return policy.temporary_memory
        return False

    @staticmethod
    def _is_control_tool(tool: Mapping[str, Any]) -> bool:
        return classify_tool(tool) == FunctionSecurity.CONTROL

    @staticmethod
    def _is_guest_unscopable_tool(tool: Mapping[str, Any]) -> bool:
        """Deny configured operations that cannot be entity-scoped reliably."""
        return classify_tool(tool) > FunctionSecurity.CONTROL

    @staticmethod
    def _guest_arguments_allowed(
        value: Any, policy: GuestCapabilityPolicy, *, control: bool
    ) -> bool:
        """Fail closed for explicit entity and broad target selectors."""
        allows = policy.allows_entity_control if control else policy.allows_entity_read

        def inspect(item: Any, key: str | None = None) -> bool:
            if isinstance(item, Mapping):
                for child_key, child in item.items():
                    normalized = str(child_key).lower()
                    if normalized in {"area_id", "area_ids", "device_id", "device_ids"}:
                        return False
                    if not inspect(child, normalized):
                        return False
                return True
            if isinstance(item, list):
                return all(inspect(child, key) for child in item)
            if key in {"entity_id", "entity_ids", "statistic_id", "statistic_ids"}:
                values = (
                    [part.strip() for part in item.split(",")]
                    if isinstance(item, str)
                    else []
                )
                return bool(values) and all(allows(entity_id) for entity_id in values)
            return True

        return inspect(value)

    def _guest_arguments_allowed_runtime(
        self, value: Any, policy: GuestCapabilityPolicy, *, control: bool
    ) -> bool:
        """Resolve broad HA selectors and require every matched entity to pass."""
        return guest_arguments_allowed_runtime(
            self.hass, value, policy, control=control
        )

    def _provider_tool_allowed(self, tool_type: str) -> bool:
        policy = self._effective_guest_policy()
        return tool_type != "web_search" or policy.web_search

    @property
    def _knowledge_available(self) -> bool:
        """Return whether an enabled, populated library is ready."""
        return bool(
            self._effective_guest_policy().knowledge_access
            and self.subentry.data.get(CONF_KNOWLEDGE_ENABLED, False)
            and self._knowledge is not None
            and self._knowledge.source_count > 0
        )

    async def _async_execute_knowledge_tool(
        self, operation: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a read-only Knowledge Library operation."""
        if operation == "search":
            _require_nonblank_query(arguments)
        if not self._knowledge_available or self._knowledge is None:
            raise RuntimeError("Knowledge Library is unavailable")
        if operation == "search":
            query = arguments.get("query")
            source_ids = arguments.get("source_ids")
            limit = arguments.get("limit", 5)
            if (
                not isinstance(query, str)
                or (
                    source_ids is not None
                    and (
                        not isinstance(source_ids, list)
                        or not all(isinstance(item, str) for item in source_ids)
                    )
                )
                or not isinstance(limit, int)
                or isinstance(limit, bool)
            ):
                raise ValueError("query, source_ids, or limit has an invalid type")
            allowed_ids, ignored_ids = self._knowledge.resolve_source_filter(source_ids)
            policy_ids = self._effective_guest_policy().knowledge_source_ids
            if policy_ids is not None:
                valid_policy_ids, _private_ignored = (
                    self._knowledge.resolve_source_filter(list(policy_ids))
                )
                requested = (
                    valid_policy_ids
                    if allowed_ids is None
                    else allowed_ids & (valid_policy_ids or set())
                )
                allowed_ids = set(requested or ())
                # Never reveal whether a supplied ID exists but is forbidden.
                ignored_ids = []
                if not allowed_ids:
                    return {
                        "results": [],
                        "source_filter": {
                            "applied_source_ids": [],
                            "ignored_source_ids": [],
                            "fell_back_to_all_sources": False,
                        },
                    }
            results = await self._knowledge.async_search(
                query, sorted(allowed_ids) if allowed_ids else None, limit
            )
            filter_requested = bool(source_ids)
            return knowledge_search_payload(
                {
                    "results": [search_result_as_dict(result) for result in results],
                    "source_filter": {
                        "applied_source_ids": sorted(allowed_ids or []),
                        "ignored_source_ids": ignored_ids,
                        "fell_back_to_all_sources": filter_requested
                        and allowed_ids is None,
                    },
                },
                filter_requested=filter_requested,
                policy_filter_applied=policy_ids is not None,
            )
        if operation == "list":
            query = arguments.get("query")
            limit = arguments.get("limit", 20)
            offset = arguments.get("offset", 0)
            if (
                (query is not None and not isinstance(query, str))
                or not isinstance(limit, int)
                or isinstance(limit, bool)
                or not isinstance(offset, int)
                or isinstance(offset, bool)
            ):
                raise ValueError("query, limit, or offset has an invalid type")
            return omit_null_paging_cursor(
                await self._knowledge.async_catalog(
                    query,
                    limit,
                    offset,
                    self._effective_guest_policy().knowledge_source_ids,
                ),
                "next_offset",
            )
        if operation == "get":
            source_id = arguments.get("source_id")
            start = arguments.get("start_character", 0)
            maximum = arguments.get("max_characters", 6_000)
            if (
                not isinstance(source_id, str)
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(maximum, int)
                or isinstance(maximum, bool)
            ):
                raise ValueError(
                    "source_id, start_character, or max_characters has an invalid type"
                )
            allowed_sources = self._effective_guest_policy().knowledge_source_ids
            if allowed_sources is not None and source_id not in allowed_sources:
                raise RuntimeError(GUEST_MODE_UNAVAILABLE)
            return omit_null_paging_cursor(
                await self._knowledge.async_get_section(source_id, start, maximum),
                "next_start_character",
            )
        raise ValueError("unknown knowledge operation")

    async def _async_execute_memory_tool(
        self,
        operation: str,
        arguments: dict[str, Any],
        llm_context: llm.LLMContext | None,
    ) -> dict[str, Any]:
        """Execute a scoped persistent-memory operation."""
        if operation == "search":
            _require_nonblank_query(arguments)
        if not memory_enabled(self.subentry.data):
            raise RuntimeError("persistent memory is disabled")
        if self._memory is None:
            raise RuntimeError("persistent memory is unavailable")
        readable_scope_ids = self._current_readable_memory_scope_ids(llm_context)
        if not readable_scope_ids:
            raise RuntimeError("persistent memory is disabled for this data scope")

        if operation in {"add", "upsert"}:
            source = arguments.get("source")
            content = arguments.get("content")
            category = arguments.get("category")
            if (
                not isinstance(content, str)
                or not isinstance(category, str)
                or not isinstance(source, str)
            ):
                raise ValueError("content, category, and source must be strings")
            if source == "implicit" and not automatic_memory_enabled(
                self.subentry.data
            ):
                raise ValueError("automatic memory creation is disabled")
            scope = _ACTIVE_SCOPE.get()
            if (
                source == "implicit"
                and scope is not None
                and scope.scope_type == "shared"
                and self.subentry.data.get(
                    CONF_SHARED_MEMORY_MODE, DEFAULT_SHARED_MEMORY_MODE
                )
                != SHARED_MEMORY_AUTOMATIC
            ):
                raise ValueError("automatic shared memory creation is disabled")
            write_scope_id = self._current_write_memory_scope_id(
                arguments.get("scope"), llm_context, source=source
            )
            method = (
                self._memory.async_upsert
                if operation == "upsert"
                else self._memory.async_add
            )
            metadata = {
                name: arguments[name]
                for name in ("importance", "subject", "key", "valid_from")
                if name in arguments
            }
            return _compact_memory_result(
                await method(write_scope_id, content, category, source, **metadata)
            )
        if operation == "search":
            query = arguments.get("query")
            category = arguments.get("category")
            limit = arguments.get("limit", 5)
            if (
                not isinstance(query, str)
                or (category is not None and not isinstance(category, str))
                or not isinstance(limit, int)
                or isinstance(limit, bool)
            ):
                raise ValueError("query, category, or limit has an invalid type")
            requested_scope = arguments.get("scope")
            search_scopes = self._filter_read_scopes(
                readable_scope_ids, requested_scope
            )
            memories = await self._async_search_memories(
                search_scopes,
                query,
                limit,
                category,
            )
            personal_id = self._personal_memory_scope_id(llm_context)
            return {
                "memories": [
                    model_memory_as_dict(
                        memory, include_scope=True, personal_scope_id=personal_id
                    )
                    for memory in memories
                ]
            }
        if operation == "list":
            category = arguments.get("category")
            limit = arguments.get("limit", 50)
            offset = arguments.get("offset", 0)
            if (
                (category is not None and not isinstance(category, str))
                or not isinstance(limit, int)
                or isinstance(limit, bool)
                or not isinstance(offset, int)
                or isinstance(offset, bool)
            ):
                raise ValueError("category, limit, or offset has an invalid type")
            list_scopes = self._filter_read_scopes(
                readable_scope_ids, arguments.get("scope")
            )
            memories = await self._memory.async_list(
                list_scopes,
                category,
                limit,
                offset,
            )
            personal_id = self._personal_memory_scope_id(llm_context)
            return {
                "memories": [
                    model_memory_as_dict(
                        memory, include_scope=True, personal_scope_id=personal_id
                    )
                    for memory in memories
                ]
            }
        if operation == "update":
            memory_id = arguments.get("memory_id")
            content = arguments.get("content")
            category = arguments.get("category")
            metadata = {
                key: arguments[key]
                for key in ("importance", "subject", "key", "valid_from")
                if arguments.get(key) is not None
            }
            clear_fields = arguments.get("clear_fields")
            if clear_fields is not None and (
                not isinstance(clear_fields, list)
                or not all(isinstance(field, str) for field in clear_fields)
            ):
                raise ValueError("clear_fields must be a list of metadata field names")
            if (
                not isinstance(memory_id, str)
                or (content is not None and not isinstance(content, str))
                or (category is not None and not isinstance(category, str))
                or (
                    content is None
                    and category is None
                    and not metadata
                    and not clear_fields
                )
            ):
                raise ValueError("memory_id and at least one valid update are required")
            write_scope_id = self._current_write_memory_scope_id(
                arguments.get("scope"), llm_context, source="explicit"
            )
            memory = await self._memory.async_update(
                write_scope_id,
                memory_id,
                content,
                category,
                **metadata,
                clear_fields=clear_fields,
            )
            return {
                "status": "updated",
                "memory": model_memory_as_dict(
                    memory,
                    include_scope=True,
                    personal_scope_id=self._personal_memory_scope_id(llm_context),
                ),
            }
        if operation == "delete":
            memory_ids = arguments.get("memory_ids")
            if not isinstance(memory_ids, list) or not all(
                isinstance(memory_id, str) for memory_id in memory_ids
            ):
                raise ValueError("memory_ids must be a list of strings")
            write_scope_id = self._current_write_memory_scope_id(
                arguments.get("scope"), llm_context, source="explicit"
            )
            deleted = await self._memory.async_delete(write_scope_id, memory_ids)
            return {"status": "deleted", "deleted": deleted}
        raise ValueError("unknown memory operation")

    def _current_memory_scope_id(
        self, llm_context: llm.LLMContext | None = None
    ) -> str | None:
        """Use the session-stable resolver for all memory operations."""
        scope = _ACTIVE_SCOPE.get()
        if scope is None:
            # Compatibility for direct service/test invocations outside the
            # conversation lifecycle. Live conversations always set a resolved
            # scope before tools are assembled.
            return memory_user_id(llm_context)
        if (
            scope.scope_type == "shared"
            and self.subentry.data.get(
                CONF_SHARED_MEMORY_MODE, DEFAULT_SHARED_MEMORY_MODE
            )
            == SHARED_MEMORY_DISABLED
        ):
            return None
        return memory_scope_id(scope)

    def _personal_memory_scope_id(
        self, llm_context: llm.LLMContext | None = None
    ) -> str | None:
        scope = _ACTIVE_SCOPE.get()
        if scope is not None:
            return scope.user_id if scope.scope_type == "user" else None
        fallback = memory_user_id(llm_context)
        return fallback if fallback != "__anonymous__" else None

    def _current_readable_memory_scope_ids(
        self, llm_context: llm.LLMContext | None = None
    ) -> list[str]:
        """Compose personal and enabled household scopes without crossing users."""
        policy = self._effective_guest_policy()
        if policy.guest_active:
            return (
                [SHARED_HOUSEHOLD_SCOPE_ID]
                if policy.shared_memory_read
                and self.subentry.data.get(
                    CONF_SHARED_MEMORY_MODE, DEFAULT_SHARED_MEMORY_MODE
                )
                != SHARED_MEMORY_DISABLED
                else []
            )
        scope = _ACTIVE_SCOPE.get()
        shared_enabled = (
            self.subentry.data.get(CONF_SHARED_MEMORY_MODE, DEFAULT_SHARED_MEMORY_MODE)
            != SHARED_MEMORY_DISABLED
        )
        if scope is None:
            fallback = memory_user_id(llm_context)
            return [fallback] if fallback else []
        if scope.scope_type == "user" and scope.user_id:
            result = [scope.user_id]
            if shared_enabled:
                result.append(SHARED_HOUSEHOLD_SCOPE_ID)
            return result
        if scope.scope_type == "shared" and shared_enabled:
            return [SHARED_HOUSEHOLD_SCOPE_ID]
        return []

    @staticmethod
    def _filter_read_scopes(readable_scope_ids: list[str], selector: Any) -> list[str]:
        if selector is None:
            return readable_scope_ids
        if selector not in {"personal", "household"}:
            raise ValueError("scope must be personal or household")
        if selector == "household":
            return [
                scope_id
                for scope_id in readable_scope_ids
                if scope_id == SHARED_HOUSEHOLD_SCOPE_ID
            ]
        return [
            scope_id
            for scope_id in readable_scope_ids
            if scope_id != SHARED_HOUSEHOLD_SCOPE_ID
        ]

    def _current_write_memory_scope_id(
        self,
        selector: Any,
        llm_context: llm.LLMContext | None,
        *,
        source: str,
    ) -> str:
        """Resolve a deliberate write target; never infer another user's owner."""
        policy = self._effective_guest_policy()
        if policy.guest_active:
            if not policy.shared_memory_write:
                raise RuntimeError(GUEST_MODE_UNAVAILABLE)
            if selector not in {None, "household"}:
                raise RuntimeError(GUEST_MODE_UNAVAILABLE)
            selector = "household"
        if selector is not None and selector not in {"personal", "household"}:
            raise ValueError("scope must be personal or household")
        scope = _ACTIVE_SCOPE.get()
        if scope is None:
            if selector == "household":
                raise ValueError("household scope requires a resolved conversation")
            return memory_user_id(llm_context)
        if scope.scope_type == "shared":
            if selector == "personal":
                raise ValueError(
                    "shared household conversations cannot write personal memory"
                )
            target = SHARED_HOUSEHOLD_SCOPE_ID
        elif scope.scope_type == "user" and scope.user_id:
            target = (
                SHARED_HOUSEHOLD_SCOPE_ID if selector == "household" else scope.user_id
            )
        else:
            raise RuntimeError("persistent memory is disabled for this data scope")
        if target == SHARED_HOUSEHOLD_SCOPE_ID:
            shared_mode = self.subentry.data.get(
                CONF_SHARED_MEMORY_MODE, DEFAULT_SHARED_MEMORY_MODE
            )
            if shared_mode == SHARED_MEMORY_DISABLED:
                raise ValueError("shared household memory is disabled")
            if source == "implicit" and shared_mode != SHARED_MEMORY_AUTOMATIC:
                raise ValueError("automatic shared memory creation is disabled")
        return target

    async def _async_execute_temporary_memory_tool(
        self, operation: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a narrow operation within the request-derived scope."""
        owner = (
            _owner_from_resolved_scope(_ACTIVE_SCOPE.get())
            if _ACTIVE_TEMPORARY_SCOPE.get() is not None
            else None
        )
        if owner is None:
            raise RuntimeError(
                "temporary memory is unavailable for this retained-data scope"
            )
        if (
            self.subentry.data.get(CONF_TEMPORARY_MEMORY, DEFAULT_TEMPORARY_MEMORY)
            == TEMPORARY_MEMORY_OFF
        ):
            raise RuntimeError("temporary memory is disabled")
        token = _ACTIVE_OWNER_SCOPE_ID.set(owner)
        try:
            if not self._effective_guest_policy().temporary_memory:
                raise RuntimeError(GUEST_MODE_UNAVAILABLE)
            if self._temporary_memory is None:
                raise RuntimeError("temporary memory is unavailable")
            scope_id = _ACTIVE_TEMPORARY_SCOPE.get()
            if scope_id is None:
                raise RuntimeError("temporary memory is unavailable for this request")
            if operation == "add":
                content = arguments.get("content")
                expires_at = arguments.get("expires_at")
                category = arguments.get("category", "general")
                if (
                    not isinstance(content, str)
                    or not isinstance(expires_at, str)
                    or not isinstance(category, str)
                ):
                    raise ValueError(
                        "content, expires_at, and category must be strings"
                    )
                return await self._temporary_memory.async_add(
                    scope_id, content, expires_at, category
                )
            if operation == "update":
                memory_id = arguments.get("memory_id")
                if not isinstance(memory_id, str):
                    raise ValueError("memory_id is required")
                record = await self._temporary_memory.async_update(
                    scope_id,
                    memory_id,
                    arguments.get("content"),
                    arguments.get("expires_at"),
                    arguments.get("category"),
                )
                return {
                    "status": "updated",
                    "memory": temporary_memory_as_dict(record),
                }
            if operation == "delete":
                memory_ids = arguments.get("memory_ids")
                if not isinstance(memory_ids, list) or not all(
                    isinstance(memory_id, str) for memory_id in memory_ids
                ):
                    raise ValueError("memory_ids must be a list of strings")
                return {
                    "status": "deleted",
                    "deleted": await self._temporary_memory.async_delete(
                        scope_id, memory_ids
                    ),
                }
            raise ValueError("unknown temporary-memory operation")
        finally:
            _ACTIVE_OWNER_SCOPE_ID.reset(token)

    async def _async_execute_archive_tool(
        self, operation: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute scoped archive, privacy, and exact-session deletion actions."""
        if operation == "search":
            _require_nonblank_query(arguments)
        try:
            if not _archive_runtime_required(self.subentry.data):
                raise RuntimeError("conversation archive is disabled")
            if self._archive is None:
                raise RuntimeError("conversation archive is unavailable")
            scope = _ACTIVE_SCOPE.get()
            active = _ACTIVE_ARCHIVE.get()
            if scope is None or active is None:
                raise RuntimeError("active conversation session is unavailable")
            session_key, session_id = active
            if operation == "search":
                if not self.subentry.data.get(
                    CONF_ARCHIVE_MODEL_SEARCH_ENABLED,
                    DEFAULT_ARCHIVE_MODEL_SEARCH_ENABLED,
                ):
                    raise RuntimeError("model archive search is disabled")
                query = arguments.get("query")
                if not isinstance(query, str):
                    raise ValueError("query is required")
                return await self._archive.async_search(
                    scope.scope_id,
                    query,
                    start_date=arguments.get("start_date"),
                    end_date=arguments.get("end_date"),
                    limit=int(arguments.get("limit", 5)),
                )
            if operation == "get":
                requested = arguments.get("session_id")
                if not isinstance(requested, str):
                    raise ValueError("session_id is required")
                return await self._archive.async_get(
                    scope.scope_id,
                    requested,
                    int(arguments.get("start_turn", 0)),
                    int(arguments.get("limit", 6)),
                )
            if operation == "private":
                return await self._archive.async_make_private(session_id)
            if operation == "resume":
                session = await self._archive.async_resume_saving(
                    session_key,
                    session_id,
                    scope,
                    shared_archive_enabled=bool(
                        self.subentry.data.get(
                            CONF_SHARED_ARCHIVE_ENABLED, DEFAULT_SHARED_ARCHIVE_ENABLED
                        )
                    ),
                )
                _ACTIVE_ARCHIVE.set((session_key, session.session_id))
                return {
                    "private_mode_enabled": False,
                    "session_id": session.session_id,
                    "future_turns_retained": session.retention_state == "retained",
                    "private_content_restored": False,
                }
            if operation == "delete_current":
                result = await self._archive.async_delete_session(
                    scope.scope_id, session_id
                )
                return {"session_id": session_id, **result}
            if operation == "delete_selected":
                session_ids = arguments.get("session_ids")
                if not isinstance(session_ids, list) or not all(
                    isinstance(value, str) for value in session_ids
                ):
                    raise ValueError("session_ids must be a list of strings")
                return await self._archive.async_delete_selected(
                    scope.scope_id,
                    session_ids,
                    confirm=arguments.get("confirm") is True,
                )
            if operation == "delete_range":
                start_date = arguments.get("start_date")
                end_date = arguments.get("end_date")
                if not isinstance(start_date, str) or not isinstance(end_date, str):
                    raise ValueError("start_date and end_date are required")
                return await self._archive.async_delete_date_range(
                    scope.scope_id,
                    start_date,
                    end_date,
                    confirm=arguments.get("confirm") is True,
                )
            raise ValueError("unknown archive operation")
        except RuntimeError, ValueError:
            raise
        except Exception:
            _LOGGER.exception("Conversation Archive tool failed")
            return {
                "status": "unavailable",
                "error": "Conversation Archive is temporarily unavailable",
            }


def _resolve_continue_conversation(
    mode: str,
    ha_default: bool,
    conditional_decision: bool | None,
) -> bool:
    """Resolve the configured continuation behavior for a successful response."""
    if mode == CONTINUE_CONVERSATION_ALWAYS:
        return True
    if mode == CONTINUE_CONVERSATION_CONDITIONAL:
        return conditional_decision is True
    return ha_default


def _get_continue_conversation_mode(options: Mapping[str, Any]) -> str:
    """Return the mode, preserving HA Default for existing config entries."""
    return str(options.get(CONF_CONTINUE_CONVERSATION, DEFAULT_CONTINUE_CONVERSATION))
