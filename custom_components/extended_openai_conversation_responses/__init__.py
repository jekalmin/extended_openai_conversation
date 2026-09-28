"""The Extended OpenAI Conversation (Responses) integration."""

from __future__ import annotations

from contextlib import suppress
import logging
from types import MappingProxyType

from openai import AsyncClient
from openai._exceptions import AuthenticationError, OpenAIError

from homeassistant.config_entries import ConfigEntry, ConfigEntryState, ConfigSubentry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .backup_transfer import setup_backup_transfer_websocket
from .const import (
    CONF_API_PROVIDER,
    CONF_API_VERSION,
    CONF_ARCHIVE_ENABLED,
    CONF_ARCHIVE_MODEL_SEARCH_ENABLED,
    CONF_ARCHIVE_RETENTION_DAYS,
    CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    CONF_BASE_URL,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_CURRENT_DATETIME_ENABLED,
    CONF_CURRENT_DATETIME_TEMPLATE,
    CONF_EXPOSED_ENTITIES_ENABLED,
    CONF_EXPOSED_ENTITIES_TEMPLATE,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_GUEST_CONTROLLABLE_AREAS,
    CONF_GUEST_CONTROLLABLE_DOMAINS,
    CONF_GUEST_CONTROLLABLE_ENTITIES,
    CONF_GUEST_CONTROLLABLE_LABELS,
    CONF_GUEST_KNOWLEDGE_ENABLED,
    CONF_GUEST_MODE_ENABLED,
    CONF_GUEST_READABLE_AREAS,
    CONF_GUEST_READABLE_DOMAINS,
    CONF_GUEST_READABLE_ENTITIES,
    CONF_GUEST_READABLE_LABELS,
    CONF_GUEST_SHARED_MEMORY_READ,
    CONF_GUEST_SHARED_MEMORY_WRITE,
    CONF_MEMORY_AUTO_CREATE,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MODE,
    CONF_ORGANIZATION,
    CONF_SHARED_ARCHIVE_ENABLED,
    CONF_SHARED_MEMORY_MODE,
    CONF_SKIP_AUTHENTICATION,
    CONF_SPEECH_PROCESSING_ENABLED,
    CONF_SPEECH_REGEX_REPLACEMENTS,
    CONF_SPEECH_STRIP_MARKDOWN,
    CONF_SPEECH_STRIP_URLS,
    CONF_USAGE_REQUEST_RETENTION_DAYS,
    CONF_USAGE_RUN_RETENTION_DAYS,
    CONF_VOICE_DEVICE_MAPPINGS,
    CONF_VOICE_SCOPE_POLICY,
    CONF_VOICE_UNMAPPED_POLICY,
    CONFIG_ENTRY_VERSION,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_AI_TASK_OPTIONS,
    DEFAULT_API_PROVIDER,
    DEFAULT_ARCHIVE_ENABLED,
    DEFAULT_ARCHIVE_MODEL_SEARCH_ENABLED,
    DEFAULT_ARCHIVE_RETENTION_DAYS,
    DEFAULT_ARCHIVE_SESSION_TIMEOUT_MINUTES,
    DEFAULT_CURRENT_DATETIME_TEMPLATE,
    DEFAULT_EXPOSED_ENTITIES_TEMPLATE,
    DEFAULT_FUNCTION_GROUPS,
    DEFAULT_SHARED_ARCHIVE_ENABLED,
    DEFAULT_SHARED_MEMORY_MODE,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_SPEECH_PROCESSING_ENABLED,
    DEFAULT_SPEECH_REGEX_REPLACEMENTS,
    DEFAULT_SPEECH_STRIP_MARKDOWN,
    DEFAULT_SPEECH_STRIP_URLS,
    DEFAULT_USAGE_REQUEST_RETENTION_DAYS,
    DEFAULT_USAGE_RUN_RETENTION_DAYS,
    DEFAULT_VOICE_SCOPE_POLICY,
    DEFAULT_VOICE_UNMAPPED_POLICY,
    DOMAIN,
    LEGACY_CONTEXT_TRUNCATE_STRATEGY,
    MEMORY_MODE_AUTOMATIC,
    MEMORY_MODE_OFF,
)
from .debug import DebugOpenAIClientProxy
from .debug_ui import async_setup_debug_ui
from .delayed_tools import async_setup_delayed_tools
from .ha_permissions import async_setup_ha_permissions
from .helpers import get_authenticated_client, supports_openai_hosted_tools
from .intercom_services import async_setup_intercom_services
from .live_subentry_updates import is_live_subentry_update
from .management_function_repair import async_prewarm_persisted_config_projection
from .management_ui import async_setup_management_ui
from .memory import get_memory_mode
from .model_catalog_manager import async_setup_model_catalog
from .native_function_schema_migration import (
    migrate_legacy_stock_native_function_tools_yaml,
)
from .openai_compat import apply_openai_compatibility
from .prompt_cache import PerformanceOpenAIClientProxy
from .provider_credentials import setup_provider_credentials_websocket
from .quiet_hours import async_get_quiet_hours
from .restore_recovery import async_recover_pending_restores
from .services import async_setup_services
from .template import async_setup_templates, async_unload_templates

_LOGGER = logging.getLogger(__name__)

_REQUEST_RULE_RUNTIMES = "extended_openai_conversation_responses.request_rule_runtimes"
_REQUEST_RULE_MANAGERS = "extended_openai_conversation_responses.request_rule_managers"

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION, Platform.SENSOR]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type ExtendedOpenAIConfigEntry = ConfigEntry[AsyncClient]


def _sole_conversation_agent(
    hass: HomeAssistant, entry: ConfigEntry
) -> ConfigSubentry | None:
    """Choose an agent only when the whole EOAI installation has exactly one."""
    agents = [
        (owner, subentry)
        for owner in hass.config_entries.async_entries(DOMAIN)
        for subentry in owner.subentries.values()
        if subentry.subentry_type == "conversation"
    ]
    if len(agents) != 1 or agents[0][0] is not entry:
        return None
    return agents[0][1]


async def _async_prewarm_sole_agent(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Prime optional Management state after setup, using current entry data."""
    if entry.state is not ConfigEntryState.LOADED:
        return
    subentry = _sole_conversation_agent(hass, entry)
    if subentry is None:
        return
    try:
        await async_prewarm_persisted_config_projection(hass, entry, subentry)
    except Exception:
        # The normal Management read remains authoritative on a warm failure.
        return


def _schedule_sole_agent_prewarm(hass: HomeAssistant, entry: ConfigEntry) -> None:
    if _sole_conversation_agent(hass, entry) is None:
        return

    removed = False

    def remove_once() -> None:
        nonlocal removed
        if not removed:
            removed = True
            remove_listener()

    def on_state_change() -> None:
        if entry.state is not ConfigEntryState.LOADED:
            return
        remove_once()
        warm = _async_prewarm_sole_agent(hass, entry)
        try:
            entry.async_create_background_task(
                hass, warm, "EOAI single-agent Management prewarm", eager_start=False
            )
        except Exception:
            warm.close()

    remove_listener = entry.async_on_state_change(on_state_change)
    entry.async_on_unload(remove_once)
    on_state_change()


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Extended OpenAI Conversation (Responses)."""
    await async_setup_model_catalog(hass)
    await async_get_quiet_hours(hass)
    apply_openai_compatibility()
    await async_setup_delayed_tools(hass)
    await async_migrate_integration(hass)
    # Resolve any interrupted cross-store restore before conversation agents load.
    await async_recover_pending_restores(hass)
    await async_setup_ha_permissions(hass)
    await async_setup_services(hass, config)
    await async_setup_intercom_services(hass)
    # Register narrow admin-only commands separately from the broad management
    # command before exposing the management panel.
    setup_provider_credentials_websocket(hass)
    setup_backup_transfer_websocket(hass)
    # Request Debug loads lazily from Management; register its asset routes
    # before exposing the panel itself.
    await async_setup_debug_ui(hass)
    await async_setup_management_ui(hass)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: ExtendedOpenAIConfigEntry
) -> bool:
    """Set up Extended OpenAI Conversation (Responses) from a config entry."""

    try:
        client = await get_authenticated_client(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
            api_provider=entry.data.get(CONF_API_PROVIDER, DEFAULT_API_PROVIDER),
        )
    except AuthenticationError as err:
        raise ConfigEntryAuthFailed("API credentials are invalid or expired") from err
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    debug_client = DebugOpenAIClientProxy(client)
    entry.runtime_data = PerformanceOpenAIClientProxy(  # type: ignore[assignment]
        debug_client,
        direct_openai=supports_openai_hosted_tools(
            entry.data.get(CONF_API_PROVIDER, DEFAULT_API_PROVIDER),
            entry.data.get(CONF_BASE_URL),
        ),
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    try:
        await async_setup_templates(hass, entry.entry_id)
        entry.async_on_unload(entry.add_update_listener(update_listener))
    except BaseException:
        try:
            await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
        finally:
            await async_unload_templates(hass, entry.entry_id)
        raise
    with suppress(Exception):
        # Optional cache preparation must never affect integration setup.
        _schedule_sole_agent_prewarm(hass, entry)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI and discard transient per-conversation routing state."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if not unloaded:
        return False
    hass_data = getattr(hass, "data", None)
    subentries = getattr(entry, "subentries", None)
    if isinstance(hass_data, dict) and subentries is not None:
        from .management_function_repair import discard_persisted_config_projection

        runtimes = hass_data.get(_REQUEST_RULE_RUNTIMES, {})
        managers = hass_data.get(_REQUEST_RULE_MANAGERS, {})
        for subentry in subentries.values():
            key = (entry.entry_id, subentry.subentry_id)
            runtimes.pop(key, None)
            # Durable managers belong to one loaded entry generation. Reload must
            # hydrate them from HA Store instead of retaining cached rule state.
            managers.pop(key, None)
            discard_persisted_config_projection(subentry)
    await async_unload_templates(hass, entry.entry_id)
    return True


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    if is_live_subentry_update():
        # Live management writes replace subentry.data in place. Conversation,
        # AI Task and sensor runtimes already consume that live object, while each
        # request boundary reconciles optional managers. Reloading here needlessly
        # reauthenticates the provider and rebuilds every platform.
        return
    await hass.config_entries.async_reload(entry.entry_id)


def _migrate_saved_native_function_schemas(data: dict) -> bool:
    """Persist exact-stock native schema upgrades in one agent data mapping."""
    migrated_yaml, changed = migrate_legacy_stock_native_function_tools_yaml(
        data.get(CONF_FUNCTION_TOOLS)
    )
    if changed:
        data[CONF_FUNCTION_TOOLS] = migrated_yaml
    return changed


async def async_migrate_integration(hass: HomeAssistant) -> None:
    """Migrate integration entry structure and exact historical stock tool schemas."""

    entries = sorted(
        hass.config_entries.async_entries(DOMAIN),
        key=lambda e: e.disabled_by is not None,
    )

    # Schema migration is intentionally independent of config-entry version. It is
    # idempotent and exact-match only, so current entries created by an older release
    # can be corrected without inventing another provider-only representation.
    for entry in entries:
        if entry.version < CONFIG_ENTRY_VERSION:
            continue
        for subentry in entry.subentries.values():
            if subentry.subentry_type != "conversation":
                continue
            data = dict(subentry.data)
            if _migrate_saved_native_function_schemas(data):
                hass.config_entries.async_update_subentry(entry, subentry, data=data)

    if not any(entry.version < CONFIG_ENTRY_VERSION for entry in entries):
        return

    for entry in entries:
        if entry.version >= CONFIG_ENTRY_VERSION:
            continue
        _LOGGER.warning(
            "Migrating Extended OpenAI Conversation (Responses) config entry %s from version %s to version %s",
            entry.entry_id,
            entry.version,
            CONFIG_ENTRY_VERSION,
        )
        if entry.version == 1:
            existing_types = {
                subentry.subentry_type for subentry in entry.subentries.values()
            }
            if "conversation" not in existing_types:
                conversation_subentry = ConfigSubentry(
                    data=entry.options,
                    subentry_type="conversation",
                    title=entry.title,
                    unique_id=None,
                )
                hass.config_entries.async_add_subentry(entry, conversation_subentry)
            if "ai_task_data" not in existing_types:
                ai_task_subentry = ConfigSubentry(
                    data=MappingProxyType(dict(DEFAULT_AI_TASK_OPTIONS)),
                    subentry_type="ai_task_data",
                    title=DEFAULT_AI_TASK_NAME,
                    unique_id=None,
                )
                hass.config_entries.async_add_subentry(entry, ai_task_subentry)
            hass.config_entries.async_update_entry(
                entry, title=entry.title, options={}, version=2
            )

        for subentry in entry.subentries.values():
            if subentry.subentry_type != "conversation":
                continue
            data = dict(subentry.data)
            _migrate_saved_native_function_schemas(data)
            mode = get_memory_mode(data)
            data[CONF_MEMORY_MODE] = mode
            data[CONF_MEMORY_ENABLED] = mode != MEMORY_MODE_OFF
            data[CONF_MEMORY_AUTO_CREATE] = mode == MEMORY_MODE_AUTOMATIC
            data.setdefault(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                LEGACY_CONTEXT_TRUNCATE_STRATEGY,
            )
            data.setdefault(CONF_ARCHIVE_ENABLED, DEFAULT_ARCHIVE_ENABLED)
            data.setdefault(CONF_ARCHIVE_RETENTION_DAYS, DEFAULT_ARCHIVE_RETENTION_DAYS)
            data.setdefault(
                CONF_ARCHIVE_MODEL_SEARCH_ENABLED,
                DEFAULT_ARCHIVE_MODEL_SEARCH_ENABLED,
            )
            data.setdefault(CONF_SHARED_ARCHIVE_ENABLED, DEFAULT_SHARED_ARCHIVE_ENABLED)
            data.setdefault(
                CONF_ARCHIVE_SESSION_TIMEOUT_MINUTES,
                DEFAULT_ARCHIVE_SESSION_TIMEOUT_MINUTES,
            )
            data.setdefault(CONF_VOICE_SCOPE_POLICY, DEFAULT_VOICE_SCOPE_POLICY)
            data.setdefault(CONF_VOICE_UNMAPPED_POLICY, DEFAULT_VOICE_UNMAPPED_POLICY)
            data.setdefault(CONF_VOICE_DEVICE_MAPPINGS, {})
            data.setdefault(CONF_SHARED_MEMORY_MODE, DEFAULT_SHARED_MEMORY_MODE)
            data.setdefault(
                CONF_USAGE_REQUEST_RETENTION_DAYS,
                DEFAULT_USAGE_REQUEST_RETENTION_DAYS,
            )
            data.setdefault(
                CONF_USAGE_RUN_RETENTION_DAYS,
                DEFAULT_USAGE_RUN_RETENTION_DAYS,
            )
            data.setdefault(
                CONF_SPEECH_PROCESSING_ENABLED,
                DEFAULT_SPEECH_PROCESSING_ENABLED,
            )
            data.setdefault(CONF_SPEECH_STRIP_MARKDOWN, DEFAULT_SPEECH_STRIP_MARKDOWN)
            data.setdefault(CONF_SPEECH_STRIP_URLS, DEFAULT_SPEECH_STRIP_URLS)
            data.setdefault(
                CONF_SPEECH_REGEX_REPLACEMENTS,
                list(DEFAULT_SPEECH_REGEX_REPLACEMENTS),
            )
            data.setdefault(CONF_FUNCTION_GROUPS, list(DEFAULT_FUNCTION_GROUPS))
            data.setdefault(CONF_GUEST_MODE_ENABLED, False)
            data.setdefault(CONF_GUEST_SHARED_MEMORY_READ, False)
            data.setdefault(CONF_GUEST_SHARED_MEMORY_WRITE, False)
            data.setdefault(CONF_GUEST_KNOWLEDGE_ENABLED, False)
            for guest_selector in (
                CONF_GUEST_READABLE_ENTITIES,
                CONF_GUEST_CONTROLLABLE_ENTITIES,
                CONF_GUEST_READABLE_DOMAINS,
                CONF_GUEST_CONTROLLABLE_DOMAINS,
                CONF_GUEST_READABLE_AREAS,
                CONF_GUEST_CONTROLLABLE_AREAS,
                CONF_GUEST_READABLE_LABELS,
                CONF_GUEST_CONTROLLABLE_LABELS,
            ):
                data.setdefault(guest_selector, [])
            data.setdefault(CONF_CURRENT_DATETIME_ENABLED, False)
            data.setdefault(
                CONF_CURRENT_DATETIME_TEMPLATE, DEFAULT_CURRENT_DATETIME_TEMPLATE
            )
            data.setdefault(CONF_EXPOSED_ENTITIES_ENABLED, False)
            data.setdefault(
                CONF_EXPOSED_ENTITIES_TEMPLATE, DEFAULT_EXPOSED_ENTITIES_TEMPLATE
            )
            hass.config_entries.async_update_subentry(entry, subentry, data=data)
        hass.config_entries.async_update_entry(entry, version=CONFIG_ENTRY_VERSION)
