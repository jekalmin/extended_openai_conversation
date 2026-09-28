"""Services for the extended openai conversation component."""

import base64
from collections.abc import Mapping
import logging
import mimetypes
from pathlib import Path
import re
import shutil
from typing import Any, cast
from urllib.parse import quote, urlparse
from uuid import uuid4

from openai._exceptions import OpenAIError
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationInput
from homeassistant.const import CONF_API_KEY
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    selector,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import async_get_integration

from .agent_config import (
    merge_agent_config,
    validate_function_groups,
    validate_function_tools,
)
from .agent_maintenance import get_agent_maintenance_gate
from .const import (
    API_MODE_OPTIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_API_PROVIDER,
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_FUNCTION_GROUPS,
    CONF_FUNCTION_TOOLS,
    CONF_ORGANIZATION,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_API_MODE,
    DEFAULT_CONF_BASE_URL,
    DOMAIN,
    GITHUB_REPO_NAME,
    GITHUB_REPO_OWNER,
    GITHUB_SKILLS_BRANCH,
    GITHUB_SKILLS_PATH,
    SERVICE_CALL_FUNCTION,
    SERVICE_DISABLE_FUNCTION_TOOLS,
    SERVICE_DOWNLOAD_SKILL,
    SERVICE_ENABLE_FUNCTION_TOOLS,
    SERVICE_GUEST_MODE_DISABLE,
    SERVICE_GUEST_MODE_UPDATE,
    SERVICE_MEMORY_CLEAR,
    SERVICE_MEMORY_DELETE,
    SERVICE_MEMORY_LIST,
    SERVICE_PROCESS,
    SERVICE_QUERY_IMAGE,
    SERVICE_RELOAD_SKILLS,
)
from .guest_mode import async_get_guest_mode
from .ha_tool_result_compat import tool_result_data
from .helpers import get_api_mode, get_authenticated_client, get_token_param_for_model
from .live_subentry_updates import update_live_subentry
from .memory import async_get_memory, memory_as_dict, memory_user_id
from .provider_errors import (
    ensure_successful_responses_result,
    log_provider_failure,
    provider_user_message,
    request_reauthentication,
)
from .request_rules import async_call_active_function
from .resource_limits import MAX_ATTACHMENT_COUNT, read_bounded_local_file
from .skill_resource_limits import (
    SkillDownloadBudget,
    async_read_bounded_json,
    async_read_bounded_response,
)

SERVICE_ENABLE_FUNCTION_GROUPS = "enable_function_groups"
SERVICE_DISABLE_FUNCTION_GROUPS = "disable_function_groups"

QUERY_IMAGE_SCHEMA = vol.Schema(
    {
        vol.Required("config_entry"): selector.ConfigEntrySelector(
            {
                "integration": DOMAIN,
            }
        ),
        vol.Required("model", default="gpt-4.1-mini"): cv.string,
        vol.Optional(CONF_API_MODE, default=DEFAULT_API_MODE): vol.In(
            [mode["key"] for mode in API_MODE_OPTIONS]
        ),
        vol.Required("prompt"): cv.string,
        vol.Required("images"): vol.All(
            cv.ensure_list,
            vol.Length(min=1, max=MAX_ATTACHMENT_COUNT),
            [{"url": cv.string}],
        ),
        vol.Optional("max_tokens", default=300): cv.positive_int,
    }
)

CHANGE_CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required("config_entry"): selector.ConfigEntrySelector(
            {
                "integration": DOMAIN,
            }
        ),
        vol.Optional(CONF_API_KEY): cv.string,
        vol.Optional(CONF_BASE_URL): cv.string,
        vol.Optional(CONF_API_VERSION): cv.string,
        vol.Optional(CONF_ORGANIZATION): cv.string,
        vol.Optional(CONF_SKIP_AUTHENTICATION): cv.boolean,
        vol.Optional(CONF_API_PROVIDER): cv.string,
    }
)

RELOAD_SKILLS_SCHEMA = vol.Schema({})
CALL_FUNCTION_SCHEMA = vol.Schema(
    {
        vol.Required("function"): cv.string,
        vol.Optional("arguments", default=dict): dict,
        vol.Optional("result_alias"): cv.string,
        vol.Optional("step_id"): cv.string,
    }
)

DOWNLOAD_SKILL_SCHEMA = vol.Schema(
    {
        vol.Required("skill_name"): cv.string,
        vol.Optional("source_ref"): vol.All(cv.string, vol.Length(min=1, max=128)),
    }
)

MEMORY_AGENT_FIELDS: dict[Any, Any] = {
    vol.Required("config_entry"): selector.ConfigEntrySelector({"integration": DOMAIN}),
    vol.Required("agent_id"): cv.string,
}

MEMORY_LIST_SCHEMA = vol.Schema(
    {
        **MEMORY_AGENT_FIELDS,
        vol.Optional("query"): cv.string,
        vol.Optional("category"): cv.string,
        vol.Optional("limit", default=50): vol.All(cv.positive_int, vol.Range(max=100)),
        vol.Optional("offset", default=0): vol.All(vol.Coerce(int), vol.Range(min=0)),
    }
)

MEMORY_DELETE_SCHEMA = vol.Schema(
    {
        **MEMORY_AGENT_FIELDS,
        vol.Required("memory_ids"): vol.All(
            cv.ensure_list, vol.Length(min=1, max=50), [cv.string]
        ),
    }
)

MEMORY_CLEAR_SCHEMA = vol.Schema(
    {
        **MEMORY_AGENT_FIELDS,
        vol.Optional("category"): cv.string,
        vol.Required("confirm", default=False): cv.boolean,
    }
)

FUNCTION_TOOL_STATE_SCHEMA = vol.Schema(
    {
        **MEMORY_AGENT_FIELDS,
        vol.Required("functions"): vol.All(
            cv.ensure_list, vol.Length(min=1, max=100), [cv.string]
        ),
    }
)

FUNCTION_GROUP_STATE_SCHEMA = vol.Schema(
    {
        **MEMORY_AGENT_FIELDS,
        vol.Required("function_groups"): vol.All(
            cv.ensure_list, vol.Length(min=1, max=50), [cv.string]
        ),
    }
)

GUEST_MODE_UPDATE_SCHEMA = vol.Schema(
    {
        **MEMORY_AGENT_FIELDS,
        vol.Optional("active_from"): cv.string,
        vol.Optional("active_until"): cv.string,
        vol.Optional("indefinite", default=False): cv.boolean,
    }
)

GUEST_MODE_DISABLE_SCHEMA = vol.Schema({**MEMORY_AGENT_FIELDS})

PROCESS_SCHEMA = vol.Schema(
    {
        vol.Required("text"): cv.string,
        vol.Optional("agent_id"): selector.EntitySelector(
            {
                "filter": {"integration": DOMAIN, "domain": "conversation"},
            }
        ),
        vol.Optional("conversation_id"): cv.string,
        vol.Optional("device_id"): cv.string,
        vol.Optional("satellite_id"): cv.string,
        vol.Optional("language"): cv.string,
    }
)

_LOGGER = logging.getLogger(__package__)
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def resolve_memory_agent(
    hass: HomeAssistant, entry_id: str, agent_reference: str
) -> tuple[str, str]:
    """Resolve a readable conversation entity or legacy subentry ID."""
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or entry.domain != DOMAIN:
        raise HomeAssistantError("Config entry not found")

    registry_entry = er.async_get(hass).async_get(agent_reference)
    if registry_entry is not None:
        if registry_entry.config_entry_id != entry_id:
            raise HomeAssistantError(
                "Conversation agent does not belong to the selected config entry"
            )
        subentry_id = registry_entry.config_subentry_id
        if subentry_id is None:
            raise HomeAssistantError("Conversation agent is not linked to a subentry")
    else:
        # Preserve compatibility with existing actions that pass the raw subentry ID.
        subentry_id = agent_reference

    subentry = entry.subentries.get(subentry_id)
    if subentry is None or subentry.subentry_type != "conversation":
        raise HomeAssistantError("Conversation agent not found")
    return entry_id, subentry_id


async def async_set_function_tools_enabled(
    hass: HomeAssistant,
    entry_id: str,
    agent_reference: str,
    function_names: list[str],
    enabled: bool,
) -> None:
    """Persist the authoritative enabled state for selected Function Tools."""
    _, subentry_id = resolve_memory_agent(hass, entry_id, agent_reference)
    async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None:
            raise HomeAssistantError("Config entry not found")
        subentry = entry.subentries[subentry_id]
        configured = validate_function_tools(subentry.data.get(CONF_FUNCTION_TOOLS, []))
        requested = list(dict.fromkeys(function_names))
        configured_names = {tool["spec"]["name"] for tool in configured}
        missing = [name for name in requested if name not in configured_names]
        if missing:
            raise HomeAssistantError(
                "Function Tool not found: " + ", ".join(sorted(missing))
            )
        for tool in configured:
            if tool["spec"]["name"] in requested:
                tool["enabled"] = enabled
        normalized = merge_agent_config(
            dict(subentry.data), {CONF_FUNCTION_TOOLS: configured}
        )
        update_live_subentry(hass, entry, subentry, data=normalized)


async def async_set_function_groups_enabled(
    hass: HomeAssistant,
    entry_id: str,
    agent_reference: str,
    group_ids: list[str],
    enabled: bool,
) -> None:
    """Persist Function Group state without changing member Function Tool state."""
    _, subentry_id = resolve_memory_agent(hass, entry_id, agent_reference)
    async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None:
            raise HomeAssistantError("Config entry not found")
        subentry = entry.subentries[subentry_id]
        configured = validate_function_tools(subentry.data.get(CONF_FUNCTION_TOOLS, []))
        groups = validate_function_groups(
            subentry.data.get(CONF_FUNCTION_GROUPS, []), configured
        )
        requested = list(dict.fromkeys(group_ids))
        configured_ids = {group["id"] for group in groups}
        missing = [group_id for group_id in requested if group_id not in configured_ids]
        if missing:
            raise HomeAssistantError(
                "Function Group not found: " + ", ".join(sorted(missing))
            )
        for group in groups:
            if group["id"] in requested:
                group["enabled"] = enabled
        normalized = merge_agent_config(
            dict(subentry.data), {CONF_FUNCTION_GROUPS: groups}
        )
        update_live_subentry(hass, entry, subentry, data=normalized)


async def async_skill_source_ref(
    hass: HomeAssistant, requested_ref: str | None = None
) -> str:
    """Resolve a reproducible release ref unless a development ref is explicit."""
    if requested_ref is not None:
        source_ref = requested_ref.strip()
        if not source_ref:
            raise HomeAssistantError("Skill source ref cannot be empty")
        return source_ref
    integration = await async_get_integration(hass, DOMAIN)
    if integration.version is not None:
        version = str(integration.version).strip()
        if version:
            return version
    # Custom development installs without version metadata retain an explicit fallback.
    return GITHUB_SKILLS_BRANCH


async def _async_require_service_admin(hass: HomeAssistant, call: ServiceCall) -> None:
    """Allow system automations and require admin rights for user-originated calls."""
    user_id = getattr(getattr(call, "context", None), "user_id", None)
    if user_id is None:
        return
    user = await hass.auth.async_get_user(user_id)
    if user is None or not user.is_admin:
        raise HomeAssistantError("Administrator permission is required")


async def async_setup_services(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up services for the extended openai conversation component."""

    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Query an image."""
        await _async_require_service_admin(hass, call)
        entry = None
        try:
            model = call.data["model"]
            api_mode = get_api_mode(call.data[CONF_API_MODE], model)
            image_params = await hass.async_add_executor_job(
                prepare_image_params, hass, call.data["images"]
            )

            entry = hass.config_entries.async_get_entry(call.data["config_entry"])
            if entry is None or entry.domain != DOMAIN:
                raise HomeAssistantError("Config entry not found")

            client = entry.runtime_data

            if api_mode == API_MODE_RESPONSES:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": call.data["prompt"]},
                            *[
                                {
                                    "type": "input_image",
                                    "image_url": image["url"],
                                    "detail": "auto",
                                }
                                for image in image_params
                            ],
                        ],
                    }
                ]
                _LOGGER.debug(
                    "Querying %s using %s with %d image(s)",
                    model,
                    api_mode,
                    len(image_params),
                )
                response = await client.responses.create(
                    model=model,
                    input=messages,
                    max_output_tokens=call.data["max_tokens"],
                    store=False,
                )
                ensure_successful_responses_result(response)
            else:
                images = [
                    {"type": "image_url", "image_url": image} for image in image_params
                ]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": call.data["prompt"]},
                            *images,
                        ],
                    }
                ]
                _LOGGER.debug(
                    "Querying %s using %s with %d image(s)",
                    model,
                    api_mode,
                    len(image_params),
                )
                token_param = get_token_param_for_model(model)
                token_kwargs = {token_param: call.data["max_tokens"]}
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **token_kwargs,
                )
            response_dict: dict = response.model_dump()
            _LOGGER.debug("Image query completed using %s", model)
        except OpenAIError as err:
            log_provider_failure(_LOGGER, "Image query provider request failed", err)
            if entry is not None:
                request_reauthentication(hass, entry, err)
            raise HomeAssistantError(
                f"Error generating image: {provider_user_message(err)}"
            ) from err

        return response_dict

    async def change_config(call: ServiceCall) -> None:
        """Change configuration."""
        await _async_require_service_admin(hass, call)
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)
        if not entry or entry.domain != DOMAIN:
            raise HomeAssistantError(f"Config entry {entry_id} not found")

        original_data = dict(entry.data)
        updates = {}
        for key in (
            CONF_API_KEY,
            CONF_BASE_URL,
            CONF_API_VERSION,
            CONF_ORGANIZATION,
            CONF_SKIP_AUTHENTICATION,
            CONF_API_PROVIDER,
        ):
            if key in call.data:
                updates[key] = call.data[key]

        if not updates:
            return

        new_data = dict(original_data)
        new_data.update(updates)

        _LOGGER.debug(
            "Updating config entry %s fields: %s",
            entry_id,
            ", ".join(sorted(updates)),
        )

        base_url = new_data.get(CONF_BASE_URL)
        if base_url == DEFAULT_CONF_BASE_URL:
            # Do not set base_url if using OpenAI for case of OpenAI's base_url change
            base_url = None
            new_data.pop(CONF_BASE_URL)

        if new_data.get(CONF_API_PROVIDER) == "azure" and not base_url:
            raise HomeAssistantError("Azure OpenAI requires a custom base URL.")

        try:
            await get_authenticated_client(
                hass=hass,
                api_key=new_data[CONF_API_KEY],
                base_url=new_data.get(CONF_BASE_URL),
                api_version=new_data.get(CONF_API_VERSION),
                organization=new_data.get(CONF_ORGANIZATION),
                skip_authentication=new_data.get(CONF_SKIP_AUTHENTICATION, False),
                api_provider=new_data.get(CONF_API_PROVIDER),
            )
        except OpenAIError as err:
            log_provider_failure(
                _LOGGER, "Provider configuration validation failed", err
            )
            raise HomeAssistantError(
                f"Provider configuration validation failed: {provider_user_message(err)}"
            ) from err

        if dict(entry.data) != original_data:
            raise HomeAssistantError(
                "Provider settings changed while this configuration was being "
                "validated. No changes were saved by this operation. Try again."
            )

        hass.config_entries.async_update_entry(entry, data=new_data)

    async def reload_skills(call: ServiceCall) -> ServiceResponse:
        """Reload skills from the user skill directory."""
        await _async_require_service_admin(hass, call)
        from .skills import SkillManager

        skill_manager = await SkillManager.async_get_instance(hass)
        await skill_manager.async_load_skills()

        return {
            "loaded_skills": len(skill_manager.get_all_skills()),
        }

    async def download_skill(call: ServiceCall) -> ServiceResponse:
        """Download a Skill, then atomically publish it into the installed catalogue."""
        await _async_require_service_admin(hass, call)
        from .skills import SkillManager

        skill_name = call.data["skill_name"].strip()
        if _SKILL_NAME_RE.fullmatch(skill_name) is None:
            raise HomeAssistantError(
                "Skill name may contain only letters, numbers, underscores, and hyphens"
            )
        source_ref = await async_skill_source_ref(hass, call.data.get("source_ref"))
        session = async_get_clientsession(hass)
        api_url = (
            f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"
            f"/contents/{GITHUB_SKILLS_PATH}/{skill_name}"
            f"?ref={quote(source_ref, safe='')}"
        )
        downloaded_files: list[str] = []
        download_budget = SkillDownloadBudget()

        def _safe_child(base: Path, name: str) -> Path:
            root = base.resolve()
            child = (root / name).resolve()
            if child != root and not child.is_relative_to(root):
                raise HomeAssistantError("Downloaded skill contains an unsafe path")
            return child

        async def _download_directory(
            url: str, local_dir: Path, depth: int = 0
        ) -> None:
            download_budget.check_directory(depth)
            async with session.get(url) as resp:
                if resp.status == 404:
                    raise HomeAssistantError(
                        f"Skill `{skill_name}` not found at source ref `{source_ref}`"
                    )
                if resp.status != 200:
                    raise HomeAssistantError(
                        f"Failed to fetch skill from GitHub (HTTP {resp.status})"
                    )
                items = await async_read_bounded_json(
                    resp, f"GitHub listing for Skill `{skill_name}`"
                )
            if not isinstance(items, list):
                raise HomeAssistantError(
                    f"Unexpected response from GitHub for skill `{skill_name}`"
                )
            for item in items:
                if not isinstance(item, Mapping):
                    raise HomeAssistantError("Unexpected item in GitHub skill response")
                item_name = item.get("name")
                item_type = item.get("type")
                if not isinstance(item_name, str):
                    raise HomeAssistantError("GitHub skill item has no valid name")
                item_path = _safe_child(local_dir, item_name)
                repo_path = str(item.get("path", item_name))
                if item_type == "file":
                    download_url = item.get("download_url")
                    if not isinstance(download_url, str):
                        raise HomeAssistantError(f"No download URL for `{repo_path}`")
                    max_bytes = download_budget.check_file(repo_path, item.get("size"))
                    async with session.get(download_url) as file_resp:
                        if file_resp.status != 200:
                            raise HomeAssistantError(
                                f"Failed to download `{repo_path}`"
                            )
                        content = await async_read_bounded_response(
                            file_resp,
                            max_bytes,
                            f"Downloaded Skill file `{repo_path}`",
                        )
                    download_budget.record_file(repo_path, len(content))
                    await hass.async_add_executor_job(
                        _write_file_sync, item_path, content
                    )
                    downloaded_files.append(repo_path)
                elif item_type == "dir":
                    child_url = item.get("url")
                    if not isinstance(child_url, str):
                        raise HomeAssistantError(f"No API URL for `{repo_path}`")
                    await _download_directory(child_url, item_path, depth + 1)

        def _write_file_sync(file_path: Path, content: bytes) -> None:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

        def _prepare_staging(root: Path, staging: Path) -> None:
            root.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)

        def _cleanup_path(path: Path) -> None:
            if path.exists():
                shutil.rmtree(path)

        skill_manager = await SkillManager.async_get_instance(hass)
        skills_root = skill_manager.user_skills_dir.resolve()
        target_dir = (skills_root / skill_name).resolve()
        if target_dir == skills_root or not target_dir.is_relative_to(skills_root):
            raise HomeAssistantError("Skill target directory is unsafe")

        staging_root = skill_manager.staging_dir.resolve()
        staging_dir = staging_root / f"{skill_name}.download-{uuid4().hex}"
        await hass.async_add_executor_job(_prepare_staging, staging_root, staging_dir)
        _LOGGER.info("Downloading skill `%s` from `%s`", skill_name, source_ref)

        try:
            await _download_directory(api_url, staging_dir)
            if not (staging_dir / "SKILL.md").is_file():
                raise HomeAssistantError(
                    f"Downloaded skill `{skill_name}` does not contain SKILL.md"
                )
            await skill_manager.async_publish_staged_skill(skill_name, staging_dir)
        except HomeAssistantError:
            await hass.async_add_executor_job(_cleanup_path, staging_dir)
            raise
        except Exception as err:
            await hass.async_add_executor_job(_cleanup_path, staging_dir)
            raise HomeAssistantError(
                f"Failed to download skill `{skill_name}`: {err}"
            ) from err

        _LOGGER.info(
            "Successfully downloaded skill `%s` (%d files)",
            skill_name,
            len(downloaded_files),
        )
        return {
            "skill_name": skill_name,
            "source_ref": source_ref,
            "downloaded_files": downloaded_files,
            "target_directory": str(target_dir),
        }

    async def memory_list(call: ServiceCall) -> ServiceResponse:
        """Inspect memories in the caller's own user scope."""
        entry_id, subentry_id = resolve_memory_agent(
            hass, call.data["config_entry"], call.data["agent_id"]
        )
        async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
            memory = await async_get_memory(hass, entry_id, subentry_id)
            user_id = memory_user_id(call)
            if query := call.data.get("query"):
                records = await memory.async_search(
                    user_id,
                    query,
                    call.data.get("category"),
                    call.data["limit"],
                )
            else:
                records = await memory.async_list(
                    user_id,
                    call.data.get("category"),
                    call.data["limit"],
                    call.data["offset"],
                )
            return cast(
                ServiceResponse,
                {"memories": [memory_as_dict(record) for record in records]},
            )

    async def memory_delete(call: ServiceCall) -> ServiceResponse:
        """Delete selected memories in the caller's own user scope."""
        entry_id, subentry_id = resolve_memory_agent(
            hass, call.data["config_entry"], call.data["agent_id"]
        )
        async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
            memory = await async_get_memory(hass, entry_id, subentry_id)
            deleted = await memory.async_delete(
                memory_user_id(call), call.data["memory_ids"]
            )
            return {"deleted": deleted}

    async def memory_clear(call: ServiceCall) -> ServiceResponse:
        """Clear memories only after explicit confirmation."""
        if call.data["confirm"] is not True:
            raise HomeAssistantError("Set confirm to true to clear memories")
        entry_id, subentry_id = resolve_memory_agent(
            hass, call.data["config_entry"], call.data["agent_id"]
        )
        async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
            memory = await async_get_memory(hass, entry_id, subentry_id)
            deleted = await memory.async_clear(
                memory_user_id(call), call.data.get("category")
            )
            return {"deleted": deleted}

    async def enable_function_tools(call: ServiceCall) -> None:
        """Enable one or more configured Function Tools."""
        await _async_require_service_admin(hass, call)
        await async_set_function_tools_enabled(
            hass,
            call.data["config_entry"],
            call.data["agent_id"],
            call.data["functions"],
            True,
        )

    async def disable_function_tools(call: ServiceCall) -> None:
        """Disable one or more configured Function Tools."""
        await _async_require_service_admin(hass, call)
        await async_set_function_tools_enabled(
            hass,
            call.data["config_entry"],
            call.data["agent_id"],
            call.data["functions"],
            False,
        )

    async def enable_function_groups(call: ServiceCall) -> None:
        """Enable one or more configured Function Groups."""
        await _async_require_service_admin(hass, call)
        await async_set_function_groups_enabled(
            hass,
            call.data["config_entry"],
            call.data["agent_id"],
            call.data["function_groups"],
            True,
        )

    async def disable_function_groups(call: ServiceCall) -> None:
        """Disable one or more configured Function Groups."""
        await _async_require_service_admin(hass, call)
        await async_set_function_groups_enabled(
            hass,
            call.data["config_entry"],
            call.data["agent_id"],
            call.data["function_groups"],
            False,
        )

    async def guest_mode_update(call: ServiceCall) -> ServiceResponse:
        """Set the full Guest Mode interval from a trusted HA action."""
        await _async_require_service_admin(hass, call)
        entry_id, subentry_id = resolve_memory_agent(
            hass, call.data["config_entry"], call.data["agent_id"]
        )
        async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
            manager = await async_get_guest_mode(hass, entry_id, subentry_id)
            try:
                return cast(
                    ServiceResponse,
                    await manager.async_update_trusted(
                        active_from=call.data.get("active_from"),
                        active_until=call.data.get("active_until"),
                        indefinite=call.data["indefinite"],
                    ),
                )
            except ValueError as err:
                raise HomeAssistantError(str(err)) from err

    def _process_agent(agent_id: str | None):
        if agent_id:
            agent = conversation.async_get_agent(hass, agent_id)
            if agent is None or not hasattr(agent, "async_process_direct"):
                raise HomeAssistantError("Extended OpenAI conversation agent not found")
            return agent
        registry = er.async_get(hass)
        agents = []
        for registry_entry in registry.entities.values():
            if (
                registry_entry.platform == DOMAIN
                and registry_entry.domain == "conversation"
            ):
                agent = conversation.async_get_agent(hass, registry_entry.entity_id)
                if agent is not None and hasattr(agent, "async_process_direct"):
                    agents.append(agent)
        if not agents:
            raise HomeAssistantError(
                "No Extended OpenAI conversation agent is available"
            )
        if len(agents) > 1:
            raise HomeAssistantError(
                "Choose a conversation agent when more than one is configured"
            )
        return agents[0]

    async def process(call: ServiceCall) -> ServiceResponse:
        """Send text directly through an Extended OpenAI entity's normal pipeline."""
        if not call.data["text"].strip():
            raise HomeAssistantError("Request text cannot be empty")
        agent = _process_agent(call.data.get("agent_id"))
        agent_id = getattr(agent, "entity_id", call.data.get("agent_id") or "")
        user_input = ConversationInput(
            text=call.data["text"],
            context=call.context,
            conversation_id=call.data.get("conversation_id"),
            device_id=call.data.get("device_id"),
            satellite_id=call.data.get("satellite_id"),
            language=call.data.get("language", hass.config.language),
            agent_id=agent_id,
        )
        try:
            result, metadata = await agent.async_process_direct(user_input)
        except Exception as err:
            if isinstance(err, HomeAssistantError):
                raise
            raise HomeAssistantError(
                f"Extended OpenAI could not process the request: {err}"
            ) from err
        speech = getattr(result.response, "speech", {}) or {}
        response_text = ""
        if isinstance(speech, Mapping):
            plain = speech.get("plain", {})
            if isinstance(plain, Mapping):
                response_text = str(plain.get("speech", ""))
        response: dict[str, Any] = {
            "response": response_text,
            "conversation_id": result.conversation_id,
            "handled_locally": bool(metadata.get("handled_locally")),
        }
        if metadata.get("matched_rule") is not None:
            response["matched_rule"] = metadata["matched_rule"]
            response["captured_values"] = metadata.get("captured_values", {})
        return response

    async def guest_mode_disable(call: ServiceCall) -> ServiceResponse:
        """End or cancel Guest Mode from a trusted HA action."""
        await _async_require_service_admin(hass, call)
        entry_id, subentry_id = resolve_memory_agent(
            hass, call.data["config_entry"], call.data["agent_id"]
        )
        async with get_agent_maintenance_gate(hass, entry_id, subentry_id).shared():
            manager = await async_get_guest_mode(hass, entry_id, subentry_id)
            return cast(ServiceResponse, await manager.async_disable_trusted())

    async def call_function(call: ServiceCall) -> ServiceResponse:
        """Bridge a native HA script action into the active configured function."""
        arguments = call.data.get("arguments", {})
        if "result_alias" in call.data:
            result = await async_call_active_function(
                call.data["function"], arguments, call.data["result_alias"]
            )
        else:
            result = await async_call_active_function(call.data["function"], arguments)
        result = tool_result_data(result, default=result)
        return cast(ServiceResponse, {"result": result})

    hass.services.async_register(
        DOMAIN,
        SERVICE_PROCESS,
        process,
        schema=PROCESS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_CALL_FUNCTION,
        call_function,
        schema=CALL_FUNCTION_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_QUERY_IMAGE,
        query_image,
        schema=QUERY_IMAGE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        "change_config",
        change_config,
        schema=CHANGE_CONFIG_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RELOAD_SKILLS,
        reload_skills,
        schema=RELOAD_SKILLS_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DOWNLOAD_SKILL,
        download_skill,
        schema=DOWNLOAD_SKILL_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_LIST,
        memory_list,
        schema=MEMORY_LIST_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_DELETE,
        memory_delete,
        schema=MEMORY_DELETE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_CLEAR,
        memory_clear,
        schema=MEMORY_CLEAR_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_ENABLE_FUNCTION_TOOLS,
        enable_function_tools,
        schema=FUNCTION_TOOL_STATE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DISABLE_FUNCTION_TOOLS,
        disable_function_tools,
        schema=FUNCTION_TOOL_STATE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_ENABLE_FUNCTION_GROUPS,
        enable_function_groups,
        schema=FUNCTION_GROUP_STATE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DISABLE_FUNCTION_GROUPS,
        disable_function_groups,
        schema=FUNCTION_GROUP_STATE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GUEST_MODE_UPDATE,
        guest_mode_update,
        schema=GUEST_MODE_UPDATE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_GUEST_MODE_DISABLE,
        guest_mode_disable,
        schema=GUEST_MODE_DISABLE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def _convert_image_param(
    hass: HomeAssistant, image: dict, total_bytes: int
) -> tuple[dict, int]:
    """Convert one local image to a bounded data URL, or preserve a remote URL."""
    result = dict(image)
    url = result["url"]

    if urlparse(url).scheme in cv.EXTERNAL_URL_PROTOCOL_SCHEMA_LIST:
        return result, 0

    if not hass.config.is_allowed_path(url):
        raise HomeAssistantError(
            f"Cannot read `{url}`, no access to path; "
            "`allowlist_external_dirs` may need to be adjusted in "
            "`configuration.yaml`"
        )

    path = Path(url)
    if not path.exists():
        raise HomeAssistantError(f"`{url}` does not exist")
    if not path.is_file():
        raise HomeAssistantError(f"`{url}` is not a file")

    mime_type, _ = mimetypes.guess_type(url)
    if mime_type is None or not mime_type.startswith("image"):
        raise HomeAssistantError(f"`{url}` is not an image")

    content = read_bounded_local_file(path, total_bytes)
    result["url"] = (
        f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"
    )
    return result, len(content)


def prepare_image_params(hass: HomeAssistant, images: list[dict]) -> list[dict]:
    """Prepare a bounded set of image parameters outside the event loop."""
    if len(images) > MAX_ATTACHMENT_COUNT:
        raise HomeAssistantError(
            f"At most {MAX_ATTACHMENT_COUNT} images can be sent in one request"
        )

    prepared: list[dict] = []
    total_bytes = 0
    for image in images:
        converted, size = _convert_image_param(hass, image, total_bytes)
        total_bytes += size
        prepared.append(converted)
    return prepared


def to_image_param(hass: HomeAssistant, image: dict) -> dict:
    """Convert a single URL to a bounded base64 encoded image if local."""
    return _convert_image_param(hass, image, 0)[0]


def encode_image(image_path: str) -> str:
    """Convert a bounded local file to base64 encoded image."""
    content = read_bounded_local_file(Path(image_path))
    return base64.b64encode(content).decode("utf-8")
