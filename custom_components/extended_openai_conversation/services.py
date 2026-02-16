"""Services for the extended openai conversation component."""

import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from openai._exceptions import OpenAIError
import voluptuous as vol

from homeassistant.const import CONF_API_KEY
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_registry as er, selector
from homeassistant.helpers.entity_platform import async_get_platforms
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_API_PROVIDER,
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_MEMORY_ENABLED,
    CONF_ORGANIZATION,
    CONF_SKIP_AUTHENTICATION,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_MEMORY_ENABLED,
    DOMAIN,
    GITHUB_REPO_NAME,
    GITHUB_REPO_OWNER,
    GITHUB_SKILLS_BRANCH,
    GITHUB_SKILLS_PATH,
    SERVICE_DOWNLOAD_SKILL,
    SERVICE_MEMORY_DELETE,
    SERVICE_MEMORY_SEARCH,
    SERVICE_MEMORY_STORE,
    SERVICE_QUERY_IMAGE,
    SERVICE_RELOAD_SKILLS,
)
from .helpers import get_authenticated_client, get_token_param_for_model

QUERY_IMAGE_SCHEMA = vol.Schema(
    {
        vol.Required("config_entry"): selector.ConfigEntrySelector(
            {
                "integration": DOMAIN,
            }
        ),
        vol.Required("model", default="gpt-4.1-mini"): cv.string,
        vol.Required("prompt"): cv.string,
        vol.Required("images"): vol.All(cv.ensure_list, [{"url": cv.string}]),
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

DOWNLOAD_SKILL_SCHEMA = vol.Schema(
    {
        vol.Required("skill_name"): cv.string,
    }
)

MEMORY_STORE_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): cv.string,
        vol.Required("content"): cv.string,
    }
)

MEMORY_SEARCH_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): cv.string,
        vol.Required("query"): cv.string,
        vol.Optional("max_results", default=5): vol.Coerce(int),
        vol.Optional("min_score", default=0.35): vol.Coerce(float),
    }
)

MEMORY_DELETE_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): cv.string,
        vol.Required("id"): vol.Coerce(int),
    }
)

_LOGGER = logging.getLogger(__package__)


def _is_memory_enabled(hass: HomeAssistant, entity_id: str) -> bool:
    """Check if memory is enabled for the given conversation entity."""
    ent_reg = er.async_get(hass)
    entity_entry = ent_reg.async_get(entity_id)
    if entity_entry is None or entity_entry.config_entry_id is None:
        return False

    config_entry = hass.config_entries.async_get_entry(entity_entry.config_entry_id)
    if config_entry is None:
        return False

    subentry_id = entity_entry.config_subentry_id
    if subentry_id is None:
        return False

    subentry = config_entry.subentries.get(subentry_id)
    if subentry is None:
        return False

    return subentry.data.get(CONF_MEMORY_ENABLED, DEFAULT_MEMORY_ENABLED)


async def async_setup_services(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up services for the extended openai conversation component."""

    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Query an image."""
        try:
            model = call.data["model"]
            images = [
                {"type": "image_url", "image_url": to_image_param(hass, image)}
                for image in call.data["images"]
            ]

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": call.data["prompt"]}, *images],
                }
            ]
            _LOGGER.info("Prompt for %s: %s", model, messages)

            entry = hass.config_entries.async_get_entry(call.data["config_entry"])
            if entry is None:
                raise HomeAssistantError("Config entry not found")

            client = entry.runtime_data

            token_param = get_token_param_for_model(model)
            token_kwargs = {token_param: call.data["max_tokens"]}

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **token_kwargs,
            )
            response_dict: dict = response.model_dump()
            _LOGGER.info("Response %s", response_dict)
        except OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return response_dict

    async def change_config(call: ServiceCall) -> None:
        """Change configuration."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)
        if not entry or entry.domain != DOMAIN:
            raise HomeAssistantError(f"Config entry {entry_id} not found")

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

        new_data = entry.data.copy()
        new_data.update(updates)

        _LOGGER.debug("Updating config entry %s with %s", entry_id, new_data)

        base_url = new_data.get(CONF_BASE_URL)
        if base_url == DEFAULT_CONF_BASE_URL:
            # Do not set base_url if using OpenAI for case of OpenAI's base_url change
            base_url = None
            new_data.pop(CONF_BASE_URL)

        if new_data.get(CONF_API_PROVIDER) == "azure" and not base_url:
            raise HomeAssistantError("Azure OpenAI requires a custom base URL.")

        await get_authenticated_client(
            hass=hass,
            api_key=new_data[CONF_API_KEY],
            base_url=new_data.get(CONF_BASE_URL),
            api_version=new_data.get(CONF_API_VERSION),
            organization=new_data.get(CONF_ORGANIZATION),
            skip_authentication=new_data.get(CONF_SKIP_AUTHENTICATION, False),
            api_provider=new_data.get(CONF_API_PROVIDER),
        )

        hass.config_entries.async_update_entry(entry, data=new_data)

    async def reload_skills(call: ServiceCall) -> ServiceResponse:
        """Reload skills from the user skill directory."""
        from .skills import SkillManager

        skill_manager = await SkillManager.async_get_instance(hass)
        await skill_manager.async_load_skills()

        return {
            "loaded_skills": len(skill_manager.get_all_skills()),
        }

    def _get_memory_manager(entity_id: str):
        """Look up MemoryManager by finding the entity on the conversation platform."""
        for platform in async_get_platforms(hass, DOMAIN):
            entity = platform.entities.get(entity_id)
            if entity is not None:
                return getattr(entity, "_memory_manager", None)
        return None

    async def memory_store(call: ServiceCall) -> ServiceResponse:
        """Store content in long-term memory for a conversation entity."""
        entity_id = call.data["entity_id"]
        if not _is_memory_enabled(hass, entity_id):
            raise HomeAssistantError(
                f"Memory is not enabled for entity '{entity_id}'. "
                "Enable memory in the conversation agent settings."
            )
        manager = _get_memory_manager(entity_id)
        if manager is None:
            raise HomeAssistantError(
                f"Memory not initialized for entity '{entity_id}'. "
                "Ensure memory is enabled for this conversation agent."
            )
        return await manager.async_store(call.data["content"])

    async def memory_search(call: ServiceCall) -> ServiceResponse:
        """Search long-term memory for a conversation entity."""
        entity_id = call.data["entity_id"]
        if not _is_memory_enabled(hass, entity_id):
            raise HomeAssistantError(
                f"Memory is not enabled for entity '{entity_id}'. "
                "Enable memory in the conversation agent settings."
            )
        manager = _get_memory_manager(entity_id)
        if manager is None:
            raise HomeAssistantError(
                f"Memory not initialized for entity '{entity_id}'. "
                "Ensure memory is enabled for this conversation agent."
            )
        results = await manager.async_search(
            query=call.data["query"],
            max_results=call.data["max_results"],
            min_score=call.data["min_score"],
        )
        return {
            "results": [
                {
                    "id": r.id,
                    "snippet": r.text,
                    "score": round(r.score, 3),
                    "created_at": r.created_at,
                }
                for r in results
            ]
        }

    async def memory_delete(call: ServiceCall) -> ServiceResponse:
        """Delete a memory entry for a conversation entity."""
        entity_id = call.data["entity_id"]
        if not _is_memory_enabled(hass, entity_id):
            raise HomeAssistantError(
                f"Memory is not enabled for entity '{entity_id}'. "
                "Enable memory in the conversation agent settings."
            )
        manager = _get_memory_manager(entity_id)
        if manager is None:
            raise HomeAssistantError(
                f"Memory not initialized for entity '{entity_id}'. "
                "Ensure memory is enabled for this conversation agent."
            )
        return await manager.async_delete(call.data["id"])

    async def download_skill(call: ServiceCall) -> ServiceResponse:
        """Download a skill from the GitHub repository."""
        from .skills import SkillManager

        skill_name = call.data["skill_name"]
        session = async_get_clientsession(hass)

        # Fetch skill directory contents from GitHub API
        api_url = (
            f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"
            f"/contents/{GITHUB_SKILLS_PATH}/{skill_name}"
            f"?ref={GITHUB_SKILLS_BRANCH}"
        )

        downloaded_files: list[str] = []

        async def _download_directory(url: str, local_dir: Path) -> None:
            """Recursively download a directory from GitHub."""
            async with session.get(url) as resp:
                if resp.status == 404:
                    raise HomeAssistantError(
                        f"Skill `{skill_name}` not found in repository"
                    )
                if resp.status != 200:
                    raise HomeAssistantError(
                        f"Failed to fetch skill from GitHub (HTTP {resp.status})"
                    )
                items = await resp.json()

            if not isinstance(items, list):
                raise HomeAssistantError(
                    f"Unexpected response from GitHub for skill `{skill_name}`"
                )

            for item in items:
                item_path = local_dir / item["name"]
                if item["type"] == "file":
                    # Download file content
                    async with session.get(item["download_url"]) as file_resp:
                        if file_resp.status != 200:
                            raise HomeAssistantError(
                                f"Failed to download `{item['path']}`"
                            )
                        content = await file_resp.read()

                    await hass.async_add_executor_job(
                        _write_file_sync, item_path, content
                    )
                    downloaded_files.append(str(item["path"]))
                elif item["type"] == "dir":
                    # Recurse into subdirectory
                    await _download_directory(item["url"], item_path)

        def _write_file_sync(file_path: Path, content: bytes) -> None:
            """Write file content to disk (run in executor)."""
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

        # Determine target directory
        skill_manager = await SkillManager.async_get_instance(hass)
        target_dir = skill_manager.user_skills_dir / skill_name

        _LOGGER.info("Downloading skill `%s` to %s", skill_name, target_dir)

        try:
            await _download_directory(api_url, target_dir)
        except HomeAssistantError:
            raise
        except Exception as err:
            raise HomeAssistantError(
                f"Failed to download skill `{skill_name}`: {err}"
            ) from err

        # Reload skills after download
        await skill_manager.async_load_skills()

        _LOGGER.info(
            "Successfully downloaded skill `%s` (%d files)",
            skill_name,
            len(downloaded_files),
        )

        return {
            "skill_name": skill_name,
            "downloaded_files": downloaded_files,
            "target_directory": str(target_dir),
        }

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
        SERVICE_MEMORY_STORE,
        memory_store,
        schema=MEMORY_STORE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_SEARCH,
        memory_search,
        schema=MEMORY_SEARCH_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_MEMORY_DELETE,
        memory_delete,
        schema=MEMORY_DELETE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )


def to_image_param(hass: HomeAssistant, image: dict) -> dict:
    """Convert url to base64 encoded image if local."""
    url = image["url"]

    if urlparse(url).scheme in cv.EXTERNAL_URL_PROTOCOL_SCHEMA_LIST:
        return image

    if not hass.config.is_allowed_path(url):
        raise HomeAssistantError(
            f"Cannot read `{url}`, no access to path; "
            "`allowlist_external_dirs` may need to be adjusted in "
            "`configuration.yaml`"
        )
    if not Path(url).exists():
        raise HomeAssistantError(f"`{url}` does not exist")
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type is None or not mime_type.startswith("image"):
        raise HomeAssistantError(f"`{url}` is not an image")

    image["url"] = f"data:{mime_type};base64,{encode_image(url)}"
    return image


def encode_image(image_path: str) -> str:
    """Convert to base64 encoded image."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
