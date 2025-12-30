"""Template functions for Extended OpenAI Conversation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.template import TemplateEnvironment

from .const import DOMAIN
from .helpers import get_exposed_entities

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)

DATA_TEMPLATE_MANAGER = "template_manager"

TEMPLATE_EXTENDED_OPENAI = "extended_openai"
TEMPLATE_GET_ENTITIES = "exposed_entities"


async def async_setup_templates(hass: HomeAssistant) -> bool:
    """Set up template functions for Extended OpenAI Conversation."""
    hass.data.setdefault(DOMAIN, {})
    if hass.data[DOMAIN].get(DATA_TEMPLATE_MANAGER):
        return True

    manager = ExtendedOpenAITemplateManager(hass)
    hass.data[DOMAIN][DATA_TEMPLATE_MANAGER] = manager
    await manager.async_setup()
    return True


async def async_unload_templates(hass: HomeAssistant) -> bool:
    """Unload template functions for Extended OpenAI Conversation."""
    if len(hass.config_entries.async_entries(DOMAIN)) == 1:
        manager = hass.data.get(DOMAIN, {}).get(DATA_TEMPLATE_MANAGER)
        if manager:
            await manager.async_on_unload()
            hass.data[DOMAIN].pop(DATA_TEMPLATE_MANAGER, None)
    return True


class ExtendedOpenAITemplateManager:
    """Class to manage template functions."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the template manager."""
        self.hass = hass
        self._extended_openai = {TEMPLATE_GET_ENTITIES: self._get_exposed_entities}
        self._original_init = None

    def _get_exposed_entities(self) -> list[dict[str, Any]]:
        return get_exposed_entities(self.hass)

    async def async_setup(self) -> None:
        """Set up the template functions."""
        _LOGGER.debug("Setting up Extended OpenAI Conversation template functions")

        # Register in existing environments
        if "template.environment" in self.hass.data:
            self.hass.data["template.environment"].globals[TEMPLATE_EXTENDED_OPENAI] = (
                self._extended_openai
            )

        # Patch TemplateEnvironment
        self._original_init = TemplateEnvironment.__init__

        def template_environment_init(
            template_env_self: TemplateEnvironment,
            hass: HomeAssistant | None,
            limited: bool | None = False,
            strict: bool | None = False,
            log_fn: Callable[[int, str], None] | None = None,
        ) -> None:
            if self._original_init:
                self._original_init(template_env_self, hass, limited, strict, log_fn)
            if hass:
                template_env_self.globals[TEMPLATE_EXTENDED_OPENAI] = (
                    self._extended_openai
                )

        TemplateEnvironment.__init__ = template_environment_init

    async def async_on_unload(self) -> None:
        """Tear down the template functions."""
        _LOGGER.debug("Tearing down Extended OpenAI Conversation template functions")

        if self._original_init:
            TemplateEnvironment.__init__ = self._original_init
            self._original_init = None

        if "template.environment" in self.hass.data:
            self.hass.data["template.environment"].globals.pop(
                TEMPLATE_EXTENDED_OPENAI, None
            )
