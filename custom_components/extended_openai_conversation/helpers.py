from abc import ABC, abstractmethod
import logging
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import template
from homeassistant.helpers.script import (
    Script,
    SCRIPT_MODE_SINGLE,
    SCRIPT_MODE_PARALLEL,
    DEFAULT_MAX,
    DEFAULT_MAX_EXCEEDED,
)

from .const import DOMAIN


_LOGGER = logging.getLogger(__name__)


def convert_to_template(
    settings, template_keys=["data", "event_data", "target", "service"]
):
    _convert_to_template(settings, template_keys, [])


def _convert_to_template(settings, template_keys, parents: list[str]):
    if isinstance(settings, dict):
        for key, value in settings.items():
            if isinstance(value, str) and (
                key in template_keys or set(parents).intersection(template_keys)
            ):
                settings[key] = template.Template(value)
            if isinstance(value, dict):
                parents.append(key)
                _convert_to_template(value, template_keys, parents)
                parents.pop()
    if isinstance(settings, list):
        for setting in settings:
            _convert_to_template(setting, template_keys, parents)


class CustomFunctionExecutor(ABC):
    def __init__(self) -> None:
        """init"""

    @abstractmethod
    async def execute(
        self,
        hass: HomeAssistant,
        custom_function,
        arguments,
        user_input: conversation.ConversationInput,
    ) -> str:
        """execute"""


class ScriptCustomFunctionExecutor(CustomFunctionExecutor):
    def __init__(self) -> None:
        """init"""

    async def execute(
        self,
        hass: HomeAssistant,
        custom_function,
        arguments,
        user_input: conversation.ConversationInput,
    ) -> str:
        _LOGGER.info("function", custom_function)
        _LOGGER.info("arguments", arguments)
        script = Script(
            hass,
            custom_function["function"]["sequence"],
            "extended_openai_conversation",
            DOMAIN,
            running_description=f"""[extended_openai_conversation] custom function {custom_function.get("spec", {}).get("name")}""",
            logger=_LOGGER,
        )

        await script.async_run(run_variables=arguments, context=user_input.context)
        return "Success"


class TemplateCustomFunctionExecutor(CustomFunctionExecutor):
    def __init__(self) -> None:
        """init"""

    async def execute(
        self,
        hass: HomeAssistant,
        custom_function,
        arguments,
        user_input: conversation.ConversationInput,
    ) -> str:
        return template.Template(
            custom_function["function"]["value_template"], hass
        ).async_render(
            arguments,
            parse_result=False,
        )
