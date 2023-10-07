from abc import ABC, abstractmethod
import yaml
import os
import logging
import re
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


class FileSettingLoader:
    def __init__(
        self,
        file_path: str,
        initial_settings: str,
        encoding: str = "utf-8",
        template_keys=["data", "event_data", "target", "service"],
    ) -> None:
        self.file_path = file_path
        self.initial_settings = initial_settings
        self.encoding = encoding
        self.settings = None
        self._create_file_if_not_exists()
        self.template_keys = template_keys

    def _create_file_if_not_exists(self):
        match = re.search(f"(.*?)([^{os.sep}]+)$", self.file_path)
        directory = match.group(1)
        file_name = match.group(2)
        assert file_name, f"파일({file_name})을 찾을 수 없습니다."

        if os.path.isdir(directory) == False:
            os.makedirs(directory)
        if os.path.isfile(self.file_path) == False:
            with open(self.file_path, "w", encoding=self.encoding) as f:
                f.write(self.initial_settings)

    def get_setting(self):
        if not self.settings:
            self.load()
        return self.settings

    def load(self):
        with open(self.file_path, encoding=self.encoding) as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
            for setting in settings:
                for function in setting["function"].values():
                    self.convert_to_template(function, [])
            self.settings = settings
            _LOGGER.debug("setting loaded: " + str(self.settings))

    def convert_to_template(self, settings, parents: list[str]):
        if isinstance(settings, dict):
            for key, value in settings.items():
                if isinstance(value, str) and (
                    key in self.template_keys
                    or set(parents).intersection(self.template_keys)
                ):
                    settings[key] = template.Template(value)
                if isinstance(value, dict):
                    parents.append(key)
                    self.convert_to_template(value, parents)
                    parents.pop()
        if isinstance(settings, list):
            for setting in settings:
                self.convert_to_template(setting, parents)


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
