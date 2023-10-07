import yaml
import os
import logging
import re
from homeassistant.helpers import template

_LOGGER = logging.getLogger(__name__)


class FileSettingLoader:
    def __init__(
        self,
        file_path: str,
        initial_settings: str,
        encoding: str = "utf-8",
        template_keys=["data", "target", "service"],
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
                for action in setting["function"]:
                    self.convert_to_template(action)
            self.settings = settings
            _LOGGER.debug("setting loaded: " + str(self.settings))

    def convert_to_template(self, action: dict):
        for key, value in action.items():
            if isinstance(value, str):
                action[key] = template.Template(value)
            if isinstance(value, dict) and key in self.template_keys:
                for k in value.keys():
                    value[k] = template.Template(value[k])
