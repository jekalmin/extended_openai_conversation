from abc import ABC, abstractmethod
import logging
import os
import yaml
import time
import asyncio

from homeassistant.components import automation
from homeassistant.components.automation.config import _async_validate_config_item
from homeassistant.const import SERVICE_RELOAD
from homeassistant.config import AUTOMATION_CONFIG_PATH
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
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from aiohttp import ClientSession


from .exceptions import (
    EntityNotFound,
    EntityNotExposed,
    CallServiceError,
    NativeNotFound,
    PineconeAuthenticationError,
)

from .const import DOMAIN, EVENT_AUTOMATION_REGISTERED


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


class PineconeStorage:
    def __init__(self, hass: HomeAssistant, api_key: str) -> None:
        """initialize pinecone storage"""
        self.hass = hass
        self.api_key = api_key
        self.host = None

    async def validate(self):
        headers = {"Api-Key": self.api_key}

        session = self.get_session()
        response = await session.get(
            "https://controller.gcp-starter.pinecone.io/databases",
            headers=headers,
            timeout=5,
        )
        if response.status == 403:
            raise PineconeAuthenticationError()

        response.raise_for_status()
        return True

    async def get_host(self):
        if self.host is not None:
            return self.host

        headers = {"Api-Key": self.api_key}

        session = self.get_session()
        response = await session.get(
            "https://controller.gcp-starter.pinecone.io/databases",
            headers=headers,
            timeout=5,
        )
        response.raise_for_status()
        response_data = await response.json()
        if "ha-openai" not in response_data:
            response = await session.post(
                "https://controller.gcp-starter.pinecone.io/databases",
                headers=headers,
                timeout=5,
                json={"name": "ha-openai", "dimension": 1536, "metric": "cosine"},
            )
            if response.status != 201:
                raise HomeAssistantError("Failed to create Pinecone index")
            await asyncio.sleep(10)

        response = await session.get(
            "https://controller.gcp-starter.pinecone.io/databases/ha-openai",
            headers=headers,
            timeout=5,
        )
        response_data = await response.json()
        self.host = response_data.get("status", {}).get("host")
        return self.host

    def get_session(self, session: ClientSession = None) -> ClientSession:
        if session is not None:
            return session
        return async_get_clientsession(self.hass)

    async def save(
        self,
        vectors: list[dict],
        session: ClientSession = None,
    ):
        host = await self.get_host()
        response = await self.get_session(session).post(
            f"https://{host}/vectors/upsert",
            headers={"Api-Key": self.api_key},
            timeout=3,
            json={"vectors": vectors},
        )
        if response.status != 200:
            raise HomeAssistantError(await response.json())
        return await response.json()

    async def query(
        self,
        topK: int,
        vector: list[float],
        includeMetadata=True,
        session: ClientSession = None,
    ):
        host = await self.get_host()
        response = await self.get_session(session).post(
            f"https://{host}/query",
            headers={"Api-Key": self.api_key},
            timeout=3,
            json={"topK": topK, "vector": vector, "includeMetadata": includeMetadata},
        )
        if response.status != 200:
            raise HomeAssistantError(await response.json())
        return await response.json()

    async def remove(
        self,
        ids: list[str],
        session: ClientSession = None,
    ):
        host = await self.get_host()
        response = await self.get_session(session).post(
            f"https://{host}/vectors/delete",
            headers={"Api-Key": self.api_key},
            timeout=3,
            json={"ids": ids},
        )
        if response.status != 200:
            raise HomeAssistantError(await response.json())
        return await response.json()


class FunctionExecutor(ABC):
    def __init__(self) -> None:
        """initialize function executor"""

    @abstractmethod
    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ) -> str:
        """execute function"""


class NativeFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize native function"""

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ) -> str:
        name = function["function"]["name"]
        if name == "execute_service":
            return await self.execute_service(
                hass, function, arguments, user_input, exposed_entities
            )
        if name == "add_automation":
            return await self.add_automation(
                hass, function, arguments, user_input, exposed_entities
            )

        raise NativeNotFound(name)

    async def execute_service(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ) -> str:
        result = []
        for service_argument in arguments.get("list", []):
            domain = service_argument["domain"]
            service = service_argument["service"]
            service_data = service_argument.get(
                "service_data", service_argument.get("data", {})
            )
            entity_id = service_data.get("entity_id", service_argument.get("entity_id"))

            if isinstance(entity_id, str):
                entity_id = [e.strip() for e in entity_id.split(",")]
            service_data["entity_id"] = entity_id

            if entity_id is None:
                raise CallServiceError(domain, service, service_data)
            if not hass.services.has_service(domain, service):
                raise ServiceNotFound(domain, service)
            if any(hass.states.get(entity) is None for entity in entity_id):
                raise EntityNotFound(entity_id)
            exposed_entity_ids = map(lambda e: e["entity_id"], exposed_entities)
            if not set(entity_id).issubset(exposed_entity_ids):
                raise EntityNotExposed(entity_id)

            try:
                await hass.services.async_call(
                    domain=domain,
                    service=service,
                    service_data=service_data,
                )
                result.append(True)
            except HomeAssistantError:
                _LOGGER.error(e)
                result.append(False)

        return str(result)

    async def add_automation(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ) -> str:
        automation_config = yaml.safe_load(arguments["automation_config"])
        config = {"id": str(round(time.time() * 1000))}
        if isinstance(automation_config, list):
            config.update(automation_config[0])
        if isinstance(automation_config, dict):
            config.update(automation_config)

        await _async_validate_config_item(hass, config, True, False)

        automations = [config]
        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "r",
            encoding="utf-8",
        ) as f:
            current_automations = yaml.safe_load(f.read())

        with open(
            os.path.join(hass.config.config_dir, AUTOMATION_CONFIG_PATH),
            "a" if current_automations else "w",
            encoding="utf-8",
        ) as f:
            raw_config = yaml.dump(automations, allow_unicode=True, sort_keys=False)
            f.write("\n" + raw_config)

        await hass.services.async_call(automation.config.DOMAIN, SERVICE_RELOAD)
        hass.bus.async_fire(
            EVENT_AUTOMATION_REGISTERED,
            {"automation_config": config, "raw_config": raw_config},
        )
        return "Success"


class ScriptFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize script function"""

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ) -> str:
        script = Script(
            hass,
            function["function"]["sequence"],
            "extended_openai_conversation",
            DOMAIN,
            running_description=f"""[extended_openai_conversation] function {function.get("spec", {}).get("name")}""",
            logger=_LOGGER,
        )

        result = await script.async_run(
            run_variables=arguments, context=user_input.context
        )
        return result.variables.get("_function_result", "Success")


class TemplateFunctionExecutor(FunctionExecutor):
    def __init__(self) -> None:
        """initialize template function"""

    async def execute(
        self,
        hass: HomeAssistant,
        function,
        arguments,
        user_input: conversation.ConversationInput,
        exposed_entities,
    ) -> str:
        return template.Template(
            function["function"]["value_template"], hass
        ).async_render(
            arguments,
            parse_result=False,
        )
