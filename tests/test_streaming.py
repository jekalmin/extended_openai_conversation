import asyncio
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "extended_openai_conversation"
)

# Provide minimal OpenAI SDK stubs for test isolation
openai_module = types.ModuleType("openai")


class _PlaceholderClient:  # pragma: no cover - simple import shim
    def __init__(self, *args, **kwargs):
        ...


openai_module.AsyncOpenAI = _PlaceholderClient
openai_module.AsyncAzureOpenAI = _PlaceholderClient

exceptions_mod = types.ModuleType("openai._exceptions")


class AuthenticationError(Exception):
    ...


class OpenAIError(Exception):
    ...


exceptions_mod.AuthenticationError = AuthenticationError
exceptions_mod.OpenAIError = OpenAIError
sys.modules.setdefault("openai._exceptions", exceptions_mod)

chat_completion_mod = types.ModuleType("openai.types.chat.chat_completion")


class ChatCompletionMessage:
    def __init__(self, **data):
        self.__dict__.update(data)

    def model_dump(self, exclude_none: bool = False):  # pragma: no cover - shim
        return dict(self.__dict__)


class Choice:
    def __init__(self, **data):
        message = data.get("message") or {}
        if not isinstance(message, ChatCompletionMessage):
            message = ChatCompletionMessage(**message)
        self.message = message
        self.finish_reason = data.get("finish_reason")
        self.index = data.get("index")

    def model_dump(self, exclude_none: bool = False):  # pragma: no cover - shim
        return {
            "message": self.message.model_dump(exclude_none=exclude_none),
            "finish_reason": self.finish_reason,
            "index": self.index,
        }


class ChatCompletion:
    def __init__(self, **data):
        raw_choices = data.get("choices", [])
        self.choices = [
            choice if isinstance(choice, Choice) else Choice(**choice)
            for choice in raw_choices
        ]
        usage = data.get("usage") or {}
        self.usage = SimpleNamespace(**usage)
        self.id = data.get("id")
        self.model = data.get("model")
        self.created = data.get("created")
        self.object = data.get("object", "chat.completion")

    @classmethod
    def model_validate(cls, data):  # pragma: no cover - shim
        return cls(**data)

    def model_dump(self, exclude_none: bool = False):  # pragma: no cover - shim
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                choice.model_dump(exclude_none=exclude_none) for choice in self.choices
            ],
            "usage": dict(vars(self.usage)),
        }


chat_completion_mod.ChatCompletion = ChatCompletion
chat_completion_mod.ChatCompletionMessage = ChatCompletionMessage
chat_completion_mod.Choice = Choice

types_mod = types.ModuleType("openai.types")
chat_mod = types.ModuleType("openai.types.chat")
chat_mod.chat_completion = chat_completion_mod
types_mod.chat = chat_mod

openai_module.types = types_mod
sys.modules.setdefault("openai.types", types_mod)
sys.modules.setdefault("openai.types.chat", chat_mod)
sys.modules.setdefault("openai.types.chat.chat_completion", chat_completion_mod)
chat_image_param_mod = types.ModuleType(
    "openai.types.chat.chat_completion_content_part_image_param"
)


class ChatCompletionContentPartImageParam(dict):  # pragma: no cover - shim
    ...


chat_image_param_mod.ChatCompletionContentPartImageParam = (
    ChatCompletionContentPartImageParam
)
sys.modules.setdefault(
    "openai.types.chat.chat_completion_content_part_image_param", chat_image_param_mod
)
sys.modules.setdefault("openai", openai_module)

yaml_module = types.ModuleType("yaml")
yaml_module.safe_load = lambda data: json.loads(data) if data else None
yaml_module.dump = lambda data, sort_keys=False: json.dumps(data)
sys.modules.setdefault("yaml", yaml_module)

bs4_module = types.ModuleType("bs4")


class BeautifulSoup:  # pragma: no cover - simple shim for imports
    def __init__(self, *args, **kwargs):
        ...

    def select(self, *args, **kwargs):
        return []


bs4_module.BeautifulSoup = BeautifulSoup
sys.modules.setdefault("bs4", bs4_module)

vol_module = types.ModuleType("voluptuous")


class Schema:  # pragma: no cover - simplified validation shim
    def __init__(self, validator=None, *_, **__):
        self._validator = validator

    def __call__(self, value):
        if callable(self._validator):
            return self._validator(value)
        return value

    def extend(self, *_args, **_kwargs):  # pragma: no cover - simplified shim
        return self


def _identity(value):  # pragma: no cover - shim helper
    return value


def Optional(key, default=None, **_kwargs):
    return key


def Required(key, default=None, **_kwargs):
    return key


def All(*validators):
    def _chain(value):
        for validator in validators:
            value = validator(value)
        return value

    return _chain


def Any(*validators, **kwargs):  # pragma: no cover - simplified shim
    return validators[0] if validators else _identity


def In(options):  # pragma: no cover - simplified shim
    return _identity


vol_module.Schema = Schema
vol_module.Optional = Optional
vol_module.Required = Required
vol_module.All = All
vol_module.Any = Any
vol_module.In = In
vol_module.Invalid = Exception
sys.modules.setdefault("voluptuous", vol_module)


def _ensure_module(name: str, module: types.ModuleType | None = None) -> types.ModuleType:
    if module is None:
        module = types.ModuleType(name)
    existing = sys.modules.setdefault(name, module)
    return existing


homeassistant_pkg = _ensure_module("homeassistant", types.ModuleType("homeassistant"))
homeassistant_pkg.__path__ = []
components_pkg = _ensure_module("homeassistant.components", types.ModuleType("homeassistant.components"))
components_pkg.__path__ = []

helpers_pkg = _ensure_module("homeassistant.helpers", types.ModuleType("homeassistant.helpers"))
helpers_pkg.__path__ = []
selector_mod = types.ModuleType("homeassistant.helpers.selector")


class _ConfigEntrySelector:
    def __init__(self, config):  # pragma: no cover - stub
        self.config = config

    def __call__(self, value):  # pragma: no cover - stub
        return value


selector_mod.ConfigEntrySelector = _ConfigEntrySelector
sys.modules["homeassistant.helpers.selector"] = selector_mod
helpers_pkg.selector = selector_mod

conversation_mod = types.ModuleType("homeassistant.components.conversation")


class _AbstractConversationAgent:
    async def async_process(self, user_input):  # pragma: no cover - stub
        raise NotImplementedError


conversation_mod.AbstractConversationAgent = _AbstractConversationAgent
conversation_mod.ConversationInput = type("ConversationInput", (), {})
sys.modules.setdefault("homeassistant.components.conversation", conversation_mod)
components_pkg.conversation = conversation_mod

ha_pkg = _ensure_module("homeassistant.components.homeassistant", types.ModuleType("homeassistant.components.homeassistant"))
ha_pkg.__path__ = []
exposed_mod = types.ModuleType("homeassistant.components.homeassistant.exposed_entities")
exposed_mod.async_should_expose = lambda *args, **kwargs: False
sys.modules.setdefault("homeassistant.components.homeassistant.exposed_entities", exposed_mod)
ha_pkg.exposed_entities = exposed_mod

automation_mod = _ensure_module("homeassistant.components.automation", types.ModuleType("homeassistant.components.automation"))
automation_config_mod = types.ModuleType("homeassistant.components.automation.config")
automation_config_mod._async_validate_config_item = lambda *args, **kwargs: None
sys.modules.setdefault("homeassistant.components.automation.config", automation_config_mod)
automation_mod.config = automation_config_mod
components_pkg.automation = automation_mod

script_component_mod = _ensure_module("homeassistant.components.script", types.ModuleType("homeassistant.components.script"))
script_config_mod = types.ModuleType("homeassistant.components.script.config")
class _FakeSchema:
    def extend(self, value):  # pragma: no cover - stub
        return self


script_config_mod.SCRIPT_ENTITY_SCHEMA = _FakeSchema()
sys.modules.setdefault("homeassistant.components.script.config", script_config_mod)
script_component_mod.config = script_config_mod
components_pkg.script = script_component_mod

for name in ("energy", "recorder", "scrape"):
    module_name = f"homeassistant.components.{name}"
    components_pkg.__dict__[name] = _ensure_module(module_name, types.ModuleType(module_name))

components_pkg.scrape.COMBINED_SCHEMA = _FakeSchema()

rest_mod = _ensure_module("homeassistant.components.rest", types.ModuleType("homeassistant.components.rest"))
rest_const_mod = types.ModuleType("homeassistant.components.rest.const")
rest_const_mod.DEFAULT_METHOD = "GET"
rest_const_mod.DEFAULT_VERIFY_SSL = True
rest_const_mod.CONF_ENCODING = "utf-8"
rest_data_mod = types.ModuleType("homeassistant.components.rest.data")
rest_data_mod.DEFAULT_TIMEOUT = 10
sys.modules.setdefault("homeassistant.components.rest.const", rest_const_mod)
sys.modules.setdefault("homeassistant.components.rest.data", rest_data_mod)
rest_mod.const = rest_const_mod
rest_mod.data = rest_data_mod
rest_mod.RESOURCE_SCHEMA = {}
components_pkg.rest = rest_mod

config_entries_mod = types.ModuleType("homeassistant.config_entries")


class _ConfigEntry:  # pragma: no cover - stub
    def __init__(self, data=None, options=None):
        self.data = data or {}
        self.options = options or {}


config_entries_mod.ConfigEntry = _ConfigEntry
sys.modules.setdefault("homeassistant.config_entries", config_entries_mod)

const_mod = types.ModuleType("homeassistant.const")
const_mod.ATTR_NAME = "name"
const_mod.CONF_API_KEY = "api_key"
const_mod.MATCH_ALL = ["*"]
const_mod.CONF_ATTRIBUTE = "attribute"
const_mod.CONF_METHOD = "method"
const_mod.CONF_NAME = "name"
const_mod.CONF_PAYLOAD = "payload"
const_mod.CONF_RESOURCE = "resource"
const_mod.CONF_RESOURCE_TEMPLATE = "resource_template"
const_mod.CONF_TIMEOUT = "timeout"
const_mod.CONF_VALUE_TEMPLATE = "value_template"
const_mod.CONF_VERIFY_SSL = "verify_ssl"
const_mod.SERVICE_RELOAD = "reload"
sys.modules.setdefault("homeassistant.const", const_mod)

core_mod = types.ModuleType("homeassistant.core")


class _HomeAssistant:  # pragma: no cover - stub
    ...


class _State:
    def __init__(self, entity_id="", state=""):
        self.entity_id = entity_id
        self.state = state


core_mod.HomeAssistant = _HomeAssistant
core_mod.State = _State
core_mod.ServiceCall = type("ServiceCall", (), {})
core_mod.ServiceResponse = type("ServiceResponse", (), {})
core_mod.SupportsResponse = type("SupportsResponse", (), {})
sys.modules["homeassistant.core"] = core_mod
homeassistant_pkg.core = core_mod

exceptions_mod = types.ModuleType("homeassistant.exceptions")


class _ConfigEntryNotReady(Exception):
    ...


class _HomeAssistantError(Exception):
    ...


class _TemplateError(Exception):
    ...


exceptions_mod.ConfigEntryNotReady = _ConfigEntryNotReady
exceptions_mod.HomeAssistantError = _HomeAssistantError
exceptions_mod.TemplateError = _TemplateError
exceptions_mod.ServiceNotFound = type("ServiceNotFound", (Exception,), {})
sys.modules.setdefault("homeassistant.exceptions", exceptions_mod)

cv_mod = types.ModuleType("homeassistant.helpers.config_validation")
cv_mod.config_entry_only_config_schema = lambda domain: lambda value: value
cv_mod.template = lambda value: value
cv_mod.ensure_list = lambda value: value if isinstance(value, list) else [value]
cv_mod.string = lambda value: value
cv_mod.positive_int = int
cv_mod.EXTERNAL_URL_PROTOCOL_SCHEMA_LIST = ["http", "https"]
sys.modules.setdefault("homeassistant.helpers.config_validation", cv_mod)

er_mod = types.ModuleType("homeassistant.helpers.entity_registry")


class _EntityRegistry:
    def async_get(self, entity_id):  # pragma: no cover - stub
        return None


er_mod.async_get = lambda hass: _EntityRegistry()
sys.modules.setdefault("homeassistant.helpers.entity_registry", er_mod)

intent_mod = types.ModuleType("homeassistant.helpers.intent")


class _IntentResponse:
    def __init__(self, language="en"):
        self.language = language

    def async_set_error(self, *args, **kwargs):
        return None

    def async_set_speech(self, *args, **kwargs):
        return None


intent_mod.IntentResponse = _IntentResponse
intent_mod.IntentResponseErrorCode = types.SimpleNamespace(UNKNOWN="unknown")
sys.modules.setdefault("homeassistant.helpers.intent", intent_mod)

template_mod = types.ModuleType("homeassistant.helpers.template")


class _Template:
    def __init__(self, text, hass):
        self._text = text

    def async_render(self, *args, **kwargs):
        return self._text


template_mod.Template = _Template
sys.modules.setdefault("homeassistant.helpers.template", template_mod)

httpx_mod = types.ModuleType("homeassistant.helpers.httpx_client")
httpx_mod.get_async_client = lambda hass: object()
sys.modules.setdefault("homeassistant.helpers.httpx_client", httpx_mod)

aiohttp_client_mod = types.ModuleType("homeassistant.helpers.aiohttp_client")


async def _fake_async_get_clientsession(hass):  # pragma: no cover - stub
    return SimpleNamespace(get=lambda *args, **kwargs: None, post=lambda *args, **kwargs: None)


aiohttp_client_mod.async_get_clientsession = _fake_async_get_clientsession
sys.modules.setdefault("homeassistant.helpers.aiohttp_client", aiohttp_client_mod)

typing_mod = types.ModuleType("homeassistant.helpers.typing")
typing_mod.ConfigType = dict
sys.modules.setdefault("homeassistant.helpers.typing", typing_mod)

ulid_mod = types.ModuleType("homeassistant.util.ulid")
ulid_mod.ulid = lambda: "test-ulid"
sys.modules.setdefault("homeassistant.util.ulid", ulid_mod)
util_pkg = _ensure_module("homeassistant.util", types.ModuleType("homeassistant.util"))
util_pkg.__path__ = []
util_pkg.ulid = ulid_mod
dt_mod = types.ModuleType("homeassistant.util.dt")
dt_mod.utcnow = lambda: None
sys.modules.setdefault("homeassistant.util.dt", dt_mod)

config_mod = _ensure_module("homeassistant.config", types.ModuleType("homeassistant.config"))
config_mod.AUTOMATION_CONFIG_PATH = "automations.yaml"

script_helpers_mod = types.ModuleType("homeassistant.helpers.script")


class _Script:
    async def async_run(self, *args, **kwargs):  # pragma: no cover - stub
        return None


script_helpers_mod.Script = _Script
sys.modules.setdefault("homeassistant.helpers.script", script_helpers_mod)

const_spec = importlib.util.spec_from_file_location(
    "custom_components.extended_openai_conversation.const",
    PACKAGE_ROOT / "const.py",
)
const_module = importlib.util.module_from_spec(const_spec)
assert const_spec and const_spec.loader
sys.modules[const_spec.name] = const_module
const_spec.loader.exec_module(const_module)

conversation = importlib.import_module(
    "custom_components.extended_openai_conversation.__init__"
)

CONF_API_KEY = "api_key"
CONF_BASE_URL = const_module.CONF_BASE_URL
CONF_CHAT_MODEL = const_module.CONF_CHAT_MODEL
CONF_CONTEXT_THRESHOLD = const_module.CONF_CONTEXT_THRESHOLD
CONF_CONTEXT_TRUNCATE_STRATEGY = const_module.CONF_CONTEXT_TRUNCATE_STRATEGY
CONF_ENABLE_STREAMING = const_module.CONF_ENABLE_STREAMING
CONF_MAX_COMPLETION_TOKENS = const_module.CONF_MAX_COMPLETION_TOKENS
CONF_MAX_TOKENS = const_module.CONF_MAX_TOKENS
CONF_PROACTIVITY_ENABLED = const_module.CONF_PROACTIVITY_ENABLED
CONF_ROUTER_FORCE_TOOLS = const_module.CONF_ROUTER_FORCE_TOOLS
CONF_TEMPERATURE = const_module.CONF_TEMPERATURE
CONF_TOP_P = const_module.CONF_TOP_P
CONF_USE_RESPONSES_API = const_module.CONF_USE_RESPONSES_API
CONF_USE_TOOLS = const_module.CONF_USE_TOOLS
DEFAULT_CONTEXT_THRESHOLD = const_module.DEFAULT_CONTEXT_THRESHOLD
DEFAULT_CONTEXT_TRUNCATE_STRATEGY = const_module.DEFAULT_CONTEXT_TRUNCATE_STRATEGY


class FakeStream:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        async def _gen():
            for event in self._events:
                yield event

        return _gen()

    async def get_final_response(self):
        return self._final


class FakeResponses:
    def __init__(self, events, final):
        self._events = events
        self._final = final
        self.stream_called = False

    def stream(self, **kwargs):
        self.stream_called = True
        return FakeStream(self._events, self._final)

    async def create(self, **kwargs):
        return self._final


class FakeClient:
    def __init__(self, events, final):
        self.responses = FakeResponses(events, final)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: pytest.fail("Chat path should not be used")
            )
        )

    def platform_headers(self):
        return {}


class FakeResponse:
    def __init__(self):
        self._payload = {
            "id": "resp_123",
            "model": "o3-test",
            "created": 0,
            "output": [],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            },
        }

    def model_dump(self):
        return self._payload


class _ImmediateResult:
    def __init__(self, result):
        self._result = result

    def __await__(self):  # pragma: no cover - simple awaitable wrapper
        if asyncio.iscoroutine(self._result):
            return self._result.__await__()

        async def _wrap():
            return self._result

        return _wrap().__await__()


class FakeHass:
    def __init__(self):
        self.config = SimpleNamespace(location_name="Home")
        self.states = SimpleNamespace(
            async_all=lambda: [],
            get=lambda entity_id: None,
        )
        self.bus = SimpleNamespace(async_fire=lambda *args, **kwargs: None)

    def async_add_executor_job(self, func, *args):
        return _ImmediateResult(func(*args))


class FakeEntry:
    def __init__(self):
        self.data = {CONF_API_KEY: "test", CONF_BASE_URL: None}
        self.options = {
            CONF_CHAT_MODEL: "o3-test",
            CONF_MAX_TOKENS: 100,
            CONF_TOP_P: 0.9,
            CONF_TEMPERATURE: 0.6,
            CONF_USE_TOOLS: False,
            CONF_USE_RESPONSES_API: True,
            CONF_MAX_COMPLETION_TOKENS: 32,
            CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
            CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            CONF_ENABLE_STREAMING: True,
            CONF_PROACTIVITY_ENABLED: False,
            CONF_ROUTER_FORCE_TOOLS: False,
        }


def test_responses_streaming_accumulates_output(monkeypatch):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello "),
        SimpleNamespace(type="response.output_text.delta", delta="world"),
        SimpleNamespace(type="response.completed", response=FakeResponse()),
    ]
    final = FakeResponse()
    fake_client = FakeClient(events, final)

    monkeypatch.setattr(conversation, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(conversation, "AsyncAzureOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(conversation, "get_async_client", lambda hass: object())

    async def run_test():
        agent = conversation.OpenAIAgent(FakeHass(), FakeEntry())

        user_input = SimpleNamespace(
            conversation_id="conv1",
            text="Hello",
            language="en",
            context=SimpleNamespace(user_id="user1"),
            device_id=None,
        )
        messages = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": "Hello"},
        ]

        result = await agent.query(user_input, messages, [], 0)

        assert fake_client.responses.stream_called
        assert result.message.content.strip() == "Hello world"

    asyncio.run(run_test())


def test_streaming_emits_confirmation_for_memory_write(monkeypatch):
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Saving note."),
        SimpleNamespace(type="response.completed", response=FakeResponse()),
    ]
    final = FakeResponse()
    fake_client = FakeClient(events, final)

    monkeypatch.setattr(conversation, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(conversation, "AsyncAzureOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(conversation, "get_async_client", lambda hass: object())

    async def run_test():
        agent = conversation.OpenAIAgent(FakeHass(), FakeEntry())

        agent.entry.options.update(
            {
                CONF_USE_RESPONSES_API: True,
                CONF_ENABLE_STREAMING: True,
                const_module.CONF_FUNCTIONS: "[]",
                const_module.CONF_SPEAK_CONFIRMATION_FIRST: True,
            }
        )

        user_input = SimpleNamespace(
            conversation_id="conv-tts",
            text="Remember that the garden needs water",
            language="en",
            context=SimpleNamespace(user_id="user1"),
            pipeline=SimpleNamespace(end_stage="tts"),
            device_id=None,
        )
        messages = [
            {"role": "system", "content": "Base"},
            {"role": "user", "content": user_input.text},
        ]

        result = await agent.query(user_input, messages, [], 0)
        assert result.message.content.startswith("Got it — saved.")

    asyncio.run(run_test())
