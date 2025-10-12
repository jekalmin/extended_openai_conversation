import asyncio
import importlib.util
import sys
import types
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "extended_openai_conversation"
)

# Prepare minimal stubs for Home Assistant modules used by memory_tools
ha_module = types.ModuleType("homeassistant")
sys.modules.setdefault("homeassistant", ha_module)

ha_core = types.ModuleType("homeassistant.core")


class HomeAssistant:  # pragma: no cover - stub for typing only
    ...


ha_core.HomeAssistant = HomeAssistant
ha_core.State = type("State", (), {})
sys.modules.setdefault("homeassistant.core", ha_core)

ha_helpers = types.ModuleType("homeassistant.helpers")
ha_helpers.__path__ = []
sys.modules.setdefault("homeassistant.helpers", ha_helpers)

aiohttp_client_module = types.ModuleType("homeassistant.helpers.aiohttp_client")
aiohttp_client_module.async_get_clientsession = lambda hass: None
sys.modules.setdefault("homeassistant.helpers.aiohttp_client", aiohttp_client_module)

aiohttp_module = types.ModuleType("aiohttp")


class _AiohttpError(Exception):
    ...


aiohttp_module.ClientError = _AiohttpError
aiohttp_module.ClientResponseError = _AiohttpError
aiohttp_module.ClientSession = type("ClientSession", (), {})
aiohttp_module.ClientTimeout = lambda **kwargs: types.SimpleNamespace(**kwargs)
sys.modules.setdefault("aiohttp", aiohttp_module)

async_timeout_module = types.ModuleType("async_timeout")


class _TimeoutContext:
    def __init__(self, *_):
        ...

    async def __aenter__(self):  # pragma: no cover - simple shim
        return self

    async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - simple shim
        return False


async_timeout_module.timeout = lambda *args, **kwargs: _TimeoutContext()
sys.modules.setdefault("async_timeout", async_timeout_module)

# Ensure the package namespace exists for relative imports
package = types.ModuleType("custom_components.extended_openai_conversation")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("custom_components.extended_openai_conversation", package)

const_spec = importlib.util.spec_from_file_location(
    "custom_components.extended_openai_conversation.const",
    PACKAGE_ROOT / "const.py",
)
const_module = importlib.util.module_from_spec(const_spec)
assert const_spec and const_spec.loader
const_spec.loader.exec_module(const_module)
sys.modules[
    "custom_components.extended_openai_conversation.const"
] = const_module

memory_tools_spec = importlib.util.spec_from_file_location(
    "custom_components.extended_openai_conversation.memory_tools",
    PACKAGE_ROOT / "memory_tools.py",
)
memory_tools = importlib.util.module_from_spec(memory_tools_spec)
assert memory_tools_spec and memory_tools_spec.loader
sys.modules[memory_tools_spec.name] = memory_tools
memory_tools_spec.loader.exec_module(memory_tools)


def test_memory_write_uses_default_namespace(monkeypatch):
    captured = {}

    async def fake_request(
        hass,
        method,
        endpoint,
        payload,
        config,
        *,
        session=None,
        operation=None,
        allow_skip_breaker=False,
    ):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"status": "ok"}

    monkeypatch.setattr(memory_tools, "_request", fake_request)
    config = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key=None,
        default_namespace="corpus",
        write_path=memory_tools.WRITE_PATH_DEFAULT,
        search_path=memory_tools.SEARCH_PATH_DEFAULT,
    )
    async def run_test():
        result = await memory_tools.async_memory_write(
            None, config, {"text": "Store this"}
        )

        assert captured["endpoint"] == memory_tools.WRITE_PATH_DEFAULT
        assert captured["payload"]["namespace"] == "corpus"
        assert "acknowledged" in result

    asyncio.run(run_test())


def test_memory_write_normalizes_text_and_importance(monkeypatch):
    captured = {}

    async def fake_request(
        hass,
        method,
        endpoint,
        payload,
        config,
        *,
        session=None,
        operation=None,
        allow_skip_breaker=False,
    ):
        captured["payload"] = payload
        return {"status": "ok"}

    monkeypatch.setattr(memory_tools, "_request", fake_request)
    config = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key=None,
        default_namespace="corpus",
        write_path=memory_tools.WRITE_PATH_DEFAULT,
        search_path=memory_tools.SEARCH_PATH_DEFAULT,
    )

    async def run_test():
        await memory_tools.async_memory_write(
            None,
            config,
            {"text": "Remember that my favorite drink is tea"},
        )

        payload = captured["payload"]
        assert payload["text"] == "My favorite drink is tea."
        assert payload["importance"] == "high"

    asyncio.run(run_test())


def test_memory_search_redacts_sensitive_data(monkeypatch):
    async def fake_request(
        hass,
        method,
        endpoint,
        payload,
        config,
        *,
        session=None,
        operation=None,
        allow_skip_breaker=False,
    ):
        return {
            "results": [
                {"text": "Email me at user@example.com", "score": 0.91},
                {"text": "Card 4242 4242 4242 4242 expires soon", "score": 0.75},
                {"text": "Call me at +1 (555) 123-4567", "score": 0.82},
                {"text": "Device IP 192.168.0.10 and token sk-test-abcdef", "score": 0.8},
                {"text": "Login via https://user:pass@example.com/home", "score": 0.78},
            ]
        }

    monkeypatch.setattr(memory_tools, "_request", fake_request)
    config = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key="token",
        default_namespace="corpus",
        write_path=memory_tools.WRITE_PATH_DEFAULT,
        search_path=memory_tools.SEARCH_PATH_DEFAULT,
    )
    async def run_test():
        text, snippets = await memory_tools.async_memory_search(
            None, config, {"query": "test", "k": 5}
        )

        assert "[REDACTED_EMAIL]" in text
        assert "[REDACTED_ACCOUNT]" in text
        assert "[REDACTED_PHONE]" in text
        assert "[REDACTED_IP]" in text
        assert "[REDACTED_TOKEN]" in text
        assert "https://[REDACTED]" in text
        assert len(snippets) == 5

    asyncio.run(run_test())


def test_memory_search_respects_token_budget(monkeypatch):
    async def fake_request(
        hass,
        method,
        endpoint,
        payload,
        config,
        *,
        session=None,
        operation=None,
        allow_skip_breaker=False,
    ):
        return {
            "results": [
                {
                    "text": "High score summary. " + "A" * 120,
                    "score": 0.95,
                },
                {
                    "text": "Medium score context. " + "B" * 120,
                    "score": 0.8,
                },
                {
                    "text": "Lowest score filler. " + "C" * 120,
                    "score": 0.6,
                },
            ]
        }

    monkeypatch.setattr(memory_tools, "_request", fake_request)
    config = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key=None,
        default_namespace="corpus",
        write_path=memory_tools.WRITE_PATH_DEFAULT,
        search_path=memory_tools.SEARCH_PATH_DEFAULT,
    )

    async def run_test():
        text, snippets = await memory_tools.async_memory_search(
            None,
            config,
            {"query": "budget", "k": 3},
            token_budget=30,
        )

        assert len(snippets) == 2
        assert "High score" in text
        assert "Medium score" in text
        assert "Lowest score" not in text

    asyncio.run(run_test())


def test_memory_paths_respect_options(monkeypatch):
    endpoints = []

    async def fake_request(
        hass,
        method,
        endpoint,
        payload,
        config,
        *,
        session=None,
        operation=None,
        allow_skip_breaker=False,
    ):
        endpoints.append(endpoint)
        if "search" in endpoint:
            return {"results": []}
        return {"status": "ok"}

    monkeypatch.setattr(memory_tools, "_request", fake_request)

    custom = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key=None,
        default_namespace="corpus",
        write_path="/alt/write",
        search_path="/alt/search",
    )
    fallback = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key=None,
        default_namespace="corpus",
        write_path="",
        search_path="",
    )

    async def run_test():
        await memory_tools.async_memory_write(None, custom, {"text": "Custom"})
        await memory_tools.async_memory_search(None, custom, {"query": "q"})
        await memory_tools.async_memory_write(None, fallback, {"text": "Default"})
        await memory_tools.async_memory_search(None, fallback, {"query": "q"})

    asyncio.run(run_test())

    assert endpoints == [
        "/alt/write",
        "/alt/search",
        memory_tools.WRITE_PATH_DEFAULT,
        memory_tools.SEARCH_PATH_DEFAULT,
    ]


def test_circuit_breaker_skips_proactive_calls(monkeypatch):
    config = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key=None,
        default_namespace="corpus",
        write_path=memory_tools.WRITE_PATH_DEFAULT,
        search_path=memory_tools.SEARCH_PATH_DEFAULT,
    )

    memory_tools._BREAKERS.clear()
    key = memory_tools._breaker_key(config, "search")
    memory_tools._BREAKERS[key] = memory_tools._BreakerState(
        open_until=memory_tools.time.monotonic() + 60
    )

    called = {"count": 0}

    async def fake_request(
        hass,
        method,
        endpoint,
        payload,
        inner_config,
        *,
        session=None,
        operation=None,
        allow_skip_breaker=False,
    ):
        called["count"] += 1
        return {"results": []}

    monkeypatch.setattr(memory_tools, "_request", fake_request)

    async def run_test():
        text, snippets = await memory_tools.async_memory_search(
            None, config, {"query": "skip"}, token_budget=10
        )
        assert text == ""
        assert snippets == []
        assert called["count"] == 0

        await memory_tools.async_memory_search(
            None, config, {"query": "forced"}, forced=True
        )
        assert called["count"] == 1

    asyncio.run(run_test())


def test_request_retries_on_server_error(monkeypatch):
    config = memory_tools.MemoryServiceConfig(
        base_url="http://example.com",
        api_key=None,
        default_namespace="corpus",
        write_path=memory_tools.WRITE_PATH_DEFAULT,
        search_path=memory_tools.SEARCH_PATH_DEFAULT,
    )

    class DummyResponseError(Exception):
        def __init__(self, *args, **kwargs):
            self.status = 503

    monkeypatch.setattr(memory_tools, "ClientResponseError", DummyResponseError)

    sleep_calls: list[float] = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(memory_tools.asyncio, "sleep", fake_sleep)

    class FakeHTTPResponse:
        def __init__(self, attempt: int):
            self.attempt = attempt

        def raise_for_status(self):
            if self.attempt < 3:
                raise DummyResponseError()

        async def json(self, content_type=None):
            return {"results": []}

    class FakeResponseContext:
        def __init__(self, attempt: int):
            self.attempt = attempt

        async def __aenter__(self):
            return FakeHTTPResponse(self.attempt)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            return FakeResponseContext(self.calls)

    fake_session = FakeSession()
    monkeypatch.setattr(
        memory_tools,
        "async_get_clientsession",
        lambda hass: fake_session,
    )

    async def run_test():
        result = await memory_tools._request(
            None,
            "POST",
            "/memories/search",
            {"query": "retry"},
            config,
            operation="search",
            allow_skip_breaker=False,
        )
        assert result == {"results": []}

    asyncio.run(run_test())

    assert fake_session.calls == 3
    assert sleep_calls == list(memory_tools._BACKOFF_DELAYS[:2])


def test_dispatch_memory_tool_unknown():
    async def run_test():
        config = memory_tools.MemoryServiceConfig(
            base_url="http://example.com",
            api_key=None,
            default_namespace="corpus",
            write_path=memory_tools.WRITE_PATH_DEFAULT,
            search_path=memory_tools.SEARCH_PATH_DEFAULT,
        )
        try:
            await memory_tools.dispatch_memory_tool(None, config, "memory.unknown", {})
        except ValueError:
            return
        raise AssertionError("Expected ValueError")

    asyncio.run(run_test())
