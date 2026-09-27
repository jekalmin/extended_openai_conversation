"""Exercise every remaining built-in Function type through public Assist and SDK wire."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from aiohttp import web
import pytest

from custom_components.extended_openai_conversation_responses.const import (
    API_MODE_CHAT_COMPLETIONS,
    API_MODE_RESPONSES,
    CONF_API_MODE,
    CONF_FUNCTION_TOOLS,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_knowledge_provider_wire_e2e import (
    _chat_sse_tool_call,
    _tool_names,
)
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _install_wire,
    _responses_sse_text,
    _responses_sse_tool_call,
    _speech,
)
from tests_stress.conftest import record

REMAINING_TYPES = (
    "rest",
    "scrape",
    "composite",
    "sqlite",
    "bash",
    "read_file",
    "write_file",
    "edit_file",
)
ERROR_CASES = (
    "rest_404",
    "scrape_missing_selector",
    "sqlite_bad_query",
    "bash_nonzero",
    "read_file_missing",
    "write_file_denied",
    "edit_file_no_match",
)

API_MODES = (API_MODE_CHAT_COMPLETIONS, API_MODE_RESPONSES)


def _provider_replies(mode: str, call_id: str, name: str, text: str) -> list[bytes]:
    if mode == API_MODE_RESPONSES:
        return [_responses_sse_tool_call(call_id, name, {}), _responses_sse_text(text)]
    return [_chat_sse_tool_call(call_id, name, {}), _chat_sse_text(text)]


def _provider_result(request: dict[str, Any], mode: str, call_id: str) -> Any:
    body = request["body"]
    if mode == API_MODE_RESPONSES:
        call = next(
            item for item in body["input"] if item.get("type") == "function_call"
        )
        assert call["call_id"] == call_id
        item = next(
            item for item in body["input"] if item.get("type") == "function_call_output"
        )
        assert item["call_id"] == call_id
        serialized = item["output"]
    else:
        assistant = next(
            message for message in body["messages"] if message.get("tool_calls")
        )
        assert assistant["tool_calls"][0]["id"] == call_id
        item = next(
            message for message in body["messages"] if message.get("role") == "tool"
        )
        assert item["tool_call_id"] == call_id
        serialized = item["content"]
    return json.loads(serialized)["result"]


def _assert_exchange(wire: Any, mode: str, name: str) -> None:
    path = "/v1/responses" if mode == API_MODE_RESPONSES else "/v1/chat/completions"
    assert [request["path"] for request in wire.requests] == [path, path]
    assert name in _tool_names(wire.requests[0]["body"], mode)


def _configuration(kind: str, root: Path, url: str) -> dict[str, Any]:
    marker = root / "marker.txt"
    if kind == "rest":
        return {"type": kind, "resource": f"{url}/json", "method": "GET"}
    if kind == "scrape":
        return {
            "type": kind,
            "resource": f"{url}/html",
            "sensor": [{"select": ".probe", "name": "probe"}],
        }
    if kind == "composite":
        return {
            "type": kind,
            "sequence": [
                {
                    "type": "template",
                    "value_template": "COMPOSITE-FIRST",
                    "response_variable": "first",
                },
                {"type": "template", "value_template": "{{ first }}-SECOND"},
            ],
        }
    if kind == "sqlite":
        return {
            "type": kind,
            "db_url": f"file:{root / 'wire.db'}",
            "query": "SELECT value FROM probes WHERE id = 1",
            "single": True,
        }
    if kind == "bash":
        return {
            "type": kind,
            "command": "printf EOAI_BASH_WIRE",
            "allow_unsafe_shell": True,
            "cwd": str(root),
        }
    if kind == "read_file":
        return {"type": kind, "path": str(marker), "allow_dir": [str(root)]}
    if kind == "write_file":
        return {
            "type": kind,
            "path": str(marker),
            "content": "EOAI_WRITE_WIRE",
            "allow_dir": [str(root)],
        }
    if kind == "edit_file":
        return {
            "type": kind,
            "path": str(marker),
            "old_text": "BEFORE_EDIT",
            "new_text": "EOAI_EDIT_WIRE",
            "allow_dir": [str(root)],
        }
    raise AssertionError(kind)


@pytest.mark.parametrize("api_mode", API_MODES)
@pytest.mark.parametrize("kind", REMAINING_TYPES)
async def test_remaining_function_type_executes_on_provider_wire(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    socket_enabled: Any,
    tmp_path: Path,
    stress_trace: list[dict],
    kind: str,
    api_mode: str,
) -> None:
    del socket_enabled  # Local HTTP fixture and HA's real REST client use loopback.
    calls: list[str] = []

    async def json_response(_request: web.Request) -> web.Response:
        calls.append("rest")
        return web.json_response({"marker": "EOAI_REST_WIRE"})

    async def html_response(_request: web.Request) -> web.Response:
        calls.append("scrape")
        return web.Response(
            text='<html><span class="probe">EOAI_SCRAPE_WIRE</span></html>',
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/json", json_response)
    app.router.add_get("/html", html_response)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        marker = tmp_path / "marker.txt"
        marker.write_text(
            "BEFORE_EDIT" if kind == "edit_file" else "EOAI_READ_WIRE",
            encoding="utf-8",
        )
        with sqlite3.connect(tmp_path / "wire.db") as db:
            db.execute("CREATE TABLE probes (id INTEGER PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO probes VALUES (1, 'EOAI_SQLITE_WIRE')")
        tool_name = f"enhanced_{kind}_wire"
        entry = _make_entry(
            f"Enhanced {kind} Function wire",
            include_ai_task=False,
            conversation_options={
                CONF_API_MODE: api_mode,
                CONF_FUNCTION_TOOLS: [
                    {
                        "spec": {
                            "name": tool_name,
                            "description": f"Execute local {kind} fixture",
                            "parameters": {"type": "object", "properties": {}},
                        },
                        "function": _configuration(kind, tmp_path, url),
                        "enabled": True,
                    }
                ],
            },
        )
        await _setup_entry(hass, entry)
        agent = conversation.async_get_agent(hass, entry.entry_id)
        assert agent is not None
        call_id = f"call-{kind}"
        wire = _install_wire(
            monkeypatch,
            agent,
            _provider_replies(api_mode, call_id, tool_name, "Fixture complete"),
        )
        response = await conversation.async_converse(
            hass=hass,
            text=f"Execute {kind}",
            conversation_id=None,
            context=Context(),
            language="en",
            agent_id=entry.entry_id,
        )
        assert _speech(response) == "Fixture complete"
        _assert_exchange(wire, api_mode, tool_name)
        result = _provider_result(wire.requests[1], api_mode, call_id)
        if isinstance(result, dict):
            assert "error" not in result, result
        if kind == "rest":
            assert json.loads(result) == {"marker": "EOAI_REST_WIRE"}
            assert calls == ["rest"]
        elif kind == "scrape":
            assert result == "EOAI_SCRAPE_WIRE"
            assert calls == ["scrape"]
        elif kind == "composite":
            assert result == "COMPOSITE-FIRST-SECOND"
        elif kind == "sqlite":
            assert result == {"value": "EOAI_SQLITE_WIRE"}
        elif kind == "bash":
            assert result == {"exit_code": 0, "stdout": "EOAI_BASH_WIRE"}
        elif kind == "read_file":
            assert result == {"content": "EOAI_READ_WIRE", "size": 14}
        elif kind == "write_file":
            assert marker.read_text(encoding="utf-8") == "EOAI_WRITE_WIRE"
            assert result == {"success": True, "path": str(marker), "bytes_written": 15}
        elif kind == "edit_file":
            assert marker.read_text(encoding="utf-8") == "EOAI_EDIT_WIRE"
            assert result == {"success": True, "path": str(marker), "replacements": 1}
        record(
            stress_trace,
            "summary",
            layer="provider-wire",
            public_turns=1,
            provider_requests=2,
            actual_function_executions=1,
            **{f"{kind}_function_executions": 1},
        )
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("api_mode", API_MODES)
@pytest.mark.parametrize("failure", ERROR_CASES)
async def test_remaining_function_errors_are_serialized_on_provider_wire(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    socket_enabled: Any,
    tmp_path: Path,
    stress_trace: list[dict],
    failure: str,
    api_mode: str,
) -> None:
    del socket_enabled
    hits: list[str] = []

    async def html_response(_request: web.Request) -> web.Response:
        hits.append("html")
        return web.Response(
            text="<html><span>Other content</span></html>", content_type="text/html"
        )

    async def missing_response(_request: web.Request) -> web.Response:
        hits.append("rest")
        return web.Response(status=404, text="404: Not Found")

    app = web.Application()
    app.router.add_get("/html", html_response)
    app.router.add_get("/missing", missing_response)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        assert site._server is not None
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        kind = failure.split("_")[0]
        if failure.startswith(("read_file", "write_file", "edit_file")):
            kind = (
                failure.rsplit("_", 1)[0]
                if failure != "edit_file_no_match"
                else "edit_file"
            )
        config = _configuration(kind, tmp_path, url)
        if failure == "rest_404":
            config["resource"] = f"{url}/missing"
        elif failure == "scrape_missing_selector":
            config["sensor"] = [{"select": ".absent", "name": "absent"}]
        elif failure == "sqlite_bad_query":
            config["query"] = "SELECT value FROM absent_table"
        elif failure == "bash_nonzero":
            config["command"] = (
                f"printf 'attempt\\n' >> '{tmp_path / 'attempts.txt'}'; printf EOAI_FAILURE >&2; exit 7"
            )
        elif failure == "read_file_missing":
            config["path"] = str(tmp_path / "absent.txt")
        elif failure == "write_file_denied":
            (tmp_path / "marker.txt").write_text("PRESERVE", encoding="utf-8")
            config["allow_dir"] = [str(tmp_path / "allowed")]
        elif failure == "edit_file_no_match":
            (tmp_path / "marker.txt").write_text("PRESERVE", encoding="utf-8")
            config["old_text"] = "ABSENT_TEXT"
        if kind == "sqlite":
            with sqlite3.connect(tmp_path / "wire.db") as db:
                db.execute("CREATE TABLE probes (id INTEGER PRIMARY KEY, value TEXT)")
        name = f"enhanced_{failure}_wire"
        entry = _make_entry(
            f"Enhanced {failure} wire error",
            include_ai_task=False,
            conversation_options={
                CONF_API_MODE: api_mode,
                CONF_FUNCTION_TOOLS: [
                    {
                        "spec": {
                            "name": name,
                            "description": f"Exercise {failure} failure",
                            "parameters": {"type": "object", "properties": {}},
                        },
                        "function": config,
                        "enabled": True,
                    }
                ],
            },
        )
        await _setup_entry(hass, entry)
        agent = conversation.async_get_agent(hass, entry.entry_id)
        assert agent is not None
        call_id = f"call-{failure}"
        wire = _install_wire(
            monkeypatch,
            agent,
            _provider_replies(api_mode, call_id, name, "Failure handled"),
        )
        response = await conversation.async_converse(
            hass=hass,
            text=f"Execute {failure}",
            conversation_id=None,
            context=Context(),
            language="en",
            agent_id=entry.entry_id,
        )
        assert _speech(response) == "Failure handled"
        _assert_exchange(wire, api_mode, name)
        result = _provider_result(wire.requests[1], api_mode, call_id)
        if failure == "bash_nonzero":
            assert result == {"exit_code": 7, "stderr": "EOAI_FAILURE", "stdout": ""}
            assert (tmp_path / "attempts.txt").read_text(
                encoding="utf-8"
            ).splitlines() == ["attempt"]
        elif failure == "scrape_missing_selector":
            assert hits == ["html"]
            assert result is None
        elif failure == "rest_404":
            assert hits == ["rest"]
            assert result == "404: Not Found"
        elif failure == "sqlite_bad_query":
            assert result == {
                "status": "error",
                "error": "SQLite query failed: no such table: absent_table",
            }
        elif failure == "read_file_missing":
            assert result == {"error": f"File not found: {tmp_path / 'absent.txt'}"}
        elif failure == "write_file_denied":
            assert result == {
                "error": f"Access denied: path '{tmp_path / 'marker.txt'}' is not in allowed directories"
            }
            assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "PRESERVE"
        else:
            assert result == {"error": "Text not found in file: ABSENT_TEXT..."}
            assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "PRESERVE"
        record(
            stress_trace,
            "summary",
            layer="provider-wire",
            public_turns=1,
            provider_requests=2,
            provider_wire_function_errors=1,
            failure=failure,
        )
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("api_mode", API_MODES)
async def test_composite_late_failure_preserves_one_completed_side_effect(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stress_trace: list[dict],
    api_mode: str,
) -> None:
    side_effects = tmp_path / "side-effects.txt"
    missing = tmp_path / "missing.txt"
    name = "composite_partial_failure_wire"
    entry = _make_entry(
        "Composite partial failure",
        include_ai_task=False,
        conversation_options={
            CONF_API_MODE: api_mode,
            CONF_FUNCTION_TOOLS: [
                {
                    "spec": {
                        "name": name,
                        "description": "Run two steps",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "function": {
                        "type": "composite",
                        "sequence": [
                            {
                                "type": "bash",
                                "command": f"printf 'first\\n' >> '{side_effects}'",
                                "allow_unsafe_shell": True,
                                "cwd": str(tmp_path),
                            },
                            {
                                "type": "read_file",
                                "path": str(missing),
                                "allow_dir": [str(tmp_path)],
                            },
                        ],
                    },
                    "enabled": True,
                }
            ],
        },
    )
    await _setup_entry(hass, entry)
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    call_id = "call-composite-partial"
    wire = _install_wire(
        monkeypatch,
        agent,
        _provider_replies(api_mode, call_id, name, "Partial failure handled"),
    )
    response = await conversation.async_converse(
        hass=hass,
        text="Run composite",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
    )
    assert _speech(response) == "Partial failure handled"
    _assert_exchange(wire, api_mode, name)
    result = _provider_result(wire.requests[1], api_mode, call_id)
    assert result == {"error": f"File not found: {missing}"}
    assert side_effects.read_text(encoding="utf-8").splitlines() == ["first"]
    record(stress_trace, "composite_partial_failure", mode=api_mode, side_effects=1)
