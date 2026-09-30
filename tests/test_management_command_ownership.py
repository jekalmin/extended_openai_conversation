"""Contracts for the one owned Management command pipeline, not isolated wrappers."""

from __future__ import annotations

import ast
import asyncio
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest

from custom_components.extended_openai_conversation_responses import (
    agent_maintenance,
    management_function_quarantine as quarantine,
    management_ui as ui,
)
from homeassistant.exceptions import HomeAssistantError


def test_production_dispatcher_and_section_owners_are_not_reassigned() -> None:
    """Features may call the API, but cannot install replacements or hidden layers."""
    root = Path(ui.__file__).parent
    protected = {"async_management_command", "_async_management_request"}
    protected.update(
        handler.__name__ for handler in ui._MANAGEMENT_SECTION_HANDLERS.values()
    )
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            for target in targets:
                for part in ast.walk(target):
                    if (
                        isinstance(part, ast.Attribute)
                        and part.attr in protected
                        or isinstance(part, ast.Name)
                        and part.id in protected
                        or isinstance(part, ast.Subscript)
                        and isinstance(part.slice, ast.Constant)
                        and part.slice.value in protected
                    ):
                        violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"setattr", "delattr"} and len(node.args) >= 2:
                    attribute = node.args[1]
                    if (
                        isinstance(attribute, ast.Constant)
                        and attribute.value in protected
                    ):
                        violations.append(f"{path.name}:{node.lineno}")
    assert violations == []
    assert ui.async_management_command.__module__ == ui.__name__
    assert not hasattr(ui.async_management_command, "__wrapped__")
    assert isinstance(ui._MANAGEMENT_SECTION_HANDLERS, MappingProxyType)
    for section, handler in ui._MANAGEMENT_SECTION_HANDLERS.items():
        assert handler is getattr(ui, f"async_{section}_command")


def test_real_installer_activation_and_repeated_setup_preserve_identity() -> None:
    """Run all real installers in a fresh process so runtime patches cannot leak to tests."""
    script = r"""
import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch
from custom_components import extended_openai_conversation_responses as integration
from custom_components.extended_openai_conversation_responses import management_ui as ui
from custom_components.extended_openai_conversation_responses import management_loading_performance as loading
from homeassistant.components import panel_custom, websocket_api

async def main():
    command = ui.async_management_command
    handlers = ui._MANAGEMENT_SECTION_HANDLERS
    snapshot = loading._agent_snapshot
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_entries.return_value = []
    hass.http.async_register_static_paths = AsyncMock()
    hass.async_add_executor_job = AsyncMock(side_effect=lambda callback: callback())
    async_io = (
        "async_setup_model_catalog", "async_get_quiet_hours", "async_setup_delayed_tools",
        "async_migrate_integration", "async_recover_pending_restores", "async_setup_ha_permissions",
        "async_setup_services", "async_setup_intercom_services",
    )
    with ExitStack() as stack:
        for name in async_io:
            stack.enter_context(patch.object(integration, name, AsyncMock()))
        stack.enter_context(patch.object(websocket_api, "async_register_command"))
        stack.enter_context(patch.object(panel_custom, "async_register_panel", AsyncMock()))
        for _ in range(2):
            assert await integration.async_setup(hass, {})
            assert ui.async_management_command is command
            assert ui._MANAGEMENT_SECTION_HANDLERS is handlers
            assert loading._agent_snapshot is snapshot
            assert not hasattr(command, "__wrapped__")
    print("stable Management ownership")

asyncio.run(main())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(ui.__file__).parents[2],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "stable Management ownership" in completed.stdout


async def test_pipeline_order_and_immutable_selected_context(
    hass, management_agent, management_message, monkeypatch
) -> None:
    events = []
    entry, subentry = management_agent

    @asynccontextmanager
    async def lease(_hass, _message):
        events.append("lease-enter")
        try:
            yield
        finally:
            events.append("lease-exit")

    def authorize(_admin, _message):
        events.append("authorize")

    def select(*_args):
        events.append("select")
        return entry, subentry

    async def handler(request):
        events.append("route")
        assert request.hass is hass
        assert request.entry is entry
        assert request.subentry is subentry
        assert request.user_id == "admin"
        with pytest.raises(FrozenInstanceError):
            request.subentry_id = "other"
        return {"config": {"marker": True}}

    def project(_hass, entry_data, result, *, action):
        events.append("project")
        assert entry_data is entry.data
        assert action == "validate"
        return {**result, "projected": True}

    monkeypatch.setattr(ui, "management_command_lease", lease)
    monkeypatch.setattr(ui, "require_management_permission", authorize)
    monkeypatch.setattr(ui, "entry_and_agent", select)
    monkeypatch.setattr(ui, "_MANAGEMENT_SECTION_HANDLERS", {"configuration": handler})
    monkeypatch.setattr(ui, "decorate_configuration_result", project)
    result = await ui.async_management_command(
        hass, "admin", True, management_message("configuration", "validate")
    )
    assert result["projected"] is True
    assert events == [
        "lease-enter",
        "authorize",
        "select",
        "route",
        "project",
        "lease-exit",
    ]


@pytest.mark.parametrize(
    "section", ["quiet_hours", "knowledge", "diagnostics", "usage"]
)
async def test_global_authorization_precedes_selection_and_validation(
    hass, section
) -> None:
    with pytest.raises(
        HomeAssistantError, match="Administrator permission is required"
    ):
        await ui.async_management_command(
            hass,
            "non-admin",
            False,
            {
                "section": section,
                "action": "invalid-action",
                "config": [],
                "revision": 123,
            },
        )
    hass.config_entries.async_get_entry.assert_not_called()
    hass.config_entries.async_update_subentry.assert_not_called()


@pytest.mark.parametrize(
    "section", ["configuration", "tools", "request_rules", "guest_mode", "backup"]
)
async def test_selected_admin_sections_reject_before_mutation_validation(
    hass, management_message, section
) -> None:
    with pytest.raises(
        HomeAssistantError, match="Administrator permission is required"
    ):
        await ui.async_management_command(
            hass,
            "non-admin",
            False,
            management_message(section, "save", config=[], revision=123),
        )
    hass.config_entries.async_update_subentry.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        {},
        {"action": 1},
        {"action": "get", "section": None},
        {"action": [], "section": "tools"},
    ],
)
async def test_invalid_common_fields_are_home_assistant_errors(hass, message) -> None:
    with pytest.raises(HomeAssistantError, match="section and action must be strings"):
        await ui.async_management_command(hass, "admin", True, message)


@pytest.mark.parametrize("section", ["unknown", "configuration", "tools"])
async def test_invalid_section_or_action_retains_meaningful_error(
    hass, management_message, section
) -> None:
    with pytest.raises(
        HomeAssistantError, match=f"Unknown {section} management action: invalid"
    ):
        await ui.async_management_command(
            hass, "admin", True, management_message(section, "invalid")
        )


async def test_other_users_memory_scope_is_rejected_before_loading(
    hass, management_message, monkeypatch
) -> None:
    load = AsyncMock()
    monkeypatch.setattr(ui, "async_get_memory", load)
    with pytest.raises(
        HomeAssistantError, match="This scope is not available to the current user"
    ):
        await ui.async_management_command(
            hass,
            "alice",
            False,
            management_message("memories", "list", scope_id="user:bob"),
        )
    load.assert_not_awaited()


async def test_pending_maintenance_blocks_dispatch_and_releases_on_error(
    hass, management_message, monkeypatch
) -> None:
    gate = agent_maintenance.get_agent_maintenance_gate(hass, "entry-1", "agent-1")
    entered = asyncio.Event()

    async def handler(_request):
        entered.set()
        raise HomeAssistantError("handler failed")

    monkeypatch.setattr(ui, "_MANAGEMENT_SECTION_HANDLERS", {"tools": handler})
    async with gate.exclusive():
        task = asyncio.create_task(
            ui.async_management_command(
                hass, "admin", True, management_message("tools", "list")
            )
        )
        await asyncio.sleep(0)
        assert not entered.is_set()
    with pytest.raises(HomeAssistantError, match="handler failed"):
        await task
    assert entered.is_set()
    async with asyncio.timeout(1), gate.exclusive():
        assert gate._active_readers == 0
    assert not quarantine._ALLOW_QUARANTINED_TOOLS.get()
    assert not quarantine._QUARANTINED_FUNCTION_NAMES.get()


@pytest.mark.parametrize(
    "section,action",
    [("configuration", "validate"), ("function_repair", "configuration_validate")],
)
async def test_configuration_projection_is_inside_maintenance_lease(
    hass, management_message, monkeypatch, section, action
) -> None:
    gate = agent_maintenance.get_agent_maintenance_gate(hass, "entry-1", "agent-1")
    handler = AsyncMock(return_value={"config": {}})
    monkeypatch.setattr(ui, "_MANAGEMENT_SECTION_HANDLERS", {section: handler})

    def projection(*_args, **_kwargs):
        assert gate._active_readers == 1
        raise HomeAssistantError("projection failed")

    monkeypatch.setattr(ui, "decorate_configuration_result", projection)
    with pytest.raises(HomeAssistantError, match="projection failed"):
        await ui.async_management_command(
            hass, "admin", True, management_message(section, action)
        )
    assert gate._active_readers == 0


async def test_speech_preview_awaits_isolated_regex_engine(
    hass, management_message, monkeypatch
) -> None:
    speech = AsyncMock(return_value="safe result")
    monkeypatch.setattr(ui, "async_process_speech_text", speech)
    result = await ui.async_management_command(
        hass,
        "admin",
        True,
        management_message(
            "configuration", "speech_preview", sample_text="sample", config={}
        ),
    )
    assert result == {"speech_text": "safe result"}
    speech.assert_awaited_once()
    assert speech.await_args.args[0] is hass
    assert speech.await_args.args[1] == "sample"


async def test_invalid_configuration_validation_has_no_projection_or_write(
    hass, management_message
) -> None:
    result = await ui.async_management_command(
        hass,
        "admin",
        True,
        management_message(
            "configuration", "validate", config={"nonexistent_option": True}
        ),
    )
    assert result["valid"] is False
    assert result["errors"]
    assert "configuration_guidance" not in result
    assert "exposed_attribute_catalog" not in result
    hass.config_entries.async_update_subentry.assert_not_called()


async def test_strict_rule_preflight_precedes_quarantine_scope(
    hass, management_message, monkeypatch
):
    """Do not accidentally move outer dependency validation into tolerant parsing."""
    seen = []
    candidate = {"name": "Rule", "action": {"actions": []}}
    monkeypatch.setattr(ui, "_prepare_request_rule", lambda *_: candidate)

    def strict(_data):
        assert not quarantine._ALLOW_QUARANTINED_TOOLS.get()
        seen.append("strict")
        return []

    async def validate(_hass, value, tools):
        assert not quarantine._ALLOW_QUARANTINED_TOOLS.get()
        assert value is candidate
        assert tools == []
        seen.append("validate")

    async def handler(request):
        assert quarantine._ALLOW_QUARANTINED_TOOLS.get()
        seen.append("route")
        return {"ok": True}

    monkeypatch.setattr(ui, "_strict_configured_function_tools", strict)
    monkeypatch.setattr(ui, "async_validate_request_rule_functions", validate)
    monkeypatch.setattr(ui, "_MANAGEMENT_SECTION_HANDLERS", {"request_rules": handler})
    assert await ui.async_management_command(
        hass, "admin", True, management_message("request_rules", "create", rule={})
    ) == {"ok": True}
    assert seen == ["strict", "validate", "route"]
    assert not quarantine._ALLOW_QUARANTINED_TOOLS.get()


@pytest.mark.parametrize(
    "address",
    [
        "https://example.com",
        "http://example.com",
        "www.example.com",
        "www.example.com/path),",
    ],
)
async def test_speech_preview_uses_shared_bare_url_cleanup(
    hass, management_message, address
) -> None:
    from custom_components.extended_openai_conversation_responses.speech import (
        process_speech_text,
    )

    config = {
        "speech_processing_enabled": True,
        "speech_strip_urls": True,
        "speech_strip_markdown": True,
        "speech_regex_replacements": [],
    }
    sample = f"Visit {address} for details."
    result = await ui.async_management_command(
        hass,
        "admin",
        True,
        management_message(
            "configuration", "speech_preview", sample_text=sample, config=config
        ),
    )
    assert (
        result["speech_text"]
        == process_speech_text(sample, config)
        == "Visit for details."
    )
