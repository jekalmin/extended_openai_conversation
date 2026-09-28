"""Enhanced-only genuine HA Management mutation matrix."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from homeassistant.core import HomeAssistant
from tests_real_ha.test_browser_backend_acceptance import (
    _run_playwright,
    _start_ws_bridge,
)
from tests_real_ha.test_management_backend_acceptance import (
    _admin_client,
    _entry,
    _setup_entry,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ENHANCED_MANAGEMENT_CONTRACT") != "1",
    reason="selected only by the Enhanced nightly/manual browser campaign",
)


@pytest.mark.asyncio
async def test_seeded_management_mutations_cross_real_websocket(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """Seed durable and active owners, then mutate them through the shipped panel."""
    from datetime import timedelta

    from custom_components.extended_openai_conversation_responses.const import (
        CONVERSATION_CONTINUITY_USER,
    )
    from custom_components.extended_openai_conversation_responses.continuity import (
        async_get_continuity,
    )
    from custom_components.extended_openai_conversation_responses.conversation_archive import (
        async_get_archive,
    )
    from custom_components.extended_openai_conversation_responses.memory import (
        ANONYMOUS_USER_ID,
        async_get_memory,
    )
    from custom_components.extended_openai_conversation_responses.scope import (
        user_scope,
    )
    from custom_components.extended_openai_conversation_responses.temporary_memory import (
        async_get_temporary_memory,
    )
    from custom_components.extended_openai_conversation_responses.usage import (
        async_get_usage,
    )
    from homeassistant.helpers import llm
    from homeassistant.util import dt as dt_util
    from tests_real_ha.test_ha_llm_tool_acceptance import (
        AcceptanceAPI,
        AcceptanceEchoTool,
    )
    from tests_real_ha.test_management_backend_acceptance import ADMIN_ID

    entry = _entry("Seeded Management Browser Acceptance")
    await _setup_entry(hass, entry)
    subentry = next(item for item in entry.subentries.values() if item.subentry_type == "conversation")
    owner_scope = f"user:{ADMIN_ID}"

    temporary = await async_get_temporary_memory(hass, entry.entry_id, subentry.subentry_id)
    expiry = (dt_util.utcnow() + timedelta(hours=1)).isoformat()
    for index in range(3):
        await temporary.async_add(
            f"conversation:nightly-{index}", f"Nightly temporary fact {index}", expiry,
            owner_scope_id=owner_scope,
        )

    memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    await memory.async_add(ANONYMOUS_USER_ID, "Nightly legacy memory", "nightly", "explicit")

    archive = await async_get_archive(hass, entry.entry_id, subentry.subentry_id)
    scope = user_scope(ADMIN_ID, source="authenticated_user")
    session = await archive.async_begin_session(
        "nightly-archive", scope, "ha-nightly-archive",
        archive_enabled=True, shared_archive_enabled=True, inactivity_minutes=30,
    )
    assert session is not None
    await archive.async_record_turn(
        session.session_id, run_id="nightly-archive-run",
        user_text="Nightly archived question", assistant_text="Nightly archived answer",
        successful=True,
    )

    continuity = async_get_continuity(hass, entry.entry_id, subentry.subentry_id)
    resolved = await continuity.async_resolve(
        CONVERSATION_CONTINUITY_USER, scope, None, None, 30,
    )
    assert resolved.key and resolved.claim_token
    await continuity.async_record_success(resolved.key, resolved.claim_token, [])

    llm.async_register_api(hass, AcceptanceAPI(hass, AcceptanceEchoTool()))

    usage = await async_get_usage(hass, entry.entry_id, subentry.subentry_id)
    async with usage.async_run(home_assistant_conversation_id="nightly-usage"):
        await usage.async_record_request(
            successful=True, provider="openai", model="gpt-5-mini",
            api_mode="responses", request_stage="initial",
        )
    assert usage.runs and usage.requests

    client = await _admin_client(hass, hass_ws_client)
    runner, backend_url = await _start_ws_bridge(client)
    try:
        await _run_playwright(
            repo_root=Path(__file__).resolve().parent.parent,
            spec="tests_browser/real-ha-seeded-management.spec.mjs",
            config="playwright.config.mjs",
            env={"REAL_HA_BACKEND_URL": backend_url},
            failure_label="Seeded Playwright genuine-HA management acceptance failed",
        )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_function_repair_mutations_cross_real_websocket(
    hass: HomeAssistant,
    hass_ws_client: Any,
) -> None:
    """The shipped repair route edits a quarantined persisted tool collection."""
    from copy import deepcopy

    import yaml

    from custom_components.extended_openai_conversation_responses.const import (
        CONF_FUNCTION_TOOLS,
    )
    from tests.test_management_function_repair import _mixed_legacy_tool_data
    from tests_real_ha.test_acceptance_lifecycle import _make_entry

    data, mixed, _valid = _mixed_legacy_tool_data()
    broken = []
    for index in range(3):
        tool = deepcopy(mixed[1])
        tool["spec"]["name"] = f"nightly_broken_tool_{index}"
        broken.append(tool)
    data[CONF_FUNCTION_TOOLS] = yaml.safe_dump(
        [broken[0], mixed[0], broken[1], broken[2]], sort_keys=False,
    )
    entry = _make_entry(
        "Browser Function Repair Acceptance", include_ai_task=False,
        conversation_options=data,
    )
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)
    runner, backend_url = await _start_ws_bridge(client)
    try:
        await _run_playwright(
            repo_root=Path(__file__).resolve().parent.parent,
            spec="tests_browser/real-ha-function-repair.spec.mjs",
            config="playwright.config.mjs",
            env={"REAL_HA_BACKEND_URL": backend_url},
            failure_label="Playwright genuine-HA Function repair acceptance failed",
        )
    finally:
        await runner.cleanup()


