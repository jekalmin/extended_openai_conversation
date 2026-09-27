"""Generated valid configurations across HA management, SDK wire and restore."""

from __future__ import annotations

from time import monotonic

from custom_components.extended_openai_conversation_responses import (
    agent_config,
    backup,
)
from custom_components.extended_openai_conversation_responses.const import (
    GUEST_POLICY_VERSION,
)
from custom_components.extended_openai_conversation_responses.model_capabilities import (
    parameter_is_allowed,
)
from custom_components.extended_openai_conversation_responses.model_catalog import (
    model_metadata,
)
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from tests_real_ha.test_acceptance_lifecycle import _make_entry, _setup_entry
from tests_real_ha.test_management_backend_acceptance import (
    _admin_client,
    _conversation_subentry,
    _fresh_reload,
    _management_call,
)
from tests_real_ha.test_provider_wire_e2e import (
    _chat_sse_text,
    _install_wire,
    _responses_sse_text,
)
from tests_stress.conftest import record
from tests_stress.generated_valid_states import generate, normalized_state


def _completed_payload(api: str, model: str) -> dict:
    if api == "responses":
        return {
            "id": "resp-coverage",
            "object": "response",
            "created_at": 1,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "id": "msg-coverage",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Coverage reply",
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
    return {
        "id": "chatcmpl-coverage",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Coverage reply"},
            }
        ],
    }


async def _converse(hass: HomeAssistant, entry, monkeypatch, api: str, model: str):
    agent = conversation.async_get_agent(hass, entry.entry_id)
    assert agent is not None
    if model_metadata(model)["streaming"]:
        reply = (
            _responses_sse_text("Coverage reply")
            if api == "responses"
            else _chat_sse_text("Coverage reply")
        )
        scripted = reply.replace(b"gpt-5.6", model.encode())
    else:
        scripted = (200, _completed_payload(api, model))
    wire = _install_wire(monkeypatch, agent, [scripted])
    result = await conversation.async_converse(
        hass=hass,
        text="Generated coverage prompt",
        conversation_id=None,
        context=Context(),
        language="en",
        agent_id=entry.entry_id,
    )
    assert result.response.error_code is None, result.response
    assert result.response.as_dict()["speech"]["plain"]["speech"] == "Coverage reply"
    assert len(wire.requests) == 1
    return wire.requests[0]


def _assert_wire(request: dict, normalized: dict, api: str) -> None:
    body = request["body"]
    model = normalized["chat_model"]
    assert request["path"] == (
        "/v1/responses" if api == "responses" else "/v1/chat/completions"
    )
    assert body["model"] == model
    assert body.get("stream", False) is model_metadata(model)["streaming"]
    assert "max_tokens" not in body
    assert "functions" not in body
    assert "function_call" not in body
    for field in ("temperature", "top_p"):
        if field in body:
            assert parameter_is_allowed(
                model, field, normalized.get("reasoning_effort")
            )
    if "service_tier" in body:
        assert body["service_tier"] in model_metadata(model)["service_tiers"]
    if normalized["web_search"]:
        assert api == "responses"
        assert any(
            tool.get("type", "").startswith("web_search")
            for tool in body.get("tools", [])
        )
    if normalized.get("reasoning_effort") is not None:
        sent_effort = (
            (body.get("reasoning") or {}).get("effort")
            if api == "responses"
            else body.get("reasoning_effort")
        )
        assert sent_effort == normalized["reasoning_effort"]


async def test_generated_valid_states_cross_real_ha_and_sdk_wire(
    hass: HomeAssistant,
    hass_ws_client,
    monkeypatch,
    stress_seed: int,
    pytestconfig,
    stress_trace: list[dict],
) -> None:
    started = monotonic()
    heavy = pytestconfig.getoption("--stress-intensity") == "heavy"
    suite = generate(stress_seed, heavy=heavy)
    entry = _make_entry(
        "Generated valid states",
        include_ai_task=False,
        conversation_options={
            "functions": [],
            "guest_policy_version": GUEST_POLICY_VERSION,
        },
    )
    await _setup_entry(hass, entry)
    client = await _admin_client(hass, hass_ws_client)
    for number, case in enumerate(suite.cases):
        case_started = monotonic()
        normalized, api = normalized_state(case)
        record(
            stress_trace,
            "generated_state_start",
            case=number,
            model=normalized["chat_model"],
            api=api,
        )
        before = await _management_call(
            client, entry=entry, section="configuration", action="get"
        )
        await _management_call(
            client,
            entry=entry,
            section="configuration",
            action="update",
            revision=before["revision"],
            config=normalized,
        )
        await _fresh_reload(hass, entry)
        current = await _management_call(
            client, entry=entry, section="configuration", action="get"
        )
        persisted = agent_config.normalize_agent_config(current["config"])
        assert persisted == normalized
        request = await _converse(
            hass, entry, monkeypatch, api, normalized["chat_model"]
        )
        _assert_wire(request, normalized, api)
        # Backup is sampled across the covering sequence: every state crosses the
        # management/reload/wire boundary, while restore exercises distinct states.
        if number % max(1, len(suite.cases) // 12) == 0:
            subentry = _conversation_subentry(entry)
            snapshot = await backup.async_collect_backup_snapshot(hass, entry, subentry)
            mutated = normalized | {
                "advanced_options": not normalized["advanced_options"]
            }
            updated = await _management_call(
                client, entry=entry, section="configuration", action="get"
            )
            await _management_call(
                client,
                entry=entry,
                section="configuration",
                action="update",
                revision=updated["revision"],
                config=mutated,
            )
            restored = await backup.async_restore_backup(
                hass, entry, subentry, snapshot
            )
            assert restored["status"] == "restored"
            await hass.async_block_till_done()
            recovered = await _management_call(
                client, entry=entry, section="configuration", action="get"
            )
            assert (
                agent_config.normalize_agent_config(recovered["config"]) == normalized
            )
            request = await _converse(
                hass, entry, monkeypatch, api, normalized["chat_model"]
            )
            _assert_wire(request, normalized, api)
        record(
            stress_trace,
            "generated_state",
            case=number,
            model=normalized["chat_model"],
            api=api,
            elapsed_seconds=round(monotonic() - case_started, 3),
        )
    record(
        stress_trace,
        "generated_suite",
        **suite.evidence(stress_seed),
        elapsed_seconds=round(monotonic() - started, 3),
    )
