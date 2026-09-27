"""Portable Request Rule packs and captured AI handoff contracts."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from custom_components.extended_openai_conversation_responses import request_rule_packs
from custom_components.extended_openai_conversation_responses.management_ui import (
    _consume_rule_pack_review,
    _register_rule_pack_review,
)
from custom_components.extended_openai_conversation_responses.request_rule_match_preview import (
    async_request_rule_match_preview,
)
from custom_components.extended_openai_conversation_responses.request_rule_packs import (
    async_append_rule_pack,
    export_rule_pack,
    validate_rule_pack,
)
from custom_components.extended_openai_conversation_responses.request_rules import (
    RequestRuleRuntime,
    RequestRules,
    async_evaluate_rule,
    validate_rule,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.typing import UNDEFINED
from tests.test_request_rules import FakeServices, MemoryStore, local_rule


def captured_routing_rule():
    rule = local_rule("Deep think", ["deep think {question}"], "sentence_pattern")
    rule["action_type"] = "model_routing"
    rule["action"] = {
        "model": "gpt-5",
        "reasoning_effort": "",
        "scope": "request",
        "reset": False,
        "continue_to_ai": True,
    }
    rule["ai_input_mode"] = "capture"
    rule["ai_input_capture"] = "question"
    return rule


def test_captured_ai_input_requires_every_variant_and_branch() -> None:
    rule = captured_routing_rule()
    assert validate_rule(rule)["ai_input_capture"] == "question"
    rule["phrases"].append("think carefully")
    with pytest.raises(ValueError, match=r"every trigger|same slots"):
        validate_rule(rule)
    rule["phrases"] = ["deep think [{question}]"]
    with pytest.raises(ValueError, match="every match"):
        validate_rule(rule)
    rule["phrases"] = ["deep think {question}"]
    rule["ai_input_capture"] = "stale"
    with pytest.raises(ValueError, match="every trigger"):
        validate_rule(rule)


async def test_captured_ai_handoff_and_preview_share_resolved_input(hass) -> None:
    stored = RequestRules(MemoryStore({"rules": [captured_routing_rule()]}))
    await stored.async_initialize()
    text = "deep think why is the sky blue?"
    evaluated = await async_evaluate_rule(
        hass, stored, RequestRuleRuntime(), text, "session"
    )
    preview = await async_request_rule_match_preview(hass, stored, text)
    assert evaluated is not None and not evaluated.consume
    assert evaluated.provider_input == "why is the sky blue?"
    assert preview["matched_rules"][0]["ai_input"] == {
        "mode": "capture",
        "capture": "question",
        "provider_input": evaluated.provider_input,
    }


async def test_captured_ai_input_survives_backup_and_old_rules_default_original() -> None:
    stored = RequestRules(MemoryStore({"rules": [captured_routing_rule()]}))
    await stored.async_initialize()
    backup = await stored.async_backup_data()
    restored = RequestRules.validate_backup_data(backup)
    assert restored["rules"][0]["ai_input_mode"] == "capture"
    assert restored["rules"][0]["ai_input_capture"] == "question"
    old = RequestRules.validate_backup_data({"rules": [local_rule()]})
    assert old["rules"][0]["ai_input_mode"] == "original"
    assert old["rules"][0]["ai_input_capture"] is None


async def test_local_captured_handoff_only_after_success(hass, monkeypatch) -> None:
    from custom_components.extended_openai_conversation_responses import request_rules

    class FakeScript:
        def __init__(self, home_assistant, sequence, *_args, **_kwargs):
            self.hass = home_assistant
            self.sequence = sequence

        async def async_run(self, _variables, _context=None):
            variables = dict(_variables)
            for action in self.sequence:
                if "variables" in action:
                    values = action["variables"]
                    variables.update(
                        values.async_simple_render(variables)
                        if hasattr(values, "async_simple_render")
                        else values
                    )
                    continue
                domain, service = action["action"].split(".", 1)
                await self.hass.services.async_call(domain, service)
            return SimpleNamespace(variables=variables, conversation_response=UNDEFINED)

        async def async_unload(self):
            pass

    async def validate_actions(_hass, actions):
        return actions

    monkeypatch.setattr(request_rules, "Script", FakeScript)
    monkeypatch.setattr(
        request_rules, "async_validate_actions_config", validate_actions
    )
    rule = local_rule("Ask", ["ask {question}"], "sentence_pattern")
    rule["action"]["continue_to_ai"] = True
    rule["ai_input_mode"] = "capture"
    rule["ai_input_capture"] = "question"
    stored = RequestRules(MemoryStore({"rules": [rule]}))
    await stored.async_initialize()
    services = FakeServices()
    hass.services = services
    success = await async_evaluate_rule(
        hass, stored, RequestRuleRuntime(), "ask why", "session"
    )
    assert success is not None and not success.consume
    assert success.provider_input == "why"
    assert len(services.calls) == 1
    hass.services = FakeServices(fail=True)
    failed = await async_evaluate_rule(
        hass, stored, RequestRuleRuntime(), "ask why", "session"
    )
    assert failed is not None and failed.consume and not failed.successful
    assert failed.provider_input is None


async def test_pack_export_import_appends_disabled_in_relative_order() -> None:
    first = local_rule("First", phrases=["first"], order=0)
    second = captured_routing_rule()
    second["order"] = 1
    second["continue_matching"] = True
    source = RequestRules(MemoryStore({"rules": [first, second]}))
    await source.async_initialize()
    pack = export_rule_pack(source, "all")
    assert pack["format"] == "extended_openai_request_rule_pack"
    assert [rule["order"] for rule in pack["rules"]] == [0, 1]
    assert pack["rules"][1]["ai_input_capture"] == "question"
    prepared = validate_rule_pack(pack)
    target = RequestRules(MemoryStore({"rules": [local_rule("Existing")]}))
    await target.async_initialize()
    result = await async_append_rule_pack(
        target, prepared, expected_revision=target.revision()
    )
    assert [rule["name"] for rule in target.snapshot()["rules"]] == [
        "Existing", "First", "Deep think"
    ]
    assert all(not rule["enabled"] for rule in result["rules"])
    assert all(rule["id"] not in {"first", "deep-think"} for rule in result["rules"])
    assert result["rules"][1]["continue_matching"] is True
    assert result["rules"][1]["ai_input_capture"] == "question"


async def test_pack_import_does_not_change_destination_wording_alternatives() -> None:
    source = RequestRules(MemoryStore({
        "wording_groups": [{"canonical": "turn on", "alternatives": ["switch on"]}],
        "rules": [local_rule("Shared", phrases=["turn on lights"])],
    }))
    await source.async_initialize()
    pack = export_rule_pack(source, "all")
    assert "wording_groups" not in pack

    target_wording = [{"canonical": "tv", "alternatives": ["television"]}]
    target = RequestRules(MemoryStore({
        "wording_groups": target_wording,
        "rules": [local_rule("Existing")],
    }))
    await target.async_initialize()
    prepared = validate_rule_pack(pack)
    await async_append_rule_pack(target, prepared, expected_revision=target.revision())
    assert target.snapshot()["wording_groups"] == target_wording


async def test_pack_group_and_selected_exports_keep_subset_order() -> None:
    first = local_rule("First", phrases=["activate lights"], order=0)
    second = local_rule("Second", phrases=["good night"], order=1)
    third = local_rule("Third", phrases=["power up lights"], order=2)
    first["group_id"] = third["group_id"] = "lighting"
    source = RequestRules(MemoryStore({
        "groups": [{"id": "lighting", "name": "Lighting"}],
        "wording_groups": [{"canonical": "activate", "alternatives": ["power up"]}],
        "rules": [first, second, third],
    }))
    await source.async_initialize()
    group = export_rule_pack(source, "group", "lighting")
    assert [rule["name"] for rule in group["rules"]] == ["First", "Third"]
    assert [rule["order"] for rule in group["rules"]] == [0, 1]
    assert [item["name"] for item in group["groups"]] == ["Lighting"]
    assert "wording_groups" not in group
    selected = export_rule_pack(source, "selected", rule_ids=[third["id"], second["id"]])
    assert [rule["name"] for rule in selected["rules"]] == ["Second", "Third"]
    assert selected["groups"][0]["name"] == "Lighting"


async def test_pack_group_id_collision_creates_a_distinct_group() -> None:
    source_rule = local_rule("Imported")
    source_rule["group_id"] = "shared-id"
    source = RequestRules(MemoryStore({
        "groups": [{"id": "shared-id", "name": "Lighting"}],
        "rules": [source_rule],
    }))
    await source.async_initialize()
    pack = validate_rule_pack(export_rule_pack(source, "all"))
    target = RequestRules(MemoryStore({
        "groups": [{"id": "shared-id", "name": "Different"}],
        "rules": [local_rule("Existing")],
    }))
    await target.async_initialize()
    result = await async_append_rule_pack(target, pack, expected_revision=target.revision())
    assert {group["name"] for group in result["groups"]} == {"Different", "Lighting"}
    assert result["rules"][0]["group_id"] != "shared-id"
    assert target.snapshot()["rules"][0]["group_id"] is None


def test_pack_rejects_newer_version_unknown_executable_and_duplicate_ids() -> None:
    rule = validate_rule(local_rule())
    pack = {
        "format": "extended_openai_request_rule_pack",
        "version": 2,
        "groups": [],
        "rules": [rule],
    }
    with pytest.raises(ValueError, match="version"):
        validate_rule_pack(pack)
    pack["version"] = 1
    pack["rules"][0]["unexpected_action"] = True
    with pytest.raises(ValueError, match="unknown rule fields"):
        validate_rule_pack(pack)
    del pack["rules"][0]["unexpected_action"]
    pack["rules"] = [rule, deepcopy(rule)]
    pack["rules"][1]["order"] = 1
    with pytest.raises(ValueError, match="duplicate rule IDs"):
        validate_rule_pack(pack)


def test_import_requires_one_review_of_the_exact_pack_and_revision() -> None:
    manager = SimpleNamespace()
    pack = {"rules": [{"name": "First"}]}
    token = _register_rule_pack_review(manager, pack, "revision-1")
    with pytest.raises(HomeAssistantError, match="Review this exact"):
        _consume_rule_pack_review(manager, token, {"rules": [{"name": "Changed"}]}, "revision-1")
    token = _register_rule_pack_review(manager, pack, "revision-1")
    _consume_rule_pack_review(manager, token, pack, "revision-1")
    with pytest.raises(HomeAssistantError, match="Review this exact"):
        _consume_rule_pack_review(manager, token, pack, "revision-1")


@pytest.mark.asyncio
async def test_pack_validation_rejects_structural_and_reference_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RequestRules(MemoryStore({"rules": [local_rule("Portable")]}))
    await source.async_initialize()
    valid = export_rule_pack(source, "all")

    malformed = deepcopy(valid)
    malformed.pop("groups")
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_rule_pack(malformed)

    malformed = deepcopy(valid)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_rule_pack(malformed)

    malformed = deepcopy(valid)
    malformed["rules"] = []
    with pytest.raises(ValueError, match="1 to 500 rules"):
        validate_rule_pack(malformed)

    malformed = deepcopy(valid)
    malformed["rules"] = "not-a-list"
    with pytest.raises(ValueError, match="1 to 500 rules"):
        validate_rule_pack(malformed)

    malformed = deepcopy(valid)
    malformed["rules"][0]["order"] = 3
    with pytest.raises(ValueError, match="priorities must be contiguous"):
        validate_rule_pack(malformed)

    malformed = deepcopy(valid)
    malformed["rules"][0]["group_id"] = "missing-group"
    with pytest.raises(ValueError, match="unknown group"):
        validate_rule_pack(malformed)

    malformed = deepcopy(valid)
    malformed["rules"][0]["matching_behavior"] = "defaults"
    with pytest.raises(ValueError, match="effective matching settings"):
        validate_rule_pack(malformed)

    malformed = deepcopy(valid)
    malformed["rules"][0]["action"].pop("continue_to_ai", None)
    with pytest.raises(ValueError, match="Continue to AI"):
        validate_rule_pack(malformed)

    with pytest.raises(ValueError, match="must be an object"):
        validate_rule_pack([])

    with pytest.raises(ValueError, match="not valid JSON"):
        validate_rule_pack("{")

    monkeypatch.setattr(request_rule_packs, "MAX_PACK_BYTES", 32)
    with pytest.raises(ValueError, match="2 MB safety limit"):
        validate_rule_pack('{"payload":"' + ("x" * 64) + '"}')


@pytest.mark.asyncio
async def test_pack_export_rejects_empty_unknown_and_duplicate_selections() -> None:
    source = RequestRules(MemoryStore({"rules": [local_rule("Only")]}))
    await source.async_initialize()

    with pytest.raises(ValueError, match="Choose at least one"):
        export_rule_pack(source, "selected", rule_ids=[])
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        export_rule_pack(source, "selected", rule_ids=["only", "only"])
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        export_rule_pack(source, "selected", rule_ids=["missing"])
    with pytest.raises(ValueError, match="Unknown group"):
        export_rule_pack(source, "group", group_id="missing")
    with pytest.raises(ValueError, match="Choose All rules"):
        export_rule_pack(source, "unexpected")


@pytest.mark.asyncio
async def test_pack_append_rolls_back_live_state_when_durable_save_fails() -> None:
    class FailingSaveStore(MemoryStore):
        fail = False

        async def async_save(self, data):
            if self.fail:
                raise OSError("simulated durable write failure")
            await super().async_save(data)

    source = RequestRules(MemoryStore({"rules": [local_rule("Imported")]}))
    await source.async_initialize()
    prepared = validate_rule_pack(export_rule_pack(source, "all"))

    store = FailingSaveStore({"rules": [local_rule("Existing")]})
    target = RequestRules(store)
    await target.async_initialize()
    before = target.snapshot()
    revision = target.revision()
    durable_before = deepcopy(store.data)
    store.fail = True

    with pytest.raises(OSError, match="durable write failure"):
        await async_append_rule_pack(
            target,
            prepared,
            expected_revision=revision,
        )

    assert target.snapshot() == before
    assert target.revision() == revision
    assert store.data == durable_before
