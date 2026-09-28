"""Home Assistant Store atomic-write acknowledgement boundary regressions."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any
from uuid import uuid4

import atomicwrites
import pytest

from custom_components.extended_openai_conversation_responses.request_rules import (
    DEFAULT_MATCHING,
    STORAGE_VERSION,
    RequestRules,
    RequestRuleStore,
)


def _rule(rule_id: str, name: str, phrase: str, order: int) -> dict[str, Any]:
    return {
        "id": rule_id,
        "name": name,
        "enabled": True,
        "phrases": [phrase],
        "match_type": "equals",
        "action_type": "local_action",
        "action": {
            "actions": [
                {
                    "domain": "script",
                    "service": "turn_on",
                    "target": {"entity_id": ["script.persistence_test"]},
                    "data": {},
                }
            ],
            "success_response": f"{name} matched",
            "failure_response": "No match",
        },
        "matching_behavior": "defaults",
        "matching": dict(DEFAULT_MATCHING),
        "order": order,
    }


def _disk_rules(manager: RequestRules) -> list[dict[str, Any]]:
    envelope = json.loads(Path(manager._store.path).read_text(encoding="utf-8"))
    return envelope["data"]["rules"]


@pytest.mark.parametrize(
    "boundary",
    ["temp_open", "file_fsync", "before_rename", "directory_fsync"],
)
async def test_ha_store_atomic_failure_reconciles_request_rules(
    hass,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    """Use HA Store and its real atomic writer against the temporary filesystem."""
    key = f"eoai.request_rules.{uuid4().hex}"
    store = RequestRuleStore(hass, STORAGE_VERSION, key)
    manager = RequestRules(store)
    await manager.async_initialize()
    rule_a = _rule("rule-a", "State A", "state a", 0)
    rule_b = _rule("rule-b", "State B", "state b", 1)
    await manager.async_create(rule_a)
    assert Path(store.path).is_relative_to(tmp_path)
    assert {rule["id"] for rule in _disk_rules(manager)} == {"rule-a"}

    original_replace = atomicwrites._replace_atomic
    original_fsync = atomicwrites._proper_fsync
    if boundary == "temp_open":
        def fail_temp_open(self: Any, *args: Any, **kwargs: Any) -> Any:
            raise OSError(errno.EIO, "injected temporary file open failure")

        monkeypatch.setattr(atomicwrites.AtomicWriter, "get_fileobject", fail_temp_open)
    elif boundary == "before_rename":
        def fail_before_rename(source: str, target: str) -> None:
            raise OSError(errno.EIO, "injected replace failure")

        monkeypatch.setattr(atomicwrites, "_replace_atomic", fail_before_rename)
    elif boundary == "directory_fsync" and sys.platform == "win32":
        def fail_after_replace(source: str, target: str) -> None:
            original_replace(source, target)
            raise OSError(errno.EIO, "injected post-replace directory sync failure")

        monkeypatch.setattr(atomicwrites, "_replace_atomic", fail_after_replace)
    else:
        def fail_fsync(fd: int) -> None:
            if boundary == "directory_fsync" and stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "injected directory fsync failure")
            if boundary == "file_fsync" and not stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "injected temporary file fsync failure")
            original_fsync(fd)

        monkeypatch.setattr(atomicwrites, "_proper_fsync", fail_fsync)

    with pytest.raises(OSError):
        await manager.async_create(rule_b)

    monkeypatch.setattr(atomicwrites, "_replace_atomic", original_replace)
    monkeypatch.setattr(atomicwrites, "_proper_fsync", original_fsync)
    committed = boundary == "directory_fsync"
    expected = {"rule-a", "rule-b"} if committed else {"rule-a"}
    assert {rule["id"] for rule in _disk_rules(manager)} == expected
    assert {rule["id"] for rule in manager.snapshot()["rules"]} == expected
    assert manager.match("state a").rule["id"] == "rule-a"
    match_b = manager.match("state b")
    assert (match_b.rule["id"] if match_b else None) == (
        "rule-b" if committed else None
    )

    # A new HA Store and manager must validate the same on-disk generation.
    reloaded = RequestRules(RequestRuleStore(hass, STORAGE_VERSION, key))
    await reloaded.async_initialize()
    assert {rule["id"] for rule in reloaded.snapshot()["rules"]} == expected
    assert {rule["id"] for rule in _disk_rules(reloaded)} == expected
    match_b = reloaded.match("state b")
    assert (match_b.rule["id"] if match_b else None) == (
        "rule-b" if committed else None
    )
