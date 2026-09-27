"""Cold-process recovery of a multi-Store restore transaction."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MappingProxyType
from unittest.mock import patch

import pytest

from tests_real_ha.process_harness import child_process_env, run_python_child
from tests_real_ha.test_immediate_tool_process_crash import (
    DOMAIN,
    _assert_source_component,
    _ensure_entry,
    _stage_component,
)

_PHASE = "RESTORE_CRASH_PHASE"
_CUT = "RESTORE_CRASH_CUT"
_CONFIG = "RESTORE_CRASH_CONFIG"
_OWNER = "restore-process-owner"
_TARGET = "restore-target"
_PREVIOUS = "restore-previous"
_SIBLING = "unrelated-agent"


async def _state(hass, entry, *, title: str | None = None):
    from custom_components.extended_openai_conversation_responses.knowledge import (
        async_get_knowledge,
    )
    from custom_components.extended_openai_conversation_responses.memory import (
        async_get_memory,
    )

    subentry = next(
        s
        for s in entry.subentries.values()
        if s.subentry_type == "conversation" and (title is None or s.title == title)
    )
    memory = await async_get_memory(hass, entry.entry_id, subentry.subentry_id)
    knowledge = await async_get_knowledge(hass, entry.entry_id, subentry.subentry_id)
    return subentry, memory, knowledge


async def _seed(hass, entry, config_dir: Path) -> None:
    from custom_components.extended_openai_conversation_responses import backup
    from homeassistant.config_entries import ConfigSubentry

    subentry, memory, knowledge = await _state(hass, entry)
    sibling = ConfigSubentry(
        data=MappingProxyType(dict(subentry.data)),
        subentry_type="conversation",
        title=_SIBLING,
        unique_id=None,
    )
    assert hass.config_entries.async_add_subentry(entry, sibling)
    await hass.async_block_till_done()
    _, sibling_memory, sibling_knowledge = await _state(hass, entry, title=_SIBLING)
    await sibling_memory.async_add(_OWNER, _SIBLING, "test", "explicit")
    await sibling_knowledge.async_create(_SIBLING, "", _SIBLING)
    await memory.async_add(_OWNER, _TARGET, "test", "explicit")
    await knowledge.async_create(_TARGET, "", _TARGET)
    target = await backup.async_collect_backup_snapshot(hass, entry, subentry)
    (config_dir / "target-backup.json").write_text(json.dumps(target), encoding="utf-8")
    for record in await memory.async_list(_OWNER):
        await memory.async_delete(_OWNER, [record.memory_id])
    for source in await knowledge.async_list():
        await knowledge.async_delete(source["source_id"])
    await memory.async_add(_OWNER, _PREVIOUS, "test", "explicit")
    await knowledge.async_create(_PREVIOUS, "", _PREVIOUS)
    assert await _observed(hass, entry) == (_PREVIOUS, _PREVIOUS)
    assert await _observed(hass, entry, title=_SIBLING) == (_SIBLING, _SIBLING)


async def _observed(hass, entry, *, title: str | None = None) -> tuple[str, str]:
    _, memory, knowledge = await _state(hass, entry, title=title)
    records = await memory.async_list(_OWNER)
    sources = await knowledge.async_list()
    assert len(records) == len(sources) == 1
    source = await knowledge.async_get(sources[0]["source_id"])
    return records[0].content, source.content


def _pause(config_dir: Path, name: str) -> None:
    (config_dir / name).write_text("reached\n", encoding="utf-8")


async def _child(config_dir: Path, phase: str, cut: str) -> None:
    from custom_components.extended_openai_conversation_responses import (
        restore_recovery,
    )
    from custom_components.extended_openai_conversation_responses.memory import (
        PersistentMemory,
    )
    from homeassistant import bootstrap, runner

    original_replace = PersistentMemory.async_replace_backup

    async def pause_recovery(self, records):
        await original_replace(self, records)
        _pause(config_dir, "recovery-paused")
        await asyncio.Event().wait()

    recovery_patch = (
        patch.object(PersistentMemory, "async_replace_backup", pause_recovery)
        if phase == "interrupt-recovery"
        else None
    )
    if recovery_patch:
        recovery_patch.start()
    try:
        hass = await bootstrap.async_setup_hass(
            runner.RuntimeConfig(config_dir=str(config_dir), skip_pip=True)
        )
        assert hass is not None
        await hass.async_start()
        entry = await _ensure_entry(hass)
        if phase == "seed":
            await _seed(hass, entry, config_dir)
        elif phase == "verify":
            expected = _TARGET if cut == "committed" else _PREVIOUS
            assert await _observed(hass, entry) == (expected, expected)
            assert await _observed(hass, entry, title=_SIBLING) == (
                _SIBLING,
                _SIBLING,
            )
            subentry, _, _ = await _state(hass, entry)
            journal = restore_recovery._journal_store(
                hass, entry.entry_id, subentry.subentry_id
            )
            assert await journal.async_load() is None
        elif phase == "restore":
            subentry, memory, _ = await _state(hass, entry)
            target = json.loads(
                (config_dir / "target-backup.json").read_text(encoding="utf-8")
            )
            original_write = restore_recovery._async_write_journal_verified
            original_memory = memory.async_replace_backup

            async def pause_journal(store, journal):
                saved = await original_write(store, journal)
                if saved and journal["phase"] == cut:
                    _pause(config_dir, "restore-paused")
                    await asyncio.Event().wait()
                return saved

            async def pause_partial(records):
                await original_memory(records)
                _pause(config_dir, "restore-paused")
                await asyncio.Event().wait()

            with patch.object(
                restore_recovery, "_async_write_journal_verified", pause_journal
            ):
                if cut == "partial":
                    with patch.object(memory, "async_replace_backup", pause_partial):
                        await restore_recovery.async_restore_backup_recoverably(
                            hass, entry, subentry, target
                        )
                else:
                    await restore_recovery.async_restore_backup_recoverably(
                        hass, entry, subentry, target
                    )
            raise AssertionError("restore returned before process kill")
        else:
            raise AssertionError(f"unexpected child phase {phase}")
        await hass.async_stop()
    finally:
        if recovery_patch:
            recovery_patch.stop()


def _env(config_dir: Path, phase: str, cut: str) -> dict[str, str]:
    return child_process_env(
        __file__, {_PHASE: phase, _CUT: cut, _CONFIG: str(config_dir)}
    )


def _run(config_dir: Path, phase: str, cut: str) -> None:
    result = run_python_child(
        __file__,
        cwd=config_dir,
        extra_env={_PHASE: phase, _CUT: cut, _CONFIG: str(config_dir)},
        timeout=75,
    )
    assert result.returncode == 0, f"{phase} failed:\n{result.stdout}\n{result.stderr}"


def _kill_at(config_dir: Path, phase: str, cut: str, marker: str) -> None:
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=config_dir,
        env=_env(config_dir, phase, cut),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(450):
            if (config_dir / marker).exists():
                break
            if process.poll() is not None:
                output, _ = process.communicate()
                raise AssertionError(f"{phase}/{cut} exited before {marker}:\n{output}")
            time.sleep(0.1)
        else:
            raise AssertionError(f"{phase}/{cut} did not reach {marker}")
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        (config_dir / marker).unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("cut", "interrupt_recovery"),
    [("applying", False), ("partial", False), ("committed", False), ("partial", True)],
)
def test_restore_transaction_converges_after_process_kills(
    tmp_path: Path, cut: str, interrupt_recovery: bool
) -> None:
    source = Path(__file__).resolve().parents[1] / "custom_components" / DOMAIN
    config_dir = tmp_path / "ha-config"
    destination = config_dir / "custom_components" / DOMAIN
    destination.parent.mkdir(parents=True)
    _stage_component(source, destination)
    config = config_dir / "configuration.yaml"
    config.write_text(
        "homeassistant:\n  name: Restore Process Boundary\n", encoding="utf-8"
    )
    _run(config_dir, "seed", cut)
    _kill_at(config_dir, "restore", cut, "restore-paused")
    if interrupt_recovery:
        _kill_at(config_dir, "interrupt-recovery", cut, "recovery-paused")
    _run(config_dir, "verify", cut)
    assert config.read_text(encoding="utf-8") == (
        "homeassistant:\n  name: Restore Process Boundary\n"
    )


if __name__ == "__main__" and os.environ.get(_PHASE):
    directory = Path(os.environ[_CONFIG]).resolve()
    sys.path.insert(0, str(directory))
    _assert_source_component(directory)
    asyncio.run(_child(directory, os.environ[_PHASE], os.environ[_CUT]))
