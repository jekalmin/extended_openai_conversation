"""Strict HA Store failure propagation and sanitization."""

from __future__ import annotations

import errno

import pytest

from homeassistant.helpers.storage import Store
from homeassistant.util.file import WriteError

from custom_components.extended_openai_conversation_responses.strict_store import (
    PropagatingWriteStore,
)


@pytest.mark.asyncio
async def test_strict_store_surfaces_errno_without_private_path_or_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_write(_self, _data):
        try:
            raise OSError(errno.ENOSPC, "disk full at /config/.storage/private")
        except OSError as cause:
            raise WriteError("writer leaked /config/.storage/private") from cause

    monkeypatch.setattr(Store, "_async_write_data", fail_write)
    store = object.__new__(PropagatingWriteStore)
    private = {"secret": "PRIVATE-CONTENT-MARKER"}

    with pytest.raises(OSError) as raised:
        await store._async_write_data(private)

    assert raised.value.errno == errno.ENOSPC
    rendered = str(raised.value)
    assert "Private storage write failed" in rendered
    assert "/config/.storage/private" not in rendered
    assert "PRIVATE-CONTENT-MARKER" not in rendered
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_strict_store_handles_writer_error_without_oserror_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_write(_self, _data):
        raise WriteError("writer failed")

    monkeypatch.setattr(Store, "_async_write_data", fail_write)
    store = object.__new__(PropagatingWriteStore)

    with pytest.raises(OSError) as raised:
        await store._async_write_data({})

    assert raised.value.errno is None
    assert "Private storage write failed" in str(raised.value)
