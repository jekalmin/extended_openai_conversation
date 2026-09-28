"""Residual private Store permission repair for historical files."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import os
import stat
from typing import Any

from homeassistant.helpers.storage import Store

_PRIVATE_STORE_MODE = 0o600


def _repair_private_store_mode(path: str) -> None:
    """Tighten one existing Store file without rewriting unchanged JSON."""
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        return
    if mode != _PRIVATE_STORE_MODE:
        os.chmod(path, _PRIVATE_STORE_MODE)


async def _async_repair_private_store_mode(store: Any) -> None:
    """Tighten permissions on a historical Home Assistant Store before loading."""
    if not isinstance(store, Store):
        return
    await store.hass.async_add_executor_job(_repair_private_store_mode, store.path)


async def _async_settle_transactional_save(
    save: Awaitable[None],
    restore_committed_state: Callable[[], None],
    remember_committed_state: Callable[[], None],
    reconcile_failed_state: Callable[[], Awaitable[None]] | None,
    invalidate_state: Callable[[], None],
) -> None:
    """Settle a write and reconcile actual disk state before reporting failure."""
    save_task = asyncio.ensure_future(save)
    cancellation: asyncio.CancelledError | None = None

    # Once live manager state has changed, caller cancellation must not abandon an
    # in-flight Store write. Keep observing it to a known result, then preserve the
    # original persistence and cancellation precedence.
    while not save_task.done():
        try:
            await asyncio.shield(save_task)
        except asyncio.CancelledError as err:
            if save_task.cancelled():
                break
            if cancellation is None:
                cancellation = err
        except Exception:
            # Inspect the finished task below so rollback and cancellation
            # precedence stay in one place.
            break

    try:
        save_task.result()
    except BaseException as err:
        if reconcile_failed_state is None:
            # Initialization repairs do not yet have a usable validator for
            # reloading. Preserve their established rollback/retry behavior.
            restore_committed_state()
        else:
            try:
                await reconcile_failed_state()
            except BaseException:
                # A failed read cannot establish which generation reached disk.
                invalidate_state()
        if cancellation is not None:
            raise cancellation from err
        raise

    remember_committed_state()
    if cancellation is not None:
        raise cancellation
