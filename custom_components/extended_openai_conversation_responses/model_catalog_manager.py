"""Fail-safe, installation-wide catalogue updates in Home Assistant storage."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import logging
import time
from typing import Any

from aiohttp import ClientError
import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.storage import Store

from .const import CONF_CHAT_MODEL, CONF_REASONING_EFFORT, DEFAULT_CHAT_MODEL, DOMAIN
from .model_capabilities import frontend_capabilities
from .model_catalog import (
    BUNDLED_CATALOG,
    MAX_CATALOG_BYTES,
    IncompatibleCatalogError,
    activate_catalog,
    all_reasoning_efforts,
    catalog_model_metadata,
    catalog_picker_models,
    catalog_reasoning_efforts,
    model_metadata,
    parse_catalog,
    validate_catalog,
    validate_catalog_transition,
    validate_or_migrate_catalog,
)
from .request_rules import SLOT_REFERENCE, async_get_request_rules
from .strict_store import PropagatingWriteStore

CATALOG_URL = (
    "https://raw.githubusercontent.com/conorod1992/extended_openai_conversation/"
    "develop/custom_components/extended_openai_conversation_responses/model_catalog.json"
)
UPDATE_INTERVAL = 24 * 60 * 60
DATA_MANAGER = f"{DOMAIN}.model_catalog"
WS_CATALOG = f"{DOMAIN}/model_catalog"
_LOGGER = logging.getLogger(__name__)


class _TransientCatalogUpdateError(Exception):
    """Remote catalogue refresh failed for an ordinary transient reason."""


class ModelCatalogManager:
    """Serialize catalogue checks/apply/reset and retain last good data."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self.store: Store[dict[str, Any]] = PropagatingWriteStore(
            hass, 1, f"{DOMAIN}.model_catalog"
        )
        self.catalog: dict[str, Any] | None = None
        self.available_catalog: dict[str, Any] | None = None
        self.etag: str | None = None
        self.last_checked = 0.0
        self.last_error: str | None = None
        self.incompatible_catalog: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    async def async_load(self) -> None:
        """Load/migrate stored data without making startup depend on the network."""
        try:
            saved = await self.store.async_load()
            migrated = False
            if saved:
                candidate = saved.get("catalog")
                if candidate is not None:
                    candidate, migrated = validate_or_migrate_catalog(candidate)
                available = saved.get("available_catalog")
                if available is not None:
                    available = validate_catalog(available)
                checked = saved.get("last_checked", 0)
                etag = saved.get("etag")
                incompatible = saved.get("incompatible_catalog")
                if incompatible is not None and (
                    not isinstance(incompatible, dict)
                    or type(incompatible.get("schema_version")) is not int
                    or type(incompatible.get("catalog_version")) is not int
                    or (
                        incompatible.get("minimum_eoai_version") is not None
                        and not isinstance(incompatible["minimum_eoai_version"], str)
                    )
                ):
                    raise ValueError("Invalid incompatible catalogue status")
                if type(checked) not in (float, int) or not 0 <= checked <= time.time():
                    raise ValueError("Invalid catalogue check time")
                if etag is not None and (
                    not isinstance(etag, str)
                    or len(etag) > 256
                    or "\n" in etag
                    or "\r" in etag
                ):
                    raise ValueError("Invalid catalogue ETag")
                stale_candidate_discarded = False
                if (
                    candidate is not None
                    and candidate["catalog_version"]
                    < BUNDLED_CATALOG["catalog_version"]
                ):
                    candidate = None
                    stale_candidate_discarded = True
                elif candidate is not None:
                    validate_catalog_transition(None, candidate)

                active = candidate or BUNDLED_CATALOG
                if available is not None:
                    if available["catalog_version"] <= active["catalog_version"]:
                        available = None
                    else:
                        validate_catalog_transition(candidate, available)
                if stale_candidate_discarded and available is None:
                    etag = None

                self.catalog = candidate
                self.available_catalog = available
                self.etag = etag
                self.last_checked = checked
                self.incompatible_catalog = incompatible
                if incompatible is not None:
                    # The response ETag belongs to rejected data, never to active data.
                    self.etag = None
                if migrated:
                    _LOGGER.debug(
                        "Migrated stored model capability catalogue to schema v4; "
                        "bundled model data is authoritative"
                    )
                    try:
                        await self._save(
                            candidate, available, etag, checked, incompatible
                        )
                    except Exception:
                        _LOGGER.warning("Unable to persist migrated model catalogue")
        except Exception:
            self.catalog = None
            self.available_catalog = None
            self.etag = None
            self.last_checked = 0.0
            self.incompatible_catalog = None
            self.last_error = (
                "Stored model data could not be loaded; using bundled data."
            )
            _LOGGER.warning(self.last_error)
        activate_catalog(self.catalog)

    def status(self) -> dict[str, Any]:
        """Return active and remotely available catalogue state."""
        active = self.catalog or BUNDLED_CATALOG
        return {
            "source": "downloaded" if self.catalog is not None else "bundled",
            "catalog_version": active["catalog_version"],
            "schema_version": active["schema_version"],
            "update_available": self.available_catalog is not None,
            "available_catalog_version": (
                self.available_catalog["catalog_version"]
                if self.available_catalog is not None
                else None
            ),
            "last_checked": self.last_checked,
            "last_error": self.last_error,
            "incompatible_catalog": self.incompatible_catalog,
        }

    async def _save(
        self,
        catalog: dict[str, Any] | None,
        available_catalog: dict[str, Any] | None,
        etag: str | None,
        checked: float,
        incompatible_catalog: dict[str, Any] | None = None,
    ) -> None:
        await self.store.async_save(
            {
                "catalog": catalog,
                "available_catalog": available_catalog,
                "etag": etag,
                "last_checked": checked,
                "incompatible_catalog": incompatible_catalog,
            }
        )

    async def _record_failed_check(self, checked: float, *, transient: bool) -> None:
        """Retain active/pending data while remembering when a check failed."""
        self.last_checked = checked
        self.last_error = "Model data check failed; the current catalogue was kept."
        if transient:
            _LOGGER.debug(self.last_error)
        else:
            _LOGGER.warning(self.last_error)
        try:
            await self._save(
                self.catalog,
                self.available_catalog,
                self.etag,
                checked,
                self.incompatible_catalog,
            )
        except Exception:
            _LOGGER.warning("Unable to persist model catalogue check time")

    async def _candidate_preserves_saved_requests(
        self, candidate: dict[str, Any]
    ) -> bool:
        """Check concrete saved agent and routing choices before publication."""
        from .request import build_provider_request_snapshot

        for entry in self.hass.config_entries.async_entries(DOMAIN):
            for subentry in entry.subentries.values():
                if subentry.subentry_type != "conversation":
                    continue
                options = subentry.data
                model = str(options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL))
                try:
                    build_provider_request_snapshot(
                        options,
                        getattr(entry, "data", {}),
                        model_capabilities=catalog_model_metadata(candidate, model),
                    )
                except Exception:
                    return False
                rules = await async_get_request_rules(
                    self.hass, entry.entry_id, subentry.subentry_id
                )
                for rule in rules.snapshot()["rules"]:
                    if rule.get("action_type") != "model_routing":
                        continue
                    action = rule.get("action", {})
                    if action.get("reset"):
                        continue
                    routed_model = action.get("model")
                    routed_effort = action.get("reasoning_effort")
                    if routed_model and SLOT_REFERENCE.search(str(routed_model)):
                        continue
                    if routed_effort and SLOT_REFERENCE.search(str(routed_effort)):
                        continue
                    if not routed_model and not routed_effort:
                        continue
                    effective = dict(options)
                    if routed_model:
                        effective[CONF_CHAT_MODEL] = routed_model
                    if routed_effort:
                        effective[CONF_REASONING_EFFORT] = routed_effort
                    try:
                        build_provider_request_snapshot(
                            effective,
                            getattr(entry, "data", {}),
                            model_capabilities=catalog_model_metadata(
                                candidate,
                                str(effective[CONF_CHAT_MODEL]),
                            ),
                        )
                    except Exception:
                        return False
        return True

    async def async_check(self, *, force: bool = False) -> dict[str, Any]:
        """Check trusted catalogue data without changing the active catalogue."""
        async with self._lock:
            now = time.time()
            if not force and now - self.last_checked < UPDATE_INTERVAL:
                return self.status()
            try:
                headers = {"If-None-Match": self.etag} if self.etag else {}
                async with asyncio.timeout(15):
                    async with async_get_clientsession(self.hass).get(
                        CATALOG_URL,
                        headers=headers,
                        allow_redirects=False,
                    ) as response:
                        if response.status == 304:
                            if not self.etag:
                                raise ValueError("Unsolicited not-modified response")
                            await self._save(
                                self.catalog,
                                self.available_catalog,
                                self.etag,
                                now,
                                self.incompatible_catalog,
                            )
                            self.last_checked = now
                            self.last_error = None
                            return self.status()
                        if response.status == 429 or response.status >= 500:
                            raise _TransientCatalogUpdateError(
                                "Catalogue service temporarily unavailable"
                            )
                        if response.status != 200:
                            raise ValueError("Catalogue HTTP check failed")
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(16384):
                            raw.extend(chunk)
                            if len(raw) > MAX_CATALOG_BYTES:
                                raise ValueError("Catalogue download too large")
                        candidate = parse_catalog(bytes(raw))
                        etag = response.headers.get("ETag")
                        if etag and (len(etag) > 256 or "\n" in etag or "\r" in etag):
                            raise ValueError("Invalid catalogue ETag")

                active = self.catalog or BUNDLED_CATALOG
                current_version = active["catalog_version"]
                if candidate["catalog_version"] < current_version:
                    raise ValueError("Catalogue version is older than current data")
                if candidate["catalog_version"] == current_version:
                    if candidate != active:
                        raise ValueError(
                            "Changed catalogue must increment catalog_version"
                        )
                    available = None
                else:
                    validate_catalog_transition(self.catalog, candidate)
                    if not await self._candidate_preserves_saved_requests(candidate):
                        raise ValueError("Catalogue update invalidates saved requests")
                    available = candidate

                await self._save(self.catalog, available, etag, now)
                self.available_catalog = available
                self.etag = etag
                self.last_checked = now
                self.last_error = None
                self.incompatible_catalog = None
            except IncompatibleCatalogError as exc:
                incompatible = {
                    "schema_version": exc.schema_version,
                    "catalog_version": exc.catalog_version,
                    "minimum_eoai_version": exc.minimum_eoai_version,
                }
                try:
                    await self._save(
                        self.catalog, self.available_catalog, None, now, incompatible
                    )
                except Exception:
                    _LOGGER.warning("Unable to persist incompatible catalogue status")
                self.etag = None
                self.last_checked = now
                self.last_error = None
                self.incompatible_catalog = incompatible
            except ClientError, TimeoutError, _TransientCatalogUpdateError:
                await self._record_failed_check(now, transient=True)
            except Exception:
                await self._record_failed_check(now, transient=False)
            return self.status()

    async def async_update(self, *, force: bool = False) -> dict[str, Any]:
        """Compatibility alias for the former update action; now check-only."""
        return await self.async_check(force=force)

    async def async_apply_update(self) -> dict[str, Any]:
        """Activate the most recently checked newer catalogue."""
        async with self._lock:
            candidate = self.available_catalog
            if candidate is None:
                self.last_error = "No model data update is available."
                return self.status()
            try:
                validate_catalog_transition(self.catalog, candidate)
                if not await self._candidate_preserves_saved_requests(candidate):
                    raise ValueError("Catalogue update invalidates saved requests")
                await self._save(
                    candidate,
                    None,
                    self.etag,
                    self.last_checked,
                    self.incompatible_catalog,
                )
            except Exception:
                self.last_error = "Model data update could not be applied; the current catalogue was kept."
                _LOGGER.warning(self.last_error)
                return self.status()
            activate_catalog(candidate)
            self.catalog = candidate
            self.available_catalog = None
            self.last_error = None
            return self.status()

    async def _bundled_reset_would_invalidate_saved_reasoning(self) -> bool:
        """Check durable agent/rule choices before narrowing to bundled data."""
        if self.catalog is None:
            return False

        bundled_efforts = set(catalog_reasoning_efforts(None))
        for entry in self.hass.config_entries.async_entries(DOMAIN):
            for subentry in entry.subentries.values():
                if subentry.subentry_type != "conversation":
                    continue

                configured_model = str(
                    subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
                )
                configured_effort = subentry.data.get(CONF_REASONING_EFFORT)
                metadata = catalog_model_metadata(None, configured_model)
                bundled_config_efforts = metadata["reasoning"]["efforts"]
                if (
                    isinstance(configured_effort, str)
                    and configured_effort
                    and (
                        not metadata["reasoning"]["supported"]
                        or configured_effort not in bundled_config_efforts
                    )
                ):
                    return True

                rules = await async_get_request_rules(
                    self.hass, entry.entry_id, subentry.subentry_id
                )
                for rule in rules.snapshot()["rules"]:
                    if rule.get("action_type") != "model_routing":
                        continue
                    action = rule.get("action", {})
                    if action.get("reset"):
                        continue
                    effort = action.get("reasoning_effort")
                    if (
                        not isinstance(effort, str)
                        or not effort
                        or SLOT_REFERENCE.search(effort)
                    ):
                        continue
                    model = action.get("model")
                    if (
                        isinstance(model, str)
                        and model
                        and not SLOT_REFERENCE.search(model)
                    ):
                        metadata = catalog_model_metadata(None, model)
                        if (
                            not metadata["reasoning"]["supported"]
                            or effort not in metadata["reasoning"]["efforts"]
                        ):
                            return True
                    elif effort not in bundled_efforts:
                        return True
        return False

    async def async_reset(self) -> dict[str, Any]:
        """Return to bundled data until a checked update is explicitly applied."""
        async with self._lock:
            if await self._bundled_reset_would_invalidate_saved_reasoning():
                self.last_error = (
                    "Model data reset was blocked because saved configuration or "
                    "Request Rules use reasoning choices unavailable in bundled data."
                )
                return self.status()

            available = self.available_catalog
            if (
                available is None
                and self.catalog is not None
                and self.catalog["catalog_version"] > BUNDLED_CATALOG["catalog_version"]
            ):
                available = self.catalog
            if (
                available is not None
                and available["catalog_version"] <= BUNDLED_CATALOG["catalog_version"]
            ):
                available = None
            etag = self.etag if available is not None else None
            try:
                await self._save(
                    None, available, etag, self.last_checked, self.incompatible_catalog
                )
            except Exception:
                self.last_error = (
                    "Model data reset failed; the current catalogue was kept."
                )
                return self.status()
            activate_catalog(None)
            self.catalog = None
            self.available_catalog = available
            self.etag = etag
            self.last_error = None
            return self.status()


@websocket_api.websocket_command(
    {
        vol.Required("type"): WS_CATALOG,
        vol.Optional("action", default="lookup"): vol.In(
            ("lookup", "check", "update", "apply", "reset")
        ),
        vol.Optional("model", default=""): vol.All(str, vol.Length(max=128)),
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_catalog(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict[str, Any]
) -> None:
    """Expose the same capability data used by request validation."""
    manager: ModelCatalogManager = hass.data[DATA_MANAGER]
    if msg["action"] in {"check", "update"}:
        status = await manager.async_check(force=True)
        if status["last_error"]:
            connection.send_error(
                msg["id"], "model_catalog_check_failed", status["last_error"]
            )
            return
    elif msg["action"] == "apply":
        status = await manager.async_apply_update()
        if status["last_error"]:
            connection.send_error(
                msg["id"], "model_catalog_apply_failed", status["last_error"]
            )
            return
    elif msg["action"] == "reset":
        status = await manager.async_reset()
        if status["last_error"]:
            connection.send_error(
                msg["id"], "model_catalog_reset_failed", status["last_error"]
            )
            return

    metadata = model_metadata(msg["model"])
    capabilities = frontend_capabilities(msg["model"], metadata=metadata)
    connection.send_result(
        msg["id"],
        {
            **manager.status(),
            "model_capabilities": capabilities,
            "model_metadata": metadata,
            "catalog_models": catalog_picker_models(manager.catalog, msg["model"]),
            "reasoning_effort_options": metadata["reasoning"]["efforts"]
            if msg["model"]
            else all_reasoning_efforts(),
        },
    )


async def async_setup_model_catalog(hass: HomeAssistant) -> None:
    """One manager and timer per HA installation, shared by all agent entries."""
    if DATA_MANAGER in hass.data:
        return
    manager = ModelCatalogManager(hass)
    hass.data[DATA_MANAGER] = manager
    await manager.async_load()
    websocket_api.async_register_command(hass, websocket_catalog)

    async def check(_now: Any) -> None:
        await manager.async_check()

    cancel = async_track_time_interval(hass, check, timedelta(hours=1))

    @callback
    def stop(_event: Any) -> None:
        cancel()
        activate_catalog(None)

    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, stop)
