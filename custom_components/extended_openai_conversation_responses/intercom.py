"""Targeted Assist Satellite announcements with non-interrupting delivery."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import re
from typing import Any
from uuid import uuid4

from homeassistant.const import ATTR_FRIENDLY_NAME, ATTR_SUPPORTED_FEATURES
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
    label_registry as lr,
)
from homeassistant.helpers.event import async_call_later, async_track_state_change_event
from homeassistant.helpers.storage import Store

from .strict_store import PropagatingWriteStore

DOMAIN = "extended_openai_conversation_responses"
DATA_KEY = f"{DOMAIN}.intercom"
STORAGE_KEY = f"{DOMAIN}.broadcast"
STORAGE_VERSION = 1
HISTORY_LIMIT = 50
DEFAULT_TTL_SECONDS = 120
IDLE_STABILITY_SECONDS = 0.5
# Matches Home Assistant's AssistSatelliteEntityFeature.ANNOUNCE without importing
# the Assist Satellite entity module (which eagerly imports the TTS/media stack).
ANNOUNCE_FEATURE = 1


@dataclass(slots=True)
class Delivery:
    entity_id: str
    status: str = "pending"
    detail: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def set(self, status: str, detail: str | None = None) -> None:
        self.status = status
        self.detail = detail
        self.updated_at = datetime.now(UTC).isoformat()


@dataclass(slots=True)
class BroadcastMessage:
    id: str
    message: str
    created_at: str
    expires_at: datetime
    source: str
    origin_entity_id: str | None
    origin_device_id: str | None
    targets: list[str]
    deliveries: dict[str, Delivery]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "message": self.message,
            "created_at": self.created_at,
            "expires_at": self.expires_at.isoformat(),
            "source": self.source,
            "origin_entity_id": self.origin_entity_id,
            "origin_device_id": self.origin_device_id,
            "targets": list(self.targets),
            "deliveries": {
                entity_id: {
                    "status": delivery.status,
                    "detail": delivery.detail,
                    "updated_at": delivery.updated_at,
                }
                for entity_id, delivery in self.deliveries.items()
            },
        }


def _aliases(value: Any) -> list[str]:
    """Return normalized Home Assistant aliases from registry entries."""
    aliases = getattr(value, "aliases", ()) or ()
    return sorted(
        {
            alias.strip()
            for alias in aliases
            if isinstance(alias, str) and alias.strip()
        },
        key=str.casefold,
    )


class IntercomManager:
    """Resolve destinations and serialize announcements per Assist satellite."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._history: deque[BroadcastMessage] = deque(maxlen=HISTORY_LIMIT)
        self._queues: dict[str, deque[BroadcastMessage]] = {}
        self._draining: set[str] = set()
        self._tracked_entities: set[str] = set()
        self._unsub_state: Any = None
        self._store: Store[dict[str, Any]] = PropagatingWriteStore(
            hass, STORAGE_VERSION, STORAGE_KEY
        )
        self._enabled = False
        self._loaded = False
        self._state_lock = asyncio.Lock()
        self._refresh_state_listener()

    async def async_initialize(self) -> None:
        """Load persisted broadcast settings once."""
        async with self._state_lock:
            if self._loaded:
                return
            stored = await self._store.async_load()
            self._enabled = bool(stored.get("enabled", False)) if stored else False
            self._loaded = True

    @property
    def enabled(self) -> bool:
        """Return whether Extended OpenAI Broadcast is enabled."""
        return self._enabled

    async def async_set_enabled(self, enabled: bool) -> None:
        """Persist the Broadcast master switch and stop pending deliveries when off."""
        async with self._state_lock:
            new_enabled = bool(enabled)
            await self._store.async_save({"enabled": new_enabled})
            self._enabled = new_enabled
            self._loaded = True
            if not self._enabled:
                for entity_id, queue in list(self._queues.items()):
                    retained: deque[BroadcastMessage] = deque()
                    for item in queue:
                        delivery = item.deliveries.get(entity_id)
                        if delivery is None:
                            continue
                        if delivery.status == "delivering":
                            retained.append(item)
                            continue
                        if delivery.status not in {"delivered", "failed", "expired"}:
                            delivery.set("expired", "broadcast_disabled")
                    queue.clear()
                    queue.extend(retained)
                    if not queue:
                        self._queues.pop(entity_id, None)

    def _satellite_entity_ids(self) -> list[str]:
        return [
            state.entity_id for state in self.hass.states.async_all("assist_satellite")
        ]

    @callback
    def _refresh_state_listener(self) -> None:
        entity_ids = set(self._satellite_entity_ids())
        if entity_ids == self._tracked_entities:
            return
        if self._unsub_state is not None:
            self._unsub_state()
        self._tracked_entities = entity_ids
        self._unsub_state = (
            async_track_state_change_event(
                self.hass, sorted(entity_ids), self._async_state_changed
            )
            if entity_ids
            else None
        )

    def _state_announce_capable(self, state: State | None) -> bool:
        if state is None:
            return False
        features = int(state.attributes.get(ATTR_SUPPORTED_FEATURES, 0) or 0)
        return bool(features & ANNOUNCE_FEATURE)

    def _entity_area_id(self, entity_id: str) -> str | None:
        entity = er.async_get(self.hass).async_get(entity_id)
        if entity is None:
            return None
        if entity.area_id:
            return entity.area_id
        if entity.device_id:
            device = dr.async_get(self.hass).async_get(entity.device_id)
            if device is not None:
                return device.area_id
        return None

    def _target_matches(
        self,
        entity_id: str,
        *,
        entity_ids: set[str],
        device_ids: set[str],
        area_ids: set[str],
        floor_ids: set[str],
        label_ids: set[str],
    ) -> bool:
        entity = er.async_get(self.hass).async_get(entity_id)
        if entity_id in entity_ids:
            return True
        if entity is None:
            return False
        if entity.device_id and entity.device_id in device_ids:
            return True
        area_id = self._entity_area_id(entity_id)
        if area_id and area_id in area_ids:
            return True
        area = ar.async_get(self.hass).async_get_area(area_id) if area_id else None
        if area and area.floor_id and area.floor_id in floor_ids:
            return True
        labels = set(entity.labels)
        if area:
            labels.update(area.labels)
        if entity.device_id:
            device = dr.async_get(self.hass).async_get(entity.device_id)
            if device:
                labels.update(device.labels)
        return bool(labels & label_ids)

    def resolve_targets(
        self,
        *,
        whole_home: bool = False,
        entity_ids: list[str] | None = None,
        device_ids: list[str] | None = None,
        area_ids: list[str] | None = None,
        floor_ids: list[str] | None = None,
        label_ids: list[str] | None = None,
        origin_entity_id: str | None = None,
        origin_device_id: str | None = None,
    ) -> list[str]:
        self._refresh_state_listener()
        entities = set(entity_ids or [])
        devices = set(device_ids or [])
        areas = set(area_ids or [])
        floors = set(floor_ids or [])
        labels = set(label_ids or [])
        registry = er.async_get(self.hass)
        result = []
        for state in self.hass.states.async_all("assist_satellite"):
            if state.entity_id == origin_entity_id:
                continue
            registry_entry = registry.async_get(state.entity_id)
            if (
                origin_device_id
                and registry_entry is not None
                and registry_entry.device_id == origin_device_id
            ):
                continue
            if not self._state_announce_capable(state):
                continue
            if whole_home or self._target_matches(
                state.entity_id,
                entity_ids=entities,
                device_ids=devices,
                area_ids=areas,
                floor_ids=floors,
                label_ids=labels,
            ):
                result.append(state.entity_id)
        return sorted(set(result))

    async def async_send(
        self,
        message: str,
        *,
        whole_home: bool = False,
        entity_ids: list[str] | None = None,
        device_ids: list[str] | None = None,
        area_ids: list[str] | None = None,
        floor_ids: list[str] | None = None,
        label_ids: list[str] | None = None,
        origin_entity_id: str | None = None,
        origin_device_id: str | None = None,
        source: str = "manual",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> dict[str, Any]:
        if not self._enabled:
            raise HomeAssistantError(
                "Broadcast is disabled. Enable it from the Extended OpenAI Overview."
            )
        text = message.strip()
        if not text:
            raise HomeAssistantError("Broadcast message cannot be empty")
        targets = self.resolve_targets(
            whole_home=whole_home,
            entity_ids=entity_ids,
            device_ids=device_ids,
            area_ids=area_ids,
            floor_ids=floor_ids,
            label_ids=label_ids,
            origin_entity_id=origin_entity_id,
            origin_device_id=origin_device_id,
        )
        if not targets:
            raise HomeAssistantError(
                "No matching announcement-capable Assist satellites found"
            )
        now = datetime.now(UTC)
        bounded_ttl = max(5, min(ttl_seconds, 3600))
        item = BroadcastMessage(
            id=uuid4().hex,
            message=text,
            created_at=now.isoformat(),
            expires_at=now + timedelta(seconds=bounded_ttl),
            source=source,
            origin_entity_id=origin_entity_id,
            origin_device_id=origin_device_id,
            targets=targets,
            deliveries={entity_id: Delivery(entity_id) for entity_id in targets},
        )
        self._history.appendleft(item)
        for entity_id in targets:
            state = self.hass.states.get(entity_id)
            if state is not None and state.state == "idle":
                item.deliveries[entity_id].set("queued_idle")
            else:
                item.deliveries[entity_id].set("queued_busy")
            self._queues.setdefault(entity_id, deque()).append(item)
            self._schedule_drain(entity_id)

        @callback
        def expire_message(_now: datetime) -> None:
            self._expire(item.id)

        async_call_later(self.hass, bounded_ttl, expire_message)
        return item.as_dict()

    @callback
    def _schedule_drain(self, entity_id: str) -> None:
        if entity_id in self._draining:
            return
        self._draining.add(entity_id)
        self.hass.async_create_task(self._async_drain(entity_id))

    async def _async_drain(self, entity_id: str) -> None:
        try:
            queue = self._queues.get(entity_id)
            while queue:
                item = queue[0]
                delivery = item.deliveries[entity_id]
                if not self._enabled:
                    delivery.set("expired", "broadcast_disabled")
                    queue.popleft()
                    continue
                if datetime.now(UTC) >= item.expires_at:
                    delivery.set("expired")
                    queue.popleft()
                    continue
                state = self.hass.states.get(entity_id)
                if state is None or state.state != "idle":
                    delivery.set("queued_busy")
                    return
                delivery.set("waiting_idle")
                await asyncio.sleep(IDLE_STABILITY_SECONDS)
                if not queue or queue[0] is not item:
                    continue
                if delivery.status == "expired":
                    queue.popleft()
                    continue
                if not self._enabled:
                    delivery.set("expired", "broadcast_disabled")
                    queue.popleft()
                    continue
                if datetime.now(UTC) >= item.expires_at:
                    delivery.set("expired")
                    queue.popleft()
                    continue
                state = self.hass.states.get(entity_id)
                if state is None or state.state != "idle":
                    delivery.set("queued_busy")
                    return
                delivery.set("delivering")
                try:
                    await self.hass.services.async_call(
                        "assist_satellite",
                        "announce",
                        {"message": item.message},
                        target={"entity_id": entity_id},
                        blocking=True,
                    )
                except Exception as err:
                    delivery.set("failed", type(err).__name__)
                else:
                    delivery.set("delivered")
                if queue and queue[0] is item:
                    queue.popleft()
            if queue is not None and not queue:
                self._queues.pop(entity_id, None)
        finally:
            self._draining.discard(entity_id)
            queue = self._queues.get(entity_id)
            if queue:
                queued_delivery = queue[0].deliveries.get(entity_id)
                state = self.hass.states.get(entity_id)
                if (
                    queued_delivery is not None
                    and queued_delivery.status == "queued_busy"
                    and state is not None
                    and state.state == "idle"
                ):
                    self._schedule_drain(entity_id)

    @callback
    def _async_state_changed(self, event: Event[Any]) -> None:
        entity_id = event.data.get("entity_id")
        new_state = event.data.get("new_state")
        if (
            entity_id in self._queues
            and new_state is not None
            and new_state.state == "idle"
        ):
            self._schedule_drain(entity_id)

    @callback
    def _expire(self, message_id: str) -> None:
        history_item = next(
            (item for item in self._history if item.id == message_id), None
        )
        if history_item is not None:
            for delivery in history_item.deliveries.values():
                if delivery.status not in {
                    "delivered",
                    "delivering",
                    "failed",
                    "expired",
                }:
                    delivery.set("expired")

        for entity_id, queue in list(self._queues.items()):
            retained: deque[BroadcastMessage] = deque()
            for item in queue:
                if item.id != message_id:
                    retained.append(item)
                    continue
                queued_delivery = item.deliveries.get(entity_id)
                if queued_delivery is None:
                    continue
                if queued_delivery.status == "delivering":
                    retained.append(item)
                    continue
                if queued_delivery.status not in {"delivered", "failed", "expired"}:
                    queued_delivery.set("expired")
            queue.clear()
            queue.extend(retained)
            if not queue:
                self._queues.pop(entity_id, None)

    def history(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self._history]

    def catalog(self) -> dict[str, Any]:
        areas = ar.async_get(self.hass)
        floors = fr.async_get(self.hass)
        labels = lr.async_get(self.hass)
        devices = dr.async_get(self.hass)
        entity_registry = er.async_get(self.hass)
        satellites = []
        relevant_area_ids: set[str] = set()
        relevant_device_ids: set[str] = set()
        relevant_label_ids: set[str] = set()
        relevant_floor_ids: set[str] = set()
        for state in self.hass.states.async_all("assist_satellite"):
            if not self._state_announce_capable(state):
                continue
            entity = entity_registry.async_get(state.entity_id)
            area_id = self._entity_area_id(state.entity_id)
            device_id = entity.device_id if entity else None
            area = areas.async_get_area(area_id) if area_id else None
            if area_id:
                relevant_area_ids.add(area_id)
            if device_id:
                relevant_device_ids.add(device_id)
            if area and area.floor_id:
                relevant_floor_ids.add(area.floor_id)
            if entity:
                relevant_label_ids.update(entity.labels)
            if area:
                relevant_label_ids.update(area.labels)
            if device_id and (device := devices.async_get(device_id)):
                relevant_label_ids.update(device.labels)
            satellites.append(
                {
                    "id": state.entity_id,
                    "name": state.attributes.get(ATTR_FRIENDLY_NAME, state.entity_id),
                    "aliases": _aliases(entity),
                    "state": state.state,
                    "area_id": area_id,
                    "device_id": device_id,
                }
            )
        return {
            "satellites": sorted(
                satellites, key=lambda item: str(item["name"]).casefold()
            ),
            "areas": sorted(
                [
                    {"id": item.id, "name": item.name, "aliases": _aliases(item)}
                    for item in areas.async_list_areas()
                    if item.id in relevant_area_ids
                ],
                key=lambda item: item["name"].casefold(),
            ),
            "floors": sorted(
                [
                    {
                        "id": item.floor_id,
                        "name": item.name,
                        "aliases": _aliases(item),
                    }
                    for item in floors.async_list_floors()
                    if item.floor_id in relevant_floor_ids
                ],
                key=lambda item: item["name"].casefold(),
            ),
            "labels": sorted(
                [
                    {
                        "id": item.label_id,
                        "name": item.name,
                        "aliases": _aliases(item),
                    }
                    for item in labels.async_list_labels()
                    if item.label_id in relevant_label_ids
                ],
                key=lambda item: item["name"].casefold(),
            ),
            "devices": sorted(
                [
                    {
                        "id": item.id,
                        "name": item.name_by_user or item.name,
                        "aliases": _aliases(item),
                    }
                    for item in (
                        devices.async_get(device_id)
                        for device_id in relevant_device_ids
                    )
                    if item is not None
                ],
                key=lambda item: str(item["name"]).casefold(),
            ),
        }

    def resolve_named_target(self, name: str) -> dict[str, Any] | None:
        wanted = re.sub(r"\s+", " ", name.strip().casefold())
        catalog = self.catalog()
        whole_home_aliases = {
            "everyone",
            "everywhere",
            "whole home",
            "the whole house",
            "all devices",
            "all speakers",
        }
        if wanted in whole_home_aliases:
            return {"whole_home": True, "name": name.strip()}
        singular = {
            "satellites": "entity_ids",
            "devices": "device_ids",
            "areas": "area_ids",
            "floors": "floor_ids",
            "labels": "label_ids",
        }
        for group, target_field in singular.items():
            for item in catalog[group]:
                names = [str(item["name"]), *(item.get("aliases") or [])]
                normalized_names = {
                    re.sub(r"\s+", " ", candidate.strip().casefold())
                    for candidate in names
                    if candidate.strip()
                }
                accepted = normalized_names | {
                    f"the {candidate}" for candidate in normalized_names
                }
                if wanted in accepted:
                    return {
                        target_field: [item["id"]],
                        "name": item["name"],
                    }
        return None


async def async_get_intercom(hass: HomeAssistant) -> IntercomManager:
    manager = hass.data.get(DATA_KEY)
    if manager is None:
        manager = IntercomManager(hass)
        hass.data[DATA_KEY] = manager
    await manager.async_initialize()
    return manager


def is_targeted_broadcast_request(text: str) -> bool:
    """Return whether wording claims a specific broadcast destination."""
    value = re.sub(r"\s+", " ", text.strip())
    return any(
        re.match(pattern, value, flags=re.IGNORECASE)
        for pattern in (
            r"^(?:broadcast|announce|send)(?: a message)? to .+$",
            r"^tell .+$",
        )
    )


def parse_targeted_broadcast(
    text: str, manager: IntercomManager
) -> tuple[dict[str, Any], str] | None:
    """Parse deliberately explicit targeted broadcast wording without an LLM call."""
    value = re.sub(r"\s+", " ", text.strip())
    patterns = (
        r"^(?:broadcast|announce|send)(?: a message)? to (?P<rest>.+)$",
        r"^tell (?P<rest>.+)$",
    )
    rest = None
    for pattern in patterns:
        match = re.match(pattern, value, flags=re.IGNORECASE)
        if match:
            rest = match.group("rest")
            break
    if rest is None:
        return None
    catalog = manager.catalog()
    candidates = [
        *(
            candidate
            for group in ("satellites", "devices", "areas", "floors", "labels")
            for item in catalog[group]
            for candidate in [str(item["name"]), *(item.get("aliases") or [])]
        ),
        "everyone",
        "everywhere",
        "whole home",
        "the whole house",
        "all devices",
        "all speakers",
    ]
    for candidate in sorted(set(candidates), key=len, reverse=True):
        target = manager.resolve_named_target(candidate)
        if target is None:
            continue
        prefix = re.escape(candidate)
        payload_match = re.match(
            rf"^(?:the )?{prefix}(?:\s+(?:that|saying|message)\s+|[:,]\s*|\s+)(?P<message>.+)$",
            rest,
            flags=re.IGNORECASE,
        )
        if payload_match:
            payload = payload_match.group("message").strip()
            target.pop("name", None)
            return target, payload
    return None
