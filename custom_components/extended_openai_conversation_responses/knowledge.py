"""On-demand local Knowledge Library for conversation agents."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import logging
import re
from typing import Any, Protocol, cast
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .persistence_hardening import _async_settle_transactional_save
from .strict_store import PropagatingWriteStore

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 2
STORAGE_KEY_PREFIX = f"{DOMAIN}.knowledge"
MAX_SOURCES_PER_AGENT = 500
MAX_TITLE_LENGTH = 120
MAX_DESCRIPTION_LENGTH = 500
MAX_CONTENT_LENGTH = 100_000
MAX_SEARCH_LIMIT = 10
MAX_CATALOG_LIMIT = 50
DEFAULT_CATALOG_LIMIT = 20
MAX_GET_CHARACTERS = 20_000
DEFAULT_GET_CHARACTERS = 6_000
CHUNK_SIZE = 2_000
CHUNK_OVERLAP = 300
MAX_EXCERPT_CHARACTERS = 2_000

KNOWLEDGE_TOOL_NAMES = {"knowledge_search", "knowledge_list", "knowledge_get"}

_TOKEN_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
_SPACE_PATTERN = re.compile(r"\s+")
_LINE_END_PATTERN = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass(slots=True, frozen=True)
class KnowledgeSource:
    """One deliberately maintained reference source."""

    source_id: str
    title: str
    description: str
    content: str
    created_at: str
    updated_at: str
    enabled: bool = True


@dataclass(slots=True, frozen=True)
class SearchResult:
    """A bounded relevant excerpt from one source."""

    source_id: str
    title: str
    description: str
    excerpt: str
    score: float
    updated_at: str


@dataclass(slots=True, frozen=True)
class _Chunk:
    source_id: str
    chunk_id: int
    start: int
    text: str
    normalized_text: str
    tokens: frozenset[str]


@dataclass(slots=True, frozen=True)
class _SourceFeatures:
    title_tokens: frozenset[str]
    description_tokens: frozenset[str]
    metadata_tokens: frozenset[str]
    normalized_title: str
    normalized_description: str
    normalized_metadata: str


class KnowledgeStorage(Protocol):
    """Persistence boundary for future alternative knowledge backends."""

    async def async_load(self) -> Mapping[str, Any] | Sequence[Any] | None:
        """Load stored knowledge data."""

    async def async_save(self, data: dict[str, Any]) -> None:
        """Persist stored knowledge data."""


class KnowledgeStore(PropagatingWriteStore):
    """Versioned Home Assistant Store backend."""

    async def _async_migrate_func(
        self, old_major_version: int, old_minor_version: int, old_data: Any
    ) -> dict[str, Any]:
        """Migrate older knowledge payloads."""
        if old_major_version in {0, 1}:
            return _migrated_storage_payload(old_data)
        raise NotImplementedError


class HomeAssistantKnowledgeStorage:
    """Private, atomic, per-agent Home Assistant storage adapter."""

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str) -> None:
        key = f"{STORAGE_KEY_PREFIX}.{entry_id}.{subentry_id}"
        self._store = KnowledgeStore(
            hass,
            STORAGE_VERSION,
            key,
            private=True,
            atomic_writes=True,
            serialize_in_event_loop=False,
        )

    async def async_load(self) -> dict[str, Any] | None:
        return await self._store.async_load()

    async def async_save(self, data: dict[str, Any]) -> None:
        await self._store.async_save(data)


class KnowledgeLibrary:
    """Concurrency-safe source collection with an in-memory lexical index."""

    def __init__(self, storage: KnowledgeStorage) -> None:
        self._storage = storage
        self._sources: dict[str, KnowledgeSource] = {}
        self._chunks: dict[tuple[str, int], _Chunk] = {}
        self._source_features: dict[str, _SourceFeatures] = {}
        self._token_index: dict[str, set[tuple[str, int]]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._initialized = False
        self._committed_state: dict[str, Any] | None = None

    async def async_initialize(self) -> None:
        """Load and index the library exactly once."""
        async with self._lock:
            if self._initialized:
                return
            try:
                data = await self._storage.async_load()
                if data is not None and not isinstance(data, Mapping):
                    raise ValueError("Knowledge Library store has invalid structure")
                if (
                    isinstance(data, Mapping)
                    and "sources" in data
                    and not isinstance(data["sources"], list)
                ):
                    raise ValueError("Knowledge Library sources have invalid structure")
                raw_sources = (
                    data.get("sources", []) if isinstance(data, Mapping) else []
                )
                for raw in raw_sources:
                    if len(self._sources) >= MAX_SOURCES_PER_AGENT:
                        _LOGGER.warning(
                            "Ignoring Knowledge Library records beyond the per-agent limit"
                        )
                        break
                    try:
                        source = _source_from_stored(raw)
                    except KeyError, TypeError, ValueError:
                        _LOGGER.warning("Ignoring malformed Knowledge Library record")
                        continue
                    if source.source_id in self._sources:
                        _LOGGER.warning(
                            "Ignoring duplicate Knowledge Library source ID"
                        )
                        continue
                    self._sources[source.source_id] = source
                    self._index(source)
                self._remember_committed_state()
                self._initialized = True
            except BaseException:
                self._sources.clear()
                self._chunks.clear()
                self._source_features.clear()
                self._token_index = defaultdict(set)
                self._initialized = False
                self._committed_state = None
                raise

    @property
    def source_count(self) -> int:
        """Return the number of model-available sources."""
        self._ensure_initialized()
        return sum(source.enabled for source in self._sources.values())

    @property
    def total_source_count(self) -> int:
        """Return all stored sources, including disabled sources."""
        self._ensure_initialized()
        return len(self._sources)

    async def async_list(self) -> list[dict[str, Any]]:
        """List all source metadata for management, including disabled sources."""
        self._ensure_initialized()
        sources = sorted(
            self._sources.values(), key=lambda source: source.updated_at, reverse=True
        )
        return [source_summary(source) for source in sources]

    async def async_get(self, source_id: str) -> KnowledgeSource:
        """Get one complete source for the management UI."""
        self._ensure_initialized()
        return self._source(source_id)

    async def async_catalog(
        self,
        query: str | None = None,
        limit: int = DEFAULT_CATALOG_LIMIT,
        offset: int = 0,
        allowed_source_ids: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """List bounded enabled source metadata without returning content."""
        self._ensure_initialized()
        if query is not None and not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise ValueError("offset must be an integer")
        limit = max(1, min(limit, MAX_CATALOG_LIMIT))
        offset = max(0, offset)
        normalized_query = _normalize(query or "")
        query_tokens = _tokens(query or "")

        sources = [
            source
            for source in self._sources.values()
            if source.enabled
            and (allowed_source_ids is None or source.source_id in allowed_source_ids)
        ]
        if normalized_query:
            sources = [
                source
                for source in sources
                if (features := self._source_features.get(source.source_id)) is not None
                and (
                    normalized_query in features.normalized_metadata
                    or bool(query_tokens & features.metadata_tokens)
                )
            ]
        sources.sort(
            key=lambda source: (source.updated_at, source.source_id), reverse=True
        )
        total = len(sources)
        selected = sources[offset : offset + limit]
        next_offset = offset + len(selected)
        has_more = next_offset < total
        return {
            "sources": [source_summary(source) for source in selected],
            "total": total,
            "offset": offset,
            "returned": len(selected),
            "has_more": has_more,
            "next_offset": next_offset if has_more else None,
        }

    async def async_create(
        self, title: str, description: str, content: str, enabled: bool = True
    ) -> KnowledgeSource:
        """Create and persist a source."""
        title, description, content = _validated_fields(title, description, content)
        enabled = _validated_enabled(enabled)
        async with self._lock:
            self._ensure_initialized()
            if len(self._sources) >= MAX_SOURCES_PER_AGENT:
                raise ValueError(
                    "knowledge source limit reached; delete a source before adding more"
                )
            timestamp = dt_util.utcnow().isoformat()
            source = KnowledgeSource(
                source_id=uuid4().hex,
                title=title,
                description=description,
                content=content,
                created_at=timestamp,
                updated_at=timestamp,
                enabled=enabled,
            )
            self._sources[source.source_id] = source
            self._index(source)
            await self._async_save_locked()
            return source

    async def async_update(
        self,
        source_id: str,
        title: str | None = None,
        description: str | None = None,
        content: str | None = None,
        enabled: bool | None = None,
        expected_revision: str | None = None,
    ) -> KnowledgeSource:
        """Update and immediately re-index one source."""
        async with self._lock:
            self._ensure_initialized()
            current = self._source(source_id)
            if (
                expected_revision is not None
                and expected_revision != current.updated_at
            ):
                raise ValueError(
                    "Knowledge source changed in another tab. Reload before saving."
                )
            new_title, new_description, new_content = _validated_fields(
                current.title if title is None else title,
                current.description if description is None else description,
                current.content if content is None else content,
            )
            new_enabled = (
                current.enabled if enabled is None else _validated_enabled(enabled)
            )
            self._unindex(source_id)
            updated = KnowledgeSource(
                source_id=current.source_id,
                title=new_title,
                description=new_description,
                content=new_content,
                created_at=current.created_at,
                updated_at=dt_util.utcnow().isoformat(),
                enabled=new_enabled,
            )
            self._sources[source_id] = updated
            self._index(updated)
            await self._async_save_locked()
            return updated

    async def async_delete(self, source_id: str) -> bool:
        """Delete one source."""
        async with self._lock:
            self._ensure_initialized()
            if source_id not in self._sources:
                return False
            self._unindex(source_id)
            del self._sources[source_id]
            await self._async_save_locked()
            return True

    async def async_search(
        self,
        query: str,
        source_ids: list[str] | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search enabled indexed chunks and return at most one excerpt per source."""
        self._ensure_initialized()
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        limit = max(1, min(limit, MAX_SEARCH_LIMIT))
        query_tokens = _tokens(query)
        normalized_query = _normalize(query)
        if not query_tokens or not normalized_query:
            return []
        allowed, _ = self.resolve_source_filter(source_ids)

        candidates: set[tuple[str, int]] = set()
        for token in query_tokens:
            candidates.update(self._token_index.get(token, set()))

        best_by_source: dict[str, tuple[float, _Chunk]] = {}
        for chunk_key in candidates:
            chunk = self._chunks[chunk_key]
            if allowed is not None and chunk.source_id not in allowed:
                continue
            features = self._source_features[chunk.source_id]
            title_tokens = features.title_tokens
            description_tokens = features.description_tokens
            normalized_title = features.normalized_title
            normalized_description = features.normalized_description
            overlap = len(query_tokens & chunk.tokens) / len(query_tokens)
            title_overlap = len(query_tokens & title_tokens) / len(query_tokens)
            description_overlap = len(query_tokens & description_tokens) / len(
                query_tokens
            )
            score = overlap + title_overlap * 8 + description_overlap * 4
            if normalized_query in normalized_title:
                score += 8
            if normalized_query in normalized_description:
                score += 5
            if normalized_query in chunk.normalized_text:
                score += 4
            current = best_by_source.get(chunk.source_id)
            if (
                current is None
                or score > current[0]
                or (score == current[0] and chunk.start < current[1].start)
            ):
                best_by_source[chunk.source_id] = (score, chunk)

        ranked = [
            (score, self._sources[source_id].updated_at, source_id, chunk)
            for source_id, (score, chunk) in best_by_source.items()
            if score > 0
        ]
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        results: list[SearchResult] = []
        for score, _, source_id, chunk in ranked[:limit]:
            source = self._sources[source_id]
            results.append(
                SearchResult(
                    source_id=source.source_id,
                    title=source.title,
                    description=source.description,
                    excerpt=chunk.text[:MAX_EXCERPT_CHARACTERS],
                    score=round(score, 4),
                    updated_at=source.updated_at,
                )
            )
        return results

    def resolve_source_filter(
        self, source_ids: list[str] | None
    ) -> tuple[set[str] | None, list[str]]:
        """Resolve model-provided IDs against enabled sources only."""
        self._ensure_initialized()
        if not source_ids:
            return None, []

        valid: set[str] = set()
        ignored: list[str] = []
        for requested_id in source_ids:
            normalized_id = requested_id.strip()
            source = self._sources.get(normalized_id)
            if normalized_id and source is not None and source.enabled:
                valid.add(normalized_id)
            else:
                ignored.append(requested_id)
        return (valid or None), ignored

    async def async_get_section(
        self,
        source_id: str,
        start_character: int = 0,
        max_characters: int = DEFAULT_GET_CHARACTERS,
    ) -> dict[str, Any]:
        """Return a bounded, pageable enabled source section for model use."""
        self._ensure_initialized()
        if not isinstance(start_character, int) or isinstance(start_character, bool):
            raise ValueError("start_character must be an integer")
        if not isinstance(max_characters, int) or isinstance(max_characters, bool):
            raise ValueError("max_characters must be an integer")
        if start_character < 0:
            raise ValueError("start_character must be at least 0")
        max_characters = max(500, min(max_characters, MAX_GET_CHARACTERS))
        source = self._available_source(source_id)
        total = len(source.content)
        start = min(start_character, total)
        content = source.content[start : start + max_characters]
        next_start = start + len(content)
        has_more = next_start < total
        return {
            "source_id": source.source_id,
            "title": source.title,
            "description": source.description,
            "content": content,
            "start_character": start,
            "returned_characters": len(content),
            "total_characters": total,
            "has_more": has_more,
            "next_start_character": next_start if has_more else None,
            "updated_at": source.updated_at,
        }

    def stats(self) -> dict[str, Any]:
        """Return non-sensitive diagnostics."""
        self._ensure_initialized()
        return {
            "knowledge_backend": "home_assistant_store",
            "knowledge_storage_version": STORAGE_VERSION,
            "knowledge_source_count": len(self._sources),
            "knowledge_enabled_source_count": self.source_count,
            "knowledge_total_character_count": sum(
                len(source.content) for source in self._sources.values()
            ),
            "knowledge_indexed_chunk_count": len(self._chunks),
        }

    async def async_backup_data(self) -> dict[str, Any]:
        """Return canonical Knowledge sources, excluding the derived index."""
        async with self._lock:
            self._ensure_initialized()
            return {"sources": [asdict(source) for source in self._sources.values()]}

    @staticmethod
    def validate_backup_data(data: Any) -> list[KnowledgeSource]:
        """Validate source material before a replacement restore."""
        if not isinstance(data, Mapping) or set(data) != {"sources"}:
            raise ValueError("Knowledge data is incomplete or corrupted")
        raw_sources = data["sources"]
        if (
            not isinstance(raw_sources, list)
            or len(raw_sources) > MAX_SOURCES_PER_AGENT
        ):
            raise ValueError("Knowledge source count is invalid")
        sources: list[KnowledgeSource] = []
        seen: set[str] = set()
        for raw in raw_sources:
            try:
                source = _source_from_stored(raw)
            except (KeyError, TypeError, ValueError) as err:
                raise ValueError("Knowledge source is invalid") from err
            if source.source_id in seen:
                raise ValueError("Knowledge source IDs must be unique")
            if (
                dt_util.parse_datetime(source.created_at) is None
                or dt_util.parse_datetime(source.updated_at) is None
            ):
                raise ValueError("Knowledge source timestamp is invalid")
            seen.add(source.source_id)
            sources.append(source)
        return sources

    async def async_replace_backup(self, sources: list[KnowledgeSource]) -> None:
        """Replace source material and rebuild the derived lexical index."""
        async with self._lock:
            self._ensure_initialized()
            self._sources = {source.source_id: source for source in sources}
            self._chunks.clear()
            self._source_features.clear()
            self._token_index.clear()
            for source in sources:
                self._index(source)
            await self._async_save_locked()

    def _source(self, source_id: str) -> KnowledgeSource:
        source = self._sources.get(source_id)
        if source is None:
            raise ValueError("knowledge source not found")
        return source

    def _available_source(self, source_id: str) -> KnowledgeSource:
        """Return a source only when it is available to model retrieval."""
        source = self._sources.get(source_id)
        if source is None or not source.enabled:
            raise ValueError("knowledge source not found")
        return source

    def _index(self, source: KnowledgeSource) -> None:
        if not source.enabled:
            return
        title_tokens = frozenset(_tokens(source.title))
        description_tokens = frozenset(_tokens(source.description))
        metadata_tokens = title_tokens | description_tokens
        normalized_title = _normalize(source.title)
        normalized_description = _normalize(source.description)
        self._source_features[source.source_id] = _SourceFeatures(
            title_tokens=title_tokens,
            description_tokens=description_tokens,
            metadata_tokens=metadata_tokens,
            normalized_title=normalized_title,
            normalized_description=normalized_description,
            normalized_metadata=" ".join(
                filter(None, (normalized_title, normalized_description))
            ),
        )
        for chunk_id, (start, text) in enumerate(_split_chunks(source.content)):
            chunk = _Chunk(
                source_id=source.source_id,
                chunk_id=chunk_id,
                start=start,
                text=text,
                normalized_text=_normalize(text),
                tokens=frozenset(_tokens(text) | metadata_tokens),
            )
            key = (source.source_id, chunk_id)
            self._chunks[key] = chunk
            for token in chunk.tokens:
                self._token_index[token].add(key)

    def _unindex(self, source_id: str) -> None:
        self._source_features.pop(source_id, None)
        keys = [key for key in self._chunks if key[0] == source_id]
        for key in keys:
            chunk = self._chunks.pop(key)
            for token in chunk.tokens:
                indexed = self._token_index.get(token)
                if indexed is None:
                    continue
                indexed.discard(key)
                if not indexed:
                    del self._token_index[token]

    async def _async_save_locked(self) -> None:
        """Settle writes and rebuild the lexical index from persisted sources."""
        await _async_settle_transactional_save(
            self._storage.async_save(
                {"sources": [asdict(source) for source in self._sources.values()]}
            ),
            self._restore_committed_state,
            self._remember_committed_state,
            self._async_reconcile_failed_save,
            self._invalidate_after_unreadable_store,
        )

    async def _async_reconcile_failed_save(self) -> None:
        """Reload validated sources and reconstruct every search index."""
        disk_state = KnowledgeLibrary(self._storage)
        await disk_state.async_initialize()
        self._sources = dict(disk_state._sources)
        self._chunks = dict(disk_state._chunks)
        self._source_features = dict(disk_state._source_features)
        self._token_index = defaultdict(
            set, {token: set(keys) for token, keys in disk_state._token_index.items()}
        )
        self._initialized = True
        self._remember_committed_state()

    def _invalidate_after_unreadable_store(self) -> None:
        """Block searches when persisted Knowledge cannot be validated."""
        self._sources.clear()
        self._chunks.clear()
        self._source_features.clear()
        self._token_index = defaultdict(set)
        self._initialized = False
        self._committed_state = None

    def _remember_committed_state(self) -> None:
        self._committed_state = {"sources": dict(self._sources)}

    def _restore_committed_state(self) -> None:
        snapshot = self._committed_state
        if snapshot is None:
            return
        self._sources = dict(snapshot["sources"])
        self._chunks.clear()
        self._source_features.clear()
        self._token_index = defaultdict(set)
        for source in self._sources.values():
            self._index(source)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Knowledge Library has not been initialized")


_KNOWLEDGE_MANAGERS = f"{DOMAIN}.knowledge_managers"


def get_loaded_knowledge(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> KnowledgeLibrary | None:
    """Return an already initialized library without creating runtime state."""
    return cast(
        KnowledgeLibrary | None,
        hass.data.get(_KNOWLEDGE_MANAGERS, {}).get((entry_id, subentry_id)),
    )


async def async_get_knowledge(
    hass: HomeAssistant, entry_id: str, subentry_id: str
) -> KnowledgeLibrary:
    """Get the shared in-process library for one conversation agent."""
    managers: dict[tuple[str, str], KnowledgeLibrary] = hass.data.setdefault(
        _KNOWLEDGE_MANAGERS, {}
    )
    key = (entry_id, subentry_id)
    if key not in managers:
        managers[key] = KnowledgeLibrary(
            HomeAssistantKnowledgeStorage(hass, entry_id, subentry_id)
        )
    library = managers[key]
    await library.async_initialize()
    return library


def knowledge_source_as_dict(source: KnowledgeSource) -> dict[str, Any]:
    """Serialize a complete source."""
    return asdict(source)


def source_summary(source: KnowledgeSource) -> dict[str, Any]:
    """Serialize metadata safe for list views."""
    return {
        "source_id": source.source_id,
        "title": source.title,
        "description": source.description,
        "enabled": source.enabled,
        "created_at": source.created_at,
        "updated_at": source.updated_at,
        "character_count": len(source.content),
    }


def search_result_as_dict(result: SearchResult) -> dict[str, Any]:
    return asdict(result)


def knowledge_tools() -> list[dict[str, Any]]:
    """Return read-only built-in model tool definitions."""
    return [
        {
            "spec": {
                "name": "knowledge_search",
                "description": (
                    "Search the agent's local Knowledge Library for relevant "
                    "deliberately maintained reference information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "description": "What information to find in the local knowledge library.",
                        },
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional exact source IDs returned by a previous "
                                "Knowledge Library tool call. Omit this field when no "
                                "exact returned IDs are available; never invent IDs or "
                                "use titles or categories as IDs."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            "function": {"type": "knowledge", "operation": "search"},
        },
        {
            "spec": {
                "name": "knowledge_list",
                "description": (
                    "List bounded Knowledge source metadata when the right search "
                    "terms or source ID are unknown. Returns titles and descriptions, "
                    "never source content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Optional short filter applied only to source titles "
                                "and descriptions. Omit it to browse available sources."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_CATALOG_LIMIT,
                            "default": DEFAULT_CATALOG_LIMIT,
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "function": {"type": "knowledge", "operation": "list"},
        },
        {
            "spec": {
                "name": "knowledge_get",
                "description": (
                    "Retrieve one Knowledge source or a bounded section after "
                    "identifying its exact source ID."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "description": "The exact knowledge-source ID to retrieve.",
                        },
                        "start_character": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                        },
                        "max_characters": {
                            "type": "integer",
                            "minimum": 500,
                            "maximum": MAX_GET_CHARACTERS,
                            "default": DEFAULT_GET_CHARACTERS,
                        },
                    },
                    "required": ["source_id"],
                    "additionalProperties": False,
                },
            },
            "function": {"type": "knowledge", "operation": "get"},
        },
    ]


def _migrated_storage_payload(old_data: Any) -> dict[str, Any]:
    """Normalize legacy stores and make previous sources explicitly enabled."""
    raw_sources = (
        old_data
        if isinstance(old_data, list)
        else old_data.get("sources", [])
        if isinstance(old_data, Mapping)
        else []
    )
    if not isinstance(raw_sources, list):
        raw_sources = []
    migrated: list[Any] = []
    for raw in raw_sources:
        if isinstance(raw, Mapping):
            record = dict(raw)
            record.setdefault("enabled", True)
            migrated.append(record)
        else:
            migrated.append(raw)
    return {"sources": migrated}


def _source_from_stored(raw: Any) -> KnowledgeSource:
    if not isinstance(raw, Mapping):
        raise ValueError("record must be an object")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    source = KnowledgeSource(
        source_id=raw["source_id"],
        title=raw["title"],
        description=raw["description"],
        content=raw["content"],
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
        enabled=enabled,
    )
    if not all(
        isinstance(value, str)
        for value in (
            source.source_id,
            source.title,
            source.description,
            source.content,
            source.created_at,
            source.updated_at,
        )
    ):
        raise ValueError("record fields must be strings")
    _validated_fields(source.title, source.description, source.content)
    if not source.source_id or len(source.source_id) > 128:
        raise ValueError("invalid source ID")
    return source


def _validated_enabled(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("enabled must be a boolean")
    return value


def _validated_fields(
    title: str, description: str, content: str
) -> tuple[str, str, str]:
    title = _clean_single_line("title", title, MAX_TITLE_LENGTH, required=True)
    description = _clean_single_line(
        "description", description, MAX_DESCRIPTION_LENGTH, required=False
    )
    content = _clean_content(content)
    return title, description, content


def _clean_single_line(name: str, value: str, limit: int, required: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    value = _SPACE_PATTERN.sub(" ", value).strip()
    minimum = 1 if required else 0
    if len(value) < minimum or len(value) > limit:
        raise ValueError(f"{name} must be {minimum} to {limit} characters")
    return value


def _clean_content(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("content must be a string")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _LINE_END_PATTERN.sub("", value).strip()
    if not value or len(value) > MAX_CONTENT_LENGTH:
        raise ValueError(f"content must be 1 to {MAX_CONTENT_LENGTH} characters")
    return value


def _tokens(value: str) -> set[str]:
    return {
        token for token in _TOKEN_PATTERN.findall(value.casefold()) if len(token) > 1
    }


def _normalize(value: str) -> str:
    return " ".join(_TOKEN_PATTERN.findall(value.casefold()))


def _split_chunks(content: str) -> list[tuple[int, str]]:
    """Split content at nearby paragraph/line boundaries with overlap."""
    chunks: list[tuple[int, str]] = []
    start = 0
    length = len(content)
    while start < length:
        target = min(start + CHUNK_SIZE, length)
        end = target
        if target < length:
            lower = max(start + CHUNK_SIZE // 2, target - 350)
            paragraph = content.rfind("\n\n", lower, target + 1)
            line = content.rfind("\n", lower, target + 1)
            boundary = paragraph if paragraph >= lower else line
            if boundary >= lower:
                end = boundary + (2 if boundary == paragraph else 1)
        text = content[start:end].strip()
        if text:
            chunks.append((start, text))
        if end >= length:
            break
        next_start = max(start + 1, end - CHUNK_OVERLAP)
        line_after_overlap = content.find("\n", next_start, end)
        start = line_after_overlap + 1 if line_after_overlap != -1 else next_start
    return chunks
