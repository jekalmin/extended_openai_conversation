"""Memory service tool specifications and execution helpers."""

from __future__ import annotations

import asyncio
import logging
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientResponseError, ClientTimeout

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_MEMORY_API_KEY,
    CONF_MEMORY_BASE_URL,
    CONF_MEMORY_DEFAULT_NAMESPACE,
    CONF_MEMORY_WRITE_PATH,
    CONF_MEMORY_SEARCH_PATH,
    DEFAULT_MEMORY_DEFAULT_NAMESPACE,
    DEFAULT_MEMORY_WRITE_PATH,
    DEFAULT_MEMORY_SEARCH_PATH,
)
from .context_composer import estimate_tokens

_LOGGER = logging.getLogger(__name__)

MEMORY_WRITE_NAME = "memory.write"
MEMORY_SEARCH_NAME = "memory.search"

WRITE_PATH_DEFAULT = DEFAULT_MEMORY_WRITE_PATH
SEARCH_PATH_DEFAULT = DEFAULT_MEMORY_SEARCH_PATH


MEMORY_WRITE_SPEC = {
    "name": MEMORY_WRITE_NAME,
    "description": "Persist durable user memories for future conversations.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The content to store."},
            "namespace": {
                "type": "string",
                "enum": ["profile", "corpus"],
                "default": "corpus",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "importance": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "default": "normal",
            },
            "ttl_days": {"type": "integer", "minimum": 0},
        },
        "required": ["text"],
    },
}


MEMORY_SEARCH_SPEC = {
    "name": MEMORY_SEARCH_NAME,
    "description": "Search stored memories for grounding context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User utterance to search for related memories.",
            },
            "k": {"type": "integer", "default": 3},
            "min_score": {"type": "number", "default": 0.70},
            "namespaces": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["profile", "corpus"],
            },
            "filters": {
                "type": "object",
                "properties": {
                    "tags_any": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "since": {"type": "string"},
                },
            },
        },
        "required": ["query"],
    },
}


MEMORY_TOOL_SPECS = [MEMORY_WRITE_SPEC, MEMORY_SEARCH_SPEC]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{0,4}:){2,7}[A-Fa-f0-9]{0,4}\b")
MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
URL_CREDENTIAL_RE = re.compile(r"https?://[^\s/:]+:[^\s@]+@[^\s]+")
API_TOKEN_RE = re.compile(
    r"\b(?:sk|pk|api|token|key)[-_A-Za-z0-9]{8,}\b", re.IGNORECASE
)
GENERIC_SECRET_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")

HIGH_IMPORTANCE_RE = re.compile(
    r"\b(favorite|always|never|default|preference)\b", re.IGNORECASE
)

REDACTION_PATTERNS = [
    (URL_CREDENTIAL_RE, "https://[REDACTED]"),
    (EMAIL_RE, "[REDACTED_EMAIL]"),
    (CREDIT_CARD_RE, "[REDACTED_ACCOUNT]"),
    (IPV4_RE, "[REDACTED_IP]"),
    (IPV6_RE, "[REDACTED_IP6]"),
    (PHONE_RE, "[REDACTED_PHONE]"),
    (MAC_RE, "[REDACTED_MAC]"),
    (API_TOKEN_RE, "[REDACTED_TOKEN]"),
    (GENERIC_SECRET_RE, "[REDACTED]"),
]

_BACKOFF_DELAYS = (0.2, 0.6, 1.5)
_BREAKER_FAILURE_WINDOW = 60
_BREAKER_OPEN_DURATION = 120


@dataclass
class _BreakerState:
    failures: int = 0
    first_failure: float | None = None
    open_until: float | None = None


_BREAKERS: dict[str, _BreakerState] = {}

NORMALIZE_PREFIX_RE = re.compile(
    r"^(remember(?: that)?|save(?: that)?|note that|add to memory)[:,\-\s]*",
    re.IGNORECASE,
)


def normalize_memory_text(text: str) -> str:
    """Normalize memory text for storage."""

    if not text:
        return ""

    stripped = text.strip()
    normalized = NORMALIZE_PREFIX_RE.sub("", stripped).strip()
    normalized = normalized or stripped
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]
    if normalized and normalized[-1] not in ".!?":
        normalized = f"{normalized}."
    return normalized


def derive_importance(text: str) -> str:
    """Infer a default importance level from the text content."""

    if not text:
        return "normal"
    return "high" if HIGH_IMPORTANCE_RE.search(text) else "normal"


@dataclass
class MemoryServiceConfig:
    """Configuration required to reach the external memory service."""

    base_url: str | None
    api_key: str | None
    default_namespace: str
    write_path: str
    search_path: str


def get_memory_service_config(options: dict[str, Any]) -> MemoryServiceConfig:
    """Build a MemoryServiceConfig from entry options."""

    return MemoryServiceConfig(
        base_url=options.get(CONF_MEMORY_BASE_URL),
        api_key=options.get(CONF_MEMORY_API_KEY),
        default_namespace=options.get(
            CONF_MEMORY_DEFAULT_NAMESPACE, DEFAULT_MEMORY_DEFAULT_NAMESPACE
        ),
        write_path=options.get(CONF_MEMORY_WRITE_PATH, WRITE_PATH_DEFAULT),
        search_path=options.get(CONF_MEMORY_SEARCH_PATH, SEARCH_PATH_DEFAULT),
    )


def redact(text: str) -> str:
    """Redact obvious secrets from snippets."""

    if not text:
        return ""
    for pattern, replacement in REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def extract_snippet(text: str, sentences: int = 3) -> str:
    """Return a short snippet with at most `sentences` sentences."""

    segments = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: Iterable[str] = [segment for segment in segments if segment]
    return " ".join(list(kept)[:sentences])


def _truncate_snippet(text: str, budget: int) -> tuple[str, int]:
    """Trim snippet text so it fits within the token budget."""

    if budget <= 0 or not text:
        return "", 0

    estimated = estimate_tokens(text)
    if estimated <= budget:
        return text, estimated

    parts = re.split(r"(\s+)", text)
    builder: list[str] = []
    for piece in parts:
        candidate = "".join(builder + [piece])
        if estimate_tokens(candidate) > budget:
            break
        builder.append(piece)

    truncated = "".join(builder).strip()
    if not truncated:
        approx_chars = max(1, budget * 4)
        truncated = text[:approx_chars].strip()
    return truncated, estimate_tokens(truncated)


def build_memory_tool_definitions() -> list[dict[str, Any]]:
    """Return tool definition payloads for OpenAI requests."""

    return [{"type": "function", "function": spec} for spec in MEMORY_TOOL_SPECS]


def is_configured(config: MemoryServiceConfig) -> bool:
    """Return True when the memory service has a base URL configured."""

    return bool(config.base_url)


def _breaker_key(config: MemoryServiceConfig, operation: str) -> str:
    base = config.base_url or "memory"
    return f"{base}:{operation}"


def _should_skip_operation(
    config: MemoryServiceConfig, operation: str, allow_skip: bool
) -> bool:
    if operation != "search" or not allow_skip:
        return False

    key = _breaker_key(config, operation)
    state = _BREAKERS.get(key)
    if not state or state.open_until is None:
        return False

    now = time.monotonic()
    if now < state.open_until:
        remaining = state.open_until - now
        _LOGGER.debug(
            "Memory service circuit breaker open for %s (%.0fs remaining)",
            key,
            remaining,
        )
        return True

    _LOGGER.debug("Memory service circuit breaker reset after cooldown for %s", key)
    _BREAKERS[key] = _BreakerState()
    return False


def _record_failure(
    config: MemoryServiceConfig, operation: str, *, allow_skip: bool
) -> None:
    if operation != "search" or not allow_skip:
        return

    key = _breaker_key(config, operation)
    state = _BREAKERS.setdefault(key, _BreakerState())
    now = time.monotonic()

    if state.first_failure is None or now - state.first_failure > _BREAKER_FAILURE_WINDOW:
        state.first_failure = now
        state.failures = 1
    else:
        state.failures += 1

    if state.failures >= 5:
        state.open_until = now + _BREAKER_OPEN_DURATION
        state.failures = 0
        state.first_failure = None
        _LOGGER.debug(
            "Memory service circuit breaker opened for %s (%.0fs)",
            key,
            _BREAKER_OPEN_DURATION,
        )


def _record_success(config: MemoryServiceConfig, operation: str) -> None:
    if operation != "search":
        return

    key = _breaker_key(config, operation)
    state = _BREAKERS.get(key)
    if not state:
        return

    if state.open_until or state.failures:
        _LOGGER.debug("Memory service circuit breaker reset for %s", key)
    _BREAKERS[key] = _BreakerState()


async def _request(
    hass: HomeAssistant,
    method: str,
    endpoint: str,
    payload: dict[str, Any],
    config: MemoryServiceConfig,
    *,
    session: aiohttp.ClientSession | None = None,
    operation: str,
    allow_skip_breaker: bool,
) -> dict[str, Any] | None:
    """Execute a request against the memory service."""

    if not config.base_url:
        _LOGGER.debug("Skipping memory call %s: no base URL configured", endpoint)
        return None

    if _should_skip_operation(config, operation, allow_skip_breaker):
        return None

    session = session or async_get_clientsession(hass)
    url = urljoin(config.base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    timeout = ClientTimeout(connect=3, sock_read=8)
    start = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(len(_BACKOFF_DELAYS) + 1):
        try:
            async with session.request(
                method, url, json=payload, headers=headers, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        except ClientResponseError as err:
            last_error = err
            status = err.status or 0
            retriable = 500 <= status < 600
            if retriable and attempt < len(_BACKOFF_DELAYS):
                await asyncio.sleep(_BACKOFF_DELAYS[attempt])
                continue
            break
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            last_error = err
            break
        else:
            duration = time.monotonic() - start
            _LOGGER.debug("Memory service %s succeeded in %.2fs", endpoint, duration)
            _record_success(config, operation)
            return data

    duration = time.monotonic() - start
    if last_error is None:
        last_error = RuntimeError("Unknown memory service failure")
    _LOGGER.debug(
        "Memory service %s failed after %.2fs: %s", endpoint, duration, last_error
    )
    _record_failure(config, operation, allow_skip=allow_skip_breaker)
    return None


async def async_memory_write(
    hass: HomeAssistant,
    config: MemoryServiceConfig,
    arguments: dict[str, Any],
    *,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Execute the memory.write tool."""

    payload = dict(arguments)
    text_value = str(payload.get("text", "") or "")
    if text_value:
        normalized_text = normalize_memory_text(text_value)
        payload["text"] = normalized_text
        payload.setdefault("importance", derive_importance(normalized_text))
    payload.setdefault("namespace", config.default_namespace)
    endpoint = config.write_path or WRITE_PATH_DEFAULT
    result = await _request(
        hass,
        "POST",
        endpoint,
        payload,
        config,
        session=session,
        operation="write",
        allow_skip_breaker=False,
    )
    if result is None:
        return "Unable to reach memory service to write entry."
    status = result.get("status") or "stored"
    return f"Memory write acknowledged ({status})."


async def async_memory_search(
    hass: HomeAssistant,
    config: MemoryServiceConfig,
    arguments: dict[str, Any],
    *,
    session: aiohttp.ClientSession | None = None,
    forced: bool = False,
    token_budget: int | None = None,
) -> tuple[str, list[str]]:
    """Execute the memory.search tool and return snippets."""

    payload = dict(arguments)
    payload.setdefault("namespaces", [config.default_namespace, "profile"])
    if _should_skip_operation(config, "search", not forced):
        return "", []

    endpoint = config.search_path or SEARCH_PATH_DEFAULT
    result = await _request(
        hass,
        "POST",
        endpoint,
        payload,
        config,
        session=session,
        operation="search",
        allow_skip_breaker=not forced,
    )
    if not result:
        return "", []
    results = result.get("results", [])
    k = arguments.get("k", 3)
    try:
        limit = max(0, int(k))
    except (TypeError, ValueError):
        limit = 0
    min_score_raw = arguments.get("min_score")
    try:
        min_score_value = float(min_score_raw)
    except (TypeError, ValueError):
        min_score_value = None
    filtered: list[dict[str, Any]] = []
    for item in results:
        score = item.get("score")
        if (
            isinstance(score, (int, float))
            and min_score_value is not None
            and score < min_score_value
        ):
            continue
        filtered.append(item)
    if limit:
        filtered = filtered[:limit]

    snippet_data: list[dict[str, Any]] = []
    for index, item in enumerate(filtered):
        raw_snippet = extract_snippet(str(item.get("text", "")))
        text = redact(raw_snippet)
        if not text:
            continue
        snippet_data.append(
            {
                "index": index,
                "text": text,
                "tokens": estimate_tokens(text),
                "score": item.get("score"),
                "weight": estimate_tokens(raw_snippet),
            }
        )

    if token_budget is not None and token_budget >= 0 and snippet_data:

        def score_value(entry: dict[str, Any]) -> float:
            score_obj = entry.get("score")
            if isinstance(score_obj, (int, float)):
                return float(score_obj)
            return float("-inf")

        def total_tokens() -> int:
            return sum(item.get("weight", item["tokens"]) for item in snippet_data)

        while total_tokens() > token_budget and len(snippet_data) > 2:
            lowest_idx, _ = min(
                enumerate(snippet_data),
                key=lambda pair: (score_value(pair[1]), pair[1]["index"]),
            )
            snippet_data.pop(lowest_idx)

        if snippet_data and total_tokens() > token_budget:
            per_snippet_budget = max(1, token_budget // len(snippet_data) or 1)
            for item in list(snippet_data):
                truncated, tokens = _truncate_snippet(item["text"], per_snippet_budget)
                if not truncated:
                    snippet_data.remove(item)
                    continue
                item["text"] = truncated
                item["tokens"] = tokens
                item["weight"] = tokens

        while total_tokens() > token_budget and len(snippet_data) > 1:
            lowest_idx, _ = min(
                enumerate(snippet_data),
                key=lambda pair: (score_value(pair[1]), pair[1]["index"]),
            )
            snippet_data.pop(lowest_idx)

        if snippet_data and total_tokens() > token_budget and len(snippet_data) == 1:
            truncated, tokens = _truncate_snippet(
                snippet_data[0]["text"], token_budget
            )
            if not truncated or tokens > token_budget:
                snippet_data.clear()
            else:
                snippet_data[0]["text"] = truncated
                snippet_data[0]["tokens"] = tokens
                snippet_data[0]["weight"] = tokens

    snippets: list[str] = []
    for item in sorted(snippet_data, key=lambda entry: entry["index"]):
        score = item.get("score")
        prefix = f"score={score:.2f}" if isinstance(score, (int, float)) else ""
        snippet = item["text"] if not prefix else f"({prefix}) {item['text']}"
        snippets.append(snippet)

    joined = "\n".join(f"- {snippet}" for snippet in snippets)
    return joined, snippets


async def dispatch_memory_tool(
    hass: HomeAssistant,
    config: MemoryServiceConfig,
    name: str,
    arguments: dict[str, Any],
) -> str:
    """Execute a memory tool by name."""

    if name == MEMORY_WRITE_NAME:
        return await async_memory_write(hass, config, arguments)
    if name == MEMORY_SEARCH_NAME:
        snippets, _ = await async_memory_search(
            hass, config, arguments, forced=True
        )
        return snippets or "No matching memories found."
    raise ValueError(f"Unknown memory tool: {name}")
