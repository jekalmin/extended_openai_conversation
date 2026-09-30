"""Opt-in, bounded request debugging for conversation agents.

The normal usage subsystem deliberately remains content-free. This module captures
full request material only while an administrator has explicitly enabled debugging
for an agent. Captures are kept in memory and disappear on Home Assistant restart.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
import hashlib
import json
import time
from typing import Any, cast
from uuid import uuid4

from homeassistant.helpers.template import Template
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .provider_errors import provider_error_metadata
from .request_diagnostics import (
    provider_diagnostics,
    record_provider_metrics,
    summary_diagnostics,
    trace_diagnostics,
)

DEBUG_DEFAULT_LIMIT = 10
DEBUG_ALLOWED_LIMITS = (5, 10, 25, 50)
DEBUG_MAX_EVENT_BYTES = 2_000_000
DEBUG_MAX_STRING_CHARACTERS = 1_000_000
_DEBUG_MANAGERS = f"{DOMAIN}.request_debug_managers"

_ACTIVE_DEBUG_TRACE: ContextVar[DebugTrace | None] = ContextVar(
    "extended_openai_request_debug_trace", default=None
)


_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "proxy_authorization",
    "x_api_key",
    "x-api-key",
}


def _iso_now() -> str:
    return dt_util.utcnow().isoformat()


def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Convert arbitrary SDK/HA values into bounded JSON-safe debug data."""
    if _depth > 20:
        return "<maximum debug serialization depth reached>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= DEBUG_MAX_STRING_CHARACTERS:
            return value
        return (
            value[:DEBUG_MAX_STRING_CHARACTERS]
            + f"\n<truncated {len(value) - DEBUG_MAX_STRING_CHARACTERS} characters>"
        )
    if isinstance(value, bytes):
        return f"<binary payload: {len(value)} bytes>"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Template):
        return value.template
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.casefold().replace("-", "_") in _SENSITIVE_KEYS:
                result[key] = "<redacted credential>"
            else:
                result[key] = _jsonable(item, _depth=_depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [_jsonable(item, _depth=_depth + 1) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(
            {item.name: getattr(value, item.name) for item in fields(value)},
            _depth=_depth + 1,
        )
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(exclude_none=True), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "as_dict"):
        try:
            return _jsonable(value.as_dict(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value), _depth=_depth + 1)
        except Exception:
            pass
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _json_characters(value: Any) -> int:
    return len(_canonical_json(value))


def _extract_usage(event: Any) -> dict[str, int] | None:
    """Extract usage while avoiding repeat normalization for hot-path dicts."""
    data = event if isinstance(event, dict) else _jsonable(event)
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    response = data.get("response")
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    def _integer(value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    input_tokens = _integer(usage.get("input_tokens")) or _integer(
        usage.get("prompt_tokens")
    )
    output_tokens = _integer(usage.get("output_tokens")) or _integer(
        usage.get("completion_tokens")
    )
    total_tokens = _integer(usage.get("total_tokens")) or input_tokens + output_tokens
    input_details = usage.get("input_tokens_details") or usage.get(
        "prompt_tokens_details"
    )
    output_details = usage.get("output_tokens_details") or usage.get(
        "completion_tokens_details"
    )
    cached = (
        _integer(input_details.get("cached_tokens"))
        if isinstance(input_details, dict)
        else 0
    )
    reasoning = (
        _integer(output_details.get("reasoning_tokens"))
        if isinstance(output_details, dict)
        else 0
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def _event_has_text(event: Any) -> bool:
    """Detect text while avoiding repeat normalization for hot-path dicts."""
    data = event if isinstance(event, dict) else _jsonable(event)
    if not isinstance(data, dict):
        return False
    event_type = str(data.get("type", ""))
    if "output_text.delta" in event_type and data.get("delta"):
        return True
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and delta.get("content"):
                return True
    return False


def _event_has_action(event: Any) -> bool:
    """Detect actions while avoiding repeat normalization for hot-path dicts."""
    data = event if isinstance(event, dict) else _jsonable(event)
    if not isinstance(data, dict):
        return False
    event_type = str(data.get("type", ""))
    if "function_call" in event_type or "web_search_call" in event_type:
        return True
    item = data.get("item")
    return isinstance(item, dict) and item.get("type") in {
        "function_call",
        "web_search_call",
    }


@dataclass(slots=True)
class DebugProviderRequest:
    """One exact provider invocation made during a debug run."""

    request_id: str
    api_surface: str
    started_at: str
    started_offset_ms: int
    request: dict[str, Any]
    metrics: dict[str, Any]
    stream_open_ms: int | None = None
    first_event_ms: int | None = None
    first_text_ms: int | None = None
    first_action_ms: int | None = None
    duration_ms: int | None = None
    successful: bool | None = None
    error_type: str | None = None
    error: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    response_events: list[Any] = field(default_factory=list)
    response_events_truncated: bool = False
    _started_monotonic: float = field(default=0.0, repr=False)
    _event_bytes: int = field(default=0, repr=False)

    def add_event(self, event: Any) -> None:
        """Retain provider events up to a hard per-request byte ceiling."""
        now_ms = int((time.monotonic() - self._started_monotonic) * 1000)
        if self.first_event_ms is None:
            self.first_event_ms = now_ms
        serialized = _jsonable(event)
        if self.first_text_ms is None and _event_has_text(serialized):
            self.first_text_ms = now_ms
        if self.first_action_ms is None and _event_has_action(serialized):
            self.first_action_ms = now_ms
        if usage := _extract_usage(serialized):
            self.usage = usage
        if self.response_events_truncated:
            return
        event_size = len(
            json.dumps(
                serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        if self._event_bytes + event_size > DEBUG_MAX_EVENT_BYTES:
            self.response_events_truncated = True
            return
        self.response_events.append(serialized)
        self._event_bytes += event_size

    def finish(self, *, successful: bool, error: BaseException | None = None) -> None:
        if self.duration_ms is None:
            self.duration_ms = int((time.monotonic() - self._started_monotonic) * 1000)
        self.successful = successful
        self.error_type = type(error).__name__ if error is not None else None
        self.error = provider_error_metadata(error) if error is not None else {}

    def as_dict(self) -> dict[str, Any]:
        data = {
            item.name: _jsonable(getattr(self, item.name))
            for item in fields(self)
            if not item.name.startswith("_")
        }
        return provider_diagnostics(self, data)


@dataclass(slots=True)
class DebugTrace:
    """Complete local troubleshooting capture for one Home Assistant turn."""

    debug_id: str
    entry_id: str
    subentry_id: str
    started_at: str
    user_input: dict[str, Any]
    incoming_conversation_id: str | None
    completed_at: str | None = None
    duration_ms: int = 0
    successful: bool | None = None
    error_type: str | None = None
    error: dict[str, Any] = field(default_factory=dict)
    usage_run_id: str | None = None
    continuity: dict[str, Any] = field(default_factory=dict)
    phases_ms: dict[str, int] = field(default_factory=dict)
    system_prompt: str | None = None
    prompt_metrics: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    provider_requests: list[DebugProviderRequest] = field(default_factory=list)
    result: Any = None
    notes: list[str] = field(default_factory=list)
    _started_monotonic: float = field(default=0.0, repr=False)

    def start_provider_request(
        self, api_surface: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> DebugProviderRequest:
        request_payload = {"args": _jsonable(args), "kwargs": _jsonable(kwargs)}
        input_value = kwargs.get("input", kwargs.get("messages"))
        tools = kwargs.get("tools")
        metrics = {
            "request_characters": _json_characters(request_payload),
            "request_sha256": _sha256(request_payload),
            "input_characters": _json_characters(input_value),
            "input_sha256": _sha256(input_value),
            "tool_count": len(tools) if isinstance(tools, list) else 0,
            "tool_characters": _json_characters(tools),
            "tools_sha256": _sha256(tools),
        }
        provider_request = DebugProviderRequest(
            request_id=uuid4().hex,
            api_surface=api_surface,
            started_at=_iso_now(),
            started_offset_ms=int((time.monotonic() - self._started_monotonic) * 1000),
            request=request_payload,
            metrics=metrics,
            _started_monotonic=time.monotonic(),
        )
        self.provider_requests.append(provider_request)
        record_provider_metrics(self, provider_request, api_surface)
        return provider_request

    def as_dict(self) -> dict[str, Any]:
        try:
            data = {
                "debug_id": self.debug_id,
                "entry_id": self.entry_id,
                "subentry_id": self.subentry_id,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "duration_ms": self.duration_ms,
                "successful": self.successful,
                "error_type": self.error_type,
                "error": _jsonable(self.error),
                "usage_run_id": self.usage_run_id,
                "user_input": _jsonable(self.user_input),
                "incoming_conversation_id": self.incoming_conversation_id,
                "continuity": _jsonable(self.continuity),
                "phases_ms": _jsonable(self.phases_ms),
                "system_prompt": self.system_prompt,
                "prompt_metrics": _jsonable(self.prompt_metrics),
                "memory": _jsonable(self.memory),
                "provider_requests": [
                    item.as_dict() for item in self.provider_requests
                ],
                "result": _jsonable(self.result),
                "notes": list(self.notes),
            }
            return trace_diagnostics(self, data)
        except Exception:
            return {
                "debug_id": str(getattr(self, "debug_id", "")),
                "diagnostics_error": "trace_serialization_failed",
            }

    def summary(self) -> dict[str, Any]:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        }
        for request in self.provider_requests:
            for key in usage:
                usage[key] += int(request.usage.get(key, 0))
        first_request = self.provider_requests[0] if self.provider_requests else None
        data = {
            "debug_id": self.debug_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "successful": self.successful,
            "error_type": self.error_type,
            "error": _jsonable(self.error),
            "usage_run_id": self.usage_run_id,
            "incoming_conversation_id": self.incoming_conversation_id,
            "resolved_conversation_id": self.continuity.get("resolved_conversation_id"),
            "continuity_resumed": self.continuity.get("resumed"),
            "continuity_mode": self.continuity.get("mode"),
            "restored_history_items": int(
                self.continuity.get("restored_history_items") or 0
            ),
            "provider_request_count": len(self.provider_requests),
            "first_text_ms": (
                first_request.first_text_ms if first_request is not None else None
            ),
            **usage,
        }
        return summary_diagnostics(self, data)


class DebugManager:
    """Per-agent in-memory, bounded debug state."""

    def __init__(self) -> None:
        self.enabled = False
        self.limit = DEBUG_DEFAULT_LIMIT
        self._runs: deque[DebugTrace] = deque(maxlen=self.limit)

    def configure(
        self, *, enabled: bool | None = None, limit: int | None = None
    ) -> None:
        if limit is not None:
            if limit not in DEBUG_ALLOWED_LIMITS:
                raise ValueError(
                    "Debug run limit must be one of "
                    + ", ".join(str(item) for item in DEBUG_ALLOWED_LIMITS)
                )
            if limit != self.limit:
                self.limit = limit
                self._runs = deque(list(self._runs)[-limit:], maxlen=limit)
        if enabled is not None:
            self.enabled = enabled

    def begin(
        self,
        *,
        entry_id: str,
        subentry_id: str,
        user_input: Any,
        incoming_conversation_id: str | None,
    ) -> DebugTrace:
        return DebugTrace(
            debug_id=uuid4().hex,
            entry_id=entry_id,
            subentry_id=subentry_id,
            started_at=_iso_now(),
            user_input=_jsonable(user_input),
            incoming_conversation_id=incoming_conversation_id,
            _started_monotonic=time.monotonic(),
        )

    def finish(
        self,
        trace: DebugTrace,
        *,
        successful: bool,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        trace.completed_at = _iso_now()
        trace.duration_ms = int((time.monotonic() - trace._started_monotonic) * 1000)
        trace.successful = successful
        if error is not None:
            trace.error_type = type(error).__name__
            trace.error = provider_error_metadata(error)
        if result is not None:
            trace.result = _jsonable(result)
        self._runs.append(trace)

    def clear(self) -> int:
        count = len(self._runs)
        self._runs.clear()
        return count

    def summaries(self) -> list[dict[str, Any]]:
        return [trace.summary() for trace in reversed(self._runs)]

    def get(self, debug_id: str) -> dict[str, Any] | None:
        trace = next((item for item in self._runs if item.debug_id == debug_id), None)
        return trace.as_dict() if trace is not None else None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "limit": self.limit,
            "count": len(self._runs),
            "allowed_limits": list(DEBUG_ALLOWED_LIMITS),
            "volatile": True,
        }


def get_debug_manager(hass: Any, entry_id: str, subentry_id: str) -> DebugManager:
    managers = hass.data.setdefault(_DEBUG_MANAGERS, {})
    key = (entry_id, subentry_id)
    if key not in managers:
        managers[key] = DebugManager()
    return cast(DebugManager, managers[key])


def current_debug_trace() -> DebugTrace | None:
    return _ACTIVE_DEBUG_TRACE.get()


def record_current_run_failure(error_type: str) -> None:
    """Observe the same handled terminal failure signal used by Usage."""
    trace = current_debug_trace()
    if trace is not None and trace.error_type is None:
        trace.error_type = error_type[:128]


def record_current_provider_failure(error: BaseException) -> None:
    """Attach a provider failure to the active opt-in debug trace, if any."""
    trace = current_debug_trace()
    if trace is None:
        return
    trace.error_type = type(error).__name__
    trace.error = provider_error_metadata(error)
    if trace.provider_requests:
        trace.provider_requests[-1].finish(successful=False, error=error)


class _DebugAsyncStream:
    """Transparent async-stream wrapper that records provider timing/events."""

    def __init__(self, delegate: Any, request: DebugProviderRequest) -> None:
        self._delegate = delegate
        self._iterator: AsyncIterator[Any] = delegate.__aiter__()
        self._request = request
        self._finished = False

    def __aiter__(self) -> _DebugAsyncStream:
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._finish(True)
            raise
        except BaseException as err:
            self._finish(False, err)
            raise
        self._request.add_event(event)
        return event

    async def __aenter__(self) -> _DebugAsyncStream:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if hasattr(self._delegate, "__aexit__"):
            await self._delegate.__aexit__(exc_type, exc, tb)
        self._finish(exc is None, exc)

    async def close(self) -> Any:
        try:
            close = getattr(self._delegate, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    return await result
                return result
            return None
        finally:
            self._finish(True)

    def _finish(self, successful: bool, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._request.finish(successful=successful, error=error)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _DebugEndpointProxy:
    def __init__(self, delegate: Any, api_surface: str) -> None:
        self._delegate = delegate
        self._api_surface = api_surface

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        trace = current_debug_trace()
        request = (
            trace.start_provider_request(self._api_surface, args, kwargs)
            if trace is not None
            else None
        )
        started = time.monotonic()
        try:
            result = await self._delegate.create(*args, **kwargs)
        except BaseException as err:
            if request is not None:
                request.finish(successful=False, error=err)
            raise
        if request is not None:
            request.stream_open_ms = int((time.monotonic() - started) * 1000)
            if hasattr(result, "__aiter__"):
                return _DebugAsyncStream(result, request)
            request.add_event(result)
            request.finish(successful=True)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _DebugChatProxy:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.completions = _DebugEndpointProxy(delegate.completions, "chat.completions")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class DebugOpenAIClientProxy:
    """Transparent OpenAI client proxy that captures exact non-secret call kwargs."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.responses = _DebugEndpointProxy(delegate.responses, "responses")
        self.chat = _DebugChatProxy(delegate.chat)
        self.embeddings = _DebugEndpointProxy(delegate.embeddings, "embeddings")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


@contextmanager
def conversation_debug_trace(
    agent: Any, user_input: Any
) -> Iterator[DebugTrace | None]:
    """Capture one opt-in request, preserving caller trace state on every exit.

    The conversation owner assigns the completed result to the yielded trace.
    Finalization happens after the inner request scopes have been restored.
    """
    manager = get_debug_manager(
        agent.hass, agent.entry.entry_id, agent.subentry.subentry_id
    )
    if not manager.enabled:
        yield None
        return
    trace = manager.begin(
        entry_id=agent.entry.entry_id,
        subentry_id=agent.subentry.subentry_id,
        user_input=user_input,
        incoming_conversation_id=user_input.conversation_id,
    )
    token = _ACTIVE_DEBUG_TRACE.set(trace)
    try:
        yield trace
    except BaseException as err:
        manager.finish(trace, successful=False, error=err)
        raise
    else:
        manager.finish(trace, successful=trace.error_type is None, result=trace.result)
    finally:
        _ACTIVE_DEBUG_TRACE.reset(token)


def record_memory_retrieval(kind: str, started: float, records: list[Any]) -> None:
    """Record the completed retrieval at its explicit runtime boundary."""
    trace = current_debug_trace()
    if trace is not None:
        trace.phases_ms[f"{kind}_memory_retrieval"] = int(
            (time.monotonic() - started) * 1000
        )
        trace.memory[f"{kind}_count"] = len(records)
        trace.memory[f"{kind}_records"] = _jsonable(records)


@contextmanager
def model_path_timing(entity: Any) -> Iterator[None]:
    """Measure the owning model operation, excluding deferred speech regex."""
    trace = current_debug_trace()
    started = time.monotonic()
    if trace is not None and entity._usage is not None:
        run = entity._usage.current_run()
        if run is not None:
            trace.usage_run_id = run.run_id
    try:
        yield
    finally:
        if trace is not None:
            trace.phases_ms["model_path_total"] = int(
                (time.monotonic() - started) * 1000
            )


def record_system_prompt(prompt: str, started: float) -> None:
    trace = current_debug_trace()
    if trace is not None:
        trace.phases_ms["system_prompt_render"] = int(
            (time.monotonic() - started) * 1000
        )
        trace.system_prompt = prompt
        trace.prompt_metrics = {
            "characters": len(prompt),
            "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }


def record_continuity_resolution(
    result: Any,
    started: float,
    mode: str,
    scope: Any,
    device_id: str | None,
    incoming_conversation_id: str | None,
    timeout_minutes: int,
    namespace: str | None,
) -> None:
    trace = current_debug_trace()
    if trace is not None:
        trace.phases_ms["continuity_resolution"] = int(
            (time.monotonic() - started) * 1000
        )
        trace.continuity = {
            "mode": mode,
            "incoming_conversation_id": incoming_conversation_id,
            "resolved_conversation_id": result.conversation_id,
            "key": result.key,
            "resumed": result.resumed,
            "restored_history_items": len(result.history),
            "timeout_minutes": timeout_minutes,
            "namespace": namespace,
            "device_id": device_id,
            "scope": _jsonable(scope),
        }
