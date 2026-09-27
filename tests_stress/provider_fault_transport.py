"""Reusable SDK-wire fault transport for nightly public Assist campaigns.

Only the outbound HTTP send is replaced. EOAI's real agent, OpenAI SDK request
serialization, streaming parser, Home Assistant services and stores remain live.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
import json
import random
from typing import Any, Literal

import httpx

from tests_real_ha.test_provider_wire_e2e import _raw_client


@dataclass(frozen=True)
class WireStep:
    kind: Literal["sse", "http", "transport", "stream_break"]
    body: bytes = b""
    status: int = 200
    error: str = ""
    delay: float = 0.0


class _BrokenStream(httpx.AsyncByteStream):
    def __init__(self, prefix: bytes, request: httpx.Request) -> None:
        self.prefix = prefix
        self.request = request

    async def __aiter__(self):
        yield self.prefix
        raise httpx.ReadError("provider stream disconnected", request=self.request)


def split_valid_sse(
    payload: bytes,
    *,
    positions: list[int] | None = None,
    sizes: list[int] | None = None,
    seed: int | None = None,
) -> list[bytes]:
    """Partition unchanged SSE bytes at explicit offsets or seeded chunk sizes."""
    if sum(option is not None for option in (positions, sizes, seed)) != 1:
        raise ValueError("Choose exactly one SSE chunking strategy")
    if seed is not None:
        rng = random.Random(seed)
        sizes = [rng.choice((1, 2, 3, 5, 13, 64)) for _ in range(len(payload))]
    if sizes is not None:
        if not sizes or any(size < 1 for size in sizes):
            raise ValueError("SSE chunk sizes must be positive")
        offsets = []
        offset = 0
        for size in sizes:
            offset += size
            if offset >= len(payload):
                break
            offsets.append(offset)
        positions = offsets
    assert positions is not None
    if positions != sorted(set(positions)) or any(
        position <= 0 or position >= len(payload) for position in positions
    ):
        raise ValueError("SSE split positions must be unique interior offsets")
    boundaries = [0, *positions, len(payload)]
    chunks = [payload[start:end] for start, end in pairwise(boundaries)]
    assert b"".join(chunks) == payload
    return chunks


class GatedSSEStream(httpx.AsyncByteStream):
    """Expose deterministic content-delivered and close boundaries to tests."""

    def __init__(self, chunks: list[bytes], *, gate_after: int | None = None) -> None:
        self.chunks = chunks
        self.gate_after = gate_after
        self.delivered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.yielded = 0

    async def __aiter__(self):
        try:
            for chunk in self.chunks:
                if self.gate_after is not None and self.yielded == self.gate_after:
                    self.delivered.set()
                    await self.release.wait()
                self.yielded += 1
                yield chunk
            self.delivered.set()
        finally:
            self.closed.set()

    async def aclose(self) -> None:
        self.closed.set()
        self.release.set()


class ProviderFaultTransport:
    """Consume deterministic wire steps and retain every serialized request."""

    def __init__(self, steps: list[WireStep]) -> None:
        self.steps = deque(steps)
        self.requests: list[dict[str, Any]] = []

    def install(self, monkeypatch: Any, agent: Any) -> None:
        client = _raw_client(agent)
        # Disable SDK automatic retries for phase-specific outcomes. The test
        # explicitly sends another public Assist turn to prove recovery.
        monkeypatch.setattr(client, "max_retries", 0)
        monkeypatch.setattr(client._client, "send", self.send)

    async def send(
        self, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        del args, kwargs
        self.requests.append(
            {"path": request.url.path, "body": json.loads(request.content.decode())}
        )
        assert self.steps, "Unexpected extra provider request"
        step = self.steps.popleft()
        if step.delay:
            await asyncio.sleep(step.delay)
        if step.kind == "transport":
            errors: dict[str, type[httpx.RequestError]] = {
                "dns": httpx.ConnectError,
                "refused": httpx.ConnectError,
                "connect_timeout": httpx.ConnectTimeout,
                "read_timeout": httpx.ReadTimeout,
                "tls": httpx.ConnectError,
                "before_headers": httpx.RemoteProtocolError,
            }
            detail = {
                "dns": "Name or service not known",
                "refused": "Connection refused",
                "connect_timeout": "Connection timed out",
                "read_timeout": "Timed out waiting for provider headers",
                "tls": "SSL certificate verify failed",
                "before_headers": "Server disconnected before response headers",
            }[step.error]
            raise errors[step.error](detail, request=request)
        if step.kind == "http":
            return httpx.Response(
                step.status,
                headers={"content-type": "application/json"},
                json={
                    "error": {
                        "message": f"seeded provider HTTP {step.status}",
                        "type": "provider_fault_acceptance",
                        "code": f"fault_{step.status}",
                    }
                },
                request=request,
            )
        if step.kind == "stream_break":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_BrokenStream(step.body, request),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=step.body,
            request=request,
        )
