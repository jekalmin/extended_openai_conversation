"""Embedding generation for memory system."""

from __future__ import annotations

import logging

from openai import AsyncClient

_LOGGER = logging.getLogger(__name__)


async def generate_embedding(
    client: AsyncClient,
    text: str,
    model: str,
) -> list[float]:
    """Generate an embedding for a single text."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception:
            if attempt == max_retries - 1:
                raise
            import asyncio

            wait_time = 2**attempt
            _LOGGER.warning(
                "Embedding API call failed (attempt %d/%d), retrying in %ds",
                attempt + 1,
                max_retries,
                wait_time,
            )
            await asyncio.sleep(wait_time)
    return []  # unreachable but satisfies type checker
