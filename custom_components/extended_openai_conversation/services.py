import json
import random

import voluptuous as vol
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    SupportsResponse,
    ServiceResponse,
)
from homeassistant.exceptions import (
    HomeAssistantError,
)
from homeassistant.helpers import (
    config_validation as cv,
    selector,
)

from .const import (
    DOMAIN,
    DATA_AGENT,
    DATA_STORAGE,
)

RANDOM_EMBEDDING = [random.random() for _ in range(1536)]

SERVICE_INDEX_INTO_PINECONE = "index_into_pinecone"
SERVICE_QUERY_PINECONE = "query_pinecone"
SERVICE_DELETE_FROM_PINECONE = "delete_from_pinecone"


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up services for the extended_openai_conversation component."""

    async def index_into_pinecone(call: ServiceCall) -> ServiceResponse:
        """Upsert Pinecone index."""

        entry_id = call.data["entry"]
        agent = hass.data[DOMAIN][entry_id][DATA_AGENT]
        exposed_entities = agent.get_exposed_entities()
        storage = hass.data[DOMAIN][entry_id].get(DATA_STORAGE)
        if storage is None:
            raise HomeAssistantError("Pinecone is not configured.")

        vectors = []
        for exposed_entity in exposed_entities:
            metadata = {
                key: exposed_entity[key]
                for key in exposed_entity
                if key not in {"state"}
            }
            embedding_result = await agent.embeddings(
                json.dumps(metadata, ensure_ascii=False)
            )

            vectors.append(
                {
                    "id": metadata["entity_id"],
                    "values": embedding_result["data"][0]["embedding"],
                    "metadata": metadata,
                }
            )

        return await storage.save(vectors)

    hass.services.async_register(
        DOMAIN,
        SERVICE_INDEX_INTO_PINECONE,
        index_into_pinecone,
        schema=vol.Schema(
            {
                vol.Required("entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )

    async def query_pinecone(call: ServiceCall) -> ServiceResponse:
        """Query from Pinecone."""

        entry_id = call.data["entry"]
        agent = hass.data[DOMAIN][entry_id][DATA_AGENT]
        prompt = call.data.get("prompt")
        storage = hass.data[DOMAIN][entry_id].get(DATA_STORAGE)
        if storage is None:
            raise HomeAssistantError("Pinecone is not configured.")

        if prompt:
            embedding_result = await agent.embeddings(prompt)
            embedding = embedding_result["data"][0]["embedding"]
        else:
            embedding = RANDOM_EMBEDDING

        result = await storage.query(
            topK=call.data.get("top_k", 100),
            vector=embedding,
        )

        score_threshold = call.data.get("score_threshold")
        if score_threshold:
            result["matches"] = [
                match
                for match in result["matches"]
                if match["score"] >= score_threshold
            ]
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_QUERY_PINECONE,
        query_pinecone,
        schema=vol.Schema(
            {
                vol.Optional("prompt"): cv.string,
                vol.Optional("top_k"): cv.Number,
                vol.Optional("score_threshold"): cv.Number,
                vol.Required("entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
            }
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )

    async def delete_from_pinecone(call: ServiceCall) -> ServiceResponse:
        """Query from Pinecone."""

        entry_id = call.data["entry"]
        storage: PineconeStorage = hass.data[DOMAIN][entry_id].get(DATA_STORAGE)
        if storage is None:
            raise HomeAssistantError("Pinecone is not configured.")

        entity_id = call.data.get("entity_id")
        if not entity_id:
            raise HomeAssistantError("entity_id is required")

        return await storage.remove(ids=entity_id)

    hass.services.async_register(
        DOMAIN,
        SERVICE_DELETE_FROM_PINECONE,
        delete_from_pinecone,
        schema=vol.Schema(
            {
                # vol.Required("entity_id"): cv.string,
                vol.Required("entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
            }
        ).extend(cv.ENTITY_SERVICE_FIELDS),
        supports_response=SupportsResponse.OPTIONAL,
    )
