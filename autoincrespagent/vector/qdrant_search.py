"""Qdrant async search helper shared across agents.

All agents that need semantic search import from here.
Gracefully returns an empty list if Qdrant is unreachable.
"""

import logging
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient

from autoincrespagent.config import settings

logger = logging.getLogger(__name__)


def build_qdrant_client() -> AsyncQdrantClient:
    """Return an async Qdrant client using settings."""
    return AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def build_embeddings() -> OllamaEmbeddings:
    """Return an Ollama embeddings instance using nomic-embed-text (768-dim)."""
    return OllamaEmbeddings(model="nomic-embed-text", base_url=settings.ollama_base_url)


async def search_collection(
    client: AsyncQdrantClient,
    embeddings: OllamaEmbeddings,
    collection: str,
    query: str,
    limit: int = 5,
    score_threshold: float = 0.3,
) -> list[dict]:
    """Embed a query and return top-k matching payloads from a Qdrant collection.

    Args:
        client:           AsyncQdrantClient instance.
        embeddings:       OllamaEmbeddings for query vectorisation.
        collection:       Qdrant collection name.
        query:            Natural language query string.
        limit:            Maximum number of results.
        score_threshold:  Minimum cosine similarity score (0–1).

    Returns:
        List of dicts with keys ``score`` and ``payload``.
        Returns empty list on any error (Qdrant unavailable, empty collection, etc.).
    """
    try:
        vector = await embeddings.aembed_query(query)
        response = await client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [{"score": round(r.score, 4), "payload": r.payload or {}} for r in response.points]
    except Exception as exc:
        logger.warning(
            f"qdrant_search: search failed on '{collection}' — {exc}. "
            "Continuing without vector search results."
        )
        return []
