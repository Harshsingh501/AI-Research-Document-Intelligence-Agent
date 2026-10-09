"""
Thin async wrapper around Ollama's embedding endpoint.

Two reasons this is its own module:
  1. Swappability — moving to OpenAI/Voyage later means editing ONE file.
  2. Batching — one HTTP call for 32 texts is far faster than 32 calls.
"""

import ollama

from app.core.config import settings

_client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL, timeout=settings.LLM_TIMEOUT_SECONDS)

BATCH_SIZE = 32


async def _embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = await _client.embed(model=settings.EMBEDDING_MODEL, input=batch)
        vectors.extend(response["embeddings"])
    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed document passages, preserving order.

    Asymmetric prefixes: passages get "search_document: ", queries get
    "search_query: ". nomic-embed-text learned that a short question and a
    long passage are different kinds of text; telling it which is which
    raised on-topic similarity by ~0.1 in our measurements. Changing the
    prefix invalidates stored vectors — see scripts/reembed.py.
    """
    return await _embed([settings.EMBEDDING_DOCUMENT_PREFIX + t for t in texts])


async def embed_query(text: str) -> list[float]:
    """Embed a single search query (with the query prefix)."""
    return (await _embed([settings.EMBEDDING_QUERY_PREFIX + text]))[0]
