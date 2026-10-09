"""
Thin async wrapper around Ollama's chat endpoint.

Same reasoning as embeddings.py: every LLM call in the app goes through this
one function, so swapping providers (or adding retries, timeouts, logging)
happens in one place.

Ollama's chat API takes OpenAI-style messages:
    [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
"""

import logging
import time
from collections.abc import AsyncIterator

import ollama
from ollama import Message

from app.core.config import settings

logger = logging.getLogger(__name__)

# timeout is passed through to httpx. Without it a hung Ollama hangs the request forever.
_client = ollama.AsyncClient(host=settings.OLLAMA_BASE_URL, timeout=settings.LLM_TIMEOUT_SECONDS)


async def chat(
    messages: list[dict],
    temperature: float = 0.1,
    format: dict | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a conversation to the LLM and return the assistant's reply text.

    temperature=0.1: RAG answers should be grounded, not creative. Low
    temperature makes the model pick the most likely tokens, which keeps it
    close to the retrieved text instead of improvising.

    format: optional JSON schema. When given, Ollama constrains generation so
    the reply is guaranteed to be JSON matching the schema.
    """
    options: dict = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens  # Ollama's name for the output-length cap
    started = time.perf_counter()
    response = await _client.chat(
        model=settings.LLM_MODEL,
        messages=messages,
        options=options,
        format=format,
    )
    elapsed = time.perf_counter() - started
    logger.info("llm: %s answered in %.1fs", settings.LLM_MODEL, elapsed)
    return response["message"]["content"]


async def chat_with_tools(messages: list, tools: list[dict], temperature: float = 0.0) -> Message:
    """
    Like `chat`, but the model may respond with tool calls instead of text.

    Returns the whole Message (not just text) because the caller needs
    `message.tool_calls` to see what the model wants to run. Temperature 0:
    tool selection should be deterministic, not creative.
    """
    started = time.perf_counter()
    response = await _client.chat(
        model=settings.LLM_MODEL,
        messages=messages,
        tools=tools,
        options={"temperature": temperature},
    )
    elapsed = time.perf_counter() - started
    n_calls = len(response.message.tool_calls or [])
    logger.info("llm: %s responded in %.1fs with %d tool call(s)", settings.LLM_MODEL, elapsed, n_calls)
    return response.message


async def chat_stream(
    messages: list[dict],
    temperature: float = 0.1,
    format: dict | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """
    Like `chat`, but yields the reply piece by piece as Ollama generates it.

    Why stream?  Total latency is the same, but PERCEIVED latency drops from
    "a minute of nothing" to "words appearing after a few seconds". For a
    CPU-bound local model that is the single biggest UX improvement available.
    """
    options: dict = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    started = time.perf_counter()
    stream = await _client.chat(
        model=settings.LLM_MODEL,
        messages=messages,
        options=options,
        format=format,
        stream=True,
    )
    pieces = 0
    async for part in stream:
        delta = part["message"]["content"]
        if delta:
            pieces += 1
            yield delta
    logger.info("llm: %s streamed %d pieces in %.1fs", settings.LLM_MODEL, pieces, time.perf_counter() - started)
