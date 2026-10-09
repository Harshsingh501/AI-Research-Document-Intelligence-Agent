"""
Split text into overlapping chunks for embedding.

Why chunk at all?  Embedding models have an input limit, and retrieval works
best on passages that cover ONE idea — a whole page mixes many ideas into one
blurry vector.

Why overlap?  A sentence straddling a boundary would otherwise be cut in half,
and neither half embeds well. Overlap guarantees every sentence appears whole
in at least one chunk.

Why break on sentences?  Pure character-count splits cut words and sentences
mid-way. We aim for `chunk_size` but back up to the nearest sentence end.
"""

import re
from dataclasses import dataclass

from app.core.config import settings

# A sentence ends at . ! or ? followed by whitespace. Good enough for prose.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TextChunk:
    index: int
    content: str
    page_number: int | None


def chunk_text(
    text: str,
    page_number: int | None = None,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
    start_index: int = 0,
) -> list[TextChunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = _SENTENCE_END.split(text)
    chunks: list[TextChunk] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        # Would adding this sentence overflow the chunk? Then close the current one.
        if current and current_len + len(sentence) + 1 > chunk_size:
            chunks.append(TextChunk(start_index + len(chunks), " ".join(current), page_number))
            # Start the next chunk with the tail of this one (the overlap).
            # Long sentences (common in papers) may not fit the overlap budget;
            # still carry the last one over unless it would dominate the chunk.
            tail = _tail_sentences(current, overlap)
            if not tail and len(current[-1]) < chunk_size // 2:
                tail = [current[-1]]
            current = tail
            current_len = sum(len(s) + 1 for s in current)
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(TextChunk(start_index + len(chunks), " ".join(current), page_number))
    return chunks


def _tail_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """Return the last sentences whose total length fits within max_chars."""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) + 1 > max_chars:
            break
        tail.insert(0, sentence)
        total += len(sentence) + 1
    return tail
