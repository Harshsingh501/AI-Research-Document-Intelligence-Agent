"""
Pydantic schemas for /ask.

AskRequest validates input BEFORE any expensive work happens: an empty
question or top_k=500 is rejected with a 422 instantly, not after a 20-second
LLM call.
"""

import uuid

from pydantic import BaseModel, Field

from app.services.retrieval import RetrievedChunk
from app.services.structured import Confidence

EXCERPT_CHARS = 300  # enough to show the user *why* a source was picked


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["What is multi-head attention?"])
    top_k: int = Field(default=5, ge=1, le=20, description="How many chunks to retrieve")
    document_id: uuid.UUID | None = Field(default=None, description="Restrict search to one document")


class SourceOut(BaseModel):
    """One cited passage. `index` matches the [n] markers in the answer."""

    index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    score: float
    excerpt: str


def sources_from_chunks(chunks: list[RetrievedChunk]) -> list["SourceOut"]:
    """RetrievedChunk → SourceOut, numbered from 1 to match [n] citations."""
    return [
        SourceOut(
            index=i,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            filename=c.filename,
            page_number=c.page_number,
            score=c.score,
            excerpt=c.content[:EXCERPT_CHARS] + ("…" if len(c.content) > EXCERPT_CHARS else ""),
        )
        for i, c in enumerate(chunks, start=1)
    ]


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    citations: list[int]          # indices into `sources` the answer actually relies on
    confidence: Confidence        # "none" = documents don't cover it / nothing to search
    short_circuited: bool         # True when the LLM was never called (fast path)
    model: str
