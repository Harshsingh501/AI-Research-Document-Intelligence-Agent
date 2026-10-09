"""
RAG orchestrator: question  →  retrieve  →  (threshold)  →  prompt  →  structured answer.

The prompt does three jobs:
  1. GROUND the model: "answer only from the context below." Without this,
     a 7B model happily blends in its own training knowledge, and you can no
     longer tell which parts of the answer came from your documents.
  2. CITE: each context block is numbered [1], [2], ... and the model is asked
     to reference those numbers. The numbers map back to the `sources` list we
     return, so the client can show "page 7 of paper.pdf" next to each claim.
  3. ABSTAIN: tell the model to say it doesn't know when the context doesn't
     cover the question. Saying "I don't know" is a feature, not a failure.

Phase 7 additions:
  - RELEVANCE THRESHOLD. If even the best chunk scores below the threshold,
    the documents almost certainly do not cover the question. We skip the LLM
    entirely: a 60-second "I don't know" becomes a 1-second one.
  - STRUCTURED OUTPUT. The model must return JSON (answer, citations,
    confidence) so we can validate it. See structured.py.

Phase 10: STREAMING. `stream_answer` yields the answer text as it is
generated. Same retrieval, same prompt, same validation at the end — only the
LLM call differs. Both entry points share `_prepare` and `_finalise` so they
cannot drift apart.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.llm import chat, chat_stream
from app.services.retrieval import RetrievedChunk, retrieve
from app.services.structured import (
    ANSWER_SCHEMA,
    AnswerStreamExtractor,
    Confidence,
    parse_structured,
    validate_citations,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a careful research assistant. Answer the user's question using ONLY the
numbered context passages provided. Rules:
- Cite passages inline using their numbers, e.g. "... attention heads [2]."
- If the context does not contain the answer, say so plainly. Do not guess and
  do not use outside knowledge.
- Be concise and factual. Do not repeat the question.

Respond as JSON with exactly these fields:
  "answer":     your answer text, with [n] citations inline
  "citations":  list of the passage numbers you used, e.g. [2, 4]
  "confidence": "high" if the passages state the answer directly,
                "medium" if you had to infer it, "low" if weakly supported"""

NOT_COVERED = "The uploaded documents do not appear to cover this question."


@dataclass
class RagResult:
    answer: str
    sources: list[RetrievedChunk]
    confidence: Confidence = "none"
    citations: list[int] = field(default_factory=list)
    short_circuited: bool = False  # True when we never called the LLM


def build_context(chunks: list[RetrievedChunk], start: int = 1) -> str:
    """Format retrieved chunks as numbered blocks the model can cite.

    `start` lets the agent keep numbering unique across several searches.
    """
    blocks = []
    for i, c in enumerate(chunks, start=start):
        page = f", page {c.page_number}" if c.page_number is not None else ""
        blocks.append(f"[{i}] ({c.filename}{page})\n{c.content}")
    return "\n\n".join(blocks)


# --- Shared pipeline pieces ---------------------------------------------------


@dataclass
class _Prepared:
    chunks: list[RetrievedChunk]
    messages: list[dict]
    early: RagResult | None  # set when we can answer without the LLM


async def _prepare(
    db: AsyncSession,
    question: str,
    top_k: int,
    document_id: uuid.UUID | None,
    history: list[dict] | None,
) -> _Prepared:
    chunks = await retrieve(db, question, top_k=top_k, document_id=document_id)

    if not chunks:
        msg = (
            "No indexed content found for that document. Check its status is 'ready'."
            if document_id is not None
            else "I don't have any documents to search. Upload a PDF first."
        )
        return _Prepared(chunks=[], messages=[], early=RagResult(answer=msg, sources=[], short_circuited=True))

    best = max(c.score for c in chunks)
    logger.info("rag: top-%d scores %s", len(chunks), [c.score for c in chunks])

    # The threshold protects against answering from irrelevant passages when
    # searching EVERYTHING. When the user scopes the question to one document
    # they have already told us where the answer lives; vague prompts like
    # "give summary" score low against any passage yet are perfectly
    # answerable, so we let the model judge instead of refusing.
    if document_id is None and best < settings.RELEVANCE_THRESHOLD:
        # Nothing relevant. Return the weak matches anyway so a UI can show
        # "closest we found was page 3 at 0.46" — useful for debugging.
        logger.info("rag: best score %.3f < threshold %.2f, skipping LLM", best, settings.RELEVANCE_THRESHOLD)
        early = RagResult(answer=NOT_COVERED, sources=chunks, confidence="none", short_circuited=True)
        return _Prepared(chunks=chunks, messages=[], early=early)

    user_prompt = f"Context:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    # System prompt, then earlier turns, then the current question with its
    # context. Earlier turns go BEFORE the context so the freshest, most
    # relevant text sits closest to where the model starts generating.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": user_prompt})
    return _Prepared(chunks=chunks, messages=messages, early=None)


def _finalise(raw: str, chunks: list[RetrievedChunk]) -> RagResult:
    parsed = parse_structured(raw)
    if parsed is None:
        # Constrained decoding should make this impossible, but never trust
        # "should". Degrade gracefully: show the raw text, flag low confidence.
        text, valid, _ = validate_citations(raw.strip(), [], len(chunks))
        return RagResult(answer=text, sources=chunks, confidence="low", citations=valid)

    text, valid, _ = validate_citations(parsed.answer, parsed.citations, len(chunks))
    confidence: Confidence = parsed.confidence
    if not valid and confidence != "low":
        # An answer that cites nothing cannot be "high" confidence, whatever
        # the model claims. (An honest "not covered" answer legitimately has
        # no citations; that is fine at "low".)
        confidence = "low"
    return RagResult(answer=text, sources=chunks, confidence=confidence, citations=valid)


# --- Entry points ---------------------------------------------------------------


async def answer_question(
    db: AsyncSession,
    question: str,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
    history: list[dict] | None = None,
) -> RagResult:
    """
    `question` should already be standalone (see history.condense_question);
    it is what gets embedded for retrieval. `history` is the earlier turns,
    included in the prompt so the model can answer in context.
    """
    prep = await _prepare(db, question, top_k, document_id, history)
    if prep.early is not None:
        return prep.early
    raw = await chat(prep.messages, format=ANSWER_SCHEMA)
    return _finalise(raw, prep.chunks)


async def stream_answer(
    db: AsyncSession,
    question: str,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[tuple[str, object]]:
    """
    Streaming twin of `answer_question`. Yields (kind, payload) events:
        ("sources", list[RetrievedChunk])   as soon as retrieval is done
        ("token",   str)                    pieces of the answer text, in order
        ("result",  RagResult)              always last, fully validated

    The model streams JSON; AnswerStreamExtractor peels the "answer" string
    out of it incrementally so the client sees prose, not braces.
    """
    prep = await _prepare(db, question, top_k, document_id, history)
    yield ("sources", prep.chunks)
    if prep.early is not None:
        yield ("result", prep.early)
        return

    extractor = AnswerStreamExtractor()
    raw: list[str] = []
    async for delta in chat_stream(prep.messages, format=ANSWER_SCHEMA):
        raw.append(delta)
        text = extractor.feed(delta)
        if text:
            yield ("token", text)
    yield ("result", _finalise("".join(raw), prep.chunks))
