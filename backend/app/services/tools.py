"""
Tools the agent can call.

Plain RAG (Phase 5) always does the same thing: retrieve, then answer. An
*agent* instead gets a menu of tools and decides, per question, which to use
and in what order. "Which papers do I have?" needs list_documents, not a
vector search. "Summarise page 3" needs the actual page, not the top-5
similar chunks. "What BLEU score did they get?" needs search_documents.

Each tool has two halves:
  1. A JSON-schema DESCRIPTION the model reads to decide when and how to
     call it. Descriptions matter enormously for a 7B model: vague text means
     wrong tool or wrong arguments.
  2. A Python IMPLEMENTATION that runs when the model asks for it. Results
     go back to the model as plain text, since that is all it can read.

Every implementation returns a string, even on error. Returning an error
message lets the model recover ("that document id doesn't exist, let me list
documents first"); raising would kill the whole request.
"""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Chunk, Document, DocumentStatus
from app.services.rag import build_context
from app.services.retrieval import RetrievedChunk, retrieve

# --- Descriptions (what the model sees) -------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search over the user's uploaded documents. Use this to find "
                "passages that answer a factual question. Returns numbered passages "
                "with filename and page number. Call it again with a rephrased query "
                "if the first results do not answer the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short, specific search query (not the full user question).",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Optional UUID to restrict the search to one document.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List all uploaded documents with their id, filename, page count and "
                "status. Use this when the user asks what documents exist, or when "
                "you need a document_id for another tool."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": (
                "Return the full text of one page of a document. Use this when the "
                "user refers to a specific page, or to read around a passage found "
                "by search_documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "UUID of the document."},
                    "page_number": {"type": "integer", "description": "1-based page number."},
                },
                "required": ["document_id", "page_number"],
            },
        },
    },
]

TOOL_NAMES = {spec["function"]["name"] for spec in TOOL_SPECS}


# --- Implementations ---------------------------------------------------------


@dataclass
class ToolContext:
    """
    Shared state for one agent run. Tools append any passages they surface
    so the final answer can cite them, even across several search calls.
    """

    db: AsyncSession
    sources: list[RetrievedChunk] = field(default_factory=list)

    def add_sources(self, chunks: list[RetrievedChunk]) -> None:
        seen = {c.chunk_id for c in self.sources}
        self.sources.extend(c for c in chunks if c.chunk_id not in seen)


def _parse_uuid(value: object) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


async def search_documents(ctx: ToolContext, query: str = "", document_id: str | None = None, **_: object) -> str:
    if not query.strip():
        return "Error: 'query' must be a non-empty string."
    doc_uuid = None
    if document_id:
        doc_uuid = _parse_uuid(document_id)
        if doc_uuid is None:
            return f"Error: '{document_id}' is not a valid document_id. Call list_documents to get valid ids."

    chunks = await retrieve(ctx.db, query, top_k=5, document_id=doc_uuid)
    if not chunks:
        return "No passages found. There may be no indexed documents, or the document_id is wrong."
    best = max(c.score for c in chunks)
    if doc_uuid is None and best < settings.RELEVANCE_THRESHOLD:  # same rule as rag.py
        # Tell the model plainly so it abstains instead of stretching weak matches.
        return (
            f"No relevant passages found (best similarity {best:.2f}, below the "
            f"{settings.RELEVANCE_THRESHOLD:.2f} relevance threshold). The documents probably do not "
            "cover this topic. Try a differently worded query once, or tell the user it is not covered."
        )

    # Number passages continuing from what the agent has already seen, so [n]
    # citations stay unique across multiple searches in one run.
    offset = len(ctx.sources)
    ctx.add_sources(chunks)
    numbered = ctx.sources[offset:]  # the genuinely new ones
    return build_context(numbered, start=offset + 1) if numbered else "Only passages you have already seen were found."


async def list_documents(ctx: ToolContext, **_: object) -> str:
    docs = (await ctx.db.scalars(select(Document).order_by(Document.created_at))).all()
    if not docs:
        return "No documents have been uploaded."
    lines = [
        f"- id={d.id} filename={d.filename!r} pages={d.page_count} status={d.status.value}"
        for d in docs
    ]
    return "\n".join(lines)


async def read_page(ctx: ToolContext, document_id: str = "", page_number: object = None, **_: object) -> str:
    doc_uuid = _parse_uuid(document_id)
    if doc_uuid is None:
        return f"Error: '{document_id}' is not a valid document_id. Call list_documents first."
    try:
        page = int(page_number)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "Error: 'page_number' must be an integer."

    doc = await ctx.db.get(Document, doc_uuid)
    if doc is None:
        return "Error: document not found. Call list_documents to see valid ids."
    if doc.status != DocumentStatus.READY:
        return f"Error: document is not ready (status={doc.status.value})."

    chunks = (
        await ctx.db.scalars(
            select(Chunk)
            .where(Chunk.document_id == doc_uuid, Chunk.page_number == page)
            .order_by(Chunk.chunk_index)
        )
    ).all()
    if not chunks:
        return f"Page {page} has no text (document has {doc.page_count} pages)."

    # Chunks overlap, so joining them repeats a little text at the seams.
    # Good enough for reading; a page-level store would be the clean fix.
    retrieved = [
        RetrievedChunk(
            chunk_id=c.id, document_id=c.document_id, filename=doc.filename,
            page_number=c.page_number, content=c.content, score=1.0,
        )
        for c in chunks
    ]
    ctx.add_sources(retrieved)
    text = " ".join(c.content for c in chunks)
    return f"{doc.filename}, page {page}:\n{text}"


TOOL_IMPLS = {
    "search_documents": search_documents,
    "list_documents": list_documents,
    "read_page": read_page,
}
