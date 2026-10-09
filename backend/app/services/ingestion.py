"""
Ingestion orchestrator: PDF file on disk  →  rows in `chunks`.

Runs as a FastAPI background task, AFTER the upload response has been sent.
Because of that it must open its OWN database session — the request's session
is already closed by the time this runs.

Status transitions:  PENDING → PROCESSING → READY
                                         ↘ FAILED (error_message set)
"""

import logging
import uuid
from pathlib import Path

from sqlalchemy import select

from app.db.models import Chunk, Document, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.services.chunker import TextChunk, chunk_text
from app.services.embeddings import embed_texts
from app.services.pdf import extract_pages

logger = logging.getLogger(__name__)


async def ingest_document(document_id: uuid.UUID, path: Path) -> None:
    async with AsyncSessionLocal() as db:
        doc = await db.scalar(select(Document).where(Document.id == document_id))
        if doc is None:
            logger.error("ingest: document %s vanished before processing", document_id)
            return

        doc.status = DocumentStatus.PROCESSING
        await db.commit()

        try:
            # 1. extract
            pages = extract_pages(path)
            if not pages:
                raise ValueError("No extractable text — is this a scanned/image-only PDF?")

            # 2. chunk, page by page so every chunk knows its page number
            chunks: list[TextChunk] = []
            for page in pages:
                chunks.extend(chunk_text(page.text, page_number=page.number, start_index=len(chunks)))

            # 3. embed (the slow part — one Ollama call per 32 chunks)
            vectors = await embed_texts([c.content for c in chunks])

            # 4. store
            db.add_all(
                Chunk(
                    document_id=doc.id,
                    chunk_index=c.index,
                    page_number=c.page_number,
                    content=c.content,
                    embedding=vec,
                )
                for c, vec in zip(chunks, vectors)
            )
            doc.page_count = len(pages)
            doc.status = DocumentStatus.READY
            await db.commit()
            logger.info("ingest: %s → %d pages, %d chunks", doc.filename, len(pages), len(chunks))

        except Exception as exc:  # noqa: BLE001 — we WANT to catch everything and record it
            logger.exception("ingest: failed for %s", doc.filename)
            await db.rollback()
            doc.status = DocumentStatus.FAILED
            doc.error_message = f"{type(exc).__name__}: {exc}"[:2000]
            await db.commit()
