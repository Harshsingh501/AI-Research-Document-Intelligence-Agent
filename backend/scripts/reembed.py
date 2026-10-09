"""
Re-embed every chunk with the current embedding settings.

Run this after changing anything that alters the vectors:
  - the embedding model,
  - the document/query prefixes (nomic-embed-text expects them),
  - the text normalisation in the ingestion pipeline.

Vectors from different settings are not comparable, so a half-migrated table
gives nonsense rankings. This script walks every chunk in batches and rewrites
its embedding in place. Content, pages and ids are untouched.

Usage (from backend/):
    .venv\\Scripts\\python.exe -m scripts.reembed
"""

import asyncio
import logging
import time

from sqlalchemy import func, select

from app.core.config import settings
from app.db.models import Chunk
from app.db.session import AsyncSessionLocal
from app.services.embeddings import embed_texts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("reembed")

BATCH = 64


async def main() -> None:
    started = time.perf_counter()
    async with AsyncSessionLocal() as db:
        total = await db.scalar(select(func.count()).select_from(Chunk))
        log.info(
            "re-embedding %d chunks with %s (doc prefix %r)",
            total, settings.EMBEDDING_MODEL, settings.EMBEDDING_DOCUMENT_PREFIX,
        )
        done = 0
        offset = 0
        while True:
            chunks = (
                await db.scalars(select(Chunk).order_by(Chunk.created_at, Chunk.id).offset(offset).limit(BATCH))
            ).all()
            if not chunks:
                break
            vectors = await embed_texts([c.content for c in chunks])
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec
            await db.commit()
            done += len(chunks)
            offset += BATCH
            log.info("  %d / %d", done, total)
    log.info("done in %.1fs", time.perf_counter() - started)


if __name__ == "__main__":
    asyncio.run(main())
