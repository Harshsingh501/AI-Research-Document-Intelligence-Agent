"""
Retrieval: question text  →  the k most relevant chunks.

This is the "R" in RAG. Two complementary searches are combined:

1. VECTOR (semantic). Embed the question with the SAME model used for the
   chunks and ask pgvector for the nearest neighbours by cosine distance
   (`<=>`). Great at meaning: "how fast is training?" finds a passage about
   "12 hours on 8 GPUs" even though no words overlap.

2. KEYWORD (lexical). Postgres full-text search: `to_tsvector(content) @@
   websearch_to_tsquery(question)` ranked by ts_rank_cd. Great at exact
   tokens embeddings blur: numbers ("28.4"), acronyms ("BLEU"), names.
   Phase 7 showed the weakness this fixes: a question about "h = 8 heads"
   missed the chunk containing exactly that.

The two ranked lists are merged with RECIPROCAL RANK FUSION:
    score(chunk) = Σ over lists  1 / (K + rank_in_list)
RRF needs no score normalisation (cosine and ts_rank live on different
scales) and rewards chunks that both searches like. K=60 is the standard
constant from the original paper; it damps the advantage of rank 1 vs 2.

The reported `score` stays cosine similarity so thresholds and the UI keep
one consistent meaning: "how semantically close is this passage".

Why cosine and not Euclidean?  Embedding models encode meaning in the
*direction* of the vector; the length carries little information. Cosine
ignores length, so two passages about the same topic score high even if one
is far wordier than the other.
"""

import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import Row, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Chunk, Document
from app.services.embeddings import embed_query

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES = 20  # how deep each list goes before fusion

_TERM = re.compile(r"[A-Za-z]{3,}|\d+(?:\.\d+)?")


def keyword_query(question: str) -> str:
    """
    Turn a natural-language question into an OR-ed term list for
    websearch_to_tsquery.

    Why OR?  websearch_to_tsquery ANDs plain words, so "How many heads does
    the base Transformer use?" requires a chunk containing ALL of base,
    transformer, head, use ... which rarely exists. OR-ing lets ts_rank_cd do
    its job: chunks matching more (and rarer, denser) terms rank higher.
    Stopwords ("how", "the") are dropped by the 'english' config anyway.
    """
    seen: dict[str, None] = {}
    for term in _TERM.findall(question):
        seen.setdefault(term.lower(), None)
    return " or ".join(seen)


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    content: str
    score: float  # cosine similarity in [-1, 1]; higher = more relevant


def _to_chunks(rows: list[Row]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            page_number=chunk.page_number,
            content=chunk.content,
            score=round(1.0 - dist, 4),
        )
        for chunk, filename, dist in rows
    ]


async def retrieve(
    db: AsyncSession,
    question: str,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
) -> list[RetrievedChunk]:
    query_vec = await embed_query(question)

    # `cosine_distance` compiles to `embedding <=> :vec`. We label it so we can
    # read it back from the row and convert distance → similarity (1 - d).
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
    base = select(Chunk, Document.filename, distance).join(Document, Chunk.document_id == Document.id)
    if document_id is not None:
        # Scope the search to one document ("ask this paper ...")
        base = base.where(Chunk.document_id == document_id)

    n = max(CANDIDATES, top_k)
    vec_rows = (await db.execute(base.order_by(distance).limit(n))).all()

    if not settings.HYBRID_SEARCH or not vec_rows:
        return _to_chunks(vec_rows[:top_k])

    # Keyword list. websearch_to_tsquery parses free text safely (quotes,
    # "or", "-") and never raises on odd input.
    tsv = func.to_tsvector("english", Chunk.content)
    tsq = func.websearch_to_tsquery("english", keyword_query(question))
    kw_rows = (
        await db.execute(base.where(tsv.op("@@")(tsq)).order_by(func.ts_rank_cd(tsv, tsq).desc()).limit(n))
    ).all()

    # Reciprocal rank fusion
    fused: dict[uuid.UUID, float] = {}
    by_id: dict[uuid.UUID, Row] = {}
    for rows in (vec_rows, kw_rows):
        for rank, row in enumerate(rows, start=1):
            cid = row[0].id
            by_id[cid] = row
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    # Tie-break by cosine distance (row[2]) so pure-keyword hits with weak
    # semantic similarity do not jump ahead of equally fused semantic hits.
    ordered = sorted(fused, key=lambda cid: (-fused[cid], by_id[cid][2]))
    logger.info("retrieve: %d vector + %d keyword candidates → %d", len(vec_rows), len(kw_rows), min(top_k, len(ordered)))
    return _to_chunks([by_id[cid] for cid in ordered[:top_k]])
