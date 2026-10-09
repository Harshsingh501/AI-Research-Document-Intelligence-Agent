"""
POST /ask — the core RAG endpoint.

Unlike /documents upload, this is NOT a background task: the client wants the
answer in the response, so the request stays open while retrieval and the LLM
run. On CPU that can be 15-30 s. Because the handler is async, other requests
(uploads, health checks) are still served while we wait on Ollama.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.schemas.ask import AskRequest, AskResponse, sources_from_chunks
from app.services.rag import answer_question

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("", response_model=AskResponse)
async def ask(body: AskRequest, db: AsyncSession = Depends(get_db)) -> AskResponse:
    result = await answer_question(
        db, body.question, top_k=body.top_k, document_id=body.document_id
    )
    return AskResponse(
        answer=result.answer,
        model=settings.LLM_MODEL,
        citations=result.citations,
        confidence=result.confidence,
        short_circuited=result.short_circuited,
        sources=sources_from_chunks(result.sources),
    )
