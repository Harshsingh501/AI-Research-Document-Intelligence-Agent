"""
POST /agent — tool-calling agent over the uploaded documents.

Compared with /ask, the model chooses what to do (search, list, read a page)
and may take several steps. Slower (2+ LLM round-trips) but handles questions
plain RAG cannot, like "what documents do I have?" or "summarise page 3".
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.schemas.agent import AgentRequest, AgentResponse, StepOut
from app.schemas.ask import sources_from_chunks
from app.services.agent import run_agent

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("", response_model=AgentResponse)
async def agent(body: AgentRequest, db: AsyncSession = Depends(get_db)) -> AgentResponse:
    result = await run_agent(db, body.question, document_id=body.document_id)
    return AgentResponse(
        answer=result.answer,
        model=settings.LLM_MODEL,
        citations=result.citations,
        confidence=result.confidence,
        fallback_used=result.fallback_used,
        fallback_reason=result.fallback_reason,
        steps=[StepOut(tool=s.tool, arguments=s.arguments, result_preview=s.result_preview) for s in result.steps],
        sources=sources_from_chunks(result.sources),
    )
