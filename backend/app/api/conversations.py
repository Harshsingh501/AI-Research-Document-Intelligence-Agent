"""
/conversations endpoints — multi-turn chat over documents.

A conversation is just an ordered list of messages. Posting a message:
  1. loads the previous turns,
  2. stores the user's message,
  3. condenses the question into a standalone one (for retrieval),
  4. runs RAG or the agent WITH the history in the prompt,
  5. stores the assistant's reply together with its sources and diagnostics.

Everything happens in one DB transaction, so a failed LLM call leaves no
half-written conversation behind.

Two flavours of the same operation:
  POST /{id}/messages          → wait, then one JSON response
  POST /{id}/messages/stream   → Server-Sent Events as work progresses

SSE is the simplest way to push a one-directional stream to a browser: a
long-lived HTTP response whose body is `event: <name>\\ndata: <json>\\n\\n`
blocks. No WebSocket handshake, works through ordinary proxies.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.db.models import Conversation, Message, MessageRole
from app.db.session import get_db
from app.schemas.ask import sources_from_chunks
from app.schemas.conversations import (
    ConversationCreate,
    ConversationDetail,
    ConversationList,
    ConversationOut,
    MessageOut,
    SendMessageRequest,
    SendMessageResponse,
)
from app.services.agent import run_agent
from app.services.history import condense_question, to_chat_messages
from app.services.rag import RagResult, answer_question, stream_answer
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])

TITLE_CHARS = 80


# --- Helpers ------------------------------------------------------------------------


async def _load(db: AsyncSession, conversation_id: uuid.UUID) -> Conversation:
    # selectinload: fetch the messages in a second query up front. Lazy loading
    # on attribute access does not work in async SQLAlchemy.
    conv = await db.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conv


def _add_user_message(db: AsyncSession, conv: Conversation, question: str) -> Message:
    user_msg = Message(conversation_id=conv.id, role=MessageRole.USER, content=question)
    db.add(user_msg)
    if conv.title is None:
        # First message doubles as the title, like most chat UIs
        conv.title = question[:TITLE_CHARS]
    return user_msg


def _rag_meta(rag: RagResult, standalone: str) -> dict:
    return {
        "mode": "ask",
        "standalone_question": standalone,
        "confidence": rag.confidence,
        "citations": rag.citations,
        "short_circuited": rag.short_circuited,
    }


async def _run_agent_mode(
    db: AsyncSession, standalone: str, body: SendMessageRequest, history: list[dict]
) -> tuple[str, list[RetrievedChunk], dict]:
    res = await run_agent(db, standalone, document_id=body.document_id, history=history)
    meta = {
        "mode": "agent",
        "standalone_question": standalone,
        "confidence": res.confidence,
        "citations": res.citations,
        "fallback_used": res.fallback_used,
        "fallback_reason": res.fallback_reason,
        "steps": [{"tool": s.tool, "arguments": s.arguments} for s in res.steps],
    }
    return res.answer, res.sources, meta


def _add_assistant_message(
    db: AsyncSession, conv: Conversation, answer: str, chunks: list[RetrievedChunk], meta: dict
) -> Message:
    assistant_msg = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content=answer,
        # mode="json" turns UUIDs into strings so the dicts are JSONB-safe
        sources=[s.model_dump(mode="json") for s in sources_from_chunks(chunks)],
        meta=meta,
    )
    db.add(assistant_msg)
    # onupdate only fires when some column changes; touching updated_at
    # explicitly keeps "most recently active" ordering correct.
    conv.updated_at = datetime.now(timezone.utc)
    return assistant_msg


# --- CRUD ------------------------------------------------------------------------------


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(body: ConversationCreate, db: AsyncSession = Depends(get_db)) -> Conversation:
    conv = Conversation(title=body.title)
    db.add(conv)
    await db.commit()
    return conv


@router.get("", response_model=ConversationList)
async def list_conversations(db: AsyncSession = Depends(get_db)) -> ConversationList:
    items = (await db.scalars(select(Conversation).order_by(Conversation.updated_at.desc()))).all()
    total = await db.scalar(select(func.count()).select_from(Conversation))
    return ConversationList(items=items, total=total or 0)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Conversation:
    return await _load(db, conversation_id)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> None:
    conv = await db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    await db.delete(conv)  # messages cascade
    await db.commit()


# --- Messages: blocking --------------------------------------------------------------


@router.post("/{conversation_id}/messages", response_model=SendMessageResponse)
async def send_message(
    conversation_id: uuid.UUID,
    body: SendMessageRequest,
    db: AsyncSession = Depends(get_db),
) -> SendMessageResponse:
    conv = await _load(db, conversation_id)
    history = to_chat_messages(conv.messages)  # BEFORE adding the new user turn
    user_msg = _add_user_message(db, conv, body.question)
    await db.flush()

    standalone = await condense_question(history, body.question)

    if body.mode == "ask":
        rag = await answer_question(
            db, standalone, top_k=body.top_k, document_id=body.document_id, history=history
        )
        answer, chunks, meta = rag.answer, rag.sources, _rag_meta(rag, standalone)
    else:
        answer, chunks, meta = await _run_agent_mode(db, standalone, body, history)

    assistant_msg = _add_assistant_message(db, conv, answer, chunks, meta)
    await db.commit()
    return SendMessageResponse(
        user_message=user_msg, assistant_message=assistant_msg, standalone_question=standalone
    )


# --- Messages: streaming --------------------------------------------------------------


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _friendly(exc: Exception) -> str:
    # The app-level exception handlers in main.py cannot help here: once the
    # stream has started the status line is already sent. So we translate
    # the common failures ourselves and emit them as an `error` event.
    if isinstance(exc, httpx.ConnectError):
        return f"Cannot reach Ollama at {settings.OLLAMA_BASE_URL}. Is 'ollama serve' running?"
    if isinstance(exc, httpx.TimeoutException):
        return f"The model did not respond within {settings.LLM_TIMEOUT_SECONDS:.0f}s."
    return f"{type(exc).__name__}: {exc}"


@router.post("/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: uuid.UUID,
    body: SendMessageRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Events, in order:
        status   {"stage": "condensing" | "retrieving" | "generating" | "agent", ...}
        sources  [SourceOut, ...]                 (ask mode, right after retrieval)
        token    {"text": "..."}                  (ask mode, many)
        done     SendMessageResponse              (always last on success)
        error    {"detail": "..."}                (instead of done, on failure)
    """
    conv = await _load(db, conversation_id)  # 404 happens before streaming starts

    async def events() -> AsyncIterator[str]:
        try:
            history = to_chat_messages(conv.messages)
            user_msg = _add_user_message(db, conv, body.question)
            await db.flush()

            if history:
                yield _sse("status", {"stage": "condensing"})
            standalone = await condense_question(history, body.question)
            yield _sse("status", {"stage": "retrieving", "standalone_question": standalone})

            if body.mode == "ask":
                rag: RagResult | None = None
                async for kind, payload in stream_answer(
                    db, standalone, top_k=body.top_k, document_id=body.document_id, history=history
                ):
                    if kind == "sources":
                        yield _sse("sources", [s.model_dump(mode="json") for s in sources_from_chunks(payload)])
                        yield _sse("status", {"stage": "generating"})
                    elif kind == "token":
                        yield _sse("token", {"text": payload})
                    else:
                        rag = payload
                assert rag is not None
                answer, chunks, meta = rag.answer, rag.sources, _rag_meta(rag, standalone)
            else:
                yield _sse("status", {"stage": "agent"})
                answer, chunks, meta = await _run_agent_mode(db, standalone, body, history)

            assistant_msg = _add_assistant_message(db, conv, answer, chunks, meta)
            await db.commit()
            done = SendMessageResponse(
                user_message=MessageOut.model_validate(user_msg),
                assistant_message=MessageOut.model_validate(assistant_msg),
                standalone_question=standalone,
            )
            yield _sse("done", done.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 — must reach the client as an event
            logger.exception("stream: failed")
            await db.rollback()
            yield _sse("error", {"detail": _friendly(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
