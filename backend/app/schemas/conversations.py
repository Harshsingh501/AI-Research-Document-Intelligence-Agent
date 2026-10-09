"""Pydantic schemas for /conversations."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import MessageRole
from app.schemas.ask import SourceOut


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    sources: list[SourceOut] | None
    meta: dict | None
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class ConversationList(BaseModel):
    items: list[ConversationOut]
    total: int


class SendMessageRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_id: uuid.UUID | None = None
    mode: Literal["ask", "agent"] = Field(
        default="ask",
        description="'ask' = plain RAG (faster). 'agent' = tool-calling agent (can list docs, read pages).",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Chunks to retrieve (ask mode only)")


class SendMessageResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
    standalone_question: str  # what was actually searched for, after condensing
