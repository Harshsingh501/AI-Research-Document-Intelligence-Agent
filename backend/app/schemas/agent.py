"""Pydantic schemas for /agent."""

import uuid

from pydantic import BaseModel, Field

from app.schemas.ask import SourceOut
from app.services.structured import Confidence


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["Which documents do I have?"])
    document_id: uuid.UUID | None = Field(default=None, description="Restrict the agent to one document")


class StepOut(BaseModel):
    """One tool call the agent made, in order. Useful for debugging and for UI."""

    tool: str
    arguments: dict
    result_preview: str


class AgentResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    steps: list[StepOut]
    citations: list[int]
    confidence: Confidence
    fallback_used: bool
    fallback_reason: str | None
    model: str
