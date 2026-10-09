"""
Pydantic schemas = the public API contract for /documents.

ORM models (app/db/models.py) describe the DATABASE shape.
Schemas describe the JSON shape clients see. Keeping them separate lets the DB
change without breaking clients, and hides internal columns.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import DocumentStatus


class DocumentOut(BaseModel):
    # from_attributes=True lets FastAPI build this directly from an ORM object
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    page_count: int | None
    status: DocumentStatus
    error_message: str | None
    created_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
