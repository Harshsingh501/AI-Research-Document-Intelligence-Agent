"""
/documents endpoints — upload, list, inspect, delete.

The upload handler does the FAST work (validate, save file, insert row) and
returns 202 Accepted immediately. The SLOW work (extract/chunk/embed) runs in
a BackgroundTask after the response is sent. The client polls GET /documents/{id}
until status is "ready" or "failed".
"""

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Document, DocumentStatus
from app.db.session import get_db
from app.schemas.documents import DocumentList, DocumentOut
from app.services.ingestion import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_TYPES = {"application/pdf"}


def _upload_path(document_id: uuid.UUID) -> Path:
    # Store by UUID, not by original filename: avoids collisions and path tricks like "../../etc"
    return Path(settings.UPLOAD_DIR) / f"{document_id}.pdf"


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> Document:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are supported")

    doc = Document(
        filename=file.filename or "upload.pdf",
        content_type=file.content_type,
        size_bytes=0,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    await db.flush()  # assigns doc.id without committing yet

    # Stream to disk in 1 MB pieces instead of loading the whole file into memory
    path = _upload_path(doc.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    async with aiofiles.open(path, "wb") as out:
        while piece := await file.read(1024 * 1024):
            size += len(piece)
            if size > MAX_UPLOAD_BYTES:
                await db.rollback()
                path.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File exceeds 25 MB")
            await out.write(piece)

    doc.size_bytes = size
    await db.commit()

    background.add_task(ingest_document, doc.id, path)
    return doc


@router.get("", response_model=DocumentList)
async def list_documents(db: AsyncSession = Depends(get_db)) -> DocumentList:
    items = (await db.scalars(select(Document).order_by(Document.created_at.desc()))).all()
    total = await db.scalar(select(func.count()).select_from(Document))
    return DocumentList(items=items, total=total or 0)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Document:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> None:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await db.delete(doc)          # chunks go with it — ON DELETE CASCADE
    await db.commit()
    _upload_path(document_id).unlink(missing_ok=True)
