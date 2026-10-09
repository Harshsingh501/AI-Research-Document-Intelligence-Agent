"""
ORM models — the four tables that make up the system.

    documents 1 ──< chunks          (a document is split into many chunks)
    conversations 1 ──< messages    (a conversation is a sequence of messages)

Deleting a document deletes its chunks (ON DELETE CASCADE); same for
conversations → messages. Chunks and messages are derived data with no
meaning on their own, so orphans would only ever be garbage.
"""

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.session import Base


# --- Mixins -----------------------------------------------------------------
# Small reusable pieces so every table gets the same id / timestamp columns.


class UUIDPrimaryKey:
    # UUIDs instead of auto-increment ints: safe to expose in URLs, and the
    # client can't enumerate /documents/1, /documents/2, ...
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Timestamped:
    # server_default=func.now(): the DATABASE stamps the time, so it's correct
    # even for rows inserted by raw SQL or another service.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --- Documents & chunks -----------------------------------------------------


class DocumentStatus(str, enum.Enum):
    """Ingestion is slow (extract → chunk → embed), so we track where it is."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status"), default=DocumentStatus.PENDING, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text)

    # cascade="all, delete-orphan" is the Python-side mirror of ON DELETE CASCADE:
    # deleting a Document via the ORM also deletes its chunks in the same flush.
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # position in the doc
    page_number: Mapped[int | None] = mapped_column(Integer)           # for citations
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # The whole point of pgvector. Dimension is fixed at table-creation time
    # and MUST match the embedding model (nomic-embed-text → 768).
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIMENSIONS), nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        # Fast "give me chunks 3..7 of document X" lookups
        Index("ix_chunks_document_id_chunk_index", "document_id", "chunk_index"),
        # HNSW = approximate nearest-neighbour index. Without it, every search
        # scans every chunk (fine for 1k chunks, painful for 1M).
        # vector_cosine_ops: optimise for the <=> (cosine distance) operator.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # Full-text (keyword) search for the hybrid retriever. Expression
        # index: Postgres stores the tsvector so it need not recompute it.
        Index("ix_chunks_content_tsv", text("to_tsvector('english', content)"), postgresql_using="gin"),
    )


# --- Conversations & messages -----------------------------------------------


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Conversation(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "conversations"

    title: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(UUIDPrimaryKey, Timestamped, Base):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Which chunks the assistant cited, e.g. [{"chunk_id": ..., "document_id": ..., "score": 0.87}]
    # JSONB: flexible shape, still queryable in SQL.
    sources: Mapped[list[dict] | None] = mapped_column(JSONB)
    # Everything else worth keeping about how the answer was produced:
    # {"confidence": "high", "citations": [1], "mode": "ask", "short_circuited": false, ...}
    # JSONB rather than one column per field: this shape will keep evolving and
    # we don't want a migration every time we add a diagnostic.
    meta: Mapped[dict | None] = mapped_column(JSONB)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
