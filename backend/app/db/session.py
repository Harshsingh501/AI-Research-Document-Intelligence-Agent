"""
Database engine and session management.

Key ideas:
- One *engine* per process: it owns the connection pool.
- One *session* per request: a short-lived unit of work. FastAPI hands it to
  endpoints through the `get_db` dependency and closes it afterwards.
- Everything is async: `await session.execute(...)` yields control to the event
  loop while Postgres works, so other requests keep being served.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,          # set True to print every SQL statement — great for learning
    pool_pre_ping=True,  # check a pooled connection is alive before reusing it
)

# expire_on_commit=False: after commit, objects keep their loaded attributes
# instead of being reloaded lazily — lazy loads don't work in async code.
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    """All ORM models inherit from this; Alembic reads Base.metadata to diff schemas."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency. Usage:

        @router.get("/things")
        async def list_things(db: AsyncSession = Depends(get_db)): ...

    The `async with` block guarantees the session is closed even if the
    endpoint raises.
    """
    async with AsyncSessionLocal() as session:
        yield session
