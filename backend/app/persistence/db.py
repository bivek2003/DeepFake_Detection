"""
Database connection and session management.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.settings import get_settings

# Create declarative base
Base = declarative_base()

# Engine cache
_engine: AsyncEngine | None = None


def create_engine() -> AsyncEngine:
    """Create a new async database engine."""
    settings = get_settings()

    # Convert postgresql:// to postgresql+asyncpg://
    database_url = settings.database_url
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    kwargs: dict = {
        "echo": settings.debug,
    }
    if "sqlite" in database_url:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10

    return create_async_engine(database_url, **kwargs)


def get_async_engine() -> AsyncEngine:
    """Get or create async database engine (cached)."""
    global _engine

    if _engine is None:
        _engine = create_engine()

    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get async session maker."""
    engine = get_async_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for dependency injection."""
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def create_tables(engine: AsyncEngine) -> None:
    """Create all database tables."""
    from app.persistence.models import Analysis, Asset, Frame  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
