from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.config import get_settings

Base = declarative_base()


def create_db_engine(url: str) -> AsyncEngine:
    """Return an async engine configured for the given database URL."""
    settings = get_settings()
    if url.startswith("sqlite"):
        return create_async_engine(
            url,
            connect_args={"timeout": settings.DB_COMMAND_TIMEOUT, "check_same_thread": False},
        )
    if url.startswith("postgresql"):
        return create_async_engine(
            url,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "command_timeout": settings.DB_COMMAND_TIMEOUT,
                "timeout": 10,
            },
        )
    return create_async_engine(url)


settings = get_settings()
engine = create_db_engine(settings.DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
