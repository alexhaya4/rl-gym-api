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
    if url.startswith("sqlite"):
        return create_async_engine(
            url,
            connect_args={"check_same_thread": False},
        )
    if url.startswith("postgresql"):
        return create_async_engine(
            url,
            pool_size=10,
            max_overflow=20,
        )
    return create_async_engine(url)


settings = get_settings()
engine = create_db_engine(settings.DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
