import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.session import Base, get_db
from app.dependencies import get_arq_redis
from app.main import app

DATABASE_URL = os.environ.get("DATABASE_URL", "")
_USE_POSTGRES = DATABASE_URL.startswith("postgresql")


@pytest.fixture(scope="session", autouse=True)
async def run_migrations() -> AsyncGenerator[None, None]:
    """Run alembic migrations once per session when using PostgreSQL."""
    if _USE_POSTGRES:
        proc = await asyncio.create_subprocess_exec(
            "alembic", "upgrade", "head",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"alembic upgrade head failed (rc={proc.returncode}): {stderr.decode()}"
            )
    yield


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    if _USE_POSTGRES:
        engine = create_async_engine(DATABASE_URL, echo=True)
    else:
        engine = create_async_engine("sqlite+aiosqlite://", echo=True)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    test_session = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with test_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    # Mock arq redis so training tests don't need a real Redis connection
    def _make_mock_job(*args, **kwargs):
        mock_job = AsyncMock()
        mock_job.job_id = str(uuid.uuid4())
        return mock_job

    mock_arq = AsyncMock()
    mock_arq.enqueue_job = AsyncMock(side_effect=_make_mock_job)

    async def override_get_arq_redis() -> AsyncGenerator[AsyncMock, None]:
        yield mock_arq

    app.dependency_overrides[get_arq_redis] = override_get_arq_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()

    if _USE_POSTGRES:
        # Truncate all tables but keep schema intact for next test
        async with engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                await conn.execute(table.delete())
    else:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()
