from app.db.session import Base, engine
from app.models import Episode, Experiment, User  # noqa: F401 — register models with Base.metadata


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
