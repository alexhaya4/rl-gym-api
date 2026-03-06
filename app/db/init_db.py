from app.db.session import Base, engine
from app.models import (  # noqa: F401 — register models with Base.metadata
    Episode,
    Experiment,
    User,
)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
