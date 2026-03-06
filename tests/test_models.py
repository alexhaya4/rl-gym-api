from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.session import Base
from app.models.episode import Episode
from app.models.experiment import Experiment
from app.models.user import User


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite://", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as sess:
        yield sess

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_create_user(session: AsyncSession) -> None:
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="fakehash",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    assert user.id is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True
    assert user.created_at is not None
    assert user.updated_at is not None


async def test_create_experiment(session: AsyncSession) -> None:
    user = User(
        username="researcher",
        email="researcher@example.com",
        hashed_password="fakehash",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    experiment = Experiment(
        name="CartPole PPO Run",
        environment_id="CartPole-v1",
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
        total_timesteps=100000,
        user_id=user.id,
    )
    session.add(experiment)
    await session.commit()
    await session.refresh(experiment)

    assert experiment.id is not None
    assert experiment.name == "CartPole PPO Run"
    assert experiment.status == "pending"
    assert experiment.user_id == user.id
    assert experiment.hyperparameters["learning_rate"] == 0.0003
    assert experiment.created_at is not None


async def test_create_episode(session: AsyncSession) -> None:
    user = User(
        username="agent",
        email="agent@example.com",
        hashed_password="fakehash",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    experiment = Experiment(
        name="DQN LunarLander",
        environment_id="LunarLander-v3",
        algorithm="DQN",
        total_timesteps=50000,
        user_id=user.id,
    )
    session.add(experiment)
    await session.commit()
    await session.refresh(experiment)

    episode = Episode(
        experiment_id=experiment.id,
        episode_number=1,
        total_reward=195.5,
        episode_length=200,
        mean_reward=195.5,
        std_reward=0.0,
    )
    session.add(episode)
    await session.commit()
    await session.refresh(episode)

    assert episode.id is not None
    assert episode.experiment_id == experiment.id
    assert episode.episode_number == 1
    assert episode.total_reward == 195.5
    assert episode.episode_length == 200
    assert episode.created_at is not None
