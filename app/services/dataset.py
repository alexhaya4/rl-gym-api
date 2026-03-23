import asyncio
import csv
import io
import json
import logging
from typing import Any

import gymnasium as gym
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetEpisode
from app.schemas.dataset import (
    DatasetCreate,
    DatasetEpisodeCreate,
    DatasetStatsResponse,
)

logger = logging.getLogger(__name__)


async def create_dataset(
    db: AsyncSession, dataset_create: DatasetCreate, user_id: int
) -> Dataset:
    """Create a dataset, auto-incrementing version if name exists."""
    result = await db.execute(
        select(func.coalesce(func.max(Dataset.version), 0)).where(
            Dataset.name == dataset_create.name
        )
    )
    next_version = result.scalar_one() + 1

    dataset = Dataset(
        name=dataset_create.name,
        version=next_version,
        description=dataset_create.description,
        environment_id=dataset_create.environment_id,
        algorithm=dataset_create.algorithm,
        storage_format=dataset_create.storage_format,
        is_public=dataset_create.is_public,
        tags=dataset_create.tags or None,
        metadata_=dataset_create.metadata or None,
        user_id=user_id,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


async def add_episodes(
    db: AsyncSession,
    dataset_id: int,
    episodes: list[DatasetEpisodeCreate],
    user_id: int,
) -> Dataset:
    """Add episodes to a dataset and update statistics."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found or not owned by user")

    total_transitions = 0
    rewards: list[float] = []
    lengths: list[int] = []

    for ep in episodes:
        db.add(DatasetEpisode(
            dataset_id=dataset_id,
            episode_number=ep.episode_number,
            total_reward=ep.total_reward,
            episode_length=ep.episode_length,
            observations=ep.observations,
            actions=ep.actions,
            rewards=ep.rewards,
            terminated=ep.terminated,
        ))
        total_transitions += ep.episode_length
        rewards.append(ep.total_reward)
        lengths.append(ep.episode_length)

    rewards_array = np.array(rewards)
    lengths_array = np.array(lengths, dtype=float)

    dataset.n_episodes = dataset.n_episodes + len(episodes)
    dataset.n_transitions = dataset.n_transitions + total_transitions
    dataset.mean_episode_reward = float(rewards_array.mean())
    dataset.std_episode_reward = float(rewards_array.std())
    dataset.mean_episode_length = float(lengths_array.mean())

    await db.commit()
    await db.refresh(dataset)
    return dataset


async def get_dataset(db: AsyncSession, dataset_id: int) -> Dataset | None:
    """Get a dataset by id."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    return result.scalar_one_or_none()


async def list_datasets(
    db: AsyncSession, user_id: int, include_public: bool = True
) -> list[Dataset]:
    """List datasets for a user, optionally including public datasets."""
    if include_public:
        query = select(Dataset).where(
            (Dataset.user_id == user_id) | (Dataset.is_public.is_(True))
        )
    else:
        query = select(Dataset).where(Dataset.user_id == user_id)
    query = query.order_by(Dataset.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


async def delete_dataset(
    db: AsyncSession, dataset_id: int, user_id: int
) -> bool:
    """Delete a dataset and its episodes."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        return False

    await db.execute(
        DatasetEpisode.__table__.delete().where(
            DatasetEpisode.dataset_id == dataset_id
        )
    )
    await db.delete(dataset)
    await db.commit()
    return True


async def get_dataset_stats(
    db: AsyncSession, dataset_id: int
) -> DatasetStatsResponse:
    """Calculate comprehensive statistics for a dataset."""
    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    ep_result = await db.execute(
        select(DatasetEpisode).where(DatasetEpisode.dataset_id == dataset_id)
    )
    episodes = list(ep_result.scalars().all())

    if not episodes:
        return DatasetStatsResponse(
            dataset_id=dataset_id,
            n_episodes=0,
            n_transitions=0,
            mean_episode_reward=None,
            std_episode_reward=None,
            min_episode_reward=None,
            max_episode_reward=None,
            mean_episode_length=None,
            reward_distribution=[],
        )

    rewards = [ep.total_reward for ep in episodes if ep.total_reward is not None]
    lengths = [ep.episode_length for ep in episodes if ep.episode_length is not None]
    rewards_array = np.array(rewards) if rewards else np.array([])
    lengths_array = np.array(lengths, dtype=float) if lengths else np.array([])

    return DatasetStatsResponse(
        dataset_id=dataset_id,
        n_episodes=len(episodes),
        n_transitions=sum(lengths) if lengths else 0,
        mean_episode_reward=float(rewards_array.mean()) if len(rewards_array) > 0 else None,
        std_episode_reward=float(rewards_array.std()) if len(rewards_array) > 0 else None,
        min_episode_reward=float(rewards_array.min()) if len(rewards_array) > 0 else None,
        max_episode_reward=float(rewards_array.max()) if len(rewards_array) > 0 else None,
        mean_episode_length=float(lengths_array.mean()) if len(lengths_array) > 0 else None,
        reward_distribution=rewards_array.tolist() if len(rewards_array) > 0 else [],
    )


async def export_dataset(
    db: AsyncSession, dataset_id: int, format: str
) -> bytes:
    """Export dataset episodes in the specified format."""
    ep_result = await db.execute(
        select(DatasetEpisode)
        .where(DatasetEpisode.dataset_id == dataset_id)
        .order_by(DatasetEpisode.episode_number)
    )
    episodes = list(ep_result.scalars().all())

    if format == "json":
        data = [
            {
                "episode_number": ep.episode_number,
                "total_reward": ep.total_reward,
                "episode_length": ep.episode_length,
                "observations": ep.observations,
                "actions": ep.actions,
                "rewards": ep.rewards,
                "terminated": ep.terminated,
            }
            for ep in episodes
        ]
        return json.dumps(data, indent=2).encode()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["episode", "reward", "length"])
        for ep in episodes:
            writer.writerow([ep.episode_number, ep.total_reward, ep.episode_length])
        return output.getvalue().encode()

    if format == "hdf5":
        import h5py

        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as f:
            for ep in episodes:
                grp = f.create_group(f"episode_{ep.episode_number}")
                if ep.observations:
                    grp.create_dataset("observations", data=np.array(ep.observations))
                if ep.actions:
                    grp.create_dataset("actions", data=np.array(ep.actions))
                if ep.rewards:
                    grp.create_dataset("rewards", data=np.array(ep.rewards))
                grp.attrs["total_reward"] = ep.total_reward or 0.0
                grp.attrs["episode_length"] = ep.episode_length or 0
                grp.attrs["terminated"] = ep.terminated
        return buffer.getvalue()

    raise ValueError(f"Unsupported format: {format}")


def _collect_episodes_sync(
    environment_id: str,
    n_episodes: int,
    model: Any = None,
) -> list[dict[str, Any]]:
    """Collect trajectory data (blocking). Uses model for inference or random policy."""
    env = gym.make(environment_id)
    collected: list[dict[str, Any]] = []

    for ep_num in range(n_episodes):
        obs, _info = env.reset()
        observations = [obs.tolist() if hasattr(obs, "tolist") else list(obs)]
        actions: list[Any] = []
        rewards: list[float] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _info = env.step(action)
            observations.append(obs.tolist() if hasattr(obs, "tolist") else list(obs))
            action_val = action.tolist() if hasattr(action, "tolist") else action
            actions.append(action_val)
            rewards.append(float(reward))

        collected.append({
            "episode_number": ep_num + 1,
            "total_reward": sum(rewards),
            "episode_length": len(rewards),
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminated": terminated,
        })

    env.close()
    return collected


async def collect_trajectory(
    db: AsyncSession,
    dataset_id: int,
    environment_id: str,
    n_episodes: int,
    algorithm: str | None,
    model_version_id: int | None,
) -> Dataset:
    """Collect trajectory data and add to dataset."""
    model = None
    if model_version_id is not None:
        from app.services.model_storage import load_model

        model, _ = await load_model(db, model_version_id)

    loop = asyncio.get_event_loop()
    raw_episodes = await loop.run_in_executor(
        None, _collect_episodes_sync, environment_id, n_episodes, model
    )

    episodes = [DatasetEpisodeCreate(**ep) for ep in raw_episodes]

    result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    return await add_episodes(db, dataset_id, episodes, dataset.user_id)
