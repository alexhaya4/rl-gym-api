import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import gymnasium as gym
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3.common.evaluation import evaluate_policy

from app.core.algorithms import get_algorithm_class, validate_algorithm_environment
from app.models.experiment import Experiment
from app.schemas.training import TrainingConfig

_training_sessions: dict[int, dict[str, Any]] = {}


def _run_training(config: TrainingConfig) -> dict[str, Any]:
    compatible, error = validate_algorithm_environment(config.algorithm, config.environment_id)
    if not compatible:
        raise ValueError(error)

    env = gym.make(config.environment_id)
    algo_class = get_algorithm_class(config.algorithm)
    model = algo_class("MlpPolicy", env, **config.hyperparameters)

    start_time = time.time()
    model.learn(total_timesteps=config.total_timesteps)
    elapsed_time = time.time() - start_time

    raw_mean, raw_std = evaluate_policy(model, env, n_eval_episodes=10)
    mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
    std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])

    env.close()

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "elapsed_time": elapsed_time,
    }


async def start_training(db: AsyncSession, config: TrainingConfig, user_id: int) -> dict[str, Any]:
    experiment = Experiment(
        name=config.experiment_name or f"{config.algorithm}_{config.environment_id}",
        environment_id=config.environment_id,
        algorithm=config.algorithm,
        status="running",
        hyperparameters=config.hyperparameters,
        total_timesteps=config.total_timesteps,
        user_id=user_id,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)

    _training_sessions[experiment.id] = {
        "experiment_id": experiment.id,
        "status": "running",
        "environment_id": config.environment_id,
        "algorithm": config.algorithm,
        "total_timesteps": config.total_timesteps,
    }

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_training, config)

        experiment.status = "completed"
        experiment.completed_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(experiment)

        _training_sessions[experiment.id] = {
            "experiment_id": experiment.id,
            "status": "completed",
            "environment_id": config.environment_id,
            "algorithm": config.algorithm,
            "total_timesteps": config.total_timesteps,
            "mean_reward": result["mean_reward"],
            "std_reward": result["std_reward"],
            "elapsed_time": result["elapsed_time"],
        }
    except Exception:
        experiment.status = "failed"
        await db.commit()
        _training_sessions[experiment.id]["status"] = "failed"
        raise

    return _training_sessions[experiment.id]


async def get_training_status(db: AsyncSession, experiment_id: int) -> dict[str, Any] | None:
    if experiment_id in _training_sessions:
        return _training_sessions[experiment_id]

    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()
    if experiment is None:
        return None

    return {
        "experiment_id": experiment.id,
        "status": experiment.status,
        "environment_id": experiment.environment_id,
        "algorithm": experiment.algorithm,
        "total_timesteps": experiment.total_timesteps,
        "mean_reward": None,
        "std_reward": None,
        "elapsed_time": None,
    }


async def list_training_sessions(db: AsyncSession, user_id: int) -> list[dict[str, Any]]:
    result = await db.execute(
        select(Experiment).where(Experiment.user_id == user_id)
    )
    experiments = result.scalars().all()

    sessions = []
    for exp in experiments:
        if exp.id in _training_sessions:
            sessions.append(_training_sessions[exp.id])
        else:
            sessions.append({
                "experiment_id": exp.id,
                "status": exp.status,
                "environment_id": exp.environment_id,
                "algorithm": exp.algorithm,
                "total_timesteps": exp.total_timesteps,
                "mean_reward": None,
                "std_reward": None,
                "elapsed_time": None,
            })
    return sessions
