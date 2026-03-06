import asyncio
import time
from datetime import datetime, timezone

import gymnasium as gym
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from app.models.experiment import Experiment
from app.schemas.training import TrainingConfig

_training_sessions: dict[int, dict] = {}

ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


def _run_training(config: TrainingConfig) -> dict:
    env = gym.make(config.environment_id)
    algo_class = ALGORITHMS[config.algorithm]
    model = algo_class("MlpPolicy", env, **config.hyperparameters)

    start_time = time.time()
    model.learn(total_timesteps=config.total_timesteps)
    elapsed_time = time.time() - start_time

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    env.close()

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "elapsed_time": elapsed_time,
    }


async def start_training(db: AsyncSession, config: TrainingConfig, user_id: int) -> dict:
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
        experiment.completed_at = datetime.now(timezone.utc)
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


async def get_training_status(db: AsyncSession, experiment_id: int) -> dict | None:
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


async def list_training_sessions(db: AsyncSession, user_id: int) -> list[dict]:
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
