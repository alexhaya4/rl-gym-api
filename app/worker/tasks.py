import asyncio
import logging
import time
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

import gymnasium as gym
from sqlalchemy import select
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from app.core.algorithms import get_algorithm_class, validate_algorithm_environment
from app.core.callbacks import WebSocketMetricsCallback, broadcast_training_metrics
from app.core.prometheus import (
    episode_reward,
    training_duration_seconds,
    training_jobs_total,
)
from app.db.session import AsyncSessionLocal
from app.models.experiment import Experiment

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _run_training_sync(
    config: dict[str, Any], callback: BaseCallback
) -> dict[str, Any]:
    """Blocking SB3 training + evaluation — meant to run in an executor."""
    compatible, error = validate_algorithm_environment(
        config["algorithm"], config["environment_id"]
    )
    if not compatible:
        raise ValueError(error)

    n_envs = config.get("n_envs", 1)
    algo_class = get_algorithm_class(config["algorithm"])

    if n_envs > 1:
        env = DummyVecEnv([
            lambda _eid=config["environment_id"]: gym.make(_eid)
            for _ in range(n_envs)
        ])
    else:
        env = gym.make(config["environment_id"])

    model = algo_class("MlpPolicy", env, **config.get("hyperparameters", {}))
    model.learn(total_timesteps=config["total_timesteps"], callback=callback)

    eval_env = gym.make(config["environment_id"])
    raw_mean, raw_std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
    std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])

    eval_env.close()
    env.close()
    return {"mean_reward": mean_reward, "std_reward": std_reward}


async def run_training_job(
    ctx: dict[str, Any],
    experiment_id: int,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """arq task: train an RL agent and persist results."""
    algorithm = config_dict.get("algorithm", "unknown")
    environment = config_dict.get("environment_id", "unknown")
    db: AsyncSession = AsyncSessionLocal()
    try:
        result = await db.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = result.scalar_one_or_none()
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = "running"
        await db.commit()

        training_jobs_total.labels(
            algorithm=algorithm, environment=environment, status="started"
        ).inc()
        start_time = time.monotonic()

        metrics_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        callback = WebSocketMetricsCallback(
            experiment_id=str(experiment_id),
            metrics_queue=metrics_queue,
        )

        broadcast_task = asyncio.create_task(
            broadcast_training_metrics(str(experiment_id), metrics_queue)
        )

        loop = asyncio.get_event_loop()
        training_result = await loop.run_in_executor(
            None, partial(_run_training_sync, config_dict, callback)
        )

        await broadcast_task

        duration = time.monotonic() - start_time
        training_jobs_total.labels(
            algorithm=algorithm, environment=environment, status="completed"
        ).inc()
        training_duration_seconds.labels(
            algorithm=algorithm, environment=environment
        ).observe(duration)
        episode_reward.labels(
            environment=environment, algorithm=algorithm
        ).observe(training_result["mean_reward"])

        experiment.mean_reward = training_result["mean_reward"]
        experiment.std_reward = training_result["std_reward"]
        experiment.status = "completed"
        experiment.completed_at = datetime.now(UTC)
        await db.commit()

        logger.info(
            "Experiment %d completed: mean_reward=%.2f",
            experiment_id,
            training_result["mean_reward"],
        )

        return {
            "experiment_id": experiment_id,
            "status": "completed",
            "mean_reward": training_result["mean_reward"],
        }
    except Exception:
        training_jobs_total.labels(
            algorithm=algorithm, environment=environment, status="failed"
        ).inc()
        try:
            result = await db.execute(
                select(Experiment).where(Experiment.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            if experiment is not None:
                experiment.status = "failed"
                await db.commit()
        except Exception:
            logger.exception("Failed to mark experiment %d as failed", experiment_id)
        raise
    finally:
        await db.close()
