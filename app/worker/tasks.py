import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import gymnasium as gym
from sqlalchemy import select
from stable_baselines3 import A2C, DQN, PPO

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3.common.evaluation import evaluate_policy

from app.db.session import AsyncSessionLocal
from app.models.experiment import Experiment

logger = logging.getLogger(__name__)

ALGORITHMS: dict[str, type] = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


def _run_training_sync(config: dict[str, Any]) -> dict[str, Any]:
    """Blocking SB3 training + evaluation — meant to run in an executor."""
    env = gym.make(config["environment_id"])
    algo_class = ALGORITHMS[config["algorithm"]]
    model = algo_class("MlpPolicy", env, **config.get("hyperparameters", {}))

    model.learn(total_timesteps=config["total_timesteps"])

    raw_mean, raw_std = evaluate_policy(model, env, n_eval_episodes=10)
    mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
    std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])

    env.close()
    return {"mean_reward": mean_reward, "std_reward": std_reward}


async def run_training_job(
    ctx: dict,
    experiment_id: int,
    config_dict: dict,
) -> dict[str, Any]:
    """arq task: train an RL agent and persist results."""
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

        loop = asyncio.get_event_loop()
        training_result = await loop.run_in_executor(
            None, _run_training_sync, config_dict
        )

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
