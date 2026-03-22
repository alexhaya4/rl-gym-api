import asyncio
import logging
from typing import Any

from prefect import task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.models.experiment import Experiment
from app.models.model_version import ModelVersion
from app.schemas.evaluation import EvaluationRequest
from app.schemas.experiment import ExperimentCreate
from app.services.evaluation import evaluate_experiment
from app.services.experiment import create_experiment
from app.services.model_storage import save_model as save_model_to_storage
from app.worker.settings import redis_settings

logger = logging.getLogger(__name__)


@task(name="create-experiment", retries=3, retry_delay_seconds=5)
async def create_experiment_task(
    db: AsyncSession, config: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Create an Experiment record using the experiment service."""
    experiment_create = ExperimentCreate(
        name=config.get("experiment_name") or f"{config['algorithm']}-{config['environment_id']}",
        environment_id=config["environment_id"],
        algorithm=config["algorithm"],
        hyperparameters=config.get("hyperparameters", {}),
        total_timesteps=config.get("total_timesteps", 10000),
    )
    experiment = await create_experiment(db, experiment_create, user_id)
    return {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
    }


@task(name="train-model", retries=2, retry_delay_seconds=30)
async def train_model_task(experiment_id: int, config: dict[str, Any]) -> dict[str, Any]:
    """Enqueue a training job via ARQ and poll until completion or timeout."""
    from arq.connections import create_pool

    pool = await create_pool(redis_settings)
    try:
        config_dict = {
            "environment_id": config["environment_id"],
            "algorithm": config["algorithm"],
            "total_timesteps": config.get("total_timesteps", 10000),
            "hyperparameters": config.get("hyperparameters", {}),
        }
        await pool.enqueue_job("run_training_job", experiment_id, config_dict)
    finally:
        await pool.aclose()

    timeout = 3600
    poll_interval = 10
    elapsed = 0.0

    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        db = AsyncSessionLocal()
        try:
            result = await db.execute(
                select(Experiment).where(Experiment.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            if experiment is None:
                raise ValueError(f"Experiment {experiment_id} not found")

            if experiment.status == "completed":
                return {
                    "experiment_id": experiment.id,
                    "status": "completed",
                    "mean_reward": experiment.mean_reward,
                    "std_reward": experiment.std_reward,
                }
            if experiment.status == "failed":
                raise RuntimeError(
                    f"Training failed for experiment {experiment_id}"
                )
        finally:
            await db.close()

    raise TimeoutError(
        f"Training for experiment {experiment_id} timed out after {timeout}s"
    )


@task(name="evaluate-model", retries=2, retry_delay_seconds=10)
async def evaluate_model_task(
    experiment_id: int, n_eval_episodes: int = 10
) -> dict[str, Any]:
    """Run evaluation using the evaluation service."""
    db = AsyncSessionLocal()
    try:
        request = EvaluationRequest(
            experiment_id=experiment_id,
            n_eval_episodes=n_eval_episodes,
        )
        response = await evaluate_experiment(db, request)
        return {
            "mean_reward": response.mean_reward,
            "std_reward": response.std_reward,
            "episodes": [ep.model_dump() for ep in response.episodes],
        }
    finally:
        await db.close()


@task(name="save-model", retries=3, retry_delay_seconds=5)
async def save_model_task(experiment_id: int, mean_reward: float) -> dict[str, Any]:
    """Save the trained model using the model storage service."""
    db = AsyncSessionLocal()
    try:
        result = await db.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = result.scalar_one_or_none()
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        from stable_baselines3 import A2C, DQN, PPO

        algorithms: dict[str, type] = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        algo_class = algorithms.get(experiment.algorithm)
        if algo_class is None:
            raise ValueError(f"Unknown algorithm: {experiment.algorithm}")

        from app.services.model_storage import load_model

        model, _ = await load_model(db, experiment_id)

        model_version = await save_model_to_storage(
            db,
            experiment_id=experiment_id,
            model=model,
            algorithm=experiment.algorithm,
            total_timesteps=experiment.total_timesteps or 0,
            mean_reward=mean_reward,
        )
        return {
            "version_id": model_version.id,
            "storage_path": model_version.storage_path,
        }
    finally:
        await db.close()


@task(name="promote-model", retries=2, retry_delay_seconds=5)
async def promote_model_task(
    model_version_id: int, threshold: float | None
) -> dict[str, Any]:
    """Promote a model version if it exceeds the reward threshold."""
    db = AsyncSessionLocal()
    try:
        result = await db.execute(
            select(ModelVersion).where(ModelVersion.id == model_version_id)
        )
        model_version = result.scalar_one_or_none()
        if model_version is None:
            raise ValueError(f"ModelVersion {model_version_id} not found")

        mean_reward = model_version.mean_reward or 0.0

        if threshold is not None and mean_reward < threshold:
            return {
                "promoted": False,
                "reason": (
                    f"mean_reward {mean_reward:.2f} below threshold {threshold:.2f}"
                ),
            }

        model_version.metadata_ = {
            **(model_version.metadata_ or {}),
            "promoted": True,
        }
        await db.commit()
        await db.refresh(model_version)

        reason = "promoted (no threshold)" if threshold is None else (
            f"mean_reward {mean_reward:.2f} >= threshold {threshold:.2f}"
        )
        logger.info(
            "Model version %d promoted: %s", model_version_id, reason
        )
        return {"promoted": True, "reason": reason}
    finally:
        await db.close()


@task(name="notify-completion")
async def notify_completion_task(
    pipeline_id: str, status: str, results: dict[str, Any]
) -> None:
    """Log pipeline completion. Extend to send email/Slack notifications."""
    logger.info(
        "Pipeline %s finished with status=%s | results=%s",
        pipeline_id,
        status,
        results,
    )
