import itertools
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ray_utils import is_ray_available
from app.models.experiment import Experiment
from app.schemas.ray_training import (
    DistributedTrainingRequest,
    DistributedTrainingResponse,
    TrialResult,
)

logger = logging.getLogger(__name__)

ALGORITHMS: dict[str, type] = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


def _train_single_trial_impl(
    environment_id: str,
    algorithm: str,
    total_timesteps: int,
    hyperparameters: dict[str, Any],
    trial_id: str,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        env = gym.make(environment_id)
        algo_class = ALGORITHMS[algorithm]
        model = algo_class("MlpPolicy", env, **hyperparameters)
        model.learn(total_timesteps=total_timesteps)

        raw_mean, raw_std = evaluate_policy(model, env, n_eval_episodes=10)
        mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
        std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])
        env.close()

        return {
            "trial_id": trial_id,
            "hyperparameters": hyperparameters,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "training_time_seconds": round(time.monotonic() - start, 2),
            "status": "completed",
        }
    except Exception:
        logger.exception("Trial %s failed", trial_id)
        return {
            "trial_id": trial_id,
            "hyperparameters": hyperparameters,
            "mean_reward": float("-inf"),
            "std_reward": 0.0,
            "training_time_seconds": round(time.monotonic() - start, 2),
            "status": "failed",
        }


# Create Ray remote version if available
if is_ray_available():
    import ray

    train_single_trial = ray.remote(_train_single_trial_impl)
else:
    train_single_trial = None


def _generate_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


async def run_distributed_training(
    request: DistributedTrainingRequest,
    db: AsyncSession,
    user_id: int,
) -> DistributedTrainingResponse:
    job_id = str(uuid.uuid4())
    experiment_name = request.experiment_name or f"distributed-{job_id[:8]}"
    started_at = datetime.now(UTC).isoformat()

    grid = {
        "learning_rate": request.hyperparameter_grid.learning_rate,
        "n_steps": request.hyperparameter_grid.n_steps,
        "batch_size": request.hyperparameter_grid.batch_size,
        "gamma": request.hyperparameter_grid.gamma,
    }
    combinations = _generate_combinations(grid)
    total_trials = len(combinations)

    results: list[TrialResult] = []

    if is_ray_available() and train_single_trial is not None:
        import ray

        logger.info(
            "Running %d trials with Ray (max_concurrent=%d)",
            total_trials,
            request.max_concurrent_trials,
        )

        pending: list[Any] = []
        combo_iter = iter(combinations)
        trial_idx = 0

        # Submit initial batch
        for combo in itertools.islice(combo_iter, request.max_concurrent_trials):
            trial_id = f"{job_id[:8]}-trial-{trial_idx}"
            ref = train_single_trial.remote(
                request.environment_id,
                request.algorithm,
                request.total_timesteps,
                combo,
                trial_id,
            )
            pending.append(ref)
            trial_idx += 1

        # Process results and submit more as slots open
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for ref in done:
                result_dict = ray.get(ref)
                results.append(TrialResult(**result_dict))

                # Submit next trial if available
                next_combo = next(combo_iter, None)
                if next_combo is not None:
                    trial_id = f"{job_id[:8]}-trial-{trial_idx}"
                    new_ref = train_single_trial.remote(
                        request.environment_id,
                        request.algorithm,
                        request.total_timesteps,
                        next_combo,
                        trial_id,
                    )
                    pending.append(new_ref)
                    trial_idx += 1
    else:
        logger.warning(
            "Ray not available, running %d trials sequentially", total_trials
        )
        for idx, combo in enumerate(combinations):
            trial_id = f"{job_id[:8]}-trial-{idx}"
            result_dict = _train_single_trial_impl(
                request.environment_id,
                request.algorithm,
                request.total_timesteps,
                combo,
                trial_id,
            )
            results.append(TrialResult(**result_dict))

    # Find best trial
    completed_results = [r for r in results if r.status == "completed"]
    best_trial: TrialResult | None = None
    best_hyperparameters: dict[str, Any] | None = None

    if completed_results:
        best_trial = max(completed_results, key=lambda r: r.mean_reward)
        best_hyperparameters = best_trial.hyperparameters

        # Create experiment record for best trial
        experiment = Experiment(
            name=experiment_name,
            environment_id=request.environment_id,
            algorithm=request.algorithm,
            hyperparameters=best_hyperparameters,
            total_timesteps=request.total_timesteps,
            status="completed",
            mean_reward=best_trial.mean_reward,
            std_reward=best_trial.std_reward,
            user_id=user_id,
            completed_at=datetime.now(UTC),
        )
        db.add(experiment)
        await db.commit()

    completed_at = datetime.now(UTC).isoformat()

    return DistributedTrainingResponse(
        job_id=job_id,
        experiment_name=experiment_name,
        total_trials=total_trials,
        status="completed",
        results=results,
        best_trial=best_trial,
        best_hyperparameters=best_hyperparameters,
        started_at=started_at,
        completed_at=completed_at,
    )
