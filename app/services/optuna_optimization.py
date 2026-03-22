import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import optuna
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3.common.evaluation import evaluate_policy

from app.core.algorithms import get_algorithm_class
from app.models.optuna_study import OptunaStudy
from app.schemas.optuna import OptimizationRequest, OptimizationResponse, TrialInfo

logger = logging.getLogger(__name__)

# In-memory cache of completed studies for history queries
_study_cache: dict[str, optuna.Study] = {}

DEFAULT_HYPERPARAMETER_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "PPO": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "n_steps": {"type": "int", "low": 128, "high": 2048},
        "batch_size": {"type": "int", "low": 32, "high": 256},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "gae_lambda": {"type": "float", "low": 0.9, "high": 1.0},
    },
    "A2C": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "n_steps": {"type": "int", "low": 5, "high": 128},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
    },
    "DQN": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "int", "low": 32, "high": 256},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "exploration_fraction": {"type": "float", "low": 0.1, "high": 0.5},
    },
    "SAC": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "int", "low": 64, "high": 512},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "tau": {"type": "float", "low": 0.001, "high": 0.1},
    },
    "TD3": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "int", "low": 64, "high": 512},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "tau": {"type": "float", "low": 0.001, "high": 0.1},
    },
    "DDPG": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "int", "low": 64, "high": 512},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "tau": {"type": "float", "low": 0.001, "high": 0.1},
    },
    "TQC": {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "batch_size": {"type": "int", "low": 64, "high": 512},
        "gamma": {"type": "float", "low": 0.9, "high": 0.9999},
        "top_quantiles_to_drop_per_net": {"type": "int", "low": 2, "high": 5},
    },
}


def _sample_hyperparameters(
    trial: optuna.Trial, space: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Sample hyperparameters from a search space definition."""
    params: dict[str, Any] = {}
    for name, config in space.items():
        param_type = config["type"]
        if param_type == "float":
            params[name] = trial.suggest_float(
                name, config["low"], config["high"], log=config.get("log", False)
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(name, config["low"], config["high"])
    return params


def create_objective(
    environment_id: str,
    algorithm: str,
    total_timesteps: int,
    n_eval_episodes: int,
    hyperparameter_space: dict[str, dict[str, Any]] | None = None,
) -> Callable[[optuna.Trial], float]:
    """Return an Optuna objective function for RL hyperparameter optimization."""
    space = hyperparameter_space or DEFAULT_HYPERPARAMETER_SPACES.get(algorithm, {})

    def objective(trial: optuna.Trial) -> float:
        params = _sample_hyperparameters(trial, space)

        env = gym.make(environment_id)
        try:
            algo_class = get_algorithm_class(algorithm)
            model = algo_class("MlpPolicy", env, **params)
            model.learn(total_timesteps=total_timesteps)

            raw_mean, _raw_std = evaluate_policy(
                model, env, n_eval_episodes=n_eval_episodes
            )
            mean_reward = (
                float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
            )

            trial.report(mean_reward, step=total_timesteps)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return mean_reward
        finally:
            env.close()

    return objective


def _run_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    timeout: int | None,
) -> None:
    """Blocking call to study.optimize — meant to run in an executor."""
    study.optimize(objective, n_trials=n_trials, timeout=timeout)


def _trial_to_info(trial: optuna.trial.FrozenTrial) -> TrialInfo:
    """Convert an Optuna FrozenTrial to a TrialInfo schema."""
    status_map = {
        optuna.trial.TrialState.COMPLETE: "completed",
        optuna.trial.TrialState.PRUNED: "pruned",
        optuna.trial.TrialState.FAIL: "failed",
        optuna.trial.TrialState.RUNNING: "running",
        optuna.trial.TrialState.WAITING: "running",
    }
    duration = (
        (trial.datetime_complete - trial.datetime_start).total_seconds()
        if trial.datetime_start and trial.datetime_complete
        else None
    )
    return TrialInfo(
        trial_number=trial.number,
        status=status_map.get(trial.state, "failed"),
        hyperparameters=trial.params,
        mean_reward=trial.value,
        duration_seconds=round(duration, 2) if duration is not None else None,
    )


def _build_response(
    study_record: OptunaStudy,
    study: optuna.Study | None = None,
) -> OptimizationResponse:
    """Build an OptimizationResponse from DB record and optional Optuna study."""
    trials: list[TrialInfo] = []
    best_trial: TrialInfo | None = None
    best_hyperparameters: dict[str, Any] | None = study_record.best_hyperparameters
    best_mean_reward: float | None = study_record.best_mean_reward
    improvement: float | None = None

    if study is not None:
        trials = [_trial_to_info(t) for t in study.trials]
        if study.best_trial is not None:
            best_trial = _trial_to_info(study.best_trial)

    return OptimizationResponse(
        study_id=study_record.study_id,
        experiment_name=study_record.study_name,
        status=study_record.status,
        n_trials=study_record.n_trials,
        n_completed=study_record.n_completed,
        n_pruned=study_record.n_pruned,
        best_trial=best_trial,
        best_hyperparameters=best_hyperparameters,
        best_mean_reward=best_mean_reward,
        trials=trials,
        started_at=study_record.created_at.isoformat(),
        completed_at=(
            study_record.updated_at.isoformat()
            if study_record.status in ("completed", "failed")
            else None
        ),
        improvement_over_default=improvement,
    )


async def run_optimization(
    db: AsyncSession, request: OptimizationRequest, user_id: int
) -> OptimizationResponse:
    """Create and run an Optuna study for hyperparameter optimization."""
    study_id = str(uuid.uuid4())
    experiment_name = request.experiment_name or (
        f"optuna-{request.algorithm}-{request.environment_id}"
    )

    # Create DB record
    study_record = OptunaStudy(
        study_id=study_id,
        study_name=experiment_name,
        environment_id=request.environment_id,
        algorithm=request.algorithm,
        status="running",
        n_trials=request.n_trials,
        user_id=user_id,
    )
    db.add(study_record)
    await db.commit()
    await db.refresh(study_record)

    # Configure Optuna study
    pruner = (
        optuna.pruners.MedianPruner() if request.pruning_enabled else optuna.pruners.NopPruner()
    )
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=experiment_name,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
    )

    objective = create_objective(
        environment_id=request.environment_id,
        algorithm=request.algorithm,
        total_timesteps=request.total_timesteps,
        n_eval_episodes=request.n_eval_episodes,
        hyperparameter_space=request.hyperparameter_space,
    )

    # Run optimization in thread pool
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None, _run_study, study, objective, request.n_trials, request.timeout_seconds
        )
        status = "completed"
    except Exception:
        logger.exception("Optuna study %s failed", study_id)
        status = "failed"

    # Compute improvement over default hyperparameters
    improvement: float | None = None
    if study.best_trial is not None:
        best_reward = study.best_trial.value
        if best_reward is not None:
            try:
                default_reward = _evaluate_default(
                    request.environment_id,
                    request.algorithm,
                    request.total_timesteps,
                    request.n_eval_episodes,
                )
                if default_reward != 0.0:
                    improvement = round(
                        ((best_reward - default_reward) / abs(default_reward)) * 100, 2
                    )
            except Exception:
                logger.warning("Failed to compute improvement over default")

    # Gather trial stats
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]

    # Update DB record
    study_record.status = status
    study_record.n_completed = len(completed_trials)
    study_record.n_pruned = len(pruned_trials)
    if study.best_trial is not None:
        study_record.best_mean_reward = study.best_trial.value
        study_record.best_hyperparameters = study.best_trial.params
    await db.commit()
    await db.refresh(study_record)

    # Cache study for history queries
    _study_cache[study_id] = study

    response = _build_response(study_record, study)
    response.improvement_over_default = improvement
    return response


def _evaluate_default(
    environment_id: str,
    algorithm: str,
    total_timesteps: int,
    n_eval_episodes: int,
) -> float:
    """Train with default hyperparameters and return mean reward."""
    env = gym.make(environment_id)
    try:
        algo_class = get_algorithm_class(algorithm)
        model = algo_class("MlpPolicy", env, learning_rate=3e-4)
        model.learn(total_timesteps=total_timesteps)
        raw_mean, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
        return float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
    finally:
        env.close()


async def get_study(
    db: AsyncSession, study_id: str, user_id: int
) -> OptimizationResponse | None:
    """Retrieve a single study by study_id."""
    result = await db.execute(
        select(OptunaStudy).where(
            OptunaStudy.study_id == study_id, OptunaStudy.user_id == user_id
        )
    )
    study_record = result.scalar_one_or_none()
    if study_record is None:
        return None
    return _build_response(study_record)


async def list_studies(
    db: AsyncSession, user_id: int
) -> list[OptimizationResponse]:
    """List all studies for a user."""
    result = await db.execute(
        select(OptunaStudy)
        .where(OptunaStudy.user_id == user_id)
        .order_by(OptunaStudy.created_at.desc())
    )
    records = list(result.scalars().all())
    return [_build_response(record) for record in records]


async def get_optimization_history(study_id: str) -> list[dict[str, Any]]:
    """Return trial history for visualization."""
    study = _study_cache.get(study_id)
    if study is None:
        return []

    status_map = {
        optuna.trial.TrialState.COMPLETE: "completed",
        optuna.trial.TrialState.PRUNED: "pruned",
        optuna.trial.TrialState.FAIL: "failed",
        optuna.trial.TrialState.RUNNING: "running",
        optuna.trial.TrialState.WAITING: "running",
    }

    return [
        {
            "trial_number": trial.number,
            "mean_reward": trial.value,
            "hyperparameters": trial.params,
            "status": status_map.get(trial.state, "failed"),
        }
        for trial in study.trials
    ]
