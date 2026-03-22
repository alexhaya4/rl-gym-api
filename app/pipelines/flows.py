import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from prefect import flow
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.experiment import Experiment
from app.pipelines.tasks import (
    create_experiment_task,
    evaluate_model_task,
    notify_completion_task,
    promote_model_task,
    save_model_task,
    train_model_task,
)
from app.schemas.pipeline import PipelineRunResponse, PipelineStepResult

logger = logging.getLogger(__name__)


async def _run_step(
    step_name: str,
    coro: Any,
) -> tuple[PipelineStepResult, Any]:
    """Execute a pipeline step, capturing timing and errors."""
    start = time.monotonic()
    try:
        result = await coro
        duration = time.monotonic() - start
        step = PipelineStepResult(
            step_name=step_name,
            status="success",
            output=result if isinstance(result, dict) else None,
            duration_seconds=round(duration, 2),
        )
        return step, result
    except Exception as exc:
        duration = time.monotonic() - start
        step = PipelineStepResult(
            step_name=step_name,
            status="failed",
            error=str(exc),
            duration_seconds=round(duration, 2),
        )
        return step, None


@flow(name="rl-training-pipeline", retries=1)
async def rl_training_pipeline(
    request: dict[str, Any], user_id: int, pipeline_id: str
) -> dict[str, Any]:
    """Full RL training pipeline: create, train, evaluate, save, promote, notify."""
    started_at = datetime.now(UTC).isoformat()
    experiment_name = request.get("experiment_name") or (
        f"{request['algorithm']}-{request['environment_id']}"
    )
    steps: list[PipelineStepResult] = []
    experiment_id: int | None = None
    model_version_id: int | None = None
    promoted = False
    status = "running"

    # Step 1: Create experiment
    db = AsyncSessionLocal()
    try:
        step, result = await _run_step(
            "create-experiment",
            create_experiment_task(db, request, user_id),
        )
        steps.append(step)
        if step.status == "failed":
            status = "failed"
            return _build_response(
                pipeline_id, experiment_name, status, steps, started_at,
                message=f"Failed at create-experiment: {step.error}",
            )
        experiment_id = result["id"]
    finally:
        await db.close()

    # Step 2: Train model
    step, result = await _run_step(
        "train-model",
        train_model_task(experiment_id, request),
    )
    steps.append(step)
    if step.status == "failed":
        status = "failed"
        await notify_completion_task(pipeline_id, status, {"error": step.error})
        return _build_response(
            pipeline_id, experiment_name, status, steps, started_at,
            experiment_id=experiment_id,
            message=f"Failed at train-model: {step.error}",
        )
    train_result = result

    # Step 3: Evaluate model
    step, result = await _run_step(
        "evaluate-model",
        evaluate_model_task(experiment_id),
    )
    steps.append(step)
    if step.status == "failed":
        status = "failed"
        await notify_completion_task(pipeline_id, status, {"error": step.error})
        return _build_response(
            pipeline_id, experiment_name, status, steps, started_at,
            experiment_id=experiment_id,
            message=f"Failed at evaluate-model: {step.error}",
        )
    eval_result = result

    # Step 4: Save model checkpoint
    mean_reward = eval_result.get("mean_reward", train_result.get("mean_reward", 0.0))
    step, result = await _run_step(
        "save-model",
        save_model_task(experiment_id, mean_reward),
    )
    steps.append(step)
    if step.status == "failed":
        status = "failed"
        await notify_completion_task(pipeline_id, status, {"error": step.error})
        return _build_response(
            pipeline_id, experiment_name, status, steps, started_at,
            experiment_id=experiment_id,
            message=f"Failed at save-model: {step.error}",
        )
    model_version_id = result["version_id"]

    # Step 5: Promote model
    threshold = request.get("min_reward_threshold")
    step, result = await _run_step(
        "promote-model",
        promote_model_task(model_version_id, threshold),
    )
    steps.append(step)
    if step.status == "success":
        promoted = result.get("promoted", False)

    # Step 6: Notify completion
    status = "completed"
    step, _ = await _run_step(
        "notify-completion",
        notify_completion_task(
            pipeline_id,
            status,
            {
                "experiment_id": experiment_id,
                "model_version_id": model_version_id,
                "promoted": promoted,
                "mean_reward": mean_reward,
            },
        ),
    )
    steps.append(step)

    return _build_response(
        pipeline_id, experiment_name, status, steps, started_at,
        experiment_id=experiment_id,
        model_version_id=model_version_id,
        promoted=promoted,
    )


@flow(name="hyperparameter-search-pipeline")
async def hyperparameter_search_pipeline(
    request: dict[str, Any], user_id: int, pipeline_id: str
) -> dict[str, Any]:
    """Pipeline: distributed hyperparameter search, save best model, promote."""
    from app.schemas.ray_training import DistributedTrainingRequest
    from app.services.ray_training import run_distributed_training

    started_at = datetime.now(UTC).isoformat()
    experiment_name = request.get("experiment_name") or (
        f"hpsearch-{request['algorithm']}-{request['environment_id']}"
    )
    steps: list[PipelineStepResult] = []
    experiment_id: int | None = None
    model_version_id: int | None = None
    promoted = False
    status = "running"

    # Step 1: Run distributed training
    db = AsyncSessionLocal()
    try:
        dist_request = DistributedTrainingRequest(
            environment_id=request["environment_id"],
            algorithm=request.get("algorithm", "PPO"),
            total_timesteps=request.get("total_timesteps", 10000),
            hyperparameter_grid=request.get("hyperparameter_grid", {}),
            max_concurrent_trials=request.get("max_concurrent_trials", 4),
            experiment_name=experiment_name,
        )

        step, result = await _run_step(
            "distributed-training",
            run_distributed_training(dist_request, db, user_id),
        )
        steps.append(step)
        if step.status == "failed":
            status = "failed"
            return _build_response(
                pipeline_id, experiment_name, status, steps, started_at,
                message=f"Failed at distributed-training: {step.error}",
            )
    finally:
        await db.close()

    # Extract best trial
    best_trial = result.get("best_trial") if isinstance(result, dict) else None
    if best_trial is None:
        status = "failed"
        steps.append(PipelineStepResult(
            step_name="extract-best-trial",
            status="failed",
            error="No completed trials found",
            duration_seconds=0.0,
        ))
        return _build_response(
            pipeline_id, experiment_name, status, steps, started_at,
            message="No completed trials found",
        )

    # Look up the experiment created by run_distributed_training
    db = AsyncSessionLocal()
    try:
        db_result = await db.execute(
            select(Experiment)
            .where(Experiment.name == experiment_name, Experiment.user_id == user_id)
            .order_by(Experiment.created_at.desc())
            .limit(1)
        )
        experiment = db_result.scalar_one_or_none()
        if experiment is not None:
            experiment_id = experiment.id
    finally:
        await db.close()

    if experiment_id is None:
        status = "failed"
        steps.append(PipelineStepResult(
            step_name="lookup-experiment",
            status="failed",
            error="Could not find experiment created by distributed training",
            duration_seconds=0.0,
        ))
        return _build_response(
            pipeline_id, experiment_name, status, steps, started_at,
            message="Experiment lookup failed",
        )

    best_mean_reward = best_trial.get("mean_reward", 0.0) if isinstance(best_trial, dict) else best_trial.mean_reward

    # Step 2: Save best model
    step, result = await _run_step(
        "save-model",
        save_model_task(experiment_id, best_mean_reward),
    )
    steps.append(step)
    if step.status == "failed":
        status = "failed"
        return _build_response(
            pipeline_id, experiment_name, status, steps, started_at,
            experiment_id=experiment_id,
            message=f"Failed at save-model: {step.error}",
        )
    model_version_id = result["version_id"]

    # Step 3: Promote model
    threshold = request.get("min_reward_threshold")
    step, result = await _run_step(
        "promote-model",
        promote_model_task(model_version_id, threshold),
    )
    steps.append(step)
    if step.status == "success":
        promoted = result.get("promoted", False)

    status = "completed"
    return _build_response(
        pipeline_id, experiment_name, status, steps, started_at,
        experiment_id=experiment_id,
        model_version_id=model_version_id,
        promoted=promoted,
    )


@flow(name="scheduled-retraining-pipeline")
async def scheduled_retraining_pipeline(
    environment_id: str, algorithm: str, user_id: int
) -> dict[str, Any]:
    """Scheduled retraining: skip if recent experiment exists, else run full pipeline."""
    pipeline_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()
    experiment_name = f"scheduled-{algorithm}-{environment_id}"
    steps: list[PipelineStepResult] = []

    # Step 1: Check for recent experiments (within last 7 days)
    start = time.monotonic()
    db = AsyncSessionLocal()
    recent_experiment: Experiment | None = None
    try:
        cutoff = datetime.now(UTC) - timedelta(days=7)
        db_result = await db.execute(
            select(Experiment)
            .where(
                Experiment.environment_id == environment_id,
                Experiment.algorithm == algorithm,
                Experiment.user_id == user_id,
                Experiment.status == "completed",
                Experiment.created_at >= cutoff,
            )
            .order_by(Experiment.created_at.desc())
            .limit(1)
        )
        recent_experiment = db_result.scalar_one_or_none()
    finally:
        await db.close()
    duration = time.monotonic() - start

    if recent_experiment is not None:
        steps.append(PipelineStepResult(
            step_name="check-recent-experiment",
            status="skipped",
            output={
                "experiment_id": recent_experiment.id,
                "created_at": recent_experiment.created_at.isoformat(),
                "mean_reward": recent_experiment.mean_reward,
            },
            duration_seconds=round(duration, 2),
        ))
        return _build_response(
            pipeline_id, experiment_name, "completed", steps, started_at,
            experiment_id=recent_experiment.id,
            message="Skipped retraining: recent experiment exists within 7 days",
        )

    steps.append(PipelineStepResult(
        step_name="check-recent-experiment",
        status="success",
        output={"message": "No recent experiment found, proceeding with training"},
        duration_seconds=round(duration, 2),
    ))

    # Step 2: Run full training pipeline
    request = {
        "environment_id": environment_id,
        "algorithm": algorithm,
        "experiment_name": experiment_name,
        "total_timesteps": 10000,
        "hyperparameters": {},
        "retrain_if_exists": True,
    }
    pipeline_result = await rl_training_pipeline(request, user_id, pipeline_id)

    # Merge steps from inner pipeline
    if isinstance(pipeline_result, dict) and "steps" in pipeline_result:
        for inner_step in pipeline_result["steps"]:
            if isinstance(inner_step, dict):
                steps.append(PipelineStepResult(**inner_step))
            else:
                steps.append(inner_step)

    return _build_response(
        pipeline_id,
        experiment_name,
        pipeline_result.get("status", "completed") if isinstance(pipeline_result, dict) else "completed",
        steps,
        started_at,
        experiment_id=pipeline_result.get("experiment_id") if isinstance(pipeline_result, dict) else None,
        model_version_id=pipeline_result.get("model_version_id") if isinstance(pipeline_result, dict) else None,
        promoted=pipeline_result.get("promoted", False) if isinstance(pipeline_result, dict) else False,
        message=pipeline_result.get("message") if isinstance(pipeline_result, dict) else None,
    )


def _build_response(
    pipeline_id: str,
    experiment_name: str,
    status: str,
    steps: list[PipelineStepResult],
    started_at: str,
    *,
    experiment_id: int | None = None,
    model_version_id: int | None = None,
    promoted: bool = False,
    message: str | None = None,
) -> dict[str, Any]:
    """Build a PipelineRunResponse as a serialisable dict."""
    completed_at = datetime.now(UTC).isoformat() if status in ("completed", "failed") else None
    response = PipelineRunResponse(
        pipeline_id=pipeline_id,
        experiment_name=experiment_name,
        status=status,
        steps=steps,
        started_at=started_at,
        completed_at=completed_at,
        experiment_id=experiment_id,
        model_version_id=model_version_id,
        promoted=promoted,
        message=message,
    )
    return response.model_dump()
