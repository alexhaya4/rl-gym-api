from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.algorithms import ALL_ALGORITHMS, validate_algorithm_environment
from app.db.session import get_db
from app.dependencies import get_arq_redis, get_current_active_user
from app.models.experiment import Experiment
from app.models.job import Job
from app.models.user import User
from app.schemas.job import JobResponse
from app.schemas.training import TrainingConfig, TrainingResult, TrainingStatus
from app.services.training import get_training_status, list_training_sessions

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/", response_model=TrainingStatus, status_code=status.HTTP_202_ACCEPTED)
async def create_training(
    config: TrainingConfig,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
    arq_redis: ArqRedis = Depends(get_arq_redis),
) -> TrainingStatus:
    if config.algorithm not in ALL_ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported algorithm: {config.algorithm}. "
            f"Available: {sorted(ALL_ALGORITHMS)}",
        )
    compatible, error = validate_algorithm_environment(
        config.algorithm, config.environment_id
    )
    if not compatible:
        raise HTTPException(status_code=400, detail=error)

    experiment = Experiment(
        name=config.experiment_name or f"{config.algorithm}_{config.environment_id}",
        environment_id=config.environment_id,
        algorithm=config.algorithm,
        status="queued",
        hyperparameters=config.hyperparameters,
        total_timesteps=config.total_timesteps,
        user_id=current_user.id,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)

    config_dict = {
        "environment_id": config.environment_id,
        "algorithm": config.algorithm,
        "total_timesteps": config.total_timesteps,
        "hyperparameters": config.hyperparameters,
        "n_envs": config.n_envs,
    }

    arq_job = await arq_redis.enqueue_job(
        "run_training_job",
        experiment.id,
        config_dict,
    )
    if arq_job is None:
        raise HTTPException(status_code=500, detail="Failed to enqueue training job")

    job = Job(
        id=arq_job.job_id,
        experiment_id=experiment.id,
        status="queued",
    )
    db.add(job)
    await db.commit()

    return TrainingStatus(
        experiment_id=experiment.id,
        status="queued",
        environment_id=config.environment_id,
        algorithm=config.algorithm,
        total_timesteps=config.total_timesteps,
        job_id=arq_job.job_id,
    )


@router.get("/", response_model=list[TrainingStatus])
async def list_trainings(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[TrainingStatus]:
    sessions = await list_training_sessions(db, current_user.id)
    return [TrainingStatus(**s) for s in sessions]


@router.get("/{experiment_id}", response_model=TrainingStatus)
async def get_training(
    experiment_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrainingStatus:
    session = await get_training_status(db, experiment_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return TrainingStatus(**session)


@router.get("/{experiment_id}/job", response_model=JobResponse)
async def get_training_job(
    experiment_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    result = await db.execute(
        select(Job).where(Job.experiment_id == experiment_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found for this experiment")
    return JobResponse.model_validate(job)


@router.get("/{experiment_id}/result", response_model=TrainingResult)
async def get_training_result(
    experiment_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrainingResult:
    session = await get_training_status(db, experiment_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not yet completed")
    return TrainingResult(**session)
