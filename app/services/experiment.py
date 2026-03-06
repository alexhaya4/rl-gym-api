from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode
from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentUpdate


async def create_experiment(
    db: AsyncSession, experiment_create: ExperimentCreate, user_id: int
) -> Experiment:
    experiment = Experiment(
        name=experiment_create.name,
        environment_id=experiment_create.environment_id,
        algorithm=experiment_create.algorithm,
        hyperparameters=experiment_create.hyperparameters,
        total_timesteps=experiment_create.total_timesteps,
        user_id=user_id,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment


async def get_experiment(
    db: AsyncSession, experiment_id: int, user_id: int
) -> Experiment | None:
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id, Experiment.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def list_experiments(
    db: AsyncSession,
    user_id: int,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
) -> tuple[list[Experiment], int]:
    query = select(Experiment).where(Experiment.user_id == user_id)
    count_query = select(func.count()).select_from(Experiment).where(
        Experiment.user_id == user_id
    )

    if status is not None:
        query = query.where(Experiment.status == status)
        count_query = count_query.where(Experiment.status == status)

    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    items = list(result.scalars().all())

    return items, total


async def update_experiment(
    db: AsyncSession, experiment_id: int, user_id: int, update: ExperimentUpdate
) -> Experiment | None:
    experiment = await get_experiment(db, experiment_id, user_id)
    if experiment is None:
        return None

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(experiment, field, value)

    await db.commit()
    await db.refresh(experiment)
    return experiment


async def delete_experiment(
    db: AsyncSession, experiment_id: int, user_id: int
) -> bool:
    experiment = await get_experiment(db, experiment_id, user_id)
    if experiment is None:
        return False

    await db.delete(experiment)
    await db.commit()
    return True


async def get_experiment_episodes(
    db: AsyncSession, experiment_id: int
) -> list[Episode]:
    result = await db.execute(
        select(Episode).where(Episode.experiment_id == experiment_id)
    )
    return list(result.scalars().all())
