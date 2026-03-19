import logging
import tempfile
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from app.config import get_settings
from app.core.storage import S3Storage, get_storage
from app.models.model_version import ModelVersion

logger = logging.getLogger(__name__)

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


async def save_model(
    db: AsyncSession,
    experiment_id: int,
    model: BaseAlgorithm,
    algorithm: str,
    total_timesteps: int,
    mean_reward: float,
) -> ModelVersion:
    """Save an SB3 model to storage and create a version record."""
    result = await db.execute(
        select(func.coalesce(func.max(ModelVersion.version), 0)).where(
            ModelVersion.experiment_id == experiment_id
        )
    )
    next_version = result.scalar_one() + 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{algorithm}.zip"
        model.save(str(tmp_path))
        data = tmp_path.read_bytes()
        file_size = len(data)

    storage = get_storage()
    settings = get_settings()
    storage_path = f"models/{experiment_id}/v{next_version}/{algorithm}.zip"
    await storage.save(storage_path, data)

    model_version = ModelVersion(
        experiment_id=experiment_id,
        version=next_version,
        storage_path=storage_path,
        storage_backend=settings.STORAGE_BACKEND,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        mean_reward=mean_reward,
        file_size_bytes=file_size,
    )
    db.add(model_version)
    await db.commit()
    await db.refresh(model_version)
    return model_version


async def load_model(
    db: AsyncSession,
    experiment_id: int,
    version: int | None = None,
) -> tuple[BaseAlgorithm, ModelVersion]:
    """Load an SB3 model from storage. Uses latest version if not specified."""
    if version is not None:
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.experiment_id == experiment_id,
                ModelVersion.version == version,
            )
        )
    else:
        result = await db.execute(
            select(ModelVersion)
            .where(ModelVersion.experiment_id == experiment_id)
            .order_by(ModelVersion.version.desc())
            .limit(1)
        )
    model_version = result.scalar_one_or_none()
    if model_version is None:
        raise ValueError(
            f"No model version found for experiment {experiment_id}"
            + (f" version {version}" if version else "")
        )

    algo_class = ALGORITHMS.get(model_version.algorithm)
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {model_version.algorithm}")

    storage = get_storage()
    data = await storage.load(model_version.storage_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{model_version.algorithm}.zip"
        tmp_path.write_bytes(data)
        model = algo_class.load(str(tmp_path))

    return model, model_version


async def list_model_versions(
    db: AsyncSession, experiment_id: int
) -> list[ModelVersion]:
    """List all model versions for an experiment."""
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.experiment_id == experiment_id)
        .order_by(ModelVersion.version.asc())
    )
    return list(result.scalars().all())


async def delete_model_version(db: AsyncSession, version_id: int) -> bool:
    """Delete a model version from storage and DB."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    model_version = result.scalar_one_or_none()
    if model_version is None:
        return False

    storage = get_storage()
    await storage.delete(model_version.storage_path)

    await db.delete(model_version)
    await db.commit()
    return True


def get_download_url(model_version: ModelVersion) -> str:
    """Generate a download URL for the model version."""
    if model_version.storage_backend == "s3":
        storage = get_storage()
        if isinstance(storage, S3Storage):
            url: str = storage._client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": storage.bucket,
                    "Key": model_version.storage_path,
                },
                ExpiresIn=3600,
            )
            return url
    return f"/api/v1/models/{model_version.id}/download"
