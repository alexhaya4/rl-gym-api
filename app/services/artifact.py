import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import get_storage
from app.models.artifact import Artifact, ArtifactLineage
from app.models.experiment import Experiment
from app.schemas.artifact import ArtifactCreate


async def create_artifact(
    db: AsyncSession,
    artifact_create: ArtifactCreate,
    user_id: int,
    file_bytes: bytes | None = None,
) -> Artifact:
    """Create an artifact record, optionally saving file data to storage."""
    storage_path: str | None = None
    checksum: str | None = None
    size_bytes: int | None = None

    if file_bytes is not None:
        checksum = hashlib.sha256(file_bytes).hexdigest()
        size_bytes = len(file_bytes)
        storage_path = (
            f"artifacts/{user_id}/{artifact_create.artifact_type}/{checksum[:16]}/"
            f"{artifact_create.name}"
        )
        storage = get_storage()
        await storage.save(storage_path, file_bytes)

    artifact = Artifact(
        name=artifact_create.name,
        artifact_type=artifact_create.artifact_type,
        storage_path=storage_path,
        checksum=checksum,
        size_bytes=size_bytes,
        metadata_=artifact_create.metadata or None,
        experiment_id=artifact_create.experiment_id,
        user_id=user_id,
    )
    db.add(artifact)
    await db.commit()
    await db.refresh(artifact)
    return artifact


async def list_artifacts(
    db: AsyncSession,
    user_id: int,
    experiment_id: int | None = None,
) -> list[Artifact]:
    """List artifacts for a user, optionally filtered by experiment."""
    query = select(Artifact).where(Artifact.user_id == user_id)
    if experiment_id is not None:
        query = query.where(Artifact.experiment_id == experiment_id)
    query = query.order_by(Artifact.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_artifact(
    db: AsyncSession, artifact_id: int, user_id: int
) -> Artifact | None:
    """Get a single artifact by id with user isolation."""
    result = await db.execute(
        select(Artifact).where(
            Artifact.id == artifact_id, Artifact.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def delete_artifact(
    db: AsyncSession, artifact_id: int, user_id: int
) -> bool:
    """Delete an artifact from DB and storage."""
    artifact = await get_artifact(db, artifact_id, user_id)
    if artifact is None:
        return False

    if artifact.storage_path:
        storage = get_storage()
        await storage.delete(artifact.storage_path)

    await db.delete(artifact)
    await db.commit()
    return True


async def set_lineage(
    db: AsyncSession,
    parent_id: int,
    child_id: int,
    relationship_type: str,
    description: str | None = None,
) -> ArtifactLineage:
    """Create a lineage record and update the child experiment's parent."""
    lineage = ArtifactLineage(
        parent_experiment_id=parent_id,
        child_experiment_id=child_id,
        relationship_type=relationship_type,
        description=description,
    )
    db.add(lineage)

    result = await db.execute(
        select(Experiment).where(Experiment.id == child_id)
    )
    child_exp = result.scalar_one_or_none()
    if child_exp is not None:
        child_exp.parent_experiment_id = parent_id

    await db.commit()
    await db.refresh(lineage)
    return lineage
