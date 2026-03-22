import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_version import ModelVersion
from app.models.registry import ModelRegistry
from app.schemas.registry import ComparisonResult, PromoteRequest, RegistryEntry

VALID_TRANSITIONS: dict[str, list[str]] = {
    "development": ["staging"],
    "staging": ["production", "archived"],
    "production": ["archived"],
    "archived": ["development"],
}


async def register_model(
    db: AsyncSession,
    model_version_id: int,
    environment_id: str,
    algorithm: str,
    user_id: int,
) -> ModelRegistry:
    """Create a new registry entry in development stage."""
    # Look up model version to get mean_reward
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == model_version_id)
    )
    model_version = result.scalar_one_or_none()
    if model_version is None:
        raise ValueError(f"ModelVersion {model_version_id} not found")

    name = f"{environment_id}-{algorithm}-{uuid.uuid4().hex[:8]}"
    entry = ModelRegistry(
        name=name,
        environment_id=environment_id,
        algorithm=algorithm,
        stage="development",
        model_version_id=model_version_id,
        mean_reward=model_version.mean_reward,
        promoted_by=user_id,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


async def promote_model(
    db: AsyncSession,
    registry_id: int,
    request: PromoteRequest,
    user_id: int,
) -> ModelRegistry:
    """Promote a registry entry to a new stage."""
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.id == registry_id)
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise ValueError(f"Registry entry {registry_id} not found")

    allowed = VALID_TRANSITIONS.get(entry.stage, [])
    if request.target_stage not in allowed:
        raise ValueError(
            f"Cannot transition from '{entry.stage}' to '{request.target_stage}'. "
            f"Allowed: {allowed}"
        )

    # If promoting to production, archive current production model
    if request.target_stage == "production":
        current_prod = await get_production_model(
            db, entry.environment_id, entry.algorithm
        )
        if current_prod is not None:
            entry.previous_production_id = current_prod.model_version_id
            current_prod.is_current = False
            current_prod.stage = "archived"

    entry.stage = request.target_stage
    entry.promoted_by = user_id
    entry.promotion_comment = request.comment
    await db.commit()
    await db.refresh(entry)
    return entry


async def rollback_production(
    db: AsyncSession,
    environment_id: str,
    algorithm: str,
    user_id: int,
) -> ModelRegistry | None:
    """Rollback to the previous production model."""
    current_prod = await get_production_model(db, environment_id, algorithm)
    if current_prod is None:
        return None

    prev_version_id = current_prod.previous_production_id
    if prev_version_id is None:
        return None

    # Find the registry entry for the previous production model
    result = await db.execute(
        select(ModelRegistry).where(
            ModelRegistry.model_version_id == prev_version_id,
            ModelRegistry.environment_id == environment_id,
            ModelRegistry.algorithm == algorithm,
        )
    )
    previous_entry = result.scalar_one_or_none()
    if previous_entry is None:
        return None

    # Demote current production to archived
    current_prod.stage = "archived"
    current_prod.is_current = False

    # Restore previous model to production
    previous_entry.stage = "production"
    previous_entry.is_current = True
    previous_entry.promoted_by = user_id
    previous_entry.promotion_comment = "Rollback from previous production"

    await db.commit()
    await db.refresh(previous_entry)
    return previous_entry


async def get_production_model(
    db: AsyncSession,
    environment_id: str,
    algorithm: str,
) -> ModelRegistry | None:
    """Return the current production model for an environment/algorithm."""
    result = await db.execute(
        select(ModelRegistry).where(
            ModelRegistry.environment_id == environment_id,
            ModelRegistry.algorithm == algorithm,
            ModelRegistry.stage == "production",
            ModelRegistry.is_current.is_(True),
        )
    )
    return result.scalar_one_or_none()


async def list_registry(
    db: AsyncSession,
    user_id: int,
    stage: str | None = None,
) -> list[ModelRegistry]:
    """List registry entries, optionally filtered by stage."""
    query = select(ModelRegistry).where(ModelRegistry.promoted_by == user_id)
    if stage is not None:
        query = query.where(ModelRegistry.stage == stage)
    query = query.order_by(ModelRegistry.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


async def compare_models(
    db: AsyncSession,
    candidate_version_id: int,
    environment_id: str,
    algorithm: str,
) -> ComparisonResult:
    """Compare a candidate model version against current production."""
    # Get candidate model version
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == candidate_version_id)
    )
    candidate_version = result.scalar_one_or_none()
    if candidate_version is None:
        raise ValueError(f"ModelVersion {candidate_version_id} not found")

    candidate_entry = RegistryEntry(
        id=0,
        name=f"{environment_id}-{algorithm}-candidate",
        environment_id=environment_id,
        algorithm=algorithm,
        stage="development",
        model_version_id=candidate_version_id,
        previous_production_id=None,
        mean_reward=candidate_version.mean_reward,
        promoted_by=None,
        promotion_comment=None,
        is_current=False,
        created_at=candidate_version.created_at,
        updated_at=candidate_version.created_at,
    )

    # Get current production model
    current_prod = await get_production_model(db, environment_id, algorithm)
    current_prod_entry: RegistryEntry | None = None
    if current_prod is not None:
        current_prod_entry = RegistryEntry.model_validate(current_prod)

    # Calculate delta and recommendation
    candidate_reward = candidate_version.mean_reward
    prod_reward = current_prod.mean_reward if current_prod is not None else None

    mean_reward_delta: float | None = None
    recommendation: str

    if candidate_reward is None:
        recommendation = "needs_evaluation"
    elif prod_reward is None:
        # No production model exists — promote
        recommendation = "promote"
    else:
        mean_reward_delta = candidate_reward - prod_reward
        if prod_reward != 0.0:
            pct_change = (mean_reward_delta / abs(prod_reward)) * 100
        else:
            pct_change = 100.0 if mean_reward_delta > 0 else -100.0

        if pct_change > 5:
            recommendation = "promote"
        elif pct_change < -5:
            recommendation = "reject"
        else:
            recommendation = "needs_evaluation"

    return ComparisonResult(
        current_production=current_prod_entry,
        candidate=candidate_entry,
        mean_reward_delta=round(mean_reward_delta, 4) if mean_reward_delta is not None else None,
        recommendation=recommendation,
    )
