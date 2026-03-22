from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.artifact import LineageGraph
from app.schemas.comparison import ComparisonRequest, ComparisonResponse, ExperimentDiff
from app.services.comparison import (
    compare_experiments,
    export_experiment_csv,
    export_experiment_json,
    get_experiment_diff,
    get_lineage_graph,
    set_experiment_tags,
)

router = APIRouter(prefix="/comparison", tags=["comparison"])


@router.post("/", response_model=ComparisonResponse)
async def compare(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ComparisonResponse:
    """Compare multiple experiments side by side."""
    try:
        return await compare_experiments(db, request.experiment_ids, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.get("/diff/{exp_id_a}/{exp_id_b}", response_model=ExperimentDiff)
async def diff(
    exp_id_a: int,
    exp_id_b: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ExperimentDiff:
    """Get diff between two experiments."""
    try:
        return await get_experiment_diff(db, exp_id_a, exp_id_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.get("/lineage/{experiment_id}", response_model=LineageGraph)
async def lineage(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> LineageGraph:
    """Get lineage graph for an experiment."""
    return await get_lineage_graph(db, experiment_id)


@router.patch("/experiments/{experiment_id}/tags")
async def update_tags(
    experiment_id: int,
    tags: list[str],
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update experiment tags."""
    updated = await set_experiment_tags(db, experiment_id, tags, current_user.id)
    if not updated:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"experiment_id": experiment_id, "tags": tags}


@router.get("/experiments/{experiment_id}/export")
async def export_experiment(
    experiment_id: int,
    format: str = "json",
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Export experiment data as CSV or JSON."""
    try:
        if format == "csv":
            csv_data = await export_experiment_csv(db, experiment_id)
            return PlainTextResponse(
                content=csv_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=experiment_{experiment_id}.csv"
                },
            )
        data = await export_experiment_json(db, experiment_id)
        return data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
