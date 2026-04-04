from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.dataset import DatasetEpisode
from app.models.user import User
from app.schemas.dataset import (
    DatasetCreate,
    DatasetEpisodeCreate,
    DatasetEpisodeResponse,
    DatasetListResponse,
    DatasetPreview,
    DatasetResponse,
    DatasetStatistics,
    DatasetStatsResponse,
    FileDatasetResponse,
)
from app.services.dataset import (
    add_episodes,
    collect_trajectory,
    create_dataset,
    delete_dataset,
    delete_file_dataset,
    export_dataset,
    get_dataset,
    get_dataset_stats,
    get_file_statistics,
    get_preview,
    list_datasets,
    upload_csv,
    validate_file,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


class CollectRequest(BaseModel):
    environment_id: str
    n_episodes: int = 10
    algorithm: str | None = None
    model_version_id: int | None = None


async def _collect_bg(
    dataset_id: int,
    environment_id: str,
    n_episodes: int,
    algorithm: str | None,
    model_version_id: int | None,
) -> None:
    from app.db.session import AsyncSessionLocal

    db = AsyncSessionLocal()
    try:
        await collect_trajectory(
            db, dataset_id, environment_id, n_episodes, algorithm, model_version_id
        )
    finally:
        await db.close()


@router.post("/", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def create(
    dataset_create: DatasetCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Create a new dataset."""
    dataset = await create_dataset(db, dataset_create, current_user.id)
    return DatasetResponse.model_validate(dataset)


@router.get("/", response_model=DatasetListResponse)
async def list_all(
    include_public: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetListResponse:
    """List datasets for the current user and optionally public datasets."""
    datasets = await list_datasets(db, current_user.id, include_public=include_public)
    return DatasetListResponse(
        items=[DatasetResponse.model_validate(d) for d in datasets],
        total=len(datasets),
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_one(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Get dataset details. Public datasets accessible without auth."""
    dataset = await get_dataset(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse.model_validate(dataset)


@router.post("/{dataset_id}/episodes", response_model=DatasetResponse)
async def add_dataset_episodes(
    dataset_id: int,
    episodes: list[DatasetEpisodeCreate],
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Add episodes to a dataset."""
    try:
        dataset = await add_episodes(db, dataset_id, episodes, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    return DatasetResponse.model_validate(dataset)


@router.get("/{dataset_id}/episodes", response_model=list[DatasetEpisodeResponse])
async def list_episodes(
    dataset_id: int,
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[DatasetEpisodeResponse]:
    """List episodes in a dataset with pagination."""
    offset = (page - 1) * page_size
    result = await db.execute(
        select(DatasetEpisode)
        .where(DatasetEpisode.dataset_id == dataset_id)
        .order_by(DatasetEpisode.episode_number)
        .offset(offset)
        .limit(page_size)
    )
    episodes = list(result.scalars().all())
    return [DatasetEpisodeResponse.model_validate(ep) for ep in episodes]


@router.get("/{dataset_id}/stats", response_model=DatasetStatsResponse)
async def stats(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetStatsResponse:
    """Get dataset statistics."""
    try:
        return await get_dataset_stats(db, dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.get("/{dataset_id}/export")
async def export(
    dataset_id: int,
    format: str = "json",
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Export dataset in the specified format."""
    dataset = await get_dataset(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        data = await export_dataset(db, dataset_id, format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    media_types = {
        "json": "application/json",
        "csv": "text/csv",
        "hdf5": "application/x-hdf5",
    }
    filename = f"{dataset.name}_v{dataset.version}.{format}"

    return Response(
        content=data,
        media_type=media_types.get(format, "application/octet-stream"),
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/{dataset_id}/collect", response_model=DatasetResponse, status_code=202)
async def collect(
    dataset_id: int,
    request: CollectRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Collect trajectories into a dataset as a background task."""
    dataset = await get_dataset(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    background_tasks.add_task(
        _collect_bg,
        dataset_id,
        request.environment_id,
        request.n_episodes,
        request.algorithm,
        request.model_version_id,
    )

    return DatasetResponse.model_validate(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a dataset and its episodes."""
    # Try file dataset first, then trajectory dataset
    if await delete_file_dataset(db, dataset_id, current_user.id):
        return
    if not await delete_dataset(db, dataset_id, current_user.id):
        raise HTTPException(status_code=404, detail="Dataset not found")


# --- File-upload endpoints ---


@router.post("/upload", response_model=FileDatasetResponse, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> FileDatasetResponse:
    """Upload a CSV/JSON/ZIP dataset file."""
    if file.filename is None:
        raise HTTPException(status_code=422, detail="Filename is required")

    content = await file.read()

    try:
        validate_file(file.filename, len(content))
    except ValueError as e:
        detail = str(e)
        if "extension" in detail:
            raise HTTPException(status_code=422, detail=detail) from None
        raise HTTPException(status_code=413, detail=detail) from None

    dataset = await upload_csv(
        db,
        file_content=content,
        filename=file.filename,
        name=name,
        description=description,
        user_id=current_user.id,
    )
    return FileDatasetResponse.from_dataset(dataset)


@router.get(
    "/file/{dataset_id}",
    response_model=FileDatasetResponse,
)
async def get_file_dataset_endpoint(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> FileDatasetResponse:
    """Get a file-uploaded dataset by ID (owner only)."""
    from app.services.dataset import get_file_dataset

    dataset = await get_file_dataset(db, dataset_id, current_user.id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return FileDatasetResponse.from_dataset(dataset)


@router.get(
    "/file/{dataset_id}/preview",
    response_model=DatasetPreview,
)
async def preview_file_dataset(
    dataset_id: int,
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetPreview:
    """Get a preview of the first N rows of a file dataset."""
    try:
        data = await get_preview(db, dataset_id, current_user.id, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    return DatasetPreview(**data)


@router.get(
    "/file/{dataset_id}/statistics",
    response_model=list[DatasetStatistics],
)
async def file_dataset_statistics(
    dataset_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[DatasetStatistics]:
    """Get per-column statistics for a CSV file dataset."""
    try:
        stats = await get_file_statistics(db, dataset_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    return [DatasetStatistics(**s) for s in stats]
