import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3.common.base_class import BaseAlgorithm

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.model_version import ModelVersion
from app.models.user import User
from app.schemas.video import VideoRequest, VideoStatus
from app.services.registry import get_production_model
from app.services.video import (
    cleanup_video,
    get_video_status,
    list_user_videos,
    record_and_encode,
    set_video_status,
)

router = APIRouter(prefix="/video", tags=["video"])


async def _load_production_model(
    db: AsyncSession,
    environment_id: str,
    algorithm: str,
) -> tuple[BaseAlgorithm, str]:
    """Look up production model and load it. Returns (model, model_path)."""
    registry_entry = await get_production_model(db, environment_id, algorithm)
    if registry_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No production model found for {environment_id}/{algorithm}",
        )

    mv_result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == registry_entry.model_version_id)
    )
    model_version = mv_result.scalar_one_or_none()
    if model_version is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model version {registry_entry.model_version_id} not found",
        )

    import stable_baselines3 as sb3

    algo_cls: Any = getattr(sb3, algorithm.upper(), None)
    if algo_cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")

    loop = asyncio.get_running_loop()
    model: BaseAlgorithm = await loop.run_in_executor(
        None, lambda: algo_cls.load(model_version.storage_path)
    )

    return model, model_version.storage_path


@router.post("/record", response_model=VideoStatus, status_code=202)
async def record_video(
    body: VideoRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> VideoStatus:
    """Start recording agent gameplay video."""
    model, _model_path = await _load_production_model(
        db, body.environment_id, body.algorithm
    )

    video_id = uuid.uuid4().hex

    await set_video_status(
        video_id,
        {
            "video_id": video_id,
            "status": "queued",
            "progress": 0.0,
            "error": None,
            "user_id": current_user.id,
        },
    )

    background_tasks.add_task(
        record_and_encode,
        video_id=video_id,
        environment_id=body.environment_id,
        model=model,
        num_episodes=body.num_episodes,
        max_steps=body.max_steps,
        fps=body.fps,
        user_id=current_user.id,
    )

    return VideoStatus(
        video_id=video_id,
        status="queued",
        progress=0.0,
        error=None,
    )


@router.get("/{video_id}/status", response_model=VideoStatus)
async def video_status(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
) -> VideoStatus:
    """Get video recording status."""
    data = await get_video_status(video_id)
    if data is None or data.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoStatus(
        video_id=data["video_id"],
        status=data["status"],
        progress=data.get("progress", 0.0),
        error=data.get("error"),
    )


@router.get("/{video_id}/download")
async def download_video(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
) -> FileResponse:
    """Download a recorded video."""
    data = await get_video_status(video_id)
    if data is None or data.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="Video not found")

    if data.get("status") != "complete":
        raise HTTPException(
            status_code=400, detail=f"Video is not ready (status: {data['status']})"
        )

    import os

    path = data.get("output_path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video file not found or expired")

    return FileResponse(
        path=path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4",
    )


@router.get("/", response_model=list[VideoStatus])
async def list_videos(
    current_user: User = Depends(get_current_active_user),
) -> list[VideoStatus]:
    """List all videos for the current user."""
    videos = await list_user_videos(current_user.id)
    return [
        VideoStatus(
            video_id=v["video_id"],
            status=v["status"],
            progress=v.get("progress", 0.0),
            error=v.get("error"),
        )
        for v in videos
    ]


@router.delete("/{video_id}", status_code=204)
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    """Delete a video file and its metadata."""
    data = await get_video_status(video_id)
    if data is None or data.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="Video not found")

    await cleanup_video(video_id)
