import asyncio
import json
import logging
import os
import time
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from app.config import get_settings

logger = logging.getLogger(__name__)

# In-memory fallback when Redis is unavailable
_memory_store: dict[str, dict[str, Any]] = {}


def _redis_client() -> Any:
    """Create a Redis client for video metadata storage."""
    import redis.asyncio as redis

    settings = get_settings()
    kwargs: dict[str, str] = {}
    if settings.REDIS_PASSWORD:
        kwargs["password"] = settings.REDIS_PASSWORD
    return redis.from_url(settings.REDIS_URL, **kwargs)  # type: ignore[attr-defined, no-untyped-call]


async def set_video_status(
    video_id: str, data: dict[str, Any], ttl: int | None = None
) -> None:
    """Store video metadata in Redis with TTL, falling back to memory."""
    if ttl is None:
        ttl = get_settings().VIDEO_CLEANUP_HOURS * 3600
    try:
        client = _redis_client()
        try:
            await client.setex(f"video:{video_id}", ttl, json.dumps(data))
        finally:
            await client.aclose()
    except Exception:
        logger.warning("Redis unavailable for video status, using in-memory fallback")
        _memory_store[video_id] = {**data, "_expires": time.time() + ttl}


async def get_video_status(video_id: str) -> dict[str, Any] | None:
    """Retrieve video metadata from Redis or memory fallback."""
    try:
        client = _redis_client()
        try:
            raw = await client.get(f"video:{video_id}")
            if raw is None:
                return None
            return json.loads(raw)  # type: ignore[no-any-return]
        finally:
            await client.aclose()
    except Exception:
        logger.warning(
            "Redis unavailable for video status read, using in-memory fallback"
        )
        entry = _memory_store.get(video_id)
        if entry is None:
            return None
        if entry.get("_expires", 0) < time.time():
            _memory_store.pop(video_id, None)
            return None
        return {k: v for k, v in entry.items() if k != "_expires"}


async def delete_video_status(video_id: str) -> None:
    """Remove video metadata from Redis or memory fallback."""
    try:
        client = _redis_client()
        try:
            await client.delete(f"video:{video_id}")
        finally:
            await client.aclose()
    except Exception:
        logger.warning(
            "Redis unavailable for video status delete, using in-memory fallback"
        )
        _memory_store.pop(video_id, None)


async def list_user_videos(user_id: int) -> list[dict[str, Any]]:
    """List all video statuses for a user. Falls back to memory store."""
    results: list[dict[str, Any]] = []
    try:
        client = _redis_client()
        try:
            keys: list[bytes] = await client.keys("video:*")
            for key in keys:
                raw = await client.get(key)
                if raw is not None:
                    data = json.loads(raw)
                    if data.get("user_id") == user_id:
                        results.append(data)
        finally:
            await client.aclose()
    except Exception:
        logger.warning("Redis unavailable for video list, using in-memory fallback")
        now = time.time()
        for vid, entry in list(_memory_store.items()):
            if entry.get("_expires", 0) < now:
                _memory_store.pop(vid, None)
                continue
            if entry.get("user_id") == user_id:
                results.append({k: v for k, v in entry.items() if k != "_expires"})
    return results


def record_episode(
    environment_id: str,
    model: BaseAlgorithm,
    max_steps: int,
) -> tuple[list[np.ndarray[Any, Any]], int, float]:
    """Record a single episode using rgb_array rendering.

    Returns (frames, total_steps, total_reward).
    """
    import gymnasium as gym

    env = gym.make(environment_id, render_mode="rgb_array")
    frames: list[np.ndarray[Any, Any]] = []
    total_reward = 0.0
    steps = 0

    try:
        obs, _info = env.reset()
        frame: Any = env.render()
        if frame is not None:
            frames.append(frame)  # type: ignore[arg-type]

        for _ in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1

            frame = env.render()
            if frame is not None:
                frames.append(frame)  # type: ignore[arg-type]

            if terminated or truncated:
                break
    finally:
        env.close()

    return frames, steps, total_reward


def encode_video(
    frames: list[np.ndarray[Any, Any]], fps: int, output_path: str
) -> None:
    """Encode frames to MP4 using imageio-ffmpeg."""
    import imageio.v3 as iio

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iio.imwrite(
        output_path,
        np.stack(frames),
        fps=fps,
        plugin="pyav",
    )


def validate_video_size(path: str) -> None:
    """Raise ValueError if video exceeds MAX_VIDEO_SIZE_MB."""
    settings = get_settings()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > settings.MAX_VIDEO_SIZE_MB:
        os.remove(path)
        raise ValueError(
            f"Video size ({size_mb:.1f} MB) exceeds maximum ({settings.MAX_VIDEO_SIZE_MB} MB)"
        )


async def record_and_encode(
    video_id: str,
    environment_id: str,
    model: BaseAlgorithm,
    num_episodes: int,
    max_steps: int,
    fps: int,
    user_id: int,
) -> None:
    """Background task: record episodes, encode video, update status."""
    settings = get_settings()
    output_path = os.path.join(settings.VIDEO_STORAGE_PATH, f"{video_id}.mp4")
    loop = asyncio.get_running_loop()

    try:
        await set_video_status(
            video_id,
            {
                "video_id": video_id,
                "status": "recording",
                "progress": 0.0,
                "error": None,
                "user_id": user_id,
            },
        )

        all_frames: list[np.ndarray[Any, Any]] = []
        total_steps = 0
        total_reward = 0.0

        for ep in range(num_episodes):
            frames, steps, reward = await loop.run_in_executor(
                None, record_episode, environment_id, model, max_steps
            )
            all_frames.extend(frames)
            total_steps += steps
            total_reward += reward

            progress = ((ep + 1) / num_episodes) * 80
            await set_video_status(
                video_id,
                {
                    "video_id": video_id,
                    "status": "recording",
                    "progress": progress,
                    "error": None,
                    "user_id": user_id,
                },
            )

        await set_video_status(
            video_id,
            {
                "video_id": video_id,
                "status": "encoding",
                "progress": 85.0,
                "error": None,
                "user_id": user_id,
            },
        )

        await loop.run_in_executor(None, encode_video, all_frames, fps, output_path)

        validate_video_size(output_path)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        duration_seconds = len(all_frames) / fps if fps > 0 else 0.0

        await set_video_status(
            video_id,
            {
                "video_id": video_id,
                "status": "complete",
                "progress": 100.0,
                "error": None,
                "user_id": user_id,
                "output_path": output_path,
                "num_episodes": num_episodes,
                "total_steps": total_steps,
                "total_reward": total_reward,
                "duration_seconds": round(duration_seconds, 2),
                "file_size_mb": round(file_size_mb, 2),
            },
        )

    except Exception as exc:
        logger.exception("Video recording failed for %s", video_id)
        await set_video_status(
            video_id,
            {
                "video_id": video_id,
                "status": "failed",
                "progress": 0.0,
                "error": str(exc),
                "user_id": user_id,
            },
        )
        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)


async def cleanup_video(video_id: str) -> bool:
    """Delete video file and Redis key. Returns True if file existed."""
    settings = get_settings()
    path = os.path.join(settings.VIDEO_STORAGE_PATH, f"{video_id}.mp4")
    status = await get_video_status(video_id)

    # Also check the stored output_path
    if status and status.get("output_path"):
        path = status["output_path"]

    await delete_video_status(video_id)

    if os.path.exists(path):
        os.remove(path)
        return True
    return False
