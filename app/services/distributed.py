import json
import logging
import time
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis

from app.config import get_settings
from app.schemas.distributed import (
    DistributedStatus,
    DistributedTrainRequest,
    DistributedTrainResponse,
)

logger = logging.getLogger(__name__)

# In-memory fallback when Redis is unavailable
_memory_store: dict[str, dict[str, Any]] = {}


def _redis_client() -> Any:
    """Create a Redis client for job metadata storage."""
    settings = get_settings()
    kwargs: dict[str, str] = {}
    if settings.REDIS_PASSWORD:
        kwargs["password"] = settings.REDIS_PASSWORD
    return aioredis.from_url(settings.REDIS_URL, **kwargs)  # type: ignore[no-untyped-call]


async def _set_job(job_id: str, data: dict[str, Any]) -> None:
    """Store job data in Redis with TTL, falling back to memory."""
    ttl = get_settings().DISTRIBUTED_JOB_TTL_HOURS * 3600
    try:
        client = _redis_client()
        try:
            await client.setex(f"distributed:{job_id}", ttl, json.dumps(data))
        finally:
            await client.aclose()
    except Exception:
        logger.warning(
            "Redis unavailable for distributed job, using in-memory fallback"
        )
        _memory_store[job_id] = {**data, "_expires": time.time() + ttl}


async def _get_job(job_id: str) -> dict[str, Any] | None:
    """Retrieve job data from Redis or memory fallback."""
    try:
        client = _redis_client()
        try:
            raw = await client.get(f"distributed:{job_id}")
            if raw is None:
                return None
            return json.loads(raw)  # type: ignore[no-any-return]
        finally:
            await client.aclose()
    except Exception:
        logger.warning(
            "Redis unavailable for distributed job read, using in-memory fallback"
        )
        entry = _memory_store.get(job_id)
        if entry is None:
            return None
        if entry.get("_expires", 0) < time.time():
            _memory_store.pop(job_id, None)
            return None
        return {k: v for k, v in entry.items() if k != "_expires"}


async def _delete_job(job_id: str) -> None:
    """Remove job data from Redis or memory fallback."""
    try:
        client = _redis_client()
        try:
            await client.delete(f"distributed:{job_id}")
        finally:
            await client.aclose()
    except Exception:
        _memory_store.pop(job_id, None)


async def _list_all_jobs() -> list[dict[str, Any]]:
    """List all distributed job entries."""
    results: list[dict[str, Any]] = []
    try:
        client = _redis_client()
        try:
            keys: list[bytes] = await client.keys("distributed:*")
            for key in keys:
                raw = await client.get(key)
                if raw is not None:
                    results.append(json.loads(raw))
        finally:
            await client.aclose()
    except Exception:
        logger.warning("Redis unavailable for job list, using in-memory fallback")
        now = time.time()
        for jid, entry in list(_memory_store.items()):
            if entry.get("_expires", 0) < now:
                _memory_store.pop(jid, None)
                continue
            results.append({k: v for k, v in entry.items() if k != "_expires"})
    return results


async def create_job(
    request: DistributedTrainRequest, user_id: int
) -> DistributedTrainResponse:
    """Create a distributed training job."""
    settings = get_settings()

    if not settings.DISTRIBUTED_ENABLED:
        raise RuntimeError("Distributed training is disabled")

    if request.num_workers > settings.DISTRIBUTED_MAX_WORKERS:
        raise ValueError(
            f"num_workers ({request.num_workers}) exceeds maximum "
            f"({settings.DISTRIBUTED_MAX_WORKERS})"
        )

    job_id = uuid4().hex
    total_envs = request.num_workers * request.num_envs_per_worker
    estimated_speedup = round(request.num_workers * 0.8, 2)

    # Initialize Ray
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Launch remote training task
    ref = _train_remote.remote(  # type: ignore[call-arg]
        job_id=job_id,
        environment_id=request.environment_id,
        algorithm=request.algorithm,
        total_timesteps=request.total_timesteps,
        num_workers=request.num_workers,
        num_envs_per_worker=request.num_envs_per_worker,
        hyperparameters=request.hyperparameters,
    )

    # Store job metadata
    await _set_job(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "user_id": user_id,
            "ref": str(ref),
            "created_at": time.time(),
            "progress": 0.0,
            "metrics": None,
            "elapsed_seconds": 0.0,
            "num_workers_active": 0,
            "error": None,
            "environment_id": request.environment_id,
            "algorithm": request.algorithm,
            "total_timesteps": request.total_timesteps,
            "num_workers": request.num_workers,
            "num_envs_per_worker": request.num_envs_per_worker,
        },
    )

    return DistributedTrainResponse(
        job_id=job_id,
        status="queued",
        num_workers=request.num_workers,
        total_envs=total_envs,
        estimated_speedup=estimated_speedup,
    )


async def get_status(job_id: str, user_id: int) -> DistributedStatus:
    """Get the status of a distributed training job."""
    data = await _get_job(job_id)
    if data is None or data.get("user_id") != user_id:
        raise FileNotFoundError("Job not found")

    return DistributedStatus(
        job_id=data["job_id"],
        status=data.get("status", "unknown"),
        progress=data.get("progress", 0.0),
        metrics=data.get("metrics"),
        elapsed_seconds=data.get("elapsed_seconds", 0.0),
        num_workers_active=data.get("num_workers_active", 0),
        error=data.get("error"),
    )


async def cancel_job(job_id: str, user_id: int) -> None:
    """Cancel a distributed training job."""
    data = await _get_job(job_id)
    if data is None or data.get("user_id") != user_id:
        raise FileNotFoundError("Job not found")

    import ray

    if ray.is_initialized() and data.get("ref"):
        try:
            # Attempt to cancel the Ray task
            ray.cancel(
                ray.ObjectRef(
                    bytes.fromhex(data["ref"].split("(")[-1].rstrip(")")[:40])
                ),
                force=True,
            )
        except Exception:
            logger.warning("Could not cancel Ray task for job %s", job_id)

    data["status"] = "cancelled"
    data["error"] = "Cancelled by user"
    await _set_job(job_id, data)


async def list_jobs(user_id: int) -> list[DistributedStatus]:
    """List all distributed training jobs for a user."""
    all_jobs = await _list_all_jobs()
    return [
        DistributedStatus(
            job_id=j["job_id"],
            status=j.get("status", "unknown"),
            progress=j.get("progress", 0.0),
            metrics=j.get("metrics"),
            elapsed_seconds=j.get("elapsed_seconds", 0.0),
            num_workers_active=j.get("num_workers_active", 0),
            error=j.get("error"),
        )
        for j in all_jobs
        if j.get("user_id") == user_id
    ]


def get_cluster_info() -> dict[str, Any]:
    """Get Ray cluster information."""
    import ray

    if not ray.is_initialized():
        return {
            "initialized": False,
            "num_cpus": 0,
            "num_gpus": 0,
            "nodes": 0,
        }

    resources = ray.cluster_resources()
    return {
        "initialized": True,
        "num_cpus": int(resources.get("CPU", 0)),
        "num_gpus": int(resources.get("GPU", 0)),
        "nodes": len(ray.nodes()),
    }


# Ray remote function — defined at module level for Ray serialization
try:
    import ray

    @ray.remote
    def _train_remote(
        job_id: str,
        environment_id: str,
        algorithm: str,
        total_timesteps: int,
        num_workers: int,
        num_envs_per_worker: int,
        hyperparameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Ray remote task for distributed training."""
        import gymnasium as gym
        import stable_baselines3 as sb3
        from stable_baselines3.common.env_util import make_vec_env

        start_time = time.time()

        try:
            total_envs = num_workers * num_envs_per_worker

            # Create vectorized environments
            vec_env = make_vec_env(environment_id, n_envs=total_envs)

            # Get algorithm class
            algo_cls = getattr(sb3, algorithm.upper(), None)
            if algo_cls is None:
                return {"status": "failed", "error": f"Unknown algorithm: {algorithm}"}

            # Create and train model
            model = algo_cls("MlpPolicy", vec_env, **hyperparameters)
            model.learn(total_timesteps=total_timesteps)

            # Evaluate
            mean_reward = 0.0
            eval_env = gym.make(environment_id)
            n_eval = 10
            for _ in range(n_eval):
                obs, _ = eval_env.reset()
                episode_reward = 0.0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += float(reward)
                    done = terminated or truncated
                mean_reward += episode_reward
            eval_env.close()
            mean_reward /= n_eval

            elapsed = time.time() - start_time

            vec_env.close()

            return {
                "status": "completed",
                "metrics": {
                    "mean_reward": round(mean_reward, 2),
                    "total_timesteps": total_timesteps,
                    "fps": round(total_timesteps / elapsed, 1) if elapsed > 0 else 0,
                },
                "elapsed_seconds": round(elapsed, 2),
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "elapsed_seconds": round(time.time() - start_time, 2),
            }

except ImportError:
    pass  # Ray not available
