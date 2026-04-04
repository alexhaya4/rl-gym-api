import asyncio
import logging
import sys
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from app.config import get_settings
from app.schemas.inference import ModelCacheInfo

logger = logging.getLogger(__name__)

# Algorithm name -> SB3 class mapping
_ALGORITHM_MAP: dict[str, str] = {
    "PPO": "stable_baselines3.PPO",
    "A2C": "stable_baselines3.A2C",
    "DQN": "stable_baselines3.DQN",
    "SAC": "stable_baselines3.SAC",
    "TD3": "stable_baselines3.TD3",
    "DDPG": "stable_baselines3.DDPG",
}


def _load_algorithm_class(algorithm: str) -> type[BaseAlgorithm]:
    """Resolve algorithm name to SB3 class."""
    import stable_baselines3 as sb3

    cls = getattr(sb3, algorithm.upper(), None)
    if cls is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return cls  # type: ignore[no-any-return,return-value]


class _CacheEntry:
    __slots__ = ("algorithm", "environment_id", "loaded_at", "memory_mb", "model")

    def __init__(
        self,
        model: BaseAlgorithm,
        algorithm: str,
        environment_id: str,
        memory_mb: float,
    ) -> None:
        self.model = model
        self.algorithm = algorithm
        self.environment_id = environment_id
        self.loaded_at = datetime.now(UTC)
        self.memory_mb = memory_mb


class ModelCache:
    """LRU cache for loaded SB3 models."""

    def __init__(self, max_size: int | None = None) -> None:
        self._max_size = max_size or get_settings().INFERENCE_CACHE_MAX_MODELS
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get_or_load_model(
        self,
        model_path: str,
        algorithm: str,
        environment_id: str = "",
    ) -> BaseAlgorithm:
        """Load a model from cache or disk. Evicts oldest entry if cache is full."""
        async with self._lock:
            if model_path in self._cache:
                self._cache.move_to_end(model_path)
                return self._cache[model_path].model

            # Load model outside the lock isn't needed for CPU-bound load
            # but we keep it simple since loads are infrequent
            algo_cls = _load_algorithm_class(algorithm)
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(None, lambda: algo_cls.load(model_path))

            # Estimate memory usage
            memory_mb = sys.getsizeof(model) / (1024 * 1024)

            if len(self._cache) >= self._max_size:
                self._evict_oldest()

            self._cache[model_path] = _CacheEntry(
                model=model,
                algorithm=algorithm,
                environment_id=environment_id,
                memory_mb=memory_mb,
            )
            logger.info("Loaded model into cache: %s (%s)", model_path, algorithm)
            return model

    def _evict_oldest(self) -> None:
        """Remove the least-recently-used model from cache."""
        if self._cache:
            key, _entry = self._cache.popitem(last=False)
            logger.info("Evicted model from cache: %s", key)

    def list_cached(self) -> list[ModelCacheInfo]:
        """Return info about all cached models."""
        return [
            ModelCacheInfo(
                model_path=path,
                algorithm=entry.algorithm,
                environment_id=entry.environment_id,
                loaded_at=entry.loaded_at,
                memory_mb=round(entry.memory_mb, 2),
            )
            for path, entry in self._cache.items()
        ]

    def clear(self) -> int:
        """Clear all cached models. Returns the number evicted."""
        count = len(self._cache)
        self._cache.clear()
        logger.info("Cleared model cache (%d models evicted)", count)
        return count

    async def predict(
        self,
        model: BaseAlgorithm,
        observation: list[float] | dict[str, Any],
        deterministic: bool = True,
    ) -> tuple[Any, float | None]:
        """Run inference on a loaded model. Returns (action, probability)."""
        loop = asyncio.get_running_loop()

        def _predict() -> tuple[Any, float | None]:
            obs = (
                np.array(observation) if isinstance(observation, list) else observation
            )
            action, _states = model.predict(obs, deterministic=deterministic)
            # Convert numpy types to Python types
            if isinstance(action, np.ndarray):
                action_val: Any = action.tolist()
            else:
                action_val = int(action)

            # Try to get action probability from the policy
            probability: float | None = None
            try:
                import torch

                obs_raw = (
                    np.array(observation)
                    if isinstance(observation, list)
                    else observation
                )
                obs_tensor = torch.as_tensor(obs_raw).float().unsqueeze(0)
                dist = model.policy.get_distribution(obs_tensor)  # type: ignore[operator]
                action_tensor = torch.as_tensor(np.array(action)).unsqueeze(0)
                log_prob = dist.log_prob(action_tensor)
                probability = float(torch.exp(log_prob).item())
            except Exception:
                pass  # Not all algorithms support action probabilities

            return action_val, probability

        return await loop.run_in_executor(None, _predict)


# Module-level singleton
model_cache = ModelCache()
