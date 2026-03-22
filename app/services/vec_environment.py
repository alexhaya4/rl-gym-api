from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from app.schemas.vec_environment import VecEnvironmentCreate
from app.services.environment import _space_to_dict

_vec_environments: dict[str, Any] = {}
_vec_configs: dict[str, dict[str, Any]] = {}


def _make_env(environment_id: str, seed: int | None = None) -> Callable[[], gym.Env[Any, Any]]:
    """Return a factory function that creates a gymnasium environment."""
    def _init() -> gym.Env[Any, Any]:
        env = gym.make(environment_id)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def create_vec_environment(vec_key: str, config: VecEnvironmentCreate) -> dict[str, Any]:
    """Create a vectorized environment with optional wrappers."""
    env_fns = [
        _make_env(config.environment_id, config.seed + i if config.seed is not None else None)
        for i in range(config.n_envs)
    ]

    if config.use_subprocess:
        vec_env: Any = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    if config.normalize_observations or config.normalize_rewards:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.normalize_observations,
            norm_reward=config.normalize_rewards,
        )

    if config.frame_stack is not None:
        vec_env = VecFrameStack(vec_env, n_stack=config.frame_stack)

    _vec_environments[vec_key] = vec_env
    _vec_configs[vec_key] = {
        "environment_id": config.environment_id,
        "n_envs": config.n_envs,
        "use_subprocess": config.use_subprocess,
        "normalize_observations": config.normalize_observations,
        "normalize_rewards": config.normalize_rewards,
        "frame_stack": config.frame_stack,
    }

    obs_space = _space_to_dict(vec_env.observation_space)
    act_space = _space_to_dict(vec_env.action_space)

    return {
        "observation_space": obs_space,
        "action_space": act_space,
    }


def step_vec_environment(vec_key: str, actions: list[Any]) -> dict[str, Any]:
    """Step all environments with the given actions."""
    vec_env = _vec_environments.get(vec_key)
    if vec_env is None:
        raise KeyError(f"Vectorized environment '{vec_key}' not found")

    actions_array = np.array(actions)
    observations, rewards, dones, infos = vec_env.step(actions_array)

    return {
        "observations": observations.tolist(),
        "rewards": rewards.tolist(),
        "terminated": dones.tolist(),
        "truncated": [False] * len(dones),
        "infos": [
            {k: v.item() if isinstance(v, (np.integer, np.floating)) else v for k, v in info.items()}
            for info in infos
        ],
        "n_envs": len(rewards),
    }


def reset_vec_environment(vec_key: str) -> dict[str, Any]:
    """Reset all environments."""
    vec_env = _vec_environments.get(vec_key)
    if vec_env is None:
        raise KeyError(f"Vectorized environment '{vec_key}' not found")

    observations = vec_env.reset()

    return {
        "observations": observations.tolist(),
        "infos": [{}] * _vec_configs[vec_key]["n_envs"],
        "n_envs": _vec_configs[vec_key]["n_envs"],
    }


def close_vec_environment(vec_key: str) -> bool:
    """Close and remove a vectorized environment."""
    vec_env = _vec_environments.pop(vec_key, None)
    _vec_configs.pop(vec_key, None)
    if vec_env is None:
        return False
    vec_env.close()
    return True


def list_vec_environments() -> list[dict[str, Any]]:
    """Return info about all active vectorized environments."""
    return [
        {"vec_key": key, **config}
        for key, config in _vec_configs.items()
    ]


def get_vec_environment_info(vec_key: str) -> dict[str, Any] | None:
    """Return info about a specific vectorized environment."""
    config = _vec_configs.get(vec_key)
    if config is None:
        return None

    vec_env = _vec_environments[vec_key]
    return {
        "vec_key": vec_key,
        **config,
        "observation_space": _space_to_dict(vec_env.observation_space),
        "action_space": _space_to_dict(vec_env.action_space),
    }
