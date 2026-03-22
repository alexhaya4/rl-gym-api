from typing import Any

import gymnasium as gym
import numpy as np

from app.core.prometheus import active_environments

_environments: dict[str, gym.Env[Any, Any]] = {}
_environment_ids: dict[str, str] = {}

AVAILABLE_ENVIRONMENTS = [
    "CartPole-v1",
    "LunarLander-v3",
    "MountainCar-v0",
    "Acrobot-v1",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "LunarLanderContinuous-v2",
]

ENVIRONMENT_ACTION_SPACES: dict[str, str] = {
    "CartPole-v1": "discrete",
    "LunarLander-v3": "discrete",
    "MountainCar-v0": "discrete",
    "Acrobot-v1": "discrete",
    "Pendulum-v1": "continuous",
    "MountainCarContinuous-v0": "continuous",
    "LunarLanderContinuous-v2": "continuous",
}


def _space_to_dict(space: gym.Space[Any]) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(space).__name__}
    if isinstance(space, gym.spaces.Discrete):
        info["n"] = int(space.n)
    elif isinstance(space, gym.spaces.Box):
        info["shape"] = list(space.shape)
        info["low"] = space.low.tolist()
        info["high"] = space.high.tolist()
    elif isinstance(space, gym.spaces.MultiDiscrete):
        info["nvec"] = space.nvec.tolist()
    return info


def _to_list(value: object) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()  # type: ignore[no-any-return]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clean_info(info: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


def create_environment(env_key: str, environment_id: str, render_mode: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    env = gym.make(environment_id, **kwargs)
    _environments[env_key] = env
    _environment_ids[env_key] = environment_id
    active_environments.inc()
    return {
        "observation_space": _space_to_dict(env.observation_space),
        "action_space": _space_to_dict(env.action_space),
    }


def get_environment(env_key: str) -> gym.Env[Any, Any] | None:
    return _environments.get(env_key)


def step_environment(env_key: str, action: int | list[float]) -> dict[str, Any]:
    env = _environments[env_key]
    observation, reward, terminated, truncated, info = env.step(action)
    return {
        "observation": _to_list(observation),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": _clean_info(info),
    }


def reset_environment(env_key: str) -> dict[str, Any]:
    env = _environments[env_key]
    observation, info = env.reset()
    return {
        "observation": _to_list(observation),
        "info": _clean_info(info),
    }


def close_environment(env_key: str) -> bool:
    env = _environments.pop(env_key, None)
    _environment_ids.pop(env_key, None)
    if env is None:
        return False
    env.close()
    active_environments.dec()
    return True


def list_environments() -> list[dict[str, str]]:
    return [
        {"env_key": key, "environment_id": _environment_ids[key]}
        for key in _environments
    ]


def get_available_environments() -> list[dict[str, str]]:
    return [
        {
            "environment_id": env_id,
            "action_space_type": ENVIRONMENT_ACTION_SPACES.get(env_id, "unknown"),
        }
        for env_id in AVAILABLE_ENVIRONMENTS
    ]
