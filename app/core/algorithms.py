from typing import Any

SUPPORTED_ALGORITHMS: dict[str, dict[str, Any]] = {
    "PPO": {
        "name": "PPO",
        "description": "Proximal Policy Optimization — best general-purpose on-policy algorithm",
        "action_space": "both",
        "paper_url": "https://arxiv.org/abs/1707.06347",
        "use_cases": ["general purpose", "robotics", "game playing", "continuous control"],
    },
    "A2C": {
        "name": "A2C",
        "description": "Advantage Actor-Critic — fast synchronous training",
        "action_space": "both",
        "paper_url": "https://arxiv.org/abs/1602.01783",
        "use_cases": ["fast training", "simple environments", "parallel collection"],
    },
    "DQN": {
        "name": "DQN",
        "description": "Deep Q-Network — classic value-based method for discrete actions",
        "action_space": "discrete",
        "paper_url": "https://arxiv.org/abs/1312.5602",
        "use_cases": ["Atari games", "discrete control", "classic RL benchmarks"],
    },
    "SAC": {
        "name": "SAC",
        "description": "Soft Actor-Critic — best sample efficiency for continuous control",
        "action_space": "continuous",
        "paper_url": "https://arxiv.org/abs/1801.01290",
        "use_cases": ["continuous control", "robotics", "sample-efficient training"],
    },
    "TD3": {
        "name": "TD3",
        "description": "Twin Delayed DDPG — stable continuous control with clipped double Q-learning",
        "action_space": "continuous",
        "paper_url": "https://arxiv.org/abs/1802.09477",
        "use_cases": ["continuous control", "robotics", "stable off-policy training"],
    },
    "DDPG": {
        "name": "DDPG",
        "description": "Deep Deterministic Policy Gradient — classic continuous control algorithm",
        "action_space": "continuous",
        "paper_url": "https://arxiv.org/abs/1509.02971",
        "use_cases": ["continuous control", "robotics", "deterministic policies"],
    },
    "TQC": {
        "name": "TQC",
        "description": "Truncated Quantile Critics — state-of-the-art continuous control (sb3-contrib)",
        "action_space": "continuous",
        "paper_url": "https://arxiv.org/abs/2005.04269",
        "use_cases": ["continuous control", "state-of-the-art performance", "distributional RL"],
    },
    "TRPO": {
        "name": "TRPO",
        "description": "Trust Region Policy Optimization — conservative policy updates (sb3-contrib)",
        "action_space": "both",
        "paper_url": "https://arxiv.org/abs/1502.05477",
        "use_cases": ["safe policy updates", "monotonic improvement", "sensitive environments"],
    },
    "RecurrentPPO": {
        "name": "RecurrentPPO",
        "description": "PPO with LSTM policy — handles partial observability (sb3-contrib)",
        "action_space": "both",
        "paper_url": "https://arxiv.org/abs/1707.06347",
        "use_cases": ["partially observable environments", "memory-dependent tasks", "sequence decisions"],
    },
    "MaskablePPO": {
        "name": "MaskablePPO",
        "description": "PPO with action masking — prevents selection of invalid actions (sb3-contrib)",
        "action_space": "discrete",
        "paper_url": "https://arxiv.org/abs/2006.14171",
        "use_cases": ["invalid action masking", "board games", "constrained action spaces"],
    },
}

DISCRETE_ALGORITHMS: set[str] = {"PPO", "A2C", "DQN", "TRPO", "RecurrentPPO", "MaskablePPO"}
CONTINUOUS_ALGORITHMS: set[str] = {"PPO", "A2C", "SAC", "TD3", "DDPG", "TQC", "TRPO", "RecurrentPPO"}
ALL_ALGORITHMS: set[str] = set(SUPPORTED_ALGORITHMS.keys())

CONTINUOUS_ENVIRONMENTS: set[str] = {
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "LunarLanderContinuous-v2",
    "BipedalWalker-v3",
    "HalfCheetah-v4",
}
DISCRETE_ENVIRONMENTS: set[str] = {
    "CartPole-v1",
    "LunarLander-v3",
    "MountainCar-v0",
    "Acrobot-v1",
}


def validate_algorithm_environment(algorithm: str, environment_id: str) -> tuple[bool, str]:
    """Check if an algorithm is compatible with an environment's action space."""
    if algorithm not in ALL_ALGORITHMS:
        return False, f"Unsupported algorithm: {algorithm}"

    algo_info = SUPPORTED_ALGORITHMS[algorithm]
    action_space = algo_info["action_space"]

    if action_space == "both":
        return True, ""

    if action_space == "discrete" and environment_id in CONTINUOUS_ENVIRONMENTS:
        return False, (
            f"Algorithm {algorithm} only supports discrete action spaces, "
            f"but {environment_id} has a continuous action space"
        )

    if action_space == "continuous" and environment_id in DISCRETE_ENVIRONMENTS:
        return False, (
            f"Algorithm {algorithm} only supports continuous action spaces, "
            f"but {environment_id} has a discrete action space"
        )

    return True, ""


def get_algorithm_class(algorithm: str) -> type:
    """Return the SB3 or sb3-contrib algorithm class."""
    if algorithm not in ALL_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    sb3_algorithms = {"PPO", "A2C", "DQN", "SAC", "TD3", "DDPG"}
    contrib_algorithms = {"TQC", "TRPO", "RecurrentPPO", "MaskablePPO"}

    if algorithm in sb3_algorithms:
        import stable_baselines3

        return getattr(stable_baselines3, algorithm)

    if algorithm in contrib_algorithms:
        import sb3_contrib

        return getattr(sb3_contrib, algorithm)

    raise ValueError(f"Unsupported algorithm: {algorithm}")
