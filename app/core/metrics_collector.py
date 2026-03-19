from typing import Any


class TrainingMetricsCollector:
    """Accumulates training metric snapshots for summary and serialization."""

    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []

    def add_snapshot(self, metrics: dict[str, Any]) -> None:
        self.snapshots.append(metrics)

    def get_summary(self) -> dict[str, Any]:
        rewards = [
            s["episode_reward"]
            for s in self.snapshots
            if s.get("episode_reward") is not None
        ]
        total_timesteps = max(
            (s.get("timestep", 0) for s in self.snapshots), default=0
        )

        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            mean_reward = sum(rewards) / len(rewards)
        else:
            min_reward = None
            max_reward = None
            mean_reward = None

        return {
            "min_episode_reward": min_reward,
            "max_episode_reward": max_reward,
            "mean_episode_reward": mean_reward,
            "total_timesteps": total_timesteps,
            "n_snapshots": len(self.snapshots),
        }

    def to_dataframe(self) -> list[dict[str, Any]]:
        return list(self.snapshots)
