import asyncio
import logging
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class WebSocketMetricsCallback(BaseCallback):
    """SB3 callback that pushes training metrics into an asyncio queue."""

    def __init__(
        self,
        experiment_id: str,
        metrics_queue: asyncio.Queue[dict[str, Any]],
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.experiment_id = experiment_id
        self.metrics_queue = metrics_queue

    def _on_step(self) -> bool:
        if self.num_timesteps % 100 != 0:
            return True

        rewards = self.locals.get("rewards", [0])
        episode_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0

        metrics: dict[str, Any] = {
            "timestep": self.num_timesteps,
            "episode_reward": episode_reward,
            "n_episodes": self.locals.get("n_episodes", 0),
        }

        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            name_to_value = self.model.logger.name_to_value
            loss = name_to_value.get("train/loss")
            if loss is not None:
                metrics["loss"] = float(loss)
            entropy = name_to_value.get("train/entropy_loss")
            if entropy is not None:
                metrics["entropy"] = float(entropy)

        if hasattr(self.model, "learning_rate"):
            metrics["learning_rate"] = float(self.model.learning_rate)

        self.metrics_queue.put_nowait(metrics)
        return True

    def _on_training_end(self) -> None:
        self.metrics_queue.put_nowait({
            "type": "training_complete",
            "experiment_id": self.experiment_id,
        })


async def broadcast_training_metrics(
    experiment_id: str,
    metrics_queue: asyncio.Queue[dict[str, Any]],
) -> None:
    """Consume metrics from the queue and broadcast via WebSocket."""
    while True:
        try:
            metrics = metrics_queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.5)
            continue

        from app.api.v1.websockets import broadcast_metrics

        if metrics.get("type") == "training_complete":
            await broadcast_metrics(experiment_id, metrics)
            break

        await broadcast_metrics(experiment_id, metrics)
