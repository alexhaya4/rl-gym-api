import asyncio
import random
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws", tags=["websockets"])

_active_connections: dict[str, WebSocket] = {}


@router.websocket("/training/{experiment_id}")
async def training_ws(websocket: WebSocket, experiment_id: str) -> None:
    await websocket.accept()
    connection_key = f"{experiment_id}:{id(websocket)}"
    _active_connections[connection_key] = websocket

    try:
        await websocket.send_json({
            "type": "connected",
            "experiment_id": experiment_id,
        })

        # Mock metrics loop for demo/testing purposes.
        # Real training metrics are delivered via broadcast_metrics()
        # from the WebSocketMetricsCallback in the arq worker.
        timestep = 0
        while True:
            await asyncio.sleep(2)
            timestep += 100
            await websocket.send_json({
                "type": "metrics",
                "experiment_id": experiment_id,
                "timestep": timestep,
                "episode_reward": round(random.uniform(-1, 1), 4),
                "loss": None,
                "entropy": None,
                "learning_rate": None,
                "n_episodes": 0,
                "timestamp": datetime.now(UTC).isoformat(),
            })
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _active_connections.pop(connection_key, None)


async def broadcast_metrics(experiment_id: str, metrics: dict[str, object]) -> None:
    """Broadcast training metrics to all WebSocket clients for an experiment.

    Real metrics are pushed by WebSocketMetricsCallback via the arq worker.
    """
    message = {
        "type": metrics.get("type", "metrics"),
        "experiment_id": experiment_id,
        "timestep": metrics.get("timestep", 0),
        "episode_reward": metrics.get("episode_reward"),
        "loss": metrics.get("loss"),
        "entropy": metrics.get("entropy"),
        "learning_rate": metrics.get("learning_rate"),
        "n_episodes": metrics.get("n_episodes", 0),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    to_remove = []
    for key, ws in _active_connections.items():
        if key.startswith(f"{experiment_id}:"):
            try:
                await ws.send_json(message)
            except Exception:
                to_remove.append(key)
    for key in to_remove:
        _active_connections.pop(key, None)
