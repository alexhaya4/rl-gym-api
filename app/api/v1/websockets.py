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

        timestep = 0
        while True:
            await asyncio.sleep(2)
            timestep += 1
            await websocket.send_json({
                "type": "metrics",
                "experiment_id": experiment_id,
                "timestep": timestep,
                "reward": round(random.uniform(-1, 1), 4),
                "timestamp": datetime.now(UTC).isoformat(),
            })
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _active_connections.pop(connection_key, None)


async def broadcast_metrics(experiment_id: str, metrics: dict) -> None:
    to_remove = []
    for key, ws in _active_connections.items():
        if key.startswith(f"{experiment_id}:"):
            try:
                await ws.send_json({
                    "type": "metrics",
                    "experiment_id": experiment_id,
                    **metrics,
                })
            except Exception:
                to_remove.append(key)
    for key in to_remove:
        _active_connections.pop(key, None)
