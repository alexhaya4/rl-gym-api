from unittest.mock import AsyncMock

from starlette.testclient import TestClient

from app.api.v1.websockets import _active_connections, broadcast_metrics
from app.main import app

client = TestClient(app)


def test_websocket_connect():
    with client.websocket_connect("/api/v1/ws/training/123") as ws:
        data = ws.receive_json()
        assert data["type"] == "connected"
        assert data["experiment_id"] == "123"


def test_websocket_metrics():
    with client.websocket_connect("/api/v1/ws/training/123") as ws:
        first = ws.receive_json()
        assert first["type"] == "connected"

        second = ws.receive_json()
        assert second["type"] == "metrics"
        assert "episode_reward" in second


def test_websocket_metrics_format():
    with client.websocket_connect("/api/v1/ws/training/456") as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connected"

        metrics = ws.receive_json()
        assert metrics["type"] == "metrics"
        assert metrics["experiment_id"] == "456"
        assert "timestep" in metrics
        assert "timestamp" in metrics
        assert "episode_reward" in metrics
        assert "loss" in metrics
        assert "entropy" in metrics
        assert "learning_rate" in metrics
        assert "n_episodes" in metrics


async def test_websocket_broadcast():
    mock_ws = AsyncMock()
    connection_key = "789:mock"
    _active_connections[connection_key] = mock_ws

    try:
        await broadcast_metrics("789", {
            "timestep": 100,
            "episode_reward": 1.5,
            "loss": 0.05,
            "entropy": 0.3,
            "learning_rate": 0.001,
            "n_episodes": 10,
        })

        mock_ws.send_json.assert_called_once()
        sent = mock_ws.send_json.call_args[0][0]
        assert sent["type"] == "metrics"
        assert sent["experiment_id"] == "789"
        assert sent["timestep"] == 100
        assert sent["episode_reward"] == 1.5
        assert sent["loss"] == 0.05
        assert sent["entropy"] == 0.3
        assert sent["learning_rate"] == 0.001
        assert sent["n_episodes"] == 10
        assert "timestamp" in sent
    finally:
        _active_connections.pop(connection_key, None)


async def test_websocket_broadcast_removes_failed_connections():
    mock_ws = AsyncMock()
    mock_ws.send_json.side_effect = Exception("connection closed")
    connection_key = "999:mock"
    _active_connections[connection_key] = mock_ws

    await broadcast_metrics("999", {"timestep": 50})

    assert connection_key not in _active_connections
