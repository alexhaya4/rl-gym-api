from starlette.testclient import TestClient

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
        assert "reward" in second
