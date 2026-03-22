from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.services import pipeline_store


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "pipeuser",
        "email": "pipeuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "pipeuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _clear_pipeline_store() -> None:
    pipeline_store._pipeline_runs.clear()


def _pipeline_payload(**overrides: object) -> dict:
    base: dict = {
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "total_timesteps": 300,
    }
    base.update(overrides)
    return base


async def test_pipeline_health(client: AsyncClient) -> None:
    response = await client.get("/api/v1/pipelines/health")
    assert response.status_code == 200
    data = response.json()
    assert "prefect_available" in data


@patch("app.api.v1.pipelines._run_training_pipeline", new_callable=AsyncMock)
async def test_trigger_pipeline(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/pipelines/run",
        json=_pipeline_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "pipeline_id" in data
    assert data["status"] == "pending"


@patch("app.api.v1.pipelines._run_training_pipeline", new_callable=AsyncMock)
async def test_get_pipeline_status(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    create_resp = await client.post(
        "/api/v1/pipelines/run",
        json=_pipeline_payload(),
        headers=auth_headers,
    )
    pipeline_id = create_resp.json()["pipeline_id"]

    response = await client.get(
        f"/api/v1/pipelines/{pipeline_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


@patch("app.api.v1.pipelines._run_training_pipeline", new_callable=AsyncMock)
async def test_list_pipelines(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    await client.post(
        "/api/v1/pipelines/run",
        json=_pipeline_payload(),
        headers=auth_headers,
    )
    await client.post(
        "/api/v1/pipelines/run",
        json=_pipeline_payload(experiment_name="second-run"),
        headers=auth_headers,
    )

    response = await client.get("/api/v1/pipelines/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2


async def test_get_nonexistent_pipeline(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/pipelines/nonexistent-id",
        headers=auth_headers,
    )
    assert response.status_code == 404


@patch("app.api.v1.pipelines._run_training_pipeline", new_callable=AsyncMock)
async def test_pipeline_response_structure(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/pipelines/run",
        json=_pipeline_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "pipeline_id" in data
    assert "status" in data
    assert "started_at" in data
    assert "steps" in data
    assert isinstance(data["steps"], list)
