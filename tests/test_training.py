import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "trainuser",
        "email": "trainuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "trainuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_start_training(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    response = await client.post("/api/v1/training/", json={
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "total_timesteps": 500,
    }, headers=auth_headers)
    assert response.status_code == 202
    data = response.json()
    assert "experiment_id" in data
    assert data["status"] == "queued"
    assert data["job_id"] is not None
    assert data["algorithm"] == "PPO"


async def test_get_training_status(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    create_resp = await client.post("/api/v1/training/", json={
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "total_timesteps": 500,
    }, headers=auth_headers)
    experiment_id = create_resp.json()["experiment_id"]

    response = await client.get(
        f"/api/v1/training/{experiment_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["experiment_id"] == experiment_id


async def test_list_training_sessions(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    await client.post("/api/v1/training/", json={
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "total_timesteps": 500,
    }, headers=auth_headers)

    await client.post("/api/v1/training/", json={
        "environment_id": "CartPole-v1",
        "algorithm": "A2C",
        "total_timesteps": 500,
    }, headers=auth_headers)

    response = await client.get("/api/v1/training/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
