import itertools

import pytest
from httpx import AsyncClient

from app.core.ray_utils import is_ray_available
from app.schemas.ray_training import HyperparameterGrid


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "rayuser",
        "email": "rayuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "rayuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_ray_status_endpoint(client: AsyncClient):
    response = await client.get("/api/v1/distributed/status")
    assert response.status_code == 200
    data = response.json()
    assert "ray_available" in data
    assert "ray_address" in data
    assert "dashboard_url" in data
    assert "active_trials" in data


def test_hyperparameter_grid_combinations():
    grid = HyperparameterGrid(
        learning_rate=[0.001, 0.01],
        n_steps=[512, 1024],
        batch_size=[32],
        gamma=[0.99],
    )
    combinations = list(itertools.product(
        grid.learning_rate,
        grid.n_steps,
        grid.batch_size,
        grid.gamma,
    ))
    assert len(combinations) == 4
    assert (0.001, 512, 32, 0.99) in combinations
    assert (0.01, 1024, 32, 0.99) in combinations


async def test_sequential_fallback_training(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.post(
        "/api/v1/distributed/train",
        json={
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "total_timesteps": 300,
            "hyperparameter_grid": {
                "learning_rate": [0.001],
                "n_steps": [512],
                "batch_size": [64],
                "gamma": [0.99],
            },
            "max_concurrent_trials": 1,
        },
        headers=auth_headers,
        timeout=120.0,
    )
    assert response.status_code == 202
    data = response.json()
    assert len(data["results"]) > 0
    assert data["results"][0]["status"] == "completed"


async def test_distributed_training_response_structure(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.post(
        "/api/v1/distributed/train",
        json={
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "total_timesteps": 300,
            "hyperparameter_grid": {
                "learning_rate": [0.001],
                "n_steps": [512],
                "batch_size": [64],
                "gamma": [0.99],
            },
            "max_concurrent_trials": 1,
        },
        headers=auth_headers,
        timeout=120.0,
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert "total_trials" in data
    assert data["total_trials"] == 1
    assert "best_trial" in data
    assert data["best_trial"] is not None
    assert "best_hyperparameters" in data
    assert data["status"] == "completed"
    assert "started_at" in data
    assert "completed_at" in data


async def test_get_nonexistent_trial(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.get(
        "/api/v1/distributed/trials/nonexistent-job-id",
        headers=auth_headers,
    )
    assert response.status_code == 404
