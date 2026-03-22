from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "mauser",
        "email": "mauser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "mauser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _training_payload(**overrides: object) -> dict:
    base: dict = {
        "environment_id": "simple_spread_v3",
        "algorithm": "PPO",
        "total_timesteps": 1000,
    }
    base.update(overrides)
    return base


async def test_list_multiagent_environments(client: AsyncClient) -> None:
    response = await client.get("/api/v1/multi-agent/environments")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    env_ids = [e["environment_id"] for e in data]
    assert "simple_spread_v3" in env_ids


@patch("app.api.v1.multi_agent._run_training_bg", new_callable=AsyncMock)
async def test_start_multiagent_training(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/multi-agent/train",
        json=_training_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "id" in data
    assert data["status"] == "pending"


@patch("app.api.v1.multi_agent._run_training_bg", new_callable=AsyncMock)
async def test_list_multiagent_experiments(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    await client.post(
        "/api/v1/multi-agent/train",
        json=_training_payload(),
        headers=auth_headers,
    )

    response = await client.get(
        "/api/v1/multi-agent/experiments",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 1


@patch("app.api.v1.multi_agent._run_training_bg", new_callable=AsyncMock)
async def test_get_multiagent_experiment(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    create_resp = await client.post(
        "/api/v1/multi-agent/train",
        json=_training_payload(),
        headers=auth_headers,
    )
    experiment_id = create_resp.json()["id"]

    response = await client.get(
        f"/api/v1/multi-agent/experiments/{experiment_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


async def test_get_nonexistent_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/multi-agent/experiments/99999",
        headers=auth_headers,
    )
    assert response.status_code == 404


@patch("app.api.v1.multi_agent._run_training_bg", new_callable=AsyncMock)
async def test_multiagent_response_structure(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/multi-agent/train",
        json=_training_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "environment_id" in data
    assert "n_agents" in data
    assert "algorithm" in data
    assert "status" in data
    assert data["environment_id"] == "simple_spread_v3"
    assert data["n_agents"] == 3
    assert data["algorithm"] == "PPO"


@patch("app.api.v1.multi_agent._run_training_bg", new_callable=AsyncMock)
async def test_invalid_environment(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/multi-agent/train",
        json=_training_payload(environment_id="invalid_env"),
        headers=auth_headers,
    )
    assert response.status_code == 400
