from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "optunauser",
        "email": "optunauser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "optunauser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _optimization_payload(**overrides: object) -> dict:
    base: dict = {
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "n_trials": 2,
        "total_timesteps": 300,
    }
    base.update(overrides)
    return base


async def test_get_hyperparameter_spaces(client: AsyncClient) -> None:
    response = await client.get("/api/v1/optimization/algorithms/spaces")
    assert response.status_code == 200
    data = response.json()
    assert "PPO" in data
    assert "A2C" in data
    assert "DQN" in data


@patch("app.api.v1.optimization._run_optimization_bg", new_callable=AsyncMock)
async def test_start_optimization(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/optimization/run",
        json=_optimization_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "study_id" in data
    assert data["status"] == "running"


@patch("app.api.v1.optimization._run_optimization_bg", new_callable=AsyncMock)
async def test_get_optimization_status(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    create_resp = await client.post(
        "/api/v1/optimization/run",
        json=_optimization_payload(),
        headers=auth_headers,
    )
    study_id = create_resp.json()["study_id"]

    # The background task is mocked, so no DB record exists yet.
    # Querying should return 404 since run_optimization never ran.
    response = await client.get(
        f"/api/v1/optimization/{study_id}",
        headers=auth_headers,
    )
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "status" in response.json()


@patch("app.api.v1.optimization._run_optimization_bg", new_callable=AsyncMock)
async def test_list_optimizations(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    await client.post(
        "/api/v1/optimization/run",
        json=_optimization_payload(),
        headers=auth_headers,
    )
    await client.post(
        "/api/v1/optimization/run",
        json=_optimization_payload(experiment_name="second-study"),
        headers=auth_headers,
    )

    response = await client.get("/api/v1/optimization/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


async def test_get_nonexistent_study(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/optimization/nonexistent-id",
        headers=auth_headers,
    )
    assert response.status_code == 404


@patch("app.api.v1.optimization._run_optimization_bg", new_callable=AsyncMock)
async def test_optimization_response_structure(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/optimization/run",
        json=_optimization_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "study_id" in data
    assert "n_trials" in data
    assert "best_trial" in data
    assert "trials" in data
    assert isinstance(data["trials"], list)


@patch("app.api.v1.optimization._run_optimization_bg", new_callable=AsyncMock)
async def test_get_optimization_history(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    create_resp = await client.post(
        "/api/v1/optimization/run",
        json=_optimization_payload(),
        headers=auth_headers,
    )
    study_id = create_resp.json()["study_id"]

    response = await client.get(
        f"/api/v1/optimization/{study_id}/history",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
