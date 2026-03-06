import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "evaluser",
        "email": "evaluser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "evaluser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_evaluate_nonexistent_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/evaluation/run",
        json={"experiment_id": 99999},
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_get_episodes_nonexistent_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/evaluation/experiments/99999/episodes",
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_evaluate_pending_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    create_resp = await client.post(
        "/api/v1/experiments",
        json={
            "name": "Pending Eval",
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "total_timesteps": 1000,
        },
        headers=auth_headers,
    )
    exp_id = create_resp.json()["id"]

    response = await client.post(
        "/api/v1/evaluation/run",
        json={"experiment_id": exp_id},
        headers=auth_headers,
    )
    assert response.status_code == 404
    assert "pending" in response.json()["detail"]


@pytest.mark.skip(reason="requires completed training, run manually")
async def test_evaluate_completed_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    # Start training to get a completed experiment
    train_resp = await client.post(
        "/api/v1/training/",
        json={
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "total_timesteps": 500,
        },
        headers=auth_headers,
    )
    exp_id = train_resp.json()["experiment_id"]

    response = await client.post(
        "/api/v1/evaluation/run",
        json={"experiment_id": exp_id, "n_eval_episodes": 3},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == exp_id
    assert len(data["episodes"]) == 3
    assert "mean_reward" in data
