import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "benchuser",
        "email": "benchuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "benchuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_get_benchmark_environments(client: AsyncClient) -> None:
    response = await client.get("/api/v1/benchmarks/environments")
    assert response.status_code == 200
    data = response.json()
    assert len(data["environments"]) > 0


async def test_get_benchmark_algorithms(client: AsyncClient) -> None:
    response = await client.get("/api/v1/benchmarks/algorithms")
    assert response.status_code == 200
    data = response.json()
    assert len(data["algorithms"]) > 0
    assert data["algorithms"][0]["name"]
    assert data["algorithms"][0]["description"]


async def test_run_benchmark(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/benchmarks/run",
        json={
            "environments": ["CartPole-v1"],
            "algorithms": ["PPO"],
            "total_timesteps": 300,
            "n_eval_episodes": 2,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_combinations"] == 1
    assert len(data["results"]) == 1
    assert "mean_reward" in data["results"][0]
    assert data["results"][0]["environment_id"] == "CartPole-v1"
    assert data["results"][0]["algorithm"] == "PPO"


async def test_run_benchmark_invalid_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/benchmarks/run",
        json={
            "environments": ["FakeEnv-v99"],
            "algorithms": ["PPO"],
            "total_timesteps": 300,
        },
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "Invalid environments" in response.json()["detail"]


async def test_run_benchmark_invalid_algorithm(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/benchmarks/run",
        json={
            "environments": ["CartPole-v1"],
            "algorithms": ["FAKE_ALGO"],
            "total_timesteps": 300,
        },
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "Invalid algorithms" in response.json()["detail"]
