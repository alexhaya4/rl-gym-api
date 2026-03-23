from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.services.pbt import _exploit, _initialize_population, _mutate


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "pbtuser",
        "email": "pbtuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "pbtuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _pbt_payload(**overrides: object) -> dict:
    base: dict = {
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "population_size": 2,
        "total_timesteps_per_member": 300,
        "exploit_interval": 150,
    }
    base.update(overrides)
    return base


@patch("app.api.v1.pbt._run_pbt_bg", new_callable=AsyncMock)
async def test_create_pbt_experiment(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/pbt/",
        json=_pbt_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "id" in data
    assert data["status"] == "pending"


@patch("app.api.v1.pbt._run_pbt_bg", new_callable=AsyncMock)
async def test_list_pbt_experiments(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    await client.post(
        "/api/v1/pbt/",
        json=_pbt_payload(),
        headers=auth_headers,
    )

    response = await client.get("/api/v1/pbt/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1


@patch("app.api.v1.pbt._run_pbt_bg", new_callable=AsyncMock)
async def test_get_pbt_experiment(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    create_resp = await client.post(
        "/api/v1/pbt/",
        json=_pbt_payload(),
        headers=auth_headers,
    )
    pbt_id = create_resp.json()["id"]

    response = await client.get(
        f"/api/v1/pbt/{pbt_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "status" in response.json()


async def test_get_nonexistent_pbt(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/pbt/99999",
        headers=auth_headers,
    )
    assert response.status_code == 404


@patch("app.api.v1.pbt._run_pbt_bg", new_callable=AsyncMock)
async def test_pbt_response_structure(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
) -> None:
    response = await client.post(
        "/api/v1/pbt/",
        json=_pbt_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 202
    data = response.json()
    assert "population_size" in data
    assert "algorithm" in data
    assert "environment_id" in data
    assert "members" in data
    assert data["population_size"] == 2
    assert data["algorithm"] == "PPO"


def test_initialize_population() -> None:
    population = _initialize_population(4, "PPO")
    assert len(population) == 4
    for hp in population:
        assert "learning_rate" in hp
        assert isinstance(hp["learning_rate"], float)


def test_exploit_copies_best_to_worst() -> None:
    rewards = [100.0, 20.0, 80.0, 10.0]
    hyperparams = [
        {"learning_rate": 0.001},
        {"learning_rate": 0.01},
        {"learning_rate": 0.002},
        {"learning_rate": 0.05},
    ]

    updated = _exploit(rewards, hyperparams, bottom_pct=0.25)

    # Member 3 (worst, reward=10) should get member 0's params (best, reward=100)
    assert updated[3]["learning_rate"] == 0.001


def test_mutate_changes_hyperparams() -> None:
    original = {"learning_rate": 0.001, "n_steps": 1024, "batch_size": 64}
    mutated = _mutate(original, mutation_rate=1.0, algorithm="PPO")

    # With mutation_rate=1.0, at least one param should change
    changed = any(mutated[k] != original[k] for k in original)
    assert changed


async def test_pbt_population_size_validation(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/pbt/",
        json=_pbt_payload(population_size=1),
        headers=auth_headers,
    )
    assert response.status_code == 422


async def test_pbt_population_size_max(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/pbt/",
        json=_pbt_payload(population_size=21),
        headers=auth_headers,
    )
    assert response.status_code == 422
