import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "compareuser",
        "email": "compareuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "compareuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _experiment_payload(name: str = "Compare Exp") -> dict:
    return {
        "name": name,
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "hyperparameters": {},
        "total_timesteps": 1000,
    }


async def _create_experiment(
    client: AsyncClient, auth_headers: dict[str, str], name: str = "Compare Exp"
) -> int:
    response = await client.post(
        "/api/v1/experiments",
        json=_experiment_payload(name),
        headers=auth_headers,
    )
    return response.json()["id"]


async def test_compare_two_experiments(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    id_a = await _create_experiment(client, auth_headers, "Exp A")
    id_b = await _create_experiment(client, auth_headers, "Exp B")

    response = await client.post(
        "/api/v1/comparison/",
        json={"experiment_ids": [id_a, id_b]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "diffs" in data
    assert len(data["diffs"]) == 1
    assert "experiments" in data
    assert len(data["experiments"]) == 2


async def test_compare_single_experiment_fails(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    id_a = await _create_experiment(client, auth_headers, "Solo Exp")

    response = await client.post(
        "/api/v1/comparison/",
        json={"experiment_ids": [id_a]},
        headers=auth_headers,
    )
    assert response.status_code == 422


async def test_get_experiment_diff(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    id_a = await _create_experiment(client, auth_headers, "Diff A")
    id_b = await _create_experiment(client, auth_headers, "Diff B")

    response = await client.get(
        f"/api/v1/comparison/diff/{id_a}/{id_b}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "winner" in data
    assert "hyperparameter_diff" in data
    assert "metrics_diff" in data


async def test_get_lineage_no_parent(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    exp_id = await _create_experiment(client, auth_headers, "Lineage Exp")

    response = await client.get(
        f"/api/v1/comparison/lineage/{exp_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) >= 1
    assert data["nodes"][0]["experiment_id"] == exp_id


async def test_set_experiment_tags(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    exp_id = await _create_experiment(client, auth_headers, "Tag Exp")

    response = await client.patch(
        f"/api/v1/comparison/experiments/{exp_id}/tags",
        json=["fast", "test"],
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tags"] == ["fast", "test"]


async def test_export_experiment_json(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    exp_id = await _create_experiment(client, auth_headers, "Export JSON Exp")

    response = await client.get(
        f"/api/v1/comparison/experiments/{exp_id}/export?format=json",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "episodes" in data


async def test_export_experiment_csv(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    exp_id = await _create_experiment(client, auth_headers, "Export CSV Exp")

    response = await client.get(
        f"/api/v1/comparison/experiments/{exp_id}/export?format=csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
