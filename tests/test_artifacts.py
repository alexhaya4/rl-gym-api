import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "artifactuser",
        "email": "artifactuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "artifactuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def _create_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> int:
    response = await client.post(
        "/api/v1/experiments",
        json={
            "name": "Artifact Test Exp",
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "hyperparameters": {},
            "total_timesteps": 1000,
        },
        headers=auth_headers,
    )
    return response.json()["id"]


async def _create_artifact(
    client: AsyncClient,
    auth_headers: dict[str, str],
    experiment_id: int | None = None,
) -> dict:
    payload: dict = {
        "name": "test-config.json",
        "artifact_type": "config",
    }
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    response = await client.post(
        "/api/v1/artifacts/",
        json=payload,
        headers=auth_headers,
    )
    return response.json()


async def test_create_artifact(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/artifacts/",
        json={"name": "my-config.json", "artifact_type": "config"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "my-config.json"
    assert data["artifact_type"] == "config"


async def test_list_artifacts_empty(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/api/v1/artifacts/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


async def test_get_artifact(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_artifact(client, auth_headers)
    artifact_id = created["id"]

    response = await client.get(
        f"/api/v1/artifacts/{artifact_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == artifact_id
    assert data["name"] == "test-config.json"


async def test_delete_artifact(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_artifact(client, auth_headers)
    artifact_id = created["id"]

    response = await client.delete(
        f"/api/v1/artifacts/{artifact_id}",
        headers=auth_headers,
    )
    assert response.status_code == 204

    response = await client.get(
        f"/api/v1/artifacts/{artifact_id}",
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_list_artifacts_by_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    exp_id = await _create_experiment(client, auth_headers)
    await _create_artifact(client, auth_headers, experiment_id=exp_id)
    await _create_artifact(client, auth_headers)  # no experiment

    response = await client.get(
        f"/api/v1/artifacts/?experiment_id={exp_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert all(a["experiment_id"] == exp_id for a in data)
