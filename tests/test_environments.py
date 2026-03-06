import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "envuser",
        "email": "envuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "envuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_get_available_environments(client: AsyncClient) -> None:
    response = await client.get("/api/v1/environments/available")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "CartPole-v1" in data


async def test_create_environment(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    response = await client.post(
        "/api/v1/environments/",
        json={"environment_id": "CartPole-v1"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert "env_key" in data
    assert data["environment_id"] == "CartPole-v1"
    assert "observation_space" in data
    assert "action_space" in data


async def test_reset_environment(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    create_resp = await client.post(
        "/api/v1/environments/",
        json={"environment_id": "CartPole-v1"},
        headers=auth_headers,
    )
    env_key = create_resp.json()["env_key"]

    response = await client.post(
        f"/api/v1/environments/{env_key}/reset",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "observation" in data
    assert isinstance(data["observation"], list)
    assert "info" in data


async def test_step_environment(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    create_resp = await client.post(
        "/api/v1/environments/",
        json={"environment_id": "CartPole-v1"},
        headers=auth_headers,
    )
    env_key = create_resp.json()["env_key"]

    await client.post(
        f"/api/v1/environments/{env_key}/reset",
        headers=auth_headers,
    )

    response = await client.post(
        f"/api/v1/environments/{env_key}/step",
        json={"action": 1},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "observation" in data
    assert "reward" in data
    assert "terminated" in data
    assert "truncated" in data
    assert "info" in data


async def test_delete_environment(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    create_resp = await client.post(
        "/api/v1/environments/",
        json={"environment_id": "CartPole-v1"},
        headers=auth_headers,
    )
    env_key = create_resp.json()["env_key"]

    response = await client.delete(
        f"/api/v1/environments/{env_key}",
        headers=auth_headers,
    )
    assert response.status_code == 204
