import pytest
from httpx import AsyncClient

from app.services.vec_environment import _vec_configs, _vec_environments


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "vecuser",
        "email": "vecuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "vecuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _clear_vec_envs() -> None:
    yield
    for key in list(_vec_environments.keys()):
        try:
            _vec_environments[key].close()
        except Exception:
            pass
    _vec_environments.clear()
    _vec_configs.clear()


async def _create_vec_env(
    client: AsyncClient, auth_headers: dict[str, str], n_envs: int = 2
) -> dict:
    response = await client.post(
        "/api/v1/vec-environments/",
        json={"environment_id": "CartPole-v1", "n_envs": n_envs},
        headers=auth_headers,
    )
    return response.json()


async def test_create_vec_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/vec-environments/",
        json={"environment_id": "CartPole-v1", "n_envs": 2},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert "vec_key" in data
    assert data["n_envs"] == 2
    assert data["environment_id"] == "CartPole-v1"
    assert data["status"] == "ready"


async def test_create_vec_environment_max_envs(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/vec-environments/",
        json={"environment_id": "CartPole-v1", "n_envs": 33},
        headers=auth_headers,
    )
    assert response.status_code == 422


async def test_reset_vec_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    data = await _create_vec_env(client, auth_headers, n_envs=2)
    vec_key = data["vec_key"]

    response = await client.post(
        f"/api/v1/vec-environments/{vec_key}/reset",
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()
    assert len(result["observations"]) == 2
    assert result["n_envs"] == 2


async def test_step_vec_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    data = await _create_vec_env(client, auth_headers, n_envs=2)
    vec_key = data["vec_key"]

    await client.post(
        f"/api/v1/vec-environments/{vec_key}/reset",
        headers=auth_headers,
    )

    response = await client.post(
        f"/api/v1/vec-environments/{vec_key}/step",
        json={"actions": [0, 1]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()
    assert len(result["rewards"]) == 2
    assert result["n_envs"] == 2


async def test_step_wrong_action_count(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    data = await _create_vec_env(client, auth_headers, n_envs=2)
    vec_key = data["vec_key"]

    await client.post(
        f"/api/v1/vec-environments/{vec_key}/reset",
        headers=auth_headers,
    )

    response = await client.post(
        f"/api/v1/vec-environments/{vec_key}/step",
        json={"actions": [0]},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "Expected 2 actions" in response.json()["detail"]


async def test_list_vec_environments(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    await _create_vec_env(client, auth_headers, n_envs=2)
    await _create_vec_env(client, auth_headers, n_envs=3)

    response = await client.get(
        "/api/v1/vec-environments/",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2


async def test_delete_vec_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    data = await _create_vec_env(client, auth_headers, n_envs=2)
    vec_key = data["vec_key"]

    response = await client.delete(
        f"/api/v1/vec-environments/{vec_key}",
        headers=auth_headers,
    )
    assert response.status_code == 204


async def test_get_nonexistent_vec_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/vec-environments/nonexistent",
        headers=auth_headers,
    )
    assert response.status_code == 404
