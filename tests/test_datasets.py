import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "datasetuser",
        "email": "datasetuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "datasetuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _dataset_payload(name: str = "cartpole-demos") -> dict:
    return {
        "name": name,
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "description": "Test dataset",
    }


def _episode_payload(episode_number: int = 1) -> dict:
    return {
        "episode_number": episode_number,
        "total_reward": 150.0 + episode_number,
        "episode_length": 100 + episode_number,
        "observations": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        "actions": [0, 1],
        "rewards": [1.0, 1.0],
        "terminated": True,
    }


async def _create_dataset(
    client: AsyncClient, auth_headers: dict[str, str], name: str = "cartpole-demos"
) -> dict:
    response = await client.post(
        "/api/v1/datasets/",
        json=_dataset_payload(name),
        headers=auth_headers,
    )
    return response.json()


async def _add_episodes(
    client: AsyncClient, auth_headers: dict[str, str], dataset_id: int, count: int = 3
) -> dict:
    episodes = [_episode_payload(i + 1) for i in range(count)]
    response = await client.post(
        f"/api/v1/datasets/{dataset_id}/episodes",
        json=episodes,
        headers=auth_headers,
    )
    return response.json()


async def test_create_dataset(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/datasets/",
        json=_dataset_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["version"] == 1
    assert data["environment_id"] == "CartPole-v1"


async def test_create_dataset_increments_version(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    await _create_dataset(client, auth_headers, "version-test")
    response = await client.post(
        "/api/v1/datasets/",
        json=_dataset_payload("version-test"),
        headers=auth_headers,
    )
    assert response.status_code == 201
    assert response.json()["version"] == 2


async def test_list_datasets(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    await _create_dataset(client, auth_headers)

    response = await client.get("/api/v1/datasets/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1


async def test_get_dataset(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_dataset(client, auth_headers)
    dataset_id = created["id"]

    response = await client.get(f"/api/v1/datasets/{dataset_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == dataset_id
    assert data["name"] == "cartpole-demos"


async def test_add_episodes(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_dataset(client, auth_headers)
    dataset_id = created["id"]

    result = await _add_episodes(client, auth_headers, dataset_id, count=3)
    assert result["n_episodes"] == 3


async def test_get_dataset_stats(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_dataset(client, auth_headers)
    dataset_id = created["id"]
    await _add_episodes(client, auth_headers, dataset_id, count=3)

    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/stats",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["n_episodes"] == 3
    assert data["mean_episode_reward"] is not None


async def test_export_dataset_json(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_dataset(client, auth_headers)
    dataset_id = created["id"]
    await _add_episodes(client, auth_headers, dataset_id, count=2)

    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/export?format=json",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


async def test_export_dataset_csv(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_dataset(client, auth_headers)
    dataset_id = created["id"]
    await _add_episodes(client, auth_headers, dataset_id, count=2)

    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/export?format=csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]


async def test_delete_dataset(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_dataset(client, auth_headers)
    dataset_id = created["id"]

    response = await client.delete(
        f"/api/v1/datasets/{dataset_id}",
        headers=auth_headers,
    )
    assert response.status_code == 204

    response = await client.get(f"/api/v1/datasets/{dataset_id}")
    assert response.status_code == 404


async def test_list_episodes(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    created = await _create_dataset(client, auth_headers)
    dataset_id = created["id"]
    await _add_episodes(client, auth_headers, dataset_id, count=3)

    response = await client.get(
        f"/api/v1/datasets/{dataset_id}/episodes",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
