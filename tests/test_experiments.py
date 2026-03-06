import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "expuser",
        "email": "expuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "expuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _experiment_payload(name: str = "Test Experiment") -> dict:
    return {
        "name": name,
        "environment_id": "CartPole-v1",
        "algorithm": "PPO",
        "hyperparameters": {},
        "total_timesteps": 1000,
    }


async def test_create_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/experiments",
        json=_experiment_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Experiment"
    assert data["environment_id"] == "CartPole-v1"
    assert data["algorithm"] == "PPO"
    assert data["status"] == "pending"
    assert "id" in data


async def test_list_experiments(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("Exp 1"),
        headers=auth_headers,
    )
    await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("Exp 2"),
        headers=auth_headers,
    )

    response = await client.get("/api/v1/experiments", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


async def test_get_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    create_resp = await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("Get Me"),
        headers=auth_headers,
    )
    exp_id = create_resp.json()["id"]

    response = await client.get(
        f"/api/v1/experiments/{exp_id}", headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == exp_id
    assert data["name"] == "Get Me"


async def test_update_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    create_resp = await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("Old Name"),
        headers=auth_headers,
    )
    exp_id = create_resp.json()["id"]

    response = await client.patch(
        f"/api/v1/experiments/{exp_id}",
        json={"name": "New Name"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


async def test_delete_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    create_resp = await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("Delete Me"),
        headers=auth_headers,
    )
    exp_id = create_resp.json()["id"]

    delete_resp = await client.delete(
        f"/api/v1/experiments/{exp_id}", headers=auth_headers
    )
    assert delete_resp.status_code == 204

    get_resp = await client.get(
        f"/api/v1/experiments/{exp_id}", headers=auth_headers
    )
    assert get_resp.status_code == 404


async def test_list_experiments_pagination(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    for i in range(3):
        await client.post(
            "/api/v1/experiments",
            json=_experiment_payload(f"Paginated {i}"),
            headers=auth_headers,
        )

    response = await client.get(
        "/api/v1/experiments",
        params={"page_size": 2},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["total"] == 3
    assert data["page"] == 1
    assert data["page_size"] == 2
