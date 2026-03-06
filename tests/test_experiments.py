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


async def test_get_nonexistent_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/experiments/99999", headers=auth_headers
    )
    assert response.status_code == 404


async def test_update_nonexistent_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.patch(
        "/api/v1/experiments/99999",
        json={"name": "Ghost"},
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_delete_nonexistent_experiment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.delete(
        "/api/v1/experiments/99999", headers=auth_headers
    )
    assert response.status_code == 404


async def test_list_experiments_with_status_filter(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("Pending One"),
        headers=auth_headers,
    )
    create_resp = await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("To Complete"),
        headers=auth_headers,
    )
    exp_id = create_resp.json()["id"]
    await client.patch(
        f"/api/v1/experiments/{exp_id}",
        json={"status": "completed"},
        headers=auth_headers,
    )

    response = await client.get(
        "/api/v1/experiments",
        params={"status": "pending"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert all(item["status"] == "pending" for item in data["items"])
    assert data["total"] >= 1


async def test_experiment_isolation(client: AsyncClient) -> None:
    # Create user1 and an experiment
    await client.post("/api/v1/auth/register", json={
        "username": "isolate1",
        "email": "isolate1@example.com",
        "password": "securepassword",
    })
    login1 = await client.post("/api/v1/auth/login", data={
        "username": "isolate1",
        "password": "securepassword",
    })
    headers1 = {"Authorization": f"Bearer {login1.json()['access_token']}"}

    create_resp = await client.post(
        "/api/v1/experiments",
        json=_experiment_payload("User1 Only"),
        headers=headers1,
    )
    exp_id = create_resp.json()["id"]

    # Create user2 and try to access user1's experiment
    await client.post("/api/v1/auth/register", json={
        "username": "isolate2",
        "email": "isolate2@example.com",
        "password": "securepassword",
    })
    login2 = await client.post("/api/v1/auth/login", data={
        "username": "isolate2",
        "password": "securepassword",
    })
    headers2 = {"Authorization": f"Bearer {login2.json()['access_token']}"}

    response = await client.get(
        f"/api/v1/experiments/{exp_id}", headers=headers2
    )
    assert response.status_code == 404
