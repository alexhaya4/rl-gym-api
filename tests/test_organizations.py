import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "orguser",
        "email": "orguser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "orguser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_create_organization(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.post(
        "/api/v1/organizations",
        json={"name": "My Org"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Org"
    assert data["slug"] == "my-org"
    assert data["plan"] == "free"
    assert data["is_active"] is True


async def test_list_organizations_empty(client: AsyncClient):
    await client.post("/api/v1/auth/register", json={
        "username": "emptyorguser",
        "email": "emptyorguser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "emptyorguser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = await client.get("/api/v1/organizations", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


async def test_get_organization(client: AsyncClient, auth_headers: dict[str, str]):
    create_resp = await client.post(
        "/api/v1/organizations",
        json={"name": "Get Org"},
        headers=auth_headers,
    )
    org_id = create_resp.json()["id"]

    response = await client.get(f"/api/v1/organizations/{org_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == org_id
    assert data["name"] == "Get Org"
    assert data["slug"] == "get-org"


async def test_create_duplicate_organization(client: AsyncClient, auth_headers: dict[str, str]):
    await client.post(
        "/api/v1/organizations",
        json={"name": "Dup Org"},
        headers=auth_headers,
    )
    response = await client.post(
        "/api/v1/organizations",
        json={"name": "Dup Org"},
        headers=auth_headers,
    )
    assert response.status_code == 409


async def test_get_organization_usage(client: AsyncClient, auth_headers: dict[str, str]):
    create_resp = await client.post(
        "/api/v1/organizations",
        json={"name": "Usage Org"},
        headers=auth_headers,
    )
    org_id = create_resp.json()["id"]

    response = await client.get(
        f"/api/v1/organizations/{org_id}/usage", headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert "month" in data
    assert data["experiments_count"] == 0
    assert data["environments_count"] == 0
    assert "limits" in data
