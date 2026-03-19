import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "billuser",
        "email": "billuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "billuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_get_plans(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.get("/api/v1/billing/plans", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    plan_names = [p["name"] for p in data]
    assert "Free" in plan_names
    assert "Pro" in plan_names
    assert "Enterprise" in plan_names


async def test_get_plans_no_auth(client: AsyncClient):
    response = await client.get("/api/v1/billing/plans")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


async def test_get_subscription(client: AsyncClient, auth_headers: dict[str, str]):
    create_resp = await client.post(
        "/api/v1/organizations",
        json={"name": "Sub Org"},
        headers=auth_headers,
    )
    org_id = create_resp.json()["id"]

    response = await client.get(
        f"/api/v1/billing/subscription/{org_id}", headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["plan"] == "free"
    assert data["status"] == "active"


async def test_create_checkout_session(client: AsyncClient, auth_headers: dict[str, str]):
    create_resp = await client.post(
        "/api/v1/organizations",
        json={"name": "Checkout Org"},
        headers=auth_headers,
    )
    org_id = create_resp.json()["id"]

    response = await client.post(
        "/api/v1/billing/checkout",
        json={
            "org_id": org_id,
            "plan": "pro",
            "success_url": "http://localhost:3000/success",
            "cancel_url": "http://localhost:3000/cancel",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "checkout_url" in data
    assert "session_id" in data


async def test_webhook_no_stripe_configured(client: AsyncClient):
    response = await client.post(
        "/api/v1/billing/webhook",
        content=b"invalid payload",
        headers={"stripe-signature": "invalid_sig"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "skipped"
