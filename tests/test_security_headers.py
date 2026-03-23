from httpx import AsyncClient


async def test_security_headers_present(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "X-Content-Type-Options" in response.headers


async def test_x_frame_options(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.headers["X-Frame-Options"] == "DENY"


async def test_request_too_large(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "username": "biguser",
            "email": "big@example.com",
            "password": "x" * (20 * 1024 * 1024),
        },
    )
    assert response.status_code == 413


async def test_custom_env_source_too_large(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "envsizeuser",
            "email": "envsize@example.com",
            "password": "securepassword",
        },
    )
    login_resp = await client.post(
        "/api/v1/auth/login",
        data={"username": "envsizeuser", "password": "securepassword"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = await client.post(
        "/api/v1/custom-environments",
        json={
            "name": "BigEnv-v0",
            "source_code": "x" * 200_000,
            "entry_point": "BigEnv",
        },
        headers=headers,
    )
    assert response.status_code == 413


async def test_metrics_endpoint_accessible_locally(client: AsyncClient) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
