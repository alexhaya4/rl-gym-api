from httpx import AsyncClient

from app.core.token_blacklist import is_token_blacklisted


async def _register_and_login(client: AsyncClient) -> str:
    """Register a user, log in, and return the access token."""
    await client.post("/api/v1/auth/register", json={
        "username": "logoutuser",
        "email": "logout@example.com",
        "password": "securepassword",
    })
    login_resp = await client.post("/api/v1/auth/login", data={
        "username": "logoutuser",
        "password": "securepassword",
    })
    return login_resp.json()["access_token"]


async def test_logout_endpoint(client: AsyncClient) -> None:
    token = await _register_and_login(client)
    response = await client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Successfully logged out"


async def test_token_blacklisted_after_logout(client: AsyncClient) -> None:
    token = await _register_and_login(client)
    await client.post(
        "/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
    )
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Token has been revoked"


async def test_is_token_blacklisted_false() -> None:
    result = await is_token_blacklisted("some-random-nonexistent-token")
    assert result is False


async def test_logout_requires_auth(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/logout")
    assert response.status_code == 401
