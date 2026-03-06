from httpx import AsyncClient


async def test_register_new_user(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepassword",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"
    assert "id" in data
    assert data["is_active"] is True


async def test_register_duplicate_email(client: AsyncClient) -> None:
    payload = {
        "username": "user1",
        "email": "dupe@example.com",
        "password": "securepassword",
    }
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201

    payload["username"] = "user2"
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]


async def test_login_valid_credentials(client: AsyncClient) -> None:
    await client.post("/api/v1/auth/register", json={
        "username": "loginuser",
        "email": "login@example.com",
        "password": "securepassword",
    })

    response = await client.post("/api/v1/auth/login", data={
        "username": "loginuser",
        "password": "securepassword",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


async def test_login_invalid_credentials(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/login", data={
        "username": "noone",
        "password": "wrongpassword",
    })
    assert response.status_code == 401


async def test_get_current_user(client: AsyncClient) -> None:
    await client.post("/api/v1/auth/register", json={
        "username": "meuser",
        "email": "me@example.com",
        "password": "securepassword",
    })

    login_response = await client.post("/api/v1/auth/login", data={
        "username": "meuser",
        "password": "securepassword",
    })
    token = login_response.json()["access_token"]

    response = await client.get("/api/v1/auth/me", headers={
        "Authorization": f"Bearer {token}",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "meuser"
    assert data["email"] == "me@example.com"
