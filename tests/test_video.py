from httpx import AsyncClient


async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "videouser",
            "email": "videouser@example.com",
            "password": "securepassword",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "videouser",
            "password": "securepassword",
        },
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_record_no_production_model(client: AsyncClient) -> None:
    """POST /video/record returns 404 when no production model exists."""
    headers = await auth_headers(client)
    response = await client.post(
        "/api/v1/video/record",
        json={
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "num_episodes": 1,
            "max_steps": 100,
        },
        headers=headers,
    )
    assert response.status_code == 404
    assert "No production model found" in response.json()["detail"]


async def test_video_status_not_found(client: AsyncClient) -> None:
    """GET /video/invalid-id/status returns 404."""
    headers = await auth_headers(client)
    response = await client.get(
        "/api/v1/video/invalid-id/status",
        headers=headers,
    )
    assert response.status_code == 404


async def test_list_videos_empty(client: AsyncClient) -> None:
    """GET /video/ returns empty list."""
    headers = await auth_headers(client)
    response = await client.get("/api/v1/video/", headers=headers)
    assert response.status_code == 200
    assert response.json() == []


async def test_video_requires_auth(client: AsyncClient) -> None:
    """All video endpoints return 401 without token."""
    assert (
        await client.post(
            "/api/v1/video/record",
            json={
                "environment_id": "CartPole-v1",
                "algorithm": "PPO",
            },
        )
    ).status_code == 401

    assert (await client.get("/api/v1/video/some-id/status")).status_code == 401
    assert (await client.get("/api/v1/video/some-id/download")).status_code == 401
    assert (await client.get("/api/v1/video/")).status_code == 401
    assert (await client.delete("/api/v1/video/some-id")).status_code == 401


async def test_video_request_validation(client: AsyncClient) -> None:
    """num_episodes > 5 returns 422."""
    headers = await auth_headers(client)
    response = await client.post(
        "/api/v1/video/record",
        json={
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "num_episodes": 10,
        },
        headers=headers,
    )
    assert response.status_code == 422


async def test_video_steps_validation(client: AsyncClient) -> None:
    """max_steps > 10000 returns 422."""
    headers = await auth_headers(client)
    response = await client.post(
        "/api/v1/video/record",
        json={
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "max_steps": 50000,
        },
        headers=headers,
    )
    assert response.status_code == 422
