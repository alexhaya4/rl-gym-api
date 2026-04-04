from httpx import AsyncClient

from app.schemas.inference import InferenceRequest


async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "inferuser",
            "email": "inferuser@example.com",
            "password": "securepassword",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "inferuser",
            "password": "securepassword",
        },
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_inference_info_no_model(client: AsyncClient) -> None:
    """GET /api/v1/inference/CartPole-v1/info returns 404 when no production model."""
    response = await client.get("/api/v1/inference/CartPole-v1/info")
    assert response.status_code == 404
    assert "No production model found" in response.json()["detail"]


async def test_inference_predict_no_model(client: AsyncClient) -> None:
    """POST /api/v1/inference/CartPole-v1/predict returns 404 when no production model."""
    headers = await auth_headers(client)
    response = await client.post(
        "/api/v1/inference/CartPole-v1/predict",
        json={"observation": [0.0, 0.0, 0.0, 0.0], "deterministic": True},
        headers=headers,
    )
    assert response.status_code == 404
    assert "No production model found" in response.json()["detail"]


async def test_inference_cache_empty(client: AsyncClient) -> None:
    """GET /api/v1/inference/cache returns empty list."""
    headers = await auth_headers(client)
    response = await client.get("/api/v1/inference/cache", headers=headers)
    assert response.status_code == 200
    assert response.json() == []


async def test_inference_cache_clear(client: AsyncClient) -> None:
    """DELETE /api/v1/inference/cache returns 200."""
    headers = await auth_headers(client)
    response = await client.delete("/api/v1/inference/cache", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Cache cleared"
    assert data["models_evicted"] == 0


async def test_inference_predict_requires_auth(client: AsyncClient) -> None:
    """POST without token returns 401."""
    response = await client.post(
        "/api/v1/inference/CartPole-v1/predict",
        json={"observation": [0.0, 0.0, 0.0, 0.0]},
    )
    assert response.status_code == 401


async def test_inference_request_schema() -> None:
    """Valid InferenceRequest validates correctly."""
    req = InferenceRequest(
        observation=[1.0, 2.0, 3.0, 4.0],
        algorithm="PPO",
        deterministic=True,
    )
    assert req.observation == [1.0, 2.0, 3.0, 4.0]
    assert req.algorithm == "PPO"
    assert req.deterministic is True

    # Default values
    req2 = InferenceRequest(observation=[0.0])
    assert req2.algorithm is None
    assert req2.deterministic is True


async def test_inference_observation_wrong_type(client: AsyncClient) -> None:
    """Invalid observation type returns 422."""
    headers = await auth_headers(client)
    response = await client.post(
        "/api/v1/inference/CartPole-v1/predict",
        json={"observation": "not-a-list-or-dict"},
        headers=headers,
    )
    assert response.status_code == 422
