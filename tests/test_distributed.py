from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "distuser",
            "email": "distuser@example.com",
            "password": "securepassword",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "distuser", "password": "securepassword"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def test_cluster_info(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """GET /distributed/cluster returns dict with initialized field."""
    with patch("app.services.distributed.get_cluster_info") as mock_info:
        mock_info.return_value = {
            "initialized": False,
            "num_cpus": 0,
            "num_gpus": 0,
            "nodes": 0,
        }
        response = await client.get(
            "/api/v1/distributed/cluster",
            headers=auth_headers,
        )
    assert response.status_code == 200
    data = response.json()
    assert "initialized" in data
    assert "num_cpus" in data
    assert "num_gpus" in data
    assert "nodes" in data


async def test_list_jobs_empty(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """GET /distributed/jobs returns empty list for a new user."""
    with patch("app.services.distributed._list_all_jobs", return_value=[]):
        response = await client.get(
            "/api/v1/distributed/jobs",
            headers=auth_headers,
        )
    assert response.status_code == 200
    assert response.json() == []


async def test_start_job_disabled(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """With DISTRIBUTED_ENABLED=False returns 503."""
    mock_settings = MagicMock()
    mock_settings.DISTRIBUTED_ENABLED = False

    with patch("app.services.distributed.get_settings", return_value=mock_settings):
        response = await client.post(
            "/api/v1/distributed/train",
            json={
                "environment_id": "CartPole-v1",
                "algorithm": "PPO",
                "num_workers": 2,
            },
            headers=auth_headers,
        )
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"].lower()


async def test_job_status_not_found(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """GET /distributed/invalid-id/status returns 404."""
    response = await client.get(
        "/api/v1/distributed/invalid-id/status",
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_cancel_not_found(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """POST /distributed/invalid-id/cancel returns 404."""
    response = await client.post(
        "/api/v1/distributed/invalid-id/cancel",
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_requires_auth(client: AsyncClient) -> None:
    """All distributed endpoints return 401 without token."""
    assert (
        await client.post(
            "/api/v1/distributed/train",
            json={"environment_id": "CartPole-v1"},
        )
    ).status_code == 401
    assert (await client.get("/api/v1/distributed/some-id/status")).status_code == 401
    assert (await client.post("/api/v1/distributed/some-id/cancel")).status_code == 401
    assert (await client.get("/api/v1/distributed/jobs")).status_code == 401
    assert (await client.get("/api/v1/distributed/cluster")).status_code == 401
