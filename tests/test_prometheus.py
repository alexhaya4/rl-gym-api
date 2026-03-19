import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_metrics_endpoint_accessible(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "python_gc_objects_collected_total" in response.text


@pytest.mark.asyncio
async def test_metrics_endpoint_no_auth_required(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_custom_metrics_registered(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "rl_gym_active_environments" in response.text


@pytest.mark.asyncio
async def test_training_metrics_registered(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "rl_gym_training_jobs_total" in response.text


@pytest.mark.asyncio
async def test_grpc_metrics_registered(client: AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "rl_gym_grpc_requests_total" in response.text
