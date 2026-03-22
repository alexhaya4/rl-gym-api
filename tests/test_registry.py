import pytest
from httpx import AsyncClient

from app.db.session import Base


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "registryuser",
        "email": "registryuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "registryuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def model_version_id(client: AsyncClient, auth_headers: dict[str, str]) -> int:
    """Create an experiment and a model version record, return the version id."""
    # Create experiment
    exp_resp = await client.post(
        "/api/v1/experiments",
        json={
            "name": "Registry Test Exp",
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "hyperparameters": {},
            "total_timesteps": 1000,
        },
        headers=auth_headers,
    )
    experiment_id = exp_resp.json()["id"]

    # Insert a model version directly via the DB override
    from app.db.session import get_db
    from app.main import app
    from app.models.model_version import ModelVersion

    db_gen = app.dependency_overrides[get_db]()
    db = await db_gen.__anext__()
    mv = ModelVersion(
        experiment_id=experiment_id,
        version=1,
        storage_path=f"models/{experiment_id}/v1/PPO.zip",
        storage_backend="local",
        algorithm="PPO",
        total_timesteps=1000,
        mean_reward=150.0,
        file_size_bytes=1024,
    )
    db.add(mv)
    await db.commit()
    await db.refresh(mv)
    version_id = mv.id
    try:
        await db_gen.__anext__()
    except StopAsyncIteration:
        pass
    return version_id


async def _register_model(
    client: AsyncClient,
    auth_headers: dict[str, str],
    model_version_id: int,
) -> dict:
    resp = await client.post(
        "/api/v1/registry/register",
        params={
            "model_version_id": model_version_id,
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
        },
        headers=auth_headers,
    )
    return resp.json() if resp.status_code == 201 else {}


async def test_register_model(
    client: AsyncClient,
    auth_headers: dict[str, str],
    model_version_id: int,
) -> None:
    response = await client.post(
        "/api/v1/registry/register",
        params={
            "model_version_id": model_version_id,
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["stage"] == "development"
    assert data["environment_id"] == "CartPole-v1"
    assert data["algorithm"] == "PPO"


async def test_list_registry_empty(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get("/api/v1/registry/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data


async def test_get_production_model_not_found(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/v1/registry/production/NonExistent-v1/PPO")
    assert response.status_code == 404


async def test_promote_invalid_transition(
    client: AsyncClient,
    auth_headers: dict[str, str],
    model_version_id: int,
) -> None:
    # Register model (development stage)
    reg_data = await _register_model(client, auth_headers, model_version_id)
    registry_id = reg_data["id"]

    # Try to promote directly to production (skipping staging)
    response = await client.post(
        f"/api/v1/registry/{registry_id}/promote",
        json={
            "model_version_id": model_version_id,
            "target_stage": "production",
        },
        headers=auth_headers,
    )
    assert response.status_code == 400


async def test_full_promotion_workflow(
    client: AsyncClient,
    auth_headers: dict[str, str],
    model_version_id: int,
) -> None:
    # Register model
    reg_data = await _register_model(client, auth_headers, model_version_id)
    registry_id = reg_data["id"]

    # Promote to staging
    response = await client.post(
        f"/api/v1/registry/{registry_id}/promote",
        json={
            "model_version_id": model_version_id,
            "target_stage": "staging",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["stage"] == "staging"

    # Promote to production
    response = await client.post(
        f"/api/v1/registry/{registry_id}/promote",
        json={
            "model_version_id": model_version_id,
            "target_stage": "production",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["stage"] == "production"


async def test_compare_models_no_production(
    client: AsyncClient,
    auth_headers: dict[str, str],
    model_version_id: int,
) -> None:
    reg_data = await _register_model(client, auth_headers, model_version_id)
    registry_id = reg_data["id"]

    response = await client.get(
        f"/api/v1/registry/{registry_id}/compare",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommendation" in data
    assert data["current_production"] is None
    assert data["recommendation"] == "promote"


async def test_list_registry_with_stage_filter(
    client: AsyncClient,
    auth_headers: dict[str, str],
    model_version_id: int,
) -> None:
    await _register_model(client, auth_headers, model_version_id)

    response = await client.get(
        "/api/v1/registry/",
        params={"stage": "development"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert all(item["stage"] == "development" for item in data["items"])
