import contextlib
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.services.ab_test import _calculate_statistics


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "abtestuser",
        "email": "abtestuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "abtestuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def two_model_versions(client: AsyncClient, auth_headers: dict[str, str]) -> tuple[int, int]:
    """Create two model version records and return their ids."""
    from app.db.session import get_db
    from app.main import app
    from app.models.model_version import ModelVersion

    exp_resp = await client.post(
        "/api/v1/experiments",
        json={
            "name": "AB Test Exp",
            "environment_id": "CartPole-v1",
            "algorithm": "PPO",
            "hyperparameters": {},
            "total_timesteps": 1000,
        },
        headers=auth_headers,
    )
    experiment_id = exp_resp.json()["id"]

    db_gen = app.dependency_overrides[get_db]()
    db = await db_gen.__anext__()

    mv_a = ModelVersion(
        experiment_id=experiment_id, version=1,
        storage_path=f"models/{experiment_id}/v1/PPO.zip",
        storage_backend="local", algorithm="PPO",
        total_timesteps=1000, mean_reward=150.0, file_size_bytes=1024,
    )
    mv_b = ModelVersion(
        experiment_id=experiment_id, version=2,
        storage_path=f"models/{experiment_id}/v2/PPO.zip",
        storage_backend="local", algorithm="PPO",
        total_timesteps=1000, mean_reward=200.0, file_size_bytes=1024,
    )
    db.add(mv_a)
    db.add(mv_b)
    await db.commit()
    await db.refresh(mv_a)
    await db.refresh(mv_b)

    with contextlib.suppress(StopAsyncIteration):
        await db_gen.__anext__()

    return mv_a.id, mv_b.id


def _ab_test_payload(version_a: int, version_b: int) -> dict:
    return {
        "name": "PPO vs PPO v2",
        "environment_id": "CartPole-v1",
        "model_version_a_id": version_a,
        "model_version_b_id": version_b,
        "n_eval_episodes_per_model": 5,
        "significance_level": 0.05,
    }


async def _create_ab_test(
    client: AsyncClient,
    auth_headers: dict[str, str],
    version_a: int,
    version_b: int,
) -> dict:
    response = await client.post(
        "/api/v1/ab-testing/",
        json=_ab_test_payload(version_a, version_b),
        headers=auth_headers,
    )
    return response.json()


async def test_create_ab_test(
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, v_b = two_model_versions
    response = await client.post(
        "/api/v1/ab-testing/",
        json=_ab_test_payload(v_a, v_b),
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "draft"
    assert data["model_version_a_id"] == v_a
    assert data["model_version_b_id"] == v_b


async def test_create_ab_test_same_models(
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, _ = two_model_versions
    response = await client.post(
        "/api/v1/ab-testing/",
        json=_ab_test_payload(v_a, v_a),
        headers=auth_headers,
    )
    assert response.status_code == 400


async def test_list_ab_tests(
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, v_b = two_model_versions
    await _create_ab_test(client, auth_headers, v_a, v_b)

    response = await client.get("/api/v1/ab-testing/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1


async def test_get_ab_test(
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, v_b = two_model_versions
    created = await _create_ab_test(client, auth_headers, v_a, v_b)
    test_id = created["id"]

    response = await client.get(
        f"/api/v1/ab-testing/{test_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == test_id
    assert data["name"] == "PPO vs PPO v2"


async def test_get_nonexistent_ab_test(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/ab-testing/99999",
        headers=auth_headers,
    )
    assert response.status_code == 404


@patch("app.api.v1.ab_testing._run_ab_test_bg", new_callable=AsyncMock)
async def test_run_ab_test(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, v_b = two_model_versions
    created = await _create_ab_test(client, auth_headers, v_a, v_b)
    test_id = created["id"]

    response = await client.post(
        f"/api/v1/ab-testing/{test_id}/run",
        headers=auth_headers,
    )
    assert response.status_code == 202


@patch("app.api.v1.ab_testing._run_ab_test_bg", new_callable=AsyncMock)
async def test_run_already_running_test(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, v_b = two_model_versions
    created = await _create_ab_test(client, auth_headers, v_a, v_b)
    test_id = created["id"]

    # First run succeeds
    await client.post(
        f"/api/v1/ab-testing/{test_id}/run",
        headers=auth_headers,
    )

    # The mock doesn't actually change status, but the endpoint checks draft status.
    # After first run call, the test is still "draft" because bg task is mocked.
    # To test properly, we need to manually change status.
    from app.db.session import get_db
    from app.main import app
    from app.models.ab_test import ABTest
    from sqlalchemy import select

    db_gen = app.dependency_overrides[get_db]()
    db = await db_gen.__anext__()
    result = await db.execute(select(ABTest).where(ABTest.id == test_id))
    ab_test = result.scalar_one()
    ab_test.status = "running"
    await db.commit()
    with contextlib.suppress(StopAsyncIteration):
        await db_gen.__anext__()

    # Second run should fail
    response = await client.post(
        f"/api/v1/ab-testing/{test_id}/run",
        headers=auth_headers,
    )
    assert response.status_code == 400


async def test_stop_ab_test(
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, v_b = two_model_versions
    created = await _create_ab_test(client, auth_headers, v_a, v_b)
    test_id = created["id"]

    response = await client.post(
        f"/api/v1/ab-testing/{test_id}/stop",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"


@patch("app.api.v1.ab_testing._run_ab_test_bg", new_callable=AsyncMock)
async def test_ab_test_statistics_structure(
    mock_run: AsyncMock,
    client: AsyncClient,
    auth_headers: dict[str, str],
    two_model_versions: tuple[int, int],
) -> None:
    v_a, v_b = two_model_versions
    created = await _create_ab_test(client, auth_headers, v_a, v_b)
    test_id = created["id"]

    # Manually insert some results to test statistics endpoint
    from app.db.session import get_db
    from app.main import app
    from app.models.ab_test import ABTestResult

    db_gen = app.dependency_overrides[get_db]()
    db = await db_gen.__anext__()
    for i in range(5):
        db.add(ABTestResult(
            ab_test_id=test_id, model_variant="a",
            episode_number=i + 1, total_reward=100.0 + i,
        ))
        db.add(ABTestResult(
            ab_test_id=test_id, model_variant="b",
            episode_number=i + 1, total_reward=110.0 + i,
        ))
    await db.commit()
    with contextlib.suppress(StopAsyncIteration):
        await db_gen.__anext__()

    response = await client.get(
        f"/api/v1/ab-testing/{test_id}/statistics",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "p_value" in data
    assert "is_significant" in data
    assert "effect_size" in data
    assert "model_a_mean_reward" in data
    assert "model_b_mean_reward" in data


def test_calculate_statistics_directly() -> None:
    # Model B clearly better
    rewards_a = [10.0, 12.0, 11.0, 9.0, 10.5] * 10
    rewards_b = [50.0, 52.0, 51.0, 49.0, 50.5] * 10

    stats = _calculate_statistics(rewards_a, rewards_b, 0.05, "ttest")
    assert stats.winner == "b"
    assert stats.is_significant is True
    assert stats.p_value is not None
    assert stats.p_value < 0.05
    assert stats.model_b_mean_reward is not None
    assert stats.model_a_mean_reward is not None
    assert stats.model_b_mean_reward > stats.model_a_mean_reward
