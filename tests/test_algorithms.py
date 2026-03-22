import pytest
from httpx import AsyncClient

from app.core.algorithms import get_algorithm_class, validate_algorithm_environment


async def test_list_algorithms(client: AsyncClient) -> None:
    response = await client.get("/api/v1/algorithms/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 10


async def test_get_ppo_details(client: AsyncClient) -> None:
    response = await client.get("/api/v1/algorithms/PPO")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PPO"
    assert data["action_space"] == "both"


async def test_get_sac_details(client: AsyncClient) -> None:
    response = await client.get("/api/v1/algorithms/SAC")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SAC"
    assert data["action_space"] == "continuous"


async def test_get_nonexistent_algorithm(client: AsyncClient) -> None:
    response = await client.get("/api/v1/algorithms/INVALID")
    assert response.status_code == 404


async def test_compatible_algorithms_cartpole(client: AsyncClient) -> None:
    response = await client.get("/api/v1/algorithms/compatible/CartPole-v1")
    assert response.status_code == 200
    data = response.json()
    names = [a["name"] for a in data]
    assert "PPO" in names
    assert "A2C" in names
    assert "DQN" in names


async def test_compatible_algorithms_pendulum(client: AsyncClient) -> None:
    response = await client.get("/api/v1/algorithms/compatible/Pendulum-v1")
    assert response.status_code == 200
    data = response.json()
    names = [a["name"] for a in data]
    assert "SAC" in names
    assert "TD3" in names
    assert "DDPG" in names
    assert "DQN" not in names


def test_validate_algorithm_environment_valid() -> None:
    valid, error = validate_algorithm_environment("PPO", "CartPole-v1")
    assert valid is True
    assert error == ""


def test_validate_algorithm_environment_invalid() -> None:
    valid, error = validate_algorithm_environment("DQN", "Pendulum-v1")
    assert valid is False
    assert "discrete" in error.lower()


def test_get_algorithm_class_ppo() -> None:
    from stable_baselines3 import PPO

    cls = get_algorithm_class("PPO")
    assert cls is PPO


def test_get_algorithm_class_sac() -> None:
    from stable_baselines3 import SAC

    cls = get_algorithm_class("SAC")
    assert cls is SAC


def test_get_algorithm_class_tqc() -> None:
    from sb3_contrib import TQC

    cls = get_algorithm_class("TQC")
    assert cls is TQC


def test_get_algorithm_class_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        get_algorithm_class("INVALID")
