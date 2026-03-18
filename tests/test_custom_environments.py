import pytest
from httpx import AsyncClient

VALID_SOURCE_CODE = """
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(4, dtype=np.float32)
        return obs, 0.0, False, False, {}
""".strip()

INVALID_SOURCE_CODE = """
import subprocess
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BadEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None):
        subprocess.run(["echo", "pwned"])
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(4, dtype=np.float32)
        return obs, 0.0, False, False, {}
""".strip()


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post("/api/v1/auth/register", json={
        "username": "customenvuser",
        "email": "customenvuser@example.com",
        "password": "securepassword",
    })
    response = await client.post("/api/v1/auth/login", data={
        "username": "customenvuser",
        "password": "securepassword",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _env_payload(name: str = "SimpleEnv-v1", source_code: str = VALID_SOURCE_CODE) -> dict:
    return {
        "name": name,
        "description": "A simple test environment",
        "source_code": source_code,
    }


async def test_register_valid_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/custom-environments",
        json=_env_payload(),
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "SimpleEnv-v1"
    assert data["is_validated"] is True
    assert data["validation_error"] is None
    assert data["entry_point"] != ""


async def test_register_invalid_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/v1/custom-environments",
        json=_env_payload(name="BadEnv-v1", source_code=INVALID_SOURCE_CODE),
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["is_validated"] is False
    assert data["validation_error"] is not None


async def test_list_custom_environments(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    await client.post(
        "/api/v1/custom-environments",
        json=_env_payload(),
        headers=auth_headers,
    )
    response = await client.get(
        "/api/v1/custom-environments",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


async def test_delete_custom_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    create_resp = await client.post(
        "/api/v1/custom-environments",
        json=_env_payload(),
        headers=auth_headers,
    )
    env_id = create_resp.json()["id"]
    response = await client.delete(
        f"/api/v1/custom-environments/{env_id}",
        headers=auth_headers,
    )
    assert response.status_code == 204


async def test_get_nonexistent_custom_environment(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    response = await client.get(
        "/api/v1/custom-environments/99999",
        headers=auth_headers,
    )
    assert response.status_code == 404
