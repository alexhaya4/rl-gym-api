from unittest.mock import patch

from app.config import get_settings
from app.core.sandbox import run_in_sandbox
from app.services.custom_environment import validate_environment_code

VALID_ENV_CODE = """
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomTestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}
"""

INVALID_SYNTAX_CODE = "def broken(:\n    pass"


def test_sandbox_config_exists():
    settings = get_settings()
    assert hasattr(settings, "SANDBOX_ENABLED")
    assert hasattr(settings, "SANDBOX_TIMEOUT_SECONDS")
    assert hasattr(settings, "SANDBOX_MEMORY_LIMIT")
    assert hasattr(settings, "SANDBOX_CPU_LIMIT")


def test_sandbox_import():
    assert callable(run_in_sandbox)


async def test_sandbox_fallback_valid_code():
    with patch.object(get_settings(), "SANDBOX_ENABLED", False):
        result = await run_in_sandbox(VALID_ENV_CODE, "TestEnv")
    assert result["valid"] is True
    assert result.get("fallback") is True


async def test_sandbox_fallback_invalid_code():
    with patch.object(get_settings(), "SANDBOX_ENABLED", False):
        result = await run_in_sandbox(INVALID_SYNTAX_CODE, "BadEnv")
    assert result["valid"] is False
    assert "Syntax error" in result["error"]
    assert result.get("fallback") is True


async def test_sandbox_fallback_dangerous_import():
    source = "import subprocess\nclass Env: pass"
    result = await validate_environment_code(source, "DangerousEnv")
    assert result[0] is False
    assert "subprocess" in result[1]


async def test_custom_env_size_limit(client):
    huge_code = "x = 1\n" * 100_001
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "sizeuser", "email": "size@test.com", "password": "testpass123"},
    )
    assert response.status_code in (200, 201)
    login_response = await client.post(
        "/api/v1/auth/login",
        data={"username": "sizeuser", "password": "testpass123"},
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = await client.post(
        "/api/v1/custom-environments",
        json={
            "name": "HugeEnv-v0",
            "description": "Too large",
            "source_code": huge_code,
        },
        headers=headers,
    )
    assert response.status_code in (413, 422)
