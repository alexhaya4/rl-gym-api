import ast
import asyncio
import json
import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "rl-gym-sandbox"
DOCKERFILE_PATH = "docker/sandbox"


def _ast_fallback(source_code: str) -> dict[str, Any]:
    """Validate source code using ast.parse as a fallback when Docker is unavailable."""
    try:
        ast.parse(source_code)
        return {
            "valid": True,
            "error": None,
            "observation_space": None,
            "action_space": None,
            "fallback": True,
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error: {e}",
            "observation_space": None,
            "action_space": None,
            "fallback": True,
        }


async def _docker_available() -> bool:
    """Check if Docker is available on the system."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return proc.returncode == 0
    except FileNotFoundError:
        return False


async def _image_exists() -> bool:
    """Check if the sandbox Docker image is already built."""
    proc = await asyncio.create_subprocess_exec(
        "docker", "images", SANDBOX_IMAGE, "-q",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return bool(stdout.strip())


async def _build_image() -> None:
    """Build the sandbox Docker image."""
    logger.info("Building sandbox Docker image...")
    proc = await asyncio.create_subprocess_exec(
        "docker", "build", "-t", SANDBOX_IMAGE, DOCKERFILE_PATH,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to build sandbox image: {stderr.decode()}")
    logger.info("Sandbox Docker image built successfully")


async def run_in_sandbox(source_code: str, env_name: str) -> dict[str, Any]:
    """Run custom environment source code in a sandboxed Docker container.

    Falls back to ast.parse validation if sandbox is disabled or Docker is unavailable.
    """
    settings = get_settings()

    if not settings.SANDBOX_ENABLED:
        logger.debug("Sandbox disabled, using ast.parse fallback")
        return _ast_fallback(source_code)

    if not await _docker_available():
        logger.warning("Docker not available, falling back to ast.parse validation")
        return _ast_fallback(source_code)

    try:
        if not await _image_exists():
            await _build_image()

        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", settings.SANDBOX_MEMORY_LIMIT,
            "--cpus", str(settings.SANDBOX_CPU_LIMIT),
            "--read-only",
            "--tmpfs", "/tmp:size=10m",
            "-i",
            SANDBOX_IMAGE,
            "python", "/sandbox/runner.py", env_name,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=source_code.encode()),
            timeout=settings.SANDBOX_TIMEOUT_SECONDS,
        )

        if proc.returncode != 0:
            logger.error("Sandbox container failed: %s", stderr.decode())
            return {
                "valid": False,
                "error": f"Sandbox execution failed: {stderr.decode().strip()}",
                "observation_space": None,
                "action_space": None,
            }

        result: dict[str, Any] = json.loads(stdout.decode())
        return result

    except TimeoutError:
        logger.warning("Sandbox execution timed out after %ds", settings.SANDBOX_TIMEOUT_SECONDS)
        proc.kill()  # type: ignore[union-attr]
        return {
            "valid": False,
            "error": "Sandbox timeout exceeded",
            "observation_space": None,
            "action_space": None,
        }
    except Exception as e:
        logger.warning("Docker error, falling back to ast.parse: %s", e)
        return _ast_fallback(source_code)
