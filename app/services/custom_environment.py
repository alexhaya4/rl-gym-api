import ast
import logging
import re
from pathlib import Path

import gymnasium as gym
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.custom_environment import CustomEnvironment
from app.schemas.custom_environment import CustomEnvironmentCreate

logger = logging.getLogger(__name__)

CUSTOM_ENVS_DIR = Path("custom_envs/")

DANGEROUS_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\b__import__\s*\(",
]


def validate_environment_code(source_code: str, name: str) -> tuple[bool, str | None]:
    """Validate that source code is safe and contains a valid environment class."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check for at least one class definition
    class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    if not class_defs:
        return False, "Source code must contain at least one class definition"

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        match = re.search(pattern, source_code)
        if match:
            return False, f"Dangerous pattern detected: {match.group()}"

    # Check imports via AST for additional safety
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("subprocess", "shutil"):
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("subprocess", "shutil"):
            return False, f"Forbidden import from: {node.module}"

    return True, None


def register_custom_environment(source_code: str, name: str) -> str:
    """Write source code to file and register with gymnasium."""
    CUSTOM_ENVS_DIR.mkdir(parents=True, exist_ok=True)

    # Write an __init__.py if it doesn't exist
    init_path = CUSTOM_ENVS_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    # Determine the class name from the source code
    tree = ast.parse(source_code)
    class_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    class_name = class_names[0]

    # Write the module file
    module_name = name.replace("-", "_").lower()
    file_path = CUSTOM_ENVS_DIR / f"{module_name}.py"
    file_path.write_text(source_code)

    entry_point = f"custom_envs.{module_name}:{class_name}"

    # Register with gymnasium (ignore if already registered)
    try:
        gym.register(id=name, entry_point=entry_point)
    except gym.error.Error:
        logger.warning("Environment %s already registered, skipping", name)

    return entry_point


async def create_custom_environment(
    db: AsyncSession, env_create: CustomEnvironmentCreate, user_id: int
) -> CustomEnvironment:
    """Validate, register, and persist a custom environment."""
    is_valid, error_message = validate_environment_code(
        env_create.source_code, env_create.name
    )

    if is_valid:
        entry_point = register_custom_environment(
            env_create.source_code, env_create.name
        )
        custom_env = CustomEnvironment(
            name=env_create.name,
            description=env_create.description,
            entry_point=entry_point,
            source_code=env_create.source_code,
            observation_space_spec=env_create.observation_space_spec,
            action_space_spec=env_create.action_space_spec,
            is_validated=True,
            validation_error=None,
            user_id=user_id,
        )
    else:
        custom_env = CustomEnvironment(
            name=env_create.name,
            description=env_create.description,
            entry_point="",
            source_code=env_create.source_code,
            observation_space_spec=env_create.observation_space_spec,
            action_space_spec=env_create.action_space_spec,
            is_validated=False,
            validation_error=error_message,
            user_id=user_id,
        )

    db.add(custom_env)
    await db.commit()
    await db.refresh(custom_env)
    return custom_env


async def list_custom_environments(
    db: AsyncSession, user_id: int
) -> list[CustomEnvironment]:
    """List all custom environments for a user."""
    result = await db.execute(
        select(CustomEnvironment).where(CustomEnvironment.user_id == user_id)
    )
    return list(result.scalars().all())


async def get_custom_environment(
    db: AsyncSession, env_id: int, user_id: int
) -> CustomEnvironment | None:
    """Get a single custom environment by ID and user."""
    result = await db.execute(
        select(CustomEnvironment).where(
            CustomEnvironment.id == env_id, CustomEnvironment.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def delete_custom_environment(
    db: AsyncSession, env_id: int, user_id: int
) -> bool:
    """Delete a custom environment from DB and remove its source file."""
    custom_env = await get_custom_environment(db, env_id, user_id)
    if custom_env is None:
        return False

    # Remove the source file if it exists
    module_name = custom_env.name.replace("-", "_").lower()
    file_path = CUSTOM_ENVS_DIR / f"{module_name}.py"
    if file_path.exists():
        file_path.unlink()

    await db.delete(custom_env)
    await db.commit()
    return True
