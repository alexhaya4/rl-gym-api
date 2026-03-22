import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.vec_environment import (
    VecEnvironmentCreate,
    VecEnvironmentResponse,
    VecResetResponse,
    VecStepRequest,
    VecStepResponse,
)
from app.services.vec_environment import (
    close_vec_environment,
    create_vec_environment,
    get_vec_environment_info,
    list_vec_environments,
    reset_vec_environment,
    step_vec_environment,
)

router = APIRouter(prefix="/vec-environments", tags=["vec-environments"])


@router.post("/", response_model=VecEnvironmentResponse, status_code=status.HTTP_201_CREATED)
async def create_vec_env(
    config: VecEnvironmentCreate,
    _current_user: User = Depends(get_current_active_user),
) -> VecEnvironmentResponse:
    """Create a vectorized environment."""
    vec_key = str(uuid.uuid4())
    try:
        info = create_vec_environment(vec_key, config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return VecEnvironmentResponse(
        vec_key=vec_key,
        environment_id=config.environment_id,
        n_envs=config.n_envs,
        use_subprocess=config.use_subprocess,
        normalize_observations=config.normalize_observations,
        normalize_rewards=config.normalize_rewards,
        frame_stack=config.frame_stack,
        observation_space=info["observation_space"],
        action_space=info["action_space"],
    )


@router.get("/", response_model=list[dict])
async def list_vec_envs(
    _current_user: User = Depends(get_current_active_user),
) -> list[dict]:
    """List all active vectorized environments."""
    return list_vec_environments()


@router.get("/{vec_key}", response_model=VecEnvironmentResponse)
async def get_vec_env(
    vec_key: str,
    _current_user: User = Depends(get_current_active_user),
) -> VecEnvironmentResponse:
    """Get vectorized environment info."""
    info = get_vec_environment_info(vec_key)
    if info is None:
        raise HTTPException(status_code=404, detail="Vectorized environment not found")
    return VecEnvironmentResponse(**info)


@router.post("/{vec_key}/reset", response_model=VecResetResponse)
async def reset_vec_env(
    vec_key: str,
    _current_user: User = Depends(get_current_active_user),
) -> VecResetResponse:
    """Reset all environments in the vector."""
    try:
        result = reset_vec_environment(vec_key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Vectorized environment not found") from None
    return VecResetResponse(**result)


@router.post("/{vec_key}/step", response_model=VecStepResponse)
async def step_vec_env(
    vec_key: str,
    body: VecStepRequest,
    _current_user: User = Depends(get_current_active_user),
) -> VecStepResponse:
    """Step all environments with one action per env."""
    info = get_vec_environment_info(vec_key)
    if info is None:
        raise HTTPException(status_code=404, detail="Vectorized environment not found")

    if len(body.actions) != info["n_envs"]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {info['n_envs']} actions, got {len(body.actions)}",
        )

    try:
        result = step_vec_environment(vec_key, body.actions)
    except KeyError:
        raise HTTPException(status_code=404, detail="Vectorized environment not found") from None
    return VecStepResponse(**result)


@router.delete("/{vec_key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vec_env(
    vec_key: str,
    _current_user: User = Depends(get_current_active_user),
) -> None:
    """Close and remove a vectorized environment."""
    if not close_vec_environment(vec_key):
        raise HTTPException(status_code=404, detail="Vectorized environment not found")
