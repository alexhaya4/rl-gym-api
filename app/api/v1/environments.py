import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.environment import (
    EnvironmentCreate,
    EnvironmentResponse,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from app.services.environment import (
    close_environment,
    create_environment,
    get_available_environments,
    get_environment,
    list_environments,
    reset_environment,
    step_environment,
)

router = APIRouter(prefix="/environments", tags=["environments"])


@router.get("/available")
async def available() -> list[str]:
    return get_available_environments()


@router.get("/")
async def list_envs(
    _current_user: User = Depends(get_current_active_user),
) -> list[dict]:
    return list_environments()


@router.post("/", response_model=EnvironmentResponse, status_code=status.HTTP_201_CREATED)
async def create_env(
    body: EnvironmentCreate,
    _current_user: User = Depends(get_current_active_user),
) -> EnvironmentResponse:
    env_key = str(uuid.uuid4())
    try:
        info = create_environment(env_key, body.environment_id, body.render_mode)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return EnvironmentResponse(
        env_key=env_key,
        environment_id=body.environment_id,
        observation_space=info["observation_space"],
        action_space=info["action_space"],
    )


@router.get("/{env_key}", response_model=EnvironmentResponse)
async def get_env(
    env_key: str,
    _current_user: User = Depends(get_current_active_user),
) -> EnvironmentResponse:
    from app.services.environment import _environment_ids, _space_to_dict

    env = get_environment(env_key)
    if env is None:
        raise HTTPException(status_code=404, detail="Environment not found")
    return EnvironmentResponse(
        env_key=env_key,
        environment_id=_environment_ids[env_key],
        observation_space=_space_to_dict(env.observation_space),
        action_space=_space_to_dict(env.action_space),
    )


@router.post("/{env_key}/reset", response_model=ResetResponse)
async def reset_env(
    env_key: str,
    _current_user: User = Depends(get_current_active_user),
) -> ResetResponse:
    if get_environment(env_key) is None:
        raise HTTPException(status_code=404, detail="Environment not found")
    result = reset_environment(env_key)
    return ResetResponse(**result)


@router.post("/{env_key}/step", response_model=StepResponse)
async def step_env(
    env_key: str,
    body: StepRequest,
    _current_user: User = Depends(get_current_active_user),
) -> StepResponse:
    if get_environment(env_key) is None:
        raise HTTPException(status_code=404, detail="Environment not found")
    try:
        result = step_environment(env_key, body.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if result["terminated"] or result["truncated"]:
        reset_environment(env_key)
    return StepResponse(**result)


@router.delete("/{env_key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_env(
    env_key: str,
    _current_user: User = Depends(get_current_active_user),
) -> None:
    if not close_environment(env_key):
        raise HTTPException(status_code=404, detail="Environment not found")
