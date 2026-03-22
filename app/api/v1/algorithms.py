from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.algorithms import (
    SUPPORTED_ALGORITHMS,
    validate_algorithm_environment,
)

router = APIRouter(prefix="/algorithms", tags=["algorithms"])


@router.get("/", response_model=list[dict[str, Any]])
async def list_algorithms() -> list[dict[str, Any]]:
    """List all supported algorithms with full details."""
    return list(SUPPORTED_ALGORITHMS.values())


@router.get("/compatible/{environment_id}", response_model=list[dict[str, Any]])
async def list_compatible_algorithms(environment_id: str) -> list[dict[str, Any]]:
    """List algorithms compatible with a given environment."""
    return [
        info
        for name, info in SUPPORTED_ALGORITHMS.items()
        if validate_algorithm_environment(name, environment_id)[0]
    ]


@router.get("/{algorithm_name}", response_model=dict[str, Any])
async def get_algorithm(algorithm_name: str) -> dict[str, Any]:
    """Get details for a specific algorithm."""
    info = SUPPORTED_ALGORITHMS.get(algorithm_name)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Algorithm '{algorithm_name}' not found. "
            f"Available: {sorted(SUPPORTED_ALGORITHMS.keys())}",
        )
    return info
