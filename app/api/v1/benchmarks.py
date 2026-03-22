
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.algorithms import ALL_ALGORITHMS, SUPPORTED_ALGORITHMS
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.benchmark import BenchmarkRequest, BenchmarkResponse
from app.services.environment import AVAILABLE_ENVIRONMENTS

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark_endpoint(
    request: BenchmarkRequest,
    current_user: User = Depends(get_current_active_user),
) -> BenchmarkResponse:
    """Run a benchmark comparing multiple algorithms across multiple environments."""
    invalid_envs = [e for e in request.environments if e not in AVAILABLE_ENVIRONMENTS]
    if invalid_envs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid environments: {invalid_envs}. "
            f"Allowed: {AVAILABLE_ENVIRONMENTS}",
        )

    invalid_algos = [a for a in request.algorithms if a not in ALL_ALGORITHMS]
    if invalid_algos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid algorithms: {invalid_algos}. "
            f"Allowed: {sorted(ALL_ALGORITHMS)}",
        )

    from app.services.benchmark import run_benchmark

    return await run_benchmark(request)


@router.get("/environments")
async def list_benchmark_environments() -> dict[str, list[str]]:
    """List available environments for benchmarking."""
    return {"environments": AVAILABLE_ENVIRONMENTS}


@router.get("/algorithms")
async def list_benchmark_algorithms() -> dict[str, list[dict[str, str]]]:
    """List supported algorithms with brief descriptions."""
    return {
        "algorithms": [
            {"name": name, "description": info["description"]}
            for name, info in SUPPORTED_ALGORITHMS.items()
        ]
    }
