from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.ab_test import ABTestResult
from app.models.user import User
from app.schemas.ab_test import (
    ABTestCreate,
    ABTestListResponse,
    ABTestResponse,
    ABTestResultResponse,
    ABTestStatistics,
)
from app.services.ab_test import (
    create_ab_test,
    get_ab_test,
    get_ab_test_statistics,
    list_ab_tests,
    run_ab_test,
    stop_ab_test,
)

router = APIRouter(prefix="/ab-testing", tags=["ab-testing"])


async def _run_ab_test_bg(ab_test_id: int, user_id: int) -> None:
    from app.db.session import AsyncSessionLocal

    db = AsyncSessionLocal()
    try:
        await run_ab_test(db, ab_test_id, user_id)
    finally:
        await db.close()


@router.post("/", response_model=ABTestResponse, status_code=status.HTTP_201_CREATED)
async def create(
    ab_test_create: ABTestCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ABTestResponse:
    """Create a new A/B test."""
    try:
        ab_test = await create_ab_test(db, ab_test_create, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return ABTestResponse.model_validate(ab_test)


@router.get("/", response_model=ABTestListResponse)
async def list_all(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ABTestListResponse:
    """List all A/B tests for the current user."""
    tests = await list_ab_tests(db, current_user.id)
    return ABTestListResponse(
        items=[ABTestResponse.model_validate(t) for t in tests],
        total=len(tests),
    )


@router.get("/{test_id}", response_model=ABTestResponse)
async def get_one(
    test_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ABTestResponse:
    """Get A/B test details with statistics."""
    ab_test = await get_ab_test(db, test_id, current_user.id)
    if ab_test is None:
        raise HTTPException(status_code=404, detail="A/B test not found")

    response = ABTestResponse.model_validate(ab_test)
    stats = await get_ab_test_statistics(db, test_id)
    response.statistics = stats
    return response


@router.post("/{test_id}/run", response_model=ABTestResponse, status_code=202)
async def run_test(
    test_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ABTestResponse:
    """Start running an A/B test."""
    ab_test = await get_ab_test(db, test_id, current_user.id)
    if ab_test is None:
        raise HTTPException(status_code=404, detail="A/B test not found")

    if ab_test.status != "draft":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run test with status '{ab_test.status}'. Must be 'draft'.",
        )

    background_tasks.add_task(_run_ab_test_bg, test_id, current_user.id)

    return ABTestResponse.model_validate(ab_test)


@router.post("/{test_id}/stop", response_model=ABTestResponse)
async def stop_test(
    test_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ABTestResponse:
    """Stop a running A/B test."""
    ab_test = await stop_ab_test(db, test_id, current_user.id)
    if ab_test is None:
        raise HTTPException(status_code=404, detail="A/B test not found")
    return ABTestResponse.model_validate(ab_test)


@router.get("/{test_id}/results", response_model=list[ABTestResultResponse])
async def get_results(
    test_id: int,
    page: int = 1,
    page_size: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[ABTestResultResponse]:
    """Get individual episode results for an A/B test."""
    ab_test = await get_ab_test(db, test_id, current_user.id)
    if ab_test is None:
        raise HTTPException(status_code=404, detail="A/B test not found")

    offset = (page - 1) * page_size
    result = await db.execute(
        select(ABTestResult)
        .where(ABTestResult.ab_test_id == test_id)
        .order_by(ABTestResult.model_variant, ABTestResult.episode_number)
        .offset(offset)
        .limit(page_size)
    )
    results = list(result.scalars().all())
    return [ABTestResultResponse.model_validate(r) for r in results]


@router.get("/{test_id}/statistics", response_model=ABTestStatistics)
async def get_statistics(
    test_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ABTestStatistics:
    """Get statistical analysis for an A/B test."""
    ab_test = await get_ab_test(db, test_id, current_user.id)
    if ab_test is None:
        raise HTTPException(status_code=404, detail="A/B test not found")

    stats = await get_ab_test_statistics(db, test_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="No results available for this test")
    return stats
