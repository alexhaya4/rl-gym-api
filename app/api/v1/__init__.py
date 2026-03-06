from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.environments import router as environments_router
from app.api.v1.training import router as training_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(environments_router)
router.include_router(training_router)
