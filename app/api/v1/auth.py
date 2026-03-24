from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token
from app.core.token_blacklist import blacklist_token, get_token_expiry
from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.token import Token
from app.schemas.user import UserCreate, UserResponse
from app.services.audit_log import log_event
from app.services.user import (
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_username,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_in: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    if await get_user_by_email(db, user_in.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if await get_user_by_username(db, user_in.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    user = await create_user(db, user_in)
    await log_event(
        db, "register", request=request,
        user_id=user.id, username=user.username,
        resource_type="user", resource_id=str(user.id), action="create",
    )
    return UserResponse.model_validate(user)


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        await log_event(
            db, "login", request=request,
            username=form_data.username, action="login", status="failure",
            details={"reason": "invalid credentials"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    await log_event(
        db, "login", request=request,
        user_id=user.id, username=user.username, action="login", status="success",
    )
    return Token(access_token=access_token)


@router.post("/logout")
async def logout(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ")
    expires_in = await get_token_expiry(token)
    await blacklist_token(token, expires_in)
    await log_event(
        db, "logout", request=request,
        user_id=current_user.id, username=current_user.username, action="logout",
    )
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def read_current_user(current_user: User = Depends(get_current_active_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)
