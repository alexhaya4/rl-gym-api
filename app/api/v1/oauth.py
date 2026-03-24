from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token
from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.oauth_account import OAuthAccount
from app.models.user import User
from app.schemas.oauth import (
    OAuthAccountResponse,
    OAuthLoginResponse,
    OAuthTokenResponse,
)
from app.services.audit_log import log_event
from app.services.oauth import (
    exchange_github_code,
    exchange_google_code,
    generate_oauth_state,
    get_github_authorization_url,
    get_github_user_info,
    get_google_authorization_url,
    get_google_user_info,
    get_or_create_oauth_user,
    verify_oauth_state,
)

router = APIRouter(prefix="/oauth", tags=["oauth"])


@router.get("/google/login", response_model=OAuthLoginResponse)
async def google_login() -> OAuthLoginResponse:
    """Initiate Google OAuth2 login flow."""
    state = generate_oauth_state("google")
    url = get_google_authorization_url(state)
    return OAuthLoginResponse(authorization_url=url, state=state)


@router.get("/google/callback", response_model=OAuthTokenResponse)
async def google_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> OAuthTokenResponse:
    """Handle Google OAuth2 callback."""
    if not verify_oauth_state(state, "google"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state",
        )

    try:
        token_data = await exchange_google_code(code)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to exchange authorization code: {e}",
        ) from None

    access_token = token_data.get("access_token", "")

    try:
        user_info = await get_google_user_info(access_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch user info: {e}",
        ) from None

    user, is_new_user = await get_or_create_oauth_user(
        db,
        provider="google",
        provider_user_id=user_info["sub"],
        email=user_info.get("email", ""),
        username=user_info.get("name", ""),
        access_token=access_token,
    )

    jwt_token = create_access_token(data={"sub": user.username})

    event_type = "register" if is_new_user else "login"
    await log_event(
        db, event_type, request=request,
        user_id=user.id, username=user.username,
        action=event_type, details={"provider": "google"},
    )

    return OAuthTokenResponse(
        access_token=jwt_token,
        user={"id": user.id, "username": user.username, "email": user.email},
        is_new_user=is_new_user,
    )


@router.get("/github/login", response_model=OAuthLoginResponse)
async def github_login() -> OAuthLoginResponse:
    """Initiate GitHub OAuth login flow."""
    state = generate_oauth_state("github")
    url = get_github_authorization_url(state)
    return OAuthLoginResponse(authorization_url=url, state=state)


@router.get("/github/callback", response_model=OAuthTokenResponse)
async def github_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> OAuthTokenResponse:
    """Handle GitHub OAuth callback."""
    if not verify_oauth_state(state, "github"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state",
        )

    try:
        token_data = await exchange_github_code(code)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to exchange authorization code: {e}",
        ) from None

    access_token = token_data.get("access_token", "")

    try:
        user_info = await get_github_user_info(access_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch user info: {e}",
        ) from None

    user, is_new_user = await get_or_create_oauth_user(
        db,
        provider="github",
        provider_user_id=str(user_info["id"]),
        email=user_info.get("email", ""),
        username=user_info.get("login", ""),
        access_token=access_token,
    )

    jwt_token = create_access_token(data={"sub": user.username})

    event_type = "register" if is_new_user else "login"
    await log_event(
        db, event_type, request=request,
        user_id=user.id, username=user.username,
        action=event_type, details={"provider": "github"},
    )

    return OAuthTokenResponse(
        access_token=jwt_token,
        user={"id": user.id, "username": user.username, "email": user.email},
        is_new_user=is_new_user,
    )


@router.get("/accounts", response_model=list[OAuthAccountResponse])
async def list_oauth_accounts(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[OAuthAccountResponse]:
    """List OAuth accounts linked to the current user."""
    result = await db.execute(
        select(OAuthAccount).where(OAuthAccount.user_id == current_user.id)
    )
    accounts = result.scalars().all()
    return [OAuthAccountResponse.model_validate(a) for a in accounts]


@router.delete("/accounts/{provider}", status_code=status.HTTP_204_NO_CONTENT)
async def unlink_oauth_account(
    provider: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Unlink an OAuth provider from the current user's account."""
    result = await db.execute(
        select(OAuthAccount).where(
            OAuthAccount.user_id == current_user.id,
            OAuthAccount.provider == provider,
        )
    )
    oauth_account = result.scalar_one_or_none()
    if oauth_account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {provider} account linked",
        )

    # Prevent unlinking if it's the only auth method
    has_password = bool(current_user.hashed_password)
    other_oauth = await db.execute(
        select(OAuthAccount).where(
            OAuthAccount.user_id == current_user.id,
            OAuthAccount.provider != provider,
        )
    )
    has_other_oauth = other_oauth.scalar_one_or_none() is not None

    if not has_password and not has_other_oauth:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot unlink the only authentication method. Set a password first.",
        )

    await db.delete(oauth_account)
    await db.commit()
