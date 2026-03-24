import logging
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.oauth_account import OAuthAccount
from app.models.user import User

logger = logging.getLogger(__name__)


def _get_serializer() -> URLSafeTimedSerializer:
    settings = get_settings()
    return URLSafeTimedSerializer(settings.OAUTH_STATE_SECRET)


def generate_oauth_state(provider: str) -> str:
    """Generate a signed OAuth state parameter."""
    serializer = _get_serializer()
    return serializer.dumps({"provider": provider, "nonce": str(uuid4())})


def verify_oauth_state(state: str, provider: str) -> bool:
    """Verify OAuth state signature and provider match."""
    serializer = _get_serializer()
    try:
        data = serializer.loads(state, max_age=600)
        return bool(data.get("provider") == provider)
    except (BadSignature, SignatureExpired):
        return False


def get_google_authorization_url(state: str) -> str:
    """Build Google OAuth2 authorization URL."""
    settings = get_settings()
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "scope": "openid email profile",
        "response_type": "code",
        "state": state,
        "access_type": "offline",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


async def exchange_google_code(code: str) -> dict[str, Any]:
    """Exchange Google authorization code for tokens."""
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result


async def get_google_user_info(access_token: str) -> dict[str, Any]:
    """Fetch Google user profile information."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result


def get_github_authorization_url(state: str) -> str:
    """Build GitHub OAuth authorization URL."""
    settings = get_settings()
    params = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": settings.GITHUB_REDIRECT_URI,
        "scope": "user:email",
        "state": state,
    }
    return f"https://github.com/login/oauth/authorize?{urlencode(params)}"


async def exchange_github_code(code: str) -> dict[str, Any]:
    """Exchange GitHub authorization code for tokens."""
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result


async def get_github_user_info(access_token: str) -> dict[str, Any]:
    """Fetch GitHub user profile and primary email."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient() as client:
        user_resp = await client.get("https://api.github.com/user", headers=headers)
        user_resp.raise_for_status()
        user_data: dict[str, Any] = user_resp.json()

        emails_resp = await client.get("https://api.github.com/user/emails", headers=headers)
        emails_resp.raise_for_status()
        emails: list[dict[str, Any]] = emails_resp.json()

        primary_email = next(
            (e["email"] for e in emails if e.get("primary") and e.get("verified")),
            None,
        )
        if primary_email:
            user_data["email"] = primary_email

        return user_data


async def get_or_create_oauth_user(
    db: AsyncSession,
    provider: str,
    provider_user_id: str,
    email: str,
    username: str,
    access_token: str,
) -> tuple[User, bool]:
    """Find or create a user from OAuth provider info.

    Returns (user, is_new_user).
    """
    # Check for existing OAuth account
    result = await db.execute(
        select(OAuthAccount).where(
            OAuthAccount.provider == provider,
            OAuthAccount.provider_user_id == provider_user_id,
        )
    )
    oauth_account = result.scalar_one_or_none()

    if oauth_account is not None:
        user_result = await db.execute(
            select(User).where(User.id == oauth_account.user_id)
        )
        user = user_result.scalar_one()
        oauth_account.access_token = access_token
        await db.commit()
        return user, False

    # Check if user with this email already exists
    user_result = await db.execute(
        select(User).where(User.email == email)
    )
    existing_user = user_result.scalar_one_or_none()

    is_new_user = False
    if existing_user is not None:
        user = existing_user
    else:
        # Ensure unique username
        base_username = username or email.split("@")[0]
        final_username = base_username
        counter = 1
        while True:
            check = await db.execute(
                select(User).where(User.username == final_username)
            )
            if check.scalar_one_or_none() is None:
                break
            final_username = f"{base_username}_{counter}"
            counter += 1

        user = User(
            username=final_username,
            email=email,
            hashed_password="",
            is_active=True,
        )
        db.add(user)
        await db.flush()
        is_new_user = True

    oauth_account = OAuthAccount(
        user_id=user.id,
        provider=provider,
        provider_user_id=provider_user_id,
        provider_email=email,
        provider_username=username,
        access_token=access_token,
    )
    db.add(oauth_account)
    await db.commit()
    await db.refresh(user)

    return user, is_new_user
