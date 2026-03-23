import logging
import time

import redis.asyncio as redis
from jose import jwt

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_EXPIRY_SECONDS = 3600

# In-memory fallback when Redis is unavailable
_memory_blacklist: dict[str, float] = {}


def _redis_client() -> redis.Redis:
    """Create a Redis client, applying REDIS_PASSWORD if configured."""
    settings = get_settings()
    kwargs: dict[str, str] = {}
    if settings.REDIS_PASSWORD:
        kwargs["password"] = settings.REDIS_PASSWORD
    client: redis.Redis = redis.from_url(  # type: ignore[no-untyped-call]
        settings.REDIS_URL, **kwargs
    )
    return client


def _purge_expired() -> None:
    """Remove expired entries from the in-memory blacklist."""
    now = time.time()
    expired = [k for k, v in _memory_blacklist.items() if v <= now]
    for k in expired:
        del _memory_blacklist[k]


async def blacklist_token(token: str, expires_in_seconds: int) -> None:
    """Store a token in the Redis blacklist with a TTL so it auto-expires.

    Falls back to in-memory storage when Redis is unavailable.
    """
    try:
        client = _redis_client()
        try:
            await client.setex(f"blacklist:{token}", expires_in_seconds, "1")
        finally:
            await client.aclose()
    except (redis.RedisError, OSError):
        logger.warning("Redis unavailable for token blacklist write, using in-memory fallback")
        _purge_expired()
        _memory_blacklist[token] = time.time() + expires_in_seconds


async def is_token_blacklisted(token: str) -> bool:
    """Check if a token is blacklisted. Falls back to in-memory store if Redis is unavailable."""
    try:
        client = _redis_client()
        try:
            result: int = await client.exists(f"blacklist:{token}")
            return result > 0
        finally:
            await client.aclose()
    except (redis.RedisError, OSError):
        logger.warning("Redis unavailable for token blacklist check, using in-memory fallback")
        _purge_expired()
        return token in _memory_blacklist


async def get_token_expiry(token: str) -> int:
    """Decode the JWT without verification to read the exp claim.

    Returns seconds until expiry, or DEFAULT_EXPIRY_SECONDS if it cannot be
    determined.
    """
    try:
        payload = jwt.decode(
            token,
            key="",
            options={"verify_signature": False, "verify_exp": False},
        )
        exp = payload.get("exp")
        if exp is not None:
            remaining = int(exp) - int(time.time())
            return max(remaining, 0)
    except Exception:
        logger.debug("Could not decode token to determine expiry", exc_info=True)
    return DEFAULT_EXPIRY_SECONDS
