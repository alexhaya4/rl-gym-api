from httpx import AsyncClient

from app.config import get_settings
from app.grpc_server.auth_interceptor import APIKeyInterceptor


async def test_hsts_header_not_in_development(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "Strict-Transport-Security" not in response.headers


async def test_csp_header_present(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "Content-Security-Policy" in response.headers


async def test_cross_origin_opener_policy(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "Cross-Origin-Opener-Policy" in response.headers


def test_db_pool_settings() -> None:
    settings = get_settings()
    assert settings.DB_POOL_SIZE >= 5
    assert settings.DB_COMMAND_TIMEOUT > 0


def test_grpc_api_key_config() -> None:
    settings = get_settings()
    assert hasattr(settings, "GRPC_API_KEY")


def test_grpc_interceptor_import() -> None:
    assert APIKeyInterceptor is not None
