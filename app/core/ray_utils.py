import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


def is_ray_available() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def get_ray_address() -> str | None:
    settings = get_settings()
    return settings.RAY_ADDRESS


async def init_ray(address: str | None = None) -> None:
    import ray

    if address is None:
        address = get_ray_address()

    ray.init(address=address, ignore_reinit_error=True)
    logger.info("Ray initialized (address=%s)", address or "local")


async def shutdown_ray() -> None:
    import ray

    ray.shutdown()
    logger.info("Ray shutdown")


def get_ray_dashboard_url() -> str | None:
    try:
        import ray

        if ray.is_initialized():
            context = ray.get_runtime_context()
            dashboard_url = ray.get_dashboard_url()
            return f"http://{dashboard_url}" if dashboard_url else None
    except Exception:
        pass
    return None
