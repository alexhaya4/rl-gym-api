import logging
from typing import Any

import grpc
import grpc.aio

from app.config import get_settings

logger = logging.getLogger(__name__)

# RPCs that do not require authentication
_PUBLIC_METHODS = frozenset({"/InferenceService/GetModelInfo"})


class APIKeyInterceptor(grpc.aio.ServerInterceptor):  # type: ignore[misc]
    """Validate x-api-key metadata on incoming gRPC requests.

    If GRPC_API_KEY is not configured (None), all requests are allowed
    (development mode). The GetModelInfo RPC is always exempt from auth.
    """

    async def intercept_service(
        self,
        continuation: Any,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Any:
        method = handler_call_details.method or ""
        if method in _PUBLIC_METHODS:
            return await continuation(handler_call_details)

        settings = get_settings()
        if settings.GRPC_API_KEY is None:
            return await continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or [])
        api_key = metadata.get("x-api-key")

        if api_key != settings.GRPC_API_KEY:
            logger.warning("gRPC auth failed: invalid or missing API key for %s", method)
            return _abort_unauthenticated

        return await continuation(handler_call_details)


async def _abort_unauthenticated(
    request: Any, context: grpc.aio.ServicerContext
) -> None:
    await context.abort(
        grpc.StatusCode.UNAUTHENTICATED,
        "Invalid or missing API key",
    )
