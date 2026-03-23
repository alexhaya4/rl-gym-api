import logging

# Ensure proto/ is importable
import sys
from concurrent.futures import ThreadPoolExecutor

import grpc

from app.config import get_settings
from app.grpc_server.auth_interceptor import APIKeyInterceptor
from app.grpc_server.servicer import InferenceServicer

sys.path.insert(0, "proto")
import inference_pb2_grpc

logger = logging.getLogger(__name__)

grpc_server: grpc.aio.Server | None = None


async def start_grpc_server(port: int = 50051) -> None:
    global grpc_server

    grpc_server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=10),
        interceptors=[APIKeyInterceptor()],
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), grpc_server
    )
    grpc_server.add_insecure_port(f"0.0.0.0:{port}")

    settings = get_settings()
    if settings.ENVIRONMENT != "production":
        try:
            from grpc_reflection.v1alpha import reflection

            service_names = (
                inference_pb2_grpc.InferenceService.service_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(service_names, grpc_server)
            logger.info("gRPC server reflection enabled (development mode)")
        except (ImportError, AttributeError):
            pass
        logger.info(
            "gRPC server started without authentication on port %d (development mode)",
            port,
        )
    else:
        logger.info(
            "gRPC server started with API key authentication on port %d", port
        )

    await grpc_server.start()


async def stop_grpc_server() -> None:
    global grpc_server

    if grpc_server is not None:
        await grpc_server.stop(grace=5)
        grpc_server = None
        logger.info("gRPC inference server stopped")
