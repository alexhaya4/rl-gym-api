import logging

# Ensure proto/ is importable
import sys
from concurrent.futures import ThreadPoolExecutor

import grpc

from app.grpc_server.servicer import InferenceServicer

sys.path.insert(0, "proto")
import inference_pb2_grpc

logger = logging.getLogger(__name__)

grpc_server: grpc.aio.Server | None = None


async def start_grpc_server(port: int = 50051) -> None:
    global grpc_server

    grpc_server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=10),
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(), grpc_server
    )
    grpc_server.add_insecure_port(f"0.0.0.0:{port}")
    await grpc_server.start()
    logger.info("gRPC inference server started on port %d", port)


async def stop_grpc_server() -> None:
    global grpc_server

    if grpc_server is not None:
        await grpc_server.stop(grace=5)
        grpc_server = None
        logger.info("gRPC inference server stopped")
