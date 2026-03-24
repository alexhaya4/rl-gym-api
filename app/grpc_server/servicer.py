import logging
import os
import sys
import time
from typing import Any

import grpc
import numpy as np
from sqlalchemy import select

from app.core.prometheus import grpc_latency_microseconds, grpc_requests_total
from app.db.session import AsyncSessionLocal
from app.models.model_version import ModelVersion
from app.services.model_storage import load_model

# Ensure proto/ is importable (works both locally and in Docker)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proto"))
import inference_pb2
import inference_pb2_grpc

logger = logging.getLogger(__name__)

# Module-level model cache: "experiment_id:version" -> loaded SB3 model
_model_cache: dict[str, Any] = {}


def _cache_key(experiment_id: int, version: int) -> str:
    return f"{experiment_id}:{version}"


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC servicer implementing the InferenceService."""

    async def Predict(  # noqa: N802
        self,
        request: inference_pb2.PredictRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.PredictResponse:
        start = time.perf_counter_ns()

        try:
            experiment_id = request.experiment_id
            version = request.model_version if request.model_version > 0 else None
            cache_key = _cache_key(experiment_id, version or 0)

            if cache_key not in _model_cache:
                async with AsyncSessionLocal() as db:
                    try:
                        model, model_version = await load_model(
                            db, experiment_id, version
                        )
                        _model_cache[cache_key] = (model, model_version)
                    except ValueError as e:
                        grpc_requests_total.labels(method="Predict", status="error").inc()
                        await context.abort(grpc.StatusCode.NOT_FOUND, str(e))

            model, model_version = _model_cache[cache_key]

            obs = np.array(list(request.observation), dtype=np.float32)
            action, _ = model.predict(obs, deterministic=request.deterministic)

            action_list = action.flatten().tolist() if hasattr(action, "flatten") else [float(action)]
            latency_us = (time.perf_counter_ns() - start) // 1000

            grpc_requests_total.labels(method="Predict", status="success").inc()
            grpc_latency_microseconds.labels(method="Predict").observe(latency_us)

            return inference_pb2.PredictResponse(
                action=action_list,
                confidence=1.0,
                latency_us=latency_us,
                model_version=str(model_version.version),
            )
        except Exception:
            grpc_requests_total.labels(method="Predict", status="error").inc()
            raise

    async def BatchPredict(  # noqa: N802
        self,
        request: inference_pb2.BatchPredictRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.BatchPredictResponse:
        start = time.perf_counter_ns()

        grpc_requests_total.labels(method="BatchPredict", status="success").inc()

        responses = []
        for req in request.requests:
            resp = await self.Predict(req, context)
            responses.append(resp)

        total_latency_us = (time.perf_counter_ns() - start) // 1000

        return inference_pb2.BatchPredictResponse(
            responses=responses,
            total_latency_us=total_latency_us,
        )

    async def GetModelInfo(  # noqa: N802
        self,
        request: inference_pb2.ModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.ModelInfoResponse:
        grpc_requests_total.labels(method="GetModelInfo", status="success").inc()

        async with AsyncSessionLocal() as db:
            query = select(ModelVersion).where(
                ModelVersion.experiment_id == request.experiment_id
            )
            if request.version > 0:
                query = query.where(ModelVersion.version == request.version)
            else:
                query = query.order_by(ModelVersion.version.desc()).limit(1)

            result = await db.execute(query)
            mv = result.scalar_one_or_none()

        if mv is None:
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"No model found for experiment {request.experiment_id}",
            )
            return inference_pb2.ModelInfoResponse()

        return inference_pb2.ModelInfoResponse(
            experiment_id=mv.experiment_id,
            version=mv.version,
            algorithm=mv.algorithm,
            mean_reward=mv.mean_reward or 0.0,
            total_timesteps=mv.total_timesteps or 0,
            created_at=mv.created_at.isoformat() if mv.created_at else "",
        )
