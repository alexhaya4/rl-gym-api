import sys

import pytest

sys.path.insert(0, "proto")
import inference_pb2
import inference_pb2_grpc


def test_grpc_server_starts():
    from app.grpc_server.server import start_grpc_server, stop_grpc_server

    assert callable(start_grpc_server)
    assert callable(stop_grpc_server)


def test_proto_imports():
    assert hasattr(inference_pb2, "PredictRequest")
    assert hasattr(inference_pb2, "PredictResponse")
    assert hasattr(inference_pb2, "BatchPredictRequest")
    assert hasattr(inference_pb2, "BatchPredictResponse")
    assert hasattr(inference_pb2, "ModelInfoRequest")
    assert hasattr(inference_pb2, "ModelInfoResponse")
    assert hasattr(inference_pb2_grpc, "InferenceServiceStub")
    assert hasattr(inference_pb2_grpc, "InferenceServiceServicer")


def test_predict_request_structure():
    req = inference_pb2.PredictRequest(
        experiment_id=1,
        observation=[0.1, 0.2, 0.3, 0.4],
        deterministic=True,
        model_version=2,
    )
    assert req.experiment_id == 1
    assert list(req.observation) == [
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
        pytest.approx(0.4),
    ]
    assert req.deterministic is True
    assert req.model_version == 2


def test_batch_predict_request_structure():
    req1 = inference_pb2.PredictRequest(
        experiment_id=1,
        observation=[0.1, 0.2, 0.3, 0.4],
        deterministic=True,
    )
    req2 = inference_pb2.PredictRequest(
        experiment_id=1,
        observation=[0.5, 0.6, 0.7, 0.8],
        deterministic=False,
    )
    batch = inference_pb2.BatchPredictRequest(
        experiment_id=1,
        requests=[req1, req2],
    )
    assert batch.experiment_id == 1
    assert len(batch.requests) == 2
    assert batch.requests[0].deterministic is True
    assert batch.requests[1].deterministic is False


def test_model_info_request_structure():
    req = inference_pb2.ModelInfoRequest(
        experiment_id=42,
        version=3,
    )
    assert req.experiment_id == 42
    assert req.version == 3
