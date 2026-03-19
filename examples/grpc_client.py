"""Example gRPC client for the RL Gym inference server.

Usage:
    python examples/grpc_client.py --experiment-id 1
    python examples/grpc_client.py --host localhost --port 50051 --experiment-id 1
"""

import argparse
import sys
import time

import grpc

# Add proto/ to path so generated modules are importable
sys.path.insert(0, "proto")
import inference_pb2
import inference_pb2_grpc


def make_predict_request(stub, experiment_id: int) -> None:
    """Send a single prediction request with a sample CartPole observation."""
    # CartPole-v1 observation space: [cart_pos, cart_vel, pole_angle, pole_vel]
    observation = [0.1, 0.2, 0.3, 0.4]

    print(f"\n--- Predict (experiment_id={experiment_id}) ---")
    print(f"Observation: {observation}")

    start = time.perf_counter()
    response = stub.Predict(
        inference_pb2.PredictRequest(
            experiment_id=experiment_id,
            observation=observation,
            deterministic=True,
            # model_version=0 means use the latest version
            model_version=0,
        )
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Action: {list(response.action)}")
    print(f"Confidence: {response.confidence}")
    print(f"Model version: {response.model_version}")
    print(f"Server latency: {response.latency_us} us")
    print(f"Round-trip time: {elapsed_ms:.2f} ms")


def make_model_info_request(stub, experiment_id: int) -> None:
    """Fetch model metadata from the server."""
    print(f"\n--- GetModelInfo (experiment_id={experiment_id}) ---")

    start = time.perf_counter()
    response = stub.GetModelInfo(
        inference_pb2.ModelInfoRequest(
            experiment_id=experiment_id,
            # version=0 means get the latest version
            version=0,
        )
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Experiment ID: {response.experiment_id}")
    print(f"Version: {response.version}")
    print(f"Algorithm: {response.algorithm}")
    print(f"Mean reward: {response.mean_reward}")
    print(f"Total timesteps: {response.total_timesteps}")
    print(f"Created at: {response.created_at}")
    print(f"Round-trip time: {elapsed_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="gRPC client for RL Gym inference server"
    )
    parser.add_argument(
        "--host", default="localhost", help="gRPC server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=50051, help="gRPC server port (default: 50051)"
    )
    parser.add_argument(
        "--experiment-id",
        type=int,
        required=True,
        help="Experiment ID to run inference on",
    )
    args = parser.parse_args()

    # Create an insecure channel to the gRPC server
    target = f"{args.host}:{args.port}"
    print(f"Connecting to gRPC server at {target}...")
    channel = grpc.insecure_channel(target)

    # Create the stub (client) for the InferenceService
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    try:
        # Fetch model info first to verify connectivity
        make_model_info_request(stub, args.experiment_id)

        # Run a prediction with a sample observation
        make_predict_request(stub, args.experiment_id)
    except grpc.RpcError as e:
        print(f"\ngRPC error: {e.code().name} - {e.details()}")
        sys.exit(1)
    finally:
        channel.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
