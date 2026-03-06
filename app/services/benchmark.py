import asyncio
import time
import uuid
from datetime import UTC, datetime

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from app.schemas.benchmark import BenchmarkRequest, BenchmarkResponse, BenchmarkResult

ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


def _train_and_evaluate(
    environment_id: str,
    algorithm: str,
    total_timesteps: int,
    n_eval_episodes: int,
) -> BenchmarkResult:
    env = gym.make(environment_id)
    try:
        algo_class = ALGORITHMS[algorithm]
        model = algo_class("MlpPolicy", env)

        start_time = time.time()
        model.learn(total_timesteps=total_timesteps)
        training_time = time.time() - start_time

        raw_mean, raw_std = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes
        )
        mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
        std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])

        return BenchmarkResult(
            environment_id=environment_id,
            algorithm=algorithm,
            mean_reward=mean_reward,
            std_reward=std_reward,
            training_time_seconds=round(training_time, 3),
            total_timesteps=total_timesteps,
        )
    finally:
        env.close()


async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    loop = asyncio.get_event_loop()
    results: list[BenchmarkResult] = []

    for environment_id in request.environments:
        for algorithm in request.algorithms:
            try:
                result = await loop.run_in_executor(
                    None,
                    _train_and_evaluate,
                    environment_id,
                    algorithm,
                    request.total_timesteps,
                    request.n_eval_episodes,
                )
                results.append(result)
            except Exception:
                results.append(
                    BenchmarkResult(
                        environment_id=environment_id,
                        algorithm=algorithm,
                        mean_reward=0.0,
                        std_reward=0.0,
                        training_time_seconds=0.0,
                        total_timesteps=request.total_timesteps,
                    )
                )

    return BenchmarkResponse(
        benchmark_id=str(uuid.uuid4()),
        results=results,
        total_combinations=len(request.environments) * len(request.algorithms),
        completed_at=datetime.now(UTC).isoformat(),
    )
