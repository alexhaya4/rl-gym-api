import asyncio
from datetime import UTC, datetime

import gymnasium as gym
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from app.models.episode import Episode
from app.models.experiment import Experiment
from app.schemas.evaluation import (
    EpisodeMetrics,
    EvaluationRequest,
    EvaluationResponse,
)

ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


def _run_evaluation(
    environment_id: str,
    algorithm: str,
    n_eval_episodes: int,
    deterministic: bool,
) -> tuple[list[float], list[int]]:
    env = gym.make(environment_id)
    try:
        algo_class = ALGORITHMS[algorithm]
        model = algo_class("MlpPolicy", env)
        model.learn(total_timesteps=1000)

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
        )

        if isinstance(episode_rewards, float):
            reward_list = [episode_rewards]
        else:
            reward_list = [float(r) for r in episode_rewards]

        if isinstance(episode_lengths, (int, float)):
            length_list = [int(episode_lengths)]
        else:
            length_list = [int(l) for l in episode_lengths]

        return reward_list, length_list
    finally:
        env.close()


async def evaluate_experiment(
    db: AsyncSession, request: EvaluationRequest
) -> EvaluationResponse:
    result = await db.execute(
        select(Experiment).where(Experiment.id == request.experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise ValueError(f"Experiment {request.experiment_id} not found")
    if experiment.status != "completed":
        raise ValueError(
            f"Experiment {request.experiment_id} has status '{experiment.status}', expected 'completed'"
        )

    environment_id = request.environment_id or experiment.environment_id
    algorithm = experiment.algorithm

    loop = asyncio.get_event_loop()
    episode_rewards, episode_lengths = await loop.run_in_executor(
        None,
        _run_evaluation,
        environment_id,
        algorithm,
        request.n_eval_episodes,
        request.deterministic,
    )

    rewards_array = np.array(episode_rewards)
    mean_reward = float(rewards_array.mean())
    std_reward = float(rewards_array.std())
    min_reward = float(rewards_array.min())
    max_reward = float(rewards_array.max())

    episodes: list[EpisodeMetrics] = []
    for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
        episodes.append(
            EpisodeMetrics(
                episode_number=i + 1,
                total_reward=reward,
                episode_length=length,
            )
        )

        db.add(
            Episode(
                experiment_id=request.experiment_id,
                episode_number=i + 1,
                total_reward=reward,
                episode_length=length,
                mean_reward=mean_reward,
                std_reward=std_reward,
            )
        )

    await db.commit()

    return EvaluationResponse(
        experiment_id=request.experiment_id,
        environment_id=environment_id,
        algorithm=algorithm,
        n_eval_episodes=request.n_eval_episodes,
        mean_reward=mean_reward,
        std_reward=std_reward,
        min_reward=min_reward,
        max_reward=max_reward,
        episodes=episodes,
        evaluated_at=datetime.now(UTC).isoformat(),
    )
