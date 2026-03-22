import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.algorithms import get_algorithm_class
from app.db.session import AsyncSessionLocal
from app.models.multi_agent import AgentPolicy, MultiAgentExperiment
from app.schemas.multi_agent import MultiAgentTrainingRequest

logger = logging.getLogger(__name__)

SUPPORTED_MULTIAGENT_ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "simple_spread_v3": {
        "type": "cooperative",
        "description": "Cooperative navigation",
        "n_agents": 3,
    },
    "simple_adversary_v3": {
        "type": "mixed",
        "description": "Adversarial task",
        "n_agents": 3,
    },
    "simple_tag_v3": {
        "type": "competitive",
        "description": "Predator-prey",
        "n_agents": 4,
    },
    "knights_archers_zombies_v10": {
        "type": "cooperative",
        "description": "Knights and archers",
        "n_agents": 4,
    },
    "pistonball_v6": {
        "type": "cooperative",
        "description": "Piston ball cooperative",
        "n_agents": 15,
    },
}


async def create_multi_agent_experiment(
    db: AsyncSession,
    request: MultiAgentTrainingRequest,
    user_id: int,
) -> MultiAgentExperiment:
    """Create a new multi-agent experiment record."""
    env_info = SUPPORTED_MULTIAGENT_ENVIRONMENTS.get(request.environment_id)
    if env_info is None:
        raise ValueError(
            f"Unsupported multi-agent environment: {request.environment_id}. "
            f"Available: {sorted(SUPPORTED_MULTIAGENT_ENVIRONMENTS.keys())}"
        )

    experiment_name = request.experiment_name or (
        f"ma-{request.algorithm}-{request.environment_id}"
    )

    experiment = MultiAgentExperiment(
        name=experiment_name,
        environment_id=request.environment_id,
        environment_type=env_info["type"],
        n_agents=env_info["n_agents"],
        algorithm=request.algorithm,
        status="pending",
        total_timesteps=request.total_timesteps,
        hyperparameters=request.hyperparameters,
        user_id=user_id,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment


def _train_independent_agents(
    experiment_id: int,
    environment_id: str,
    algorithm: str,
    total_timesteps: int,
    hyperparameters: dict[str, Any],
    n_eval_episodes: int,
    shared_policy: bool = False,
) -> dict[str, Any]:
    """Train agents in a PettingZoo environment using SB3 via SuperSuit."""
    import supersuit as ss
    from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
    from stable_baselines3.common.evaluation import evaluate_policy

    env_modules: dict[str, Any] = {
        "simple_spread_v3": simple_spread_v3,
        "simple_adversary_v3": simple_adversary_v3,
        "simple_tag_v3": simple_tag_v3,
    }

    env_module = env_modules.get(environment_id)
    if env_module is None:
        raise ValueError(f"Training not implemented for {environment_id}")

    start_time = time.monotonic()
    algo_class = get_algorithm_class(algorithm)

    # Create parallel env for training
    env = env_module.parallel_env()
    agents = env.possible_agents[:]
    env.close()

    agent_results: dict[str, dict[str, Any]] = {}

    if shared_policy:
        # Shared policy: train one model using vectorized env
        train_env = env_module.parallel_env()
        vec_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
        vec_env = ss.concat_vec_envs_v1(vec_env, 1, base_class="stable_baselines3")

        model = algo_class("MlpPolicy", vec_env, **hyperparameters)
        model.learn(total_timesteps=total_timesteps)

        raw_mean, raw_std = evaluate_policy(
            model, vec_env, n_eval_episodes=n_eval_episodes
        )
        mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
        std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])

        vec_env.close()

        for agent_id in agents:
            agent_results[agent_id] = {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "shared": True,
            }
    else:
        # Independent policies: train one model per agent
        for agent_id in agents:
            try:
                train_env = env_module.parallel_env()
                vec_env = ss.pettingzoo_env_to_vec_env_v1(train_env)
                vec_env = ss.concat_vec_envs_v1(vec_env, 1, base_class="stable_baselines3")

                model = algo_class("MlpPolicy", vec_env, **hyperparameters)
                model.learn(total_timesteps=total_timesteps // len(agents))

                raw_mean, raw_std = evaluate_policy(
                    model, vec_env, n_eval_episodes=n_eval_episodes
                )
                mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
                std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])

                vec_env.close()

                agent_results[agent_id] = {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "shared": False,
                }
            except Exception:
                logger.exception("Failed to train agent %s", agent_id)
                agent_results[agent_id] = {
                    "mean_reward": 0.0,
                    "std_reward": 0.0,
                    "shared": False,
                    "error": True,
                }

    # Compute team metrics
    all_rewards = [r["mean_reward"] for r in agent_results.values()]
    rewards_array = np.array(all_rewards)
    team_mean = float(rewards_array.mean())
    team_std = float(rewards_array.std())
    training_time = time.monotonic() - start_time

    return {
        "agent_results": agent_results,
        "mean_team_reward": team_mean,
        "std_team_reward": team_std,
        "n_episodes": n_eval_episodes,
        "training_time_seconds": round(training_time, 2),
    }


async def run_multi_agent_training(
    db: AsyncSession,
    experiment_id: int,
    request: MultiAgentTrainingRequest,
) -> None:
    """Run multi-agent training and update the experiment record."""
    result = await db.execute(
        select(MultiAgentExperiment).where(MultiAgentExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise ValueError(f"MultiAgentExperiment {experiment_id} not found")

    experiment.status = "running"
    await db.commit()

    try:
        loop = asyncio.get_event_loop()
        training_result = await loop.run_in_executor(
            None,
            _train_independent_agents,
            experiment_id,
            request.environment_id,
            request.algorithm,
            request.total_timesteps,
            request.hyperparameters,
            request.n_eval_episodes,
            request.shared_policy,
        )

        # Create AgentPolicy records
        env_info = SUPPORTED_MULTIAGENT_ENVIRONMENTS.get(request.environment_id, {})
        env_type = env_info.get("type", "cooperative")
        for agent_id, agent_data in training_result["agent_results"].items():
            policy = AgentPolicy(
                experiment_id=experiment_id,
                agent_id=agent_id,
                role=env_type,
                mean_reward=agent_data["mean_reward"],
                std_reward=agent_data["std_reward"],
            )
            db.add(policy)

        experiment.mean_team_reward = training_result["mean_team_reward"]
        experiment.std_team_reward = training_result["std_team_reward"]
        experiment.n_episodes = training_result["n_episodes"]
        experiment.status = "completed"
        experiment.completed_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(experiment)

        logger.info(
            "Multi-agent experiment %d completed: mean_team_reward=%.2f",
            experiment_id,
            training_result["mean_team_reward"],
        )
    except Exception:
        logger.exception("Multi-agent experiment %d failed", experiment_id)
        experiment.status = "failed"
        await db.commit()
        raise


async def get_multi_agent_experiment(
    db: AsyncSession, experiment_id: int, user_id: int
) -> MultiAgentExperiment | None:
    """Retrieve a single multi-agent experiment with user isolation."""
    result = await db.execute(
        select(MultiAgentExperiment).where(
            MultiAgentExperiment.id == experiment_id,
            MultiAgentExperiment.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def list_multi_agent_experiments(
    db: AsyncSession, user_id: int
) -> list[MultiAgentExperiment]:
    """List all multi-agent experiments for a user."""
    result = await db.execute(
        select(MultiAgentExperiment)
        .where(MultiAgentExperiment.user_id == user_id)
        .order_by(MultiAgentExperiment.created_at.desc())
    )
    return list(result.scalars().all())


async def get_available_multi_agent_environments() -> list[dict[str, Any]]:
    """Return list of supported multi-agent environments."""
    return [
        {"environment_id": env_id, **info}
        for env_id, info in SUPPORTED_MULTIAGENT_ENVIRONMENTS.items()
    ]
