import asyncio
import logging
import random
from datetime import UTC, datetime
from typing import Any

import gymnasium as gym
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from stable_baselines3.common.evaluation import evaluate_policy

from app.core.algorithms import get_algorithm_class
from app.models.pbt import PBTExperiment, PBTMember
from app.schemas.pbt import PBTRequest
from app.services.optuna_optimization import DEFAULT_HYPERPARAMETER_SPACES

logger = logging.getLogger(__name__)


def _initialize_population(
    population_size: int, algorithm: str
) -> list[dict[str, Any]]:
    """Create random hyperparameter sets for the population."""
    space = DEFAULT_HYPERPARAMETER_SPACES.get(algorithm, {})
    population: list[dict[str, Any]] = []

    for _ in range(population_size):
        params: dict[str, Any] = {}
        for name, config in space.items():
            if config["type"] == "float":
                if config.get("log"):
                    import math
                    log_low = math.log(config["low"])
                    log_high = math.log(config["high"])
                    params[name] = math.exp(random.uniform(log_low, log_high))
                else:
                    params[name] = random.uniform(config["low"], config["high"])
            elif config["type"] == "int":
                params[name] = random.randint(config["low"], config["high"])
        population.append(params)

    return population


def _train_member(
    environment_id: str,
    algorithm: str,
    hyperparameters: dict[str, Any],
    timesteps: int,
) -> tuple[float, float]:
    """Train a single population member and return (mean_reward, std_reward)."""
    env = gym.make(environment_id)
    try:
        algo_class = get_algorithm_class(algorithm)
        model = algo_class("MlpPolicy", env, **hyperparameters)
        model.learn(total_timesteps=timesteps)

        raw_mean, raw_std = evaluate_policy(model, env, n_eval_episodes=5)
        mean_reward = float(raw_mean) if isinstance(raw_mean, float) else float(raw_mean[0])
        std_reward = float(raw_std) if isinstance(raw_std, (float, int)) else float(raw_std[0])
        return mean_reward, std_reward
    finally:
        env.close()


def _exploit(
    population_rewards: list[float],
    population_hyperparams: list[dict[str, Any]],
    bottom_pct: float = 0.2,
) -> list[dict[str, Any]]:
    """Copy hyperparameters from top performers to bottom performers."""
    n = len(population_rewards)
    n_replace = max(1, int(n * bottom_pct))

    ranked = sorted(range(n), key=lambda i: population_rewards[i])
    bottom_indices = ranked[:n_replace]
    top_indices = ranked[-n_replace:]

    updated = [dict(hp) for hp in population_hyperparams]
    for bottom_idx, top_idx in zip(bottom_indices, top_indices, strict=False):
        updated[bottom_idx] = dict(population_hyperparams[top_idx])

    return updated


def _mutate(
    hyperparameters: dict[str, Any],
    mutation_rate: float,
    algorithm: str,
) -> dict[str, Any]:
    """Mutate hyperparameters with probability mutation_rate."""
    space = DEFAULT_HYPERPARAMETER_SPACES.get(algorithm, {})
    mutated = dict(hyperparameters)

    for name, value in mutated.items():
        if random.random() >= mutation_rate:
            continue

        config = space.get(name)
        if config is None:
            continue

        factor = random.uniform(0.8, 1.2)

        if config["type"] == "float":
            new_val = value * factor
            mutated[name] = max(config["low"], min(config["high"], new_val))
        elif config["type"] == "int":
            new_val = int(value * factor)
            mutated[name] = max(config["low"], min(config["high"], new_val))

    return mutated


async def run_pbt(
    db: AsyncSession,
    pbt_experiment_id: int,
    request: PBTRequest,
) -> None:
    """Run Population Based Training."""
    result = await db.execute(
        select(PBTExperiment).where(PBTExperiment.id == pbt_experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise ValueError(f"PBTExperiment {pbt_experiment_id} not found")

    experiment.status = "running"
    await db.commit()

    try:
        # Initialize population
        population = _initialize_population(request.population_size, request.algorithm)

        # Create member records
        members: list[PBTMember] = []
        for i, hp in enumerate(population):
            member = PBTMember(
                pbt_experiment_id=pbt_experiment_id,
                member_index=i,
                hyperparameters=hp,
            )
            db.add(member)
            members.append(member)
        await db.commit()
        for m in members:
            await db.refresh(m)

        # Training loop
        n_generations = max(
            1, request.total_timesteps_per_member // request.exploit_interval
        )

        for gen in range(n_generations):
            # Train all members in parallel
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    None,
                    _train_member,
                    request.environment_id,
                    request.algorithm,
                    population[i],
                    request.exploit_interval,
                )
                for i in range(request.population_size)
            ]
            results = await asyncio.gather(*tasks)

            # Update member rewards
            rewards: list[float] = []
            for i, (mean_r, std_r) in enumerate(results):
                members[i].mean_reward = mean_r
                members[i].std_reward = std_r
                rewards.append(mean_r)
            await db.commit()

            # Exploit: copy best to worst
            population = _exploit(rewards, population)

            # Mutate exploited members
            ranked = sorted(range(len(rewards)), key=lambda idx: rewards[idx])
            n_replace = max(1, int(len(rewards) * 0.2))
            for idx in ranked[:n_replace]:
                population[idx] = _mutate(population[idx], request.mutation_rate, request.algorithm)
                members[idx].hyperparameters = population[idx]
                members[idx].n_exploits += 1
                members[idx].n_mutations += 1

            experiment.n_generations = gen + 1
            await db.commit()

        # Identify best member
        best_idx = max(range(len(rewards)), key=lambda idx: rewards[idx])
        for i, m in enumerate(members):
            m.is_best = i == best_idx
        await db.commit()

        experiment.best_mean_reward = rewards[best_idx]
        experiment.best_hyperparameters = population[best_idx]
        experiment.status = "completed"
        experiment.completed_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(experiment)

        logger.info(
            "PBT experiment %d completed: best_mean_reward=%.2f",
            pbt_experiment_id,
            rewards[best_idx],
        )

    except Exception:
        logger.exception("PBT experiment %d failed", pbt_experiment_id)
        experiment.status = "failed"
        await db.commit()
        raise


async def create_pbt_experiment(
    db: AsyncSession, request: PBTRequest, user_id: int
) -> PBTExperiment:
    """Create a new PBT experiment."""
    experiment_name = request.experiment_name or (
        f"pbt-{request.algorithm}-{request.environment_id}"
    )
    experiment = PBTExperiment(
        name=experiment_name,
        environment_id=request.environment_id,
        algorithm=request.algorithm,
        population_size=request.population_size,
        total_timesteps_per_member=request.total_timesteps_per_member,
        exploit_interval=request.exploit_interval,
        mutation_rate=request.mutation_rate,
        status="pending",
        user_id=user_id,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return experiment


async def get_pbt_experiment(
    db: AsyncSession, pbt_id: int, user_id: int
) -> PBTExperiment | None:
    """Get a PBT experiment with user isolation."""
    result = await db.execute(
        select(PBTExperiment).where(
            PBTExperiment.id == pbt_id, PBTExperiment.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def list_pbt_experiments(
    db: AsyncSession, user_id: int
) -> list[PBTExperiment]:
    """List all PBT experiments for a user."""
    result = await db.execute(
        select(PBTExperiment)
        .where(PBTExperiment.user_id == user_id)
        .order_by(PBTExperiment.created_at.desc())
    )
    return list(result.scalars().all())
