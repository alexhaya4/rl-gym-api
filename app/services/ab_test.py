import asyncio
import logging
import math
from datetime import UTC, datetime
from typing import Any

import gymnasium as gym
import numpy as np
from scipy import stats
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ab_test import ABTest, ABTestResult
from app.schemas.ab_test import ABTestCreate, ABTestStatistics

logger = logging.getLogger(__name__)


async def create_ab_test(
    db: AsyncSession, ab_test_create: ABTestCreate, user_id: int
) -> ABTest:
    """Create a new A/B test."""
    if ab_test_create.model_version_a_id == ab_test_create.model_version_b_id:
        raise ValueError("model_version_a_id and model_version_b_id must be different")

    ab_test = ABTest(
        name=ab_test_create.name,
        description=ab_test_create.description,
        environment_id=ab_test_create.environment_id,
        model_version_a_id=ab_test_create.model_version_a_id,
        model_version_b_id=ab_test_create.model_version_b_id,
        traffic_split_a=ab_test_create.traffic_split_a,
        n_eval_episodes_per_model=ab_test_create.n_eval_episodes_per_model,
        significance_level=ab_test_create.significance_level,
        statistical_test=ab_test_create.statistical_test,
        status="draft",
        user_id=user_id,
    )
    db.add(ab_test)
    await db.commit()
    await db.refresh(ab_test)
    return ab_test


def _run_ab_evaluation(
    environment_id: str,
    model_version_a_id: int,
    model_version_b_id: int,
    n_episodes: int,
) -> dict[str, Any]:
    """Run evaluation episodes for both model variants."""
    rewards_a: list[float] = []
    rewards_b: list[float] = []
    lengths_a: list[int] = []
    lengths_b: list[int] = []

    for _variant, rewards_list, lengths_list in [
        ("a", rewards_a, lengths_a),
        ("b", rewards_b, lengths_b),
    ]:
        env = gym.make(environment_id)
        for _ in range(n_episodes):
            _obs, _info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                _obs, reward, terminated, truncated, _info = env.step(action)
                episode_reward += float(reward)
                episode_length += 1

            rewards_list.append(episode_reward)
            lengths_list.append(episode_length)
        env.close()

    return {
        "rewards_a": rewards_a,
        "rewards_b": rewards_b,
        "lengths_a": lengths_a,
        "lengths_b": lengths_b,
    }


def _calculate_statistics(
    rewards_a: list[float],
    rewards_b: list[float],
    significance_level: float,
    statistical_test: str,
) -> ABTestStatistics:
    """Calculate statistical comparison between two reward distributions."""
    arr_a = np.array(rewards_a)
    arr_b = np.array(rewards_b)

    mean_a = float(arr_a.mean()) if len(arr_a) > 0 else None
    std_a = float(arr_a.std()) if len(arr_a) > 0 else None
    mean_b = float(arr_b.mean()) if len(arr_b) > 0 else None
    std_b = float(arr_b.std()) if len(arr_b) > 0 else None

    p_value: float | None = None
    test_statistic: float | None = None

    if len(arr_a) >= 2 and len(arr_b) >= 2:
        if statistical_test == "mannwhitney":
            stat, p = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        else:
            stat, p = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        test_statistic = float(stat)
        p_value = float(p)

    is_significant = p_value is not None and p_value < significance_level

    # Cohen's d effect size
    effect_size: float | None = None
    if std_a is not None and std_b is not None and mean_a is not None and mean_b is not None:
        pooled_std = math.sqrt((std_a**2 + std_b**2) / 2)
        if pooled_std > 0:
            effect_size = round((mean_b - mean_a) / pooled_std, 4)

    # Determine winner
    winner: str | None = None
    if is_significant and mean_a is not None and mean_b is not None:
        winner = "a" if mean_a > mean_b else "b"

    return ABTestStatistics(
        model_a_mean_reward=round(mean_a, 4) if mean_a is not None else None,
        model_a_std_reward=round(std_a, 4) if std_a is not None else None,
        model_a_n_episodes=len(rewards_a),
        model_b_mean_reward=round(mean_b, 4) if mean_b is not None else None,
        model_b_std_reward=round(std_b, 4) if std_b is not None else None,
        model_b_n_episodes=len(rewards_b),
        p_value=round(p_value, 6) if p_value is not None else None,
        test_statistic=round(test_statistic, 4) if test_statistic is not None else None,
        is_significant=is_significant,
        winner=winner,
        confidence_level=1 - significance_level,
        effect_size=effect_size,
    )


async def run_ab_test(
    db: AsyncSession, ab_test_id: int, user_id: int
) -> ABTest:
    """Execute an A/B test evaluation."""
    result = await db.execute(
        select(ABTest).where(ABTest.id == ab_test_id, ABTest.user_id == user_id)
    )
    ab_test = result.scalar_one_or_none()
    if ab_test is None:
        raise ValueError(f"ABTest {ab_test_id} not found")

    ab_test.status = "running"
    ab_test.started_at = datetime.now(UTC)
    await db.commit()

    try:
        loop = asyncio.get_event_loop()
        eval_results = await loop.run_in_executor(
            None,
            _run_ab_evaluation,
            ab_test.environment_id,
            ab_test.model_version_a_id,
            ab_test.model_version_b_id,
            ab_test.n_eval_episodes_per_model,
        )

        # Save individual results
        for i, reward in enumerate(eval_results["rewards_a"]):
            db.add(ABTestResult(
                ab_test_id=ab_test_id,
                model_variant="a",
                episode_number=i + 1,
                total_reward=reward,
                episode_length=eval_results["lengths_a"][i],
            ))
        for i, reward in enumerate(eval_results["rewards_b"]):
            db.add(ABTestResult(
                ab_test_id=ab_test_id,
                model_variant="b",
                episode_number=i + 1,
                total_reward=reward,
                episode_length=eval_results["lengths_b"][i],
            ))

        # Calculate statistics
        statistics = _calculate_statistics(
            eval_results["rewards_a"],
            eval_results["rewards_b"],
            ab_test.significance_level,
            ab_test.statistical_test,
        )

        ab_test.winner = statistics.winner
        ab_test.p_value = statistics.p_value
        ab_test.test_statistic = statistics.test_statistic
        ab_test.status = "completed"
        ab_test.completed_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(ab_test)

        logger.info(
            "ABTest %d completed: winner=%s, p_value=%s",
            ab_test_id,
            statistics.winner,
            statistics.p_value,
        )
        return ab_test

    except Exception:
        logger.exception("ABTest %d failed", ab_test_id)
        ab_test.status = "failed"
        await db.commit()
        raise


async def get_ab_test(
    db: AsyncSession, ab_test_id: int, user_id: int
) -> ABTest | None:
    """Get a single A/B test with user isolation."""
    result = await db.execute(
        select(ABTest).where(ABTest.id == ab_test_id, ABTest.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def list_ab_tests(
    db: AsyncSession, user_id: int
) -> list[ABTest]:
    """List all A/B tests for a user."""
    result = await db.execute(
        select(ABTest)
        .where(ABTest.user_id == user_id)
        .order_by(ABTest.created_at.desc())
    )
    return list(result.scalars().all())


async def stop_ab_test(
    db: AsyncSession, ab_test_id: int, user_id: int
) -> ABTest | None:
    """Stop a running A/B test."""
    result = await db.execute(
        select(ABTest).where(ABTest.id == ab_test_id, ABTest.user_id == user_id)
    )
    ab_test = result.scalar_one_or_none()
    if ab_test is None:
        return None

    if ab_test.status not in ("running", "draft"):
        return ab_test

    ab_test.status = "stopped"
    ab_test.completed_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(ab_test)
    return ab_test


async def get_ab_test_statistics(
    db: AsyncSession, ab_test_id: int
) -> ABTestStatistics | None:
    """Recalculate statistics from stored A/B test results."""
    result = await db.execute(
        select(ABTest).where(ABTest.id == ab_test_id)
    )
    ab_test = result.scalar_one_or_none()
    if ab_test is None:
        return None

    results_result = await db.execute(
        select(ABTestResult).where(ABTestResult.ab_test_id == ab_test_id)
    )
    results = list(results_result.scalars().all())

    rewards_a = [r.total_reward for r in results if r.model_variant == "a"]
    rewards_b = [r.total_reward for r in results if r.model_variant == "b"]

    if not rewards_a and not rewards_b:
        return None

    return _calculate_statistics(
        rewards_a,
        rewards_b,
        ab_test.significance_level,
        ab_test.statistical_test,
    )
