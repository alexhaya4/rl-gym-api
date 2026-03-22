import csv
import io
from itertools import combinations
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode
from app.models.experiment import Experiment
from app.schemas.artifact import LineageGraph, LineageNode
from app.schemas.comparison import ComparisonResponse, ExperimentDiff


def _experiment_to_dict(exp: Experiment) -> dict[str, Any]:
    return {
        "id": exp.id,
        "name": exp.name,
        "environment_id": exp.environment_id,
        "algorithm": exp.algorithm,
        "status": exp.status,
        "hyperparameters": exp.hyperparameters or {},
        "total_timesteps": exp.total_timesteps,
        "mean_reward": exp.mean_reward,
        "std_reward": exp.std_reward,
        "tags": exp.tags or [],
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
    }


def _compute_diff(exp_a: Experiment, exp_b: Experiment) -> ExperimentDiff:
    """Compute pairwise diff between two experiments."""
    hp_a = exp_a.hyperparameters or {}
    hp_b = exp_b.hyperparameters or {}
    all_keys = set(hp_a.keys()) | set(hp_b.keys())
    hyperparameter_diff: dict[str, Any] = {}
    for key in all_keys:
        val_a = hp_a.get(key)
        val_b = hp_b.get(key)
        if val_a != val_b:
            hyperparameter_diff[key] = {"a": val_a, "b": val_b}

    metrics_diff: dict[str, Any] = {
        "mean_reward": {"a": exp_a.mean_reward, "b": exp_b.mean_reward},
        "std_reward": {"a": exp_a.std_reward, "b": exp_b.std_reward},
    }

    reward_a = exp_a.mean_reward
    reward_b = exp_b.mean_reward
    winner: str | None = None
    improvement_pct: float | None = None

    if reward_a is not None and reward_b is not None:
        if reward_b > reward_a:
            winner = "b"
        elif reward_a > reward_b:
            winner = "a"

        if reward_a != 0.0:
            improvement_pct = round(((reward_b - reward_a) / abs(reward_a)) * 100, 2)

    return ExperimentDiff(
        experiment_id_a=exp_a.id,
        experiment_id_b=exp_b.id,
        name_a=exp_a.name,
        name_b=exp_b.name,
        hyperparameter_diff=hyperparameter_diff,
        metrics_diff=metrics_diff,
        status_a=exp_a.status,
        status_b=exp_b.status,
        winner=winner,
        improvement_pct=improvement_pct,
    )


async def compare_experiments(
    db: AsyncSession, experiment_ids: list[int], user_id: int
) -> ComparisonResponse:
    """Compare multiple experiments and return pairwise diffs."""
    result = await db.execute(
        select(Experiment).where(
            Experiment.id.in_(experiment_ids),
            Experiment.user_id == user_id,
        )
    )
    experiments = list(result.scalars().all())

    if len(experiments) != len(experiment_ids):
        found_ids = {e.id for e in experiments}
        missing = [eid for eid in experiment_ids if eid not in found_ids]
        raise ValueError(f"Experiments not found or not owned by user: {missing}")

    exp_map = {e.id: e for e in experiments}
    diffs: list[ExperimentDiff] = []
    for id_a, id_b in combinations(experiment_ids, 2):
        diffs.append(_compute_diff(exp_map[id_a], exp_map[id_b]))

    completed = [e for e in experiments if e.mean_reward is not None]
    best_id: int | None = None
    if completed:
        best = max(completed, key=lambda e: e.mean_reward)  # type: ignore[arg-type]
        best_id = best.id

    return ComparisonResponse(
        experiments=[_experiment_to_dict(exp_map[eid]) for eid in experiment_ids],
        diffs=diffs,
        best_experiment_id=best_id,
    )


async def get_experiment_diff(
    db: AsyncSession, exp_id_a: int, exp_id_b: int
) -> ExperimentDiff:
    """Get a detailed diff between two experiments."""
    result_a = await db.execute(
        select(Experiment).where(Experiment.id == exp_id_a)
    )
    exp_a = result_a.scalar_one_or_none()
    if exp_a is None:
        raise ValueError(f"Experiment {exp_id_a} not found")

    result_b = await db.execute(
        select(Experiment).where(Experiment.id == exp_id_b)
    )
    exp_b = result_b.scalar_one_or_none()
    if exp_b is None:
        raise ValueError(f"Experiment {exp_id_b} not found")

    return _compute_diff(exp_a, exp_b)


async def get_lineage_graph(
    db: AsyncSession, experiment_id: int
) -> LineageGraph:
    """Build a lineage graph traversing parent/child relationships."""
    nodes: list[LineageNode] = []
    edges: list[dict[str, Any]] = []
    visited: set[int] = set()
    root_id: int | None = None

    async def _traverse_up(exp_id: int) -> None:
        nonlocal root_id
        if exp_id in visited:
            return
        visited.add(exp_id)

        result = await db.execute(
            select(Experiment).where(Experiment.id == exp_id)
        )
        exp = result.scalar_one_or_none()
        if exp is None:
            return

        # Find children
        child_result = await db.execute(
            select(Experiment.id).where(Experiment.parent_experiment_id == exp_id)
        )
        children = [row[0] for row in child_result.all()]

        nodes.append(LineageNode(
            experiment_id=exp.id,
            experiment_name=exp.name,
            algorithm=exp.algorithm,
            environment_id=exp.environment_id,
            mean_reward=exp.mean_reward,
            parent_id=exp.parent_experiment_id,
            children=children,
            tags=exp.tags or [],
        ))

        if exp.parent_experiment_id is not None:
            edges.append({
                "from": exp.parent_experiment_id,
                "to": exp.id,
                "type": "derived_from",
            })
            await _traverse_up(exp.parent_experiment_id)
        else:
            root_id = exp.id

        for child_id in children:
            if child_id not in visited:
                edges.append({
                    "from": exp.id,
                    "to": child_id,
                    "type": "derived_from",
                })
                await _traverse_up(child_id)

    await _traverse_up(experiment_id)

    return LineageGraph(
        nodes=nodes,
        edges=edges,
        root_experiment_id=root_id,
    )


async def set_experiment_tags(
    db: AsyncSession, experiment_id: int, tags: list[str], user_id: int
) -> bool:
    """Update experiment tags."""
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id, Experiment.user_id == user_id
        )
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        return False

    experiment.tags = tags
    await db.commit()
    return True


async def export_experiment_csv(db: AsyncSession, experiment_id: int) -> str:
    """Export experiment episodes as CSV string."""
    result = await db.execute(
        select(Episode)
        .where(Episode.experiment_id == experiment_id)
        .order_by(Episode.episode_number)
    )
    episodes = list(result.scalars().all())

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["episode_number", "total_reward", "episode_length", "mean_reward"])
    for ep in episodes:
        writer.writerow([ep.episode_number, ep.total_reward, ep.episode_length, ep.mean_reward])

    return output.getvalue()


async def export_experiment_json(
    db: AsyncSession, experiment_id: int
) -> dict[str, Any]:
    """Export full experiment data as dict."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id} not found")

    ep_result = await db.execute(
        select(Episode)
        .where(Episode.experiment_id == experiment_id)
        .order_by(Episode.episode_number)
    )
    episodes = list(ep_result.scalars().all())

    return {
        **_experiment_to_dict(experiment),
        "episodes": [
            {
                "episode_number": ep.episode_number,
                "total_reward": ep.total_reward,
                "episode_length": ep.episode_length,
                "mean_reward": ep.mean_reward,
                "std_reward": ep.std_reward,
            }
            for ep in episodes
        ],
    }
