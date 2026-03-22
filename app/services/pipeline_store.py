from typing import Any

_pipeline_runs: dict[str, dict[str, Any]] = {}


def store_pipeline_run(pipeline_id: str, data: dict[str, Any]) -> None:
    _pipeline_runs[pipeline_id] = data


def get_pipeline_run(pipeline_id: str) -> dict[str, Any] | None:
    return _pipeline_runs.get(pipeline_id)


def list_pipeline_runs(user_id: int) -> list[dict[str, Any]]:
    return [
        run for run in _pipeline_runs.values()
        if run.get("user_id") == user_id
    ]


def update_pipeline_run(pipeline_id: str, updates: dict[str, Any]) -> None:
    if pipeline_id in _pipeline_runs:
        _pipeline_runs[pipeline_id].update(updates)
