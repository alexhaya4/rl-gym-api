_pipeline_runs: dict[str, dict] = {}


def store_pipeline_run(pipeline_id: str, data: dict) -> None:
    _pipeline_runs[pipeline_id] = data


def get_pipeline_run(pipeline_id: str) -> dict | None:
    return _pipeline_runs.get(pipeline_id)


def list_pipeline_runs(user_id: int) -> list[dict]:
    return [
        run for run in _pipeline_runs.values()
        if run.get("user_id") == user_id
    ]


def update_pipeline_run(pipeline_id: str, updates: dict) -> None:
    if pipeline_id in _pipeline_runs:
        _pipeline_runs[pipeline_id].update(updates)
