import io

import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "mluser",
            "email": "mluser@example.com",
            "password": "securepassword",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "mluser", "password": "securepassword"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def auth_headers_b(client: AsyncClient) -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": "mluser2",
            "email": "mluser2@example.com",
            "password": "securepassword",
        },
    )
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "mluser2", "password": "securepassword"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# Classification CSV: 4 features, 1 binary target
CLASSIFICATION_CSV = (
    "f1,f2,f3,f4,target\n"
    "1.0,2.0,3.0,4.0,0\n"
    "2.0,3.0,4.0,5.0,1\n"
    "3.0,4.0,5.0,6.0,0\n"
    "4.0,5.0,6.0,7.0,1\n"
    "5.0,6.0,7.0,8.0,0\n"
    "6.0,7.0,8.0,9.0,1\n"
    "1.5,2.5,3.5,4.5,0\n"
    "2.5,3.5,4.5,5.5,1\n"
    "3.5,4.5,5.5,6.5,0\n"
    "4.5,5.5,6.5,7.5,1\n"
    "5.5,6.5,7.5,8.5,0\n"
    "6.5,7.5,8.5,9.5,1\n"
    "1.2,2.2,3.2,4.2,0\n"
    "2.2,3.2,4.2,5.2,1\n"
    "3.2,4.2,5.2,6.2,0\n"
    "4.2,5.2,6.2,7.2,1\n"
    "5.2,6.2,7.2,8.2,0\n"
    "6.2,7.2,8.2,9.2,1\n"
    "1.8,2.8,3.8,4.8,0\n"
    "2.8,3.8,4.8,5.8,1\n"
)

# Regression CSV
REGRESSION_CSV = (
    "x1,x2,y\n"
    "1.0,2.0,3.0\n"
    "2.0,4.0,6.0\n"
    "3.0,6.0,9.0\n"
    "4.0,8.0,12.0\n"
    "5.0,10.0,15.0\n"
    "6.0,12.0,18.0\n"
    "7.0,14.0,21.0\n"
    "8.0,16.0,24.0\n"
    "9.0,18.0,27.0\n"
    "10.0,20.0,30.0\n"
)

# Clustering CSV (no target)
CLUSTERING_CSV = (
    "a,b\n"
    "1.0,1.0\n"
    "1.1,1.1\n"
    "1.2,0.9\n"
    "5.0,5.0\n"
    "5.1,5.1\n"
    "5.2,4.9\n"
    "9.0,9.0\n"
    "9.1,9.1\n"
    "9.2,8.9\n"
)


async def _upload_csv(
    client: AsyncClient,
    auth_headers: dict[str, str],
    name: str,
    csv_content: str,
) -> int:
    """Upload a CSV and return dataset_id."""
    response = await client.post(
        "/api/v1/datasets/upload",
        files={"file": ("data.csv", io.BytesIO(csv_content.encode()), "text/csv")},
        data={"name": name},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def test_list_algorithms(client: AsyncClient) -> None:
    """GET /ml/algorithms returns dict with 4 task types."""
    response = await client.get("/api/v1/ml/algorithms")
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "regression" in data
    assert "clustering" in data
    assert "dimensionality_reduction" in data
    assert "RandomForestClassifier" in data["classification"]


async def test_train_classification(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Train RandomForestClassifier on CSV dataset."""
    dataset_id = await _upload_csv(client, auth_headers, "clf-data", CLASSIFICATION_CSV)
    response = await client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": dataset_id,
            "algorithm": "RandomForestClassifier",
            "target_column": "target",
            "task_type": "classification",
            "hyperparameters": {"n_estimators": 10, "random_state": 42},
        },
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["algorithm"] == "RandomForestClassifier"
    assert data["task_type"] == "classification"
    assert "accuracy" in data["metrics"]
    assert "f1" in data["metrics"]
    assert "confusion_matrix" in data["metrics"]
    assert data["feature_importance"] is not None
    assert data["training_time_seconds"] > 0


async def test_train_regression(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Train LinearRegression on CSV dataset."""
    dataset_id = await _upload_csv(client, auth_headers, "reg-data", REGRESSION_CSV)
    response = await client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": dataset_id,
            "algorithm": "LinearRegression",
            "target_column": "y",
            "task_type": "regression",
        },
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert "mse" in data["metrics"]
    assert "r2" in data["metrics"]
    assert data["metrics"]["r2"] > 0.9  # Perfect linear relationship


async def test_train_clustering(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Train KMeans on CSV dataset."""
    dataset_id = await _upload_csv(client, auth_headers, "cluster-data", CLUSTERING_CSV)
    response = await client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": dataset_id,
            "algorithm": "KMeans",
            "task_type": "clustering",
            "hyperparameters": {"n_clusters": 3, "random_state": 42},
        },
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert "silhouette_score" in data["metrics"]
    assert "n_clusters" in data["metrics"]
    assert data["metrics"]["n_clusters"] == 3


async def test_predict(client: AsyncClient, auth_headers: dict[str, str]) -> None:
    """Train model then predict on new features."""
    dataset_id = await _upload_csv(
        client, auth_headers, "pred-data", CLASSIFICATION_CSV
    )
    train_resp = await client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": dataset_id,
            "algorithm": "RandomForestClassifier",
            "target_column": "target",
            "task_type": "classification",
            "hyperparameters": {"n_estimators": 10, "random_state": 42},
        },
        headers=auth_headers,
    )
    assert train_resp.status_code == 201
    model_id = train_resp.json()["model_id"]

    pred_resp = await client.post(
        "/api/v1/ml/predict",
        json={
            "model_id": model_id,
            "features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        },
        headers=auth_headers,
    )
    assert pred_resp.status_code == 200
    data = pred_resp.json()
    assert len(data["predictions"]) == 2
    assert data["probabilities"] is not None
    assert data["inference_time_ms"] >= 0


async def test_predict_wrong_shape(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Features shape mismatch returns 422."""
    dataset_id = await _upload_csv(
        client, auth_headers, "shape-data", CLASSIFICATION_CSV
    )
    train_resp = await client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": dataset_id,
            "algorithm": "RandomForestClassifier",
            "target_column": "target",
            "task_type": "classification",
            "hyperparameters": {"n_estimators": 10, "random_state": 42},
        },
        headers=auth_headers,
    )
    model_id = train_resp.json()["model_id"]

    pred_resp = await client.post(
        "/api/v1/ml/predict",
        json={
            "model_id": model_id,
            "features": [[1.0, 2.0]],  # Should be 4 features
        },
        headers=auth_headers,
    )
    assert pred_resp.status_code == 422


async def test_train_nonexistent_dataset(
    client: AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Training on nonexistent dataset returns 404."""
    response = await client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": 99999,
            "algorithm": "RandomForestClassifier",
            "target_column": "target",
            "task_type": "classification",
        },
        headers=auth_headers,
    )
    assert response.status_code == 404


async def test_owner_only_access(
    client: AsyncClient,
    auth_headers: dict[str, str],
    auth_headers_b: dict[str, str],
) -> None:
    """User cannot access other user's model."""
    dataset_id = await _upload_csv(
        client, auth_headers, "owner-data", CLASSIFICATION_CSV
    )
    train_resp = await client.post(
        "/api/v1/ml/train",
        json={
            "dataset_id": dataset_id,
            "algorithm": "RandomForestClassifier",
            "target_column": "target",
            "task_type": "classification",
            "hyperparameters": {"n_estimators": 10, "random_state": 42},
        },
        headers=auth_headers,
    )
    model_id = train_resp.json()["model_id"]

    # User B tries to access
    response = await client.get(
        f"/api/v1/ml/models/{model_id}",
        headers=auth_headers_b,
    )
    assert response.status_code == 404

    response = await client.post(
        "/api/v1/ml/predict",
        json={"model_id": model_id, "features": [[1.0, 2.0, 3.0, 4.0]]},
        headers=auth_headers_b,
    )
    assert response.status_code == 404
