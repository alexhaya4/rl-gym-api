import logging
import os
import time
import uuid
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import cluster, decomposition, ensemble, linear_model, neighbors, svm, tree
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.dataset import Dataset
from app.models.ml_model import MLModel
from app.schemas.ml import (
    MLPredictRequest,
    MLPredictResponse,
    MLTrainRequest,
    MLTrainResponse,
)

logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS: dict[str, list[str]] = {
    "classification": [
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "SVC",
        "KNeighborsClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
    ],
    "regression": [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "SVR",
        "KNeighborsRegressor",
        "LinearRegression",
        "DecisionTreeRegressor",
    ],
    "clustering": ["KMeans", "DBSCAN", "AgglomerativeClustering"],
    "dimensionality_reduction": ["PCA", "TruncatedSVD"],
}

_ALGORITHM_MAP: dict[str, Any] = {
    # Classification
    "RandomForestClassifier": ensemble.RandomForestClassifier,
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier,
    "SVC": svm.SVC,
    "KNeighborsClassifier": neighbors.KNeighborsClassifier,
    "LogisticRegression": linear_model.LogisticRegression,
    "DecisionTreeClassifier": tree.DecisionTreeClassifier,
    # Regression
    "RandomForestRegressor": ensemble.RandomForestRegressor,
    "GradientBoostingRegressor": ensemble.GradientBoostingRegressor,
    "SVR": svm.SVR,
    "KNeighborsRegressor": neighbors.KNeighborsRegressor,
    "LinearRegression": linear_model.LinearRegression,
    "DecisionTreeRegressor": tree.DecisionTreeRegressor,
    # Clustering
    "KMeans": cluster.KMeans,
    "DBSCAN": cluster.DBSCAN,
    "AgglomerativeClustering": cluster.AgglomerativeClustering,
    # Dimensionality reduction
    "PCA": decomposition.PCA,
    "TruncatedSVD": decomposition.TruncatedSVD,
}


async def train(
    request: MLTrainRequest, user_id: int, db: AsyncSession
) -> MLTrainResponse:
    """Train a scikit-learn model on a file dataset."""
    # Validate task_type and algorithm
    if request.task_type not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unknown task type: {request.task_type}")
    if request.algorithm not in SUPPORTED_ALGORITHMS[request.task_type]:
        raise ValueError(
            f"Algorithm '{request.algorithm}' not supported for {request.task_type}. "
            f"Supported: {SUPPORTED_ALGORITHMS[request.task_type]}"
        )

    # Load dataset
    result = await db.execute(
        select(Dataset).where(
            Dataset.id == request.dataset_id,
            Dataset.user_id == user_id,
        )
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise FileNotFoundError(
            f"Dataset {request.dataset_id} not found or not owned by user"
        )
    if not dataset.file_path or not os.path.exists(dataset.file_path):
        raise FileNotFoundError("Dataset file not found on disk")

    # Read CSV
    df = pd.read_csv(dataset.file_path)
    original_len = len(df)
    df = df.dropna()
    nan_rows_dropped = original_len - len(df)

    if len(df) == 0:
        raise ValueError("Dataset is empty after dropping NaN rows")

    # Prepare features and target
    is_supervised = request.task_type in ("classification", "regression")

    if is_supervised:
        if request.target_column is None:
            raise ValueError("target_column is required for supervised learning")
        if request.target_column not in df.columns:
            raise ValueError(f"Target column '{request.target_column}' not in dataset")
        y = df[request.target_column]
        X = df.drop(columns=[request.target_column])  # noqa: N806
        # Select only numeric columns for features
        X = X.select_dtypes(include=[np.number])  # noqa: N806
        if X.empty:
            raise ValueError("No numeric feature columns found")
        feature_columns = X.columns.tolist()
    else:
        X = df.select_dtypes(include=[np.number])  # noqa: N806
        if X.empty:
            raise ValueError("No numeric columns found")
        feature_columns = X.columns.tolist()
        y = None

    # Instantiate model
    algo_cls = _ALGORITHM_MAP[request.algorithm]
    model = algo_cls()
    if request.hyperparameters:
        model.set_params(**request.hyperparameters)

    # For SVC, enable probability if classification
    if request.algorithm == "SVC" and request.task_type == "classification":
        model.set_params(probability=True)

    start_time = time.perf_counter()

    # Train
    metrics: dict[str, Any] = {}
    import asyncio

    loop = asyncio.get_running_loop()

    if is_supervised:
        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
            X, y, test_size=request.test_split, random_state=42
        )

        def _fit_supervised() -> None:
            model.fit(X_train, y_train)

        await loop.run_in_executor(None, _fit_supervised)
        y_pred = model.predict(X_test)

        if request.task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(
                    precision_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "f1": float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
        else:
            mse = float(mean_squared_error(y_test, y_pred))
            metrics = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }
    else:
        # Unsupervised
        if request.task_type == "clustering":

            def _fit_clustering() -> None:
                model.fit(X)

            await loop.run_in_executor(None, _fit_clustering)
            labels = model.labels_
            n_labels = len(set(labels)) - (1 if -1 in labels else 0)
            if n_labels > 1:
                metrics["silhouette_score"] = float(silhouette_score(X, labels))
                metrics["calinski_harabasz_score"] = float(
                    calinski_harabasz_score(X, labels)
                )
            metrics["n_clusters"] = n_labels
        elif request.task_type == "dimensionality_reduction":

            def _fit_dr() -> None:
                model.fit(X)

            await loop.run_in_executor(None, _fit_dr)
            if hasattr(model, "explained_variance_ratio_"):
                metrics["explained_variance_ratio"] = (
                    model.explained_variance_ratio_.tolist()
                )
                metrics["total_explained_variance"] = float(
                    sum(model.explained_variance_ratio_)
                )

    training_time = time.perf_counter() - start_time

    # Feature importance
    feature_importance: list[dict[str, Any]] | None = None
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        feature_importance = [
            {"feature": col, "importance": float(imp)}
            for col, imp in zip(feature_columns, importances, strict=False)
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    # Save model
    settings = get_settings()
    user_dir = os.path.join(settings.DATASET_STORAGE_PATH, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    model_filename = f"ml_{uuid.uuid4().hex}.joblib"
    model_path = os.path.join(user_dir, model_filename)
    joblib.dump(model, model_path)

    # Create DB record
    ml_model = MLModel(
        name=f"{request.algorithm}_{dataset.name}",
        algorithm=request.algorithm,
        task_type=request.task_type,
        dataset_id=request.dataset_id,
        model_path=model_path,
        metrics=metrics,
        hyperparameters=request.hyperparameters or None,
        feature_columns=feature_columns,
        target_column=request.target_column,
        owner_id=user_id,
    )
    db.add(ml_model)
    await db.commit()
    await db.refresh(ml_model)

    return MLTrainResponse(
        model_id=ml_model.id,
        algorithm=request.algorithm,
        task_type=request.task_type,
        metrics=metrics,
        training_time_seconds=round(training_time, 4),
        feature_importance=feature_importance,
        nan_rows_dropped=nan_rows_dropped,
    )


async def predict(
    request: MLPredictRequest, user_id: int, db: AsyncSession
) -> MLPredictResponse:
    """Run predictions using a trained ML model."""
    result = await db.execute(
        select(MLModel).where(
            MLModel.id == request.model_id,
            MLModel.owner_id == user_id,
        )
    )
    ml_model = result.scalar_one_or_none()
    if ml_model is None:
        raise FileNotFoundError(
            f"Model {request.model_id} not found or not owned by user"
        )

    if not os.path.exists(ml_model.model_path):
        raise FileNotFoundError("Model file not found on disk")

    model = joblib.load(ml_model.model_path)

    # Validate feature shape
    if ml_model.feature_columns:
        expected_features = len(ml_model.feature_columns)
        for i, row in enumerate(request.features):
            if len(row) != expected_features:
                raise ValueError(
                    f"Row {i}: expected {expected_features} features, got {len(row)}"
                )

    features = np.array(request.features)

    start = time.perf_counter()
    predictions_raw = model.predict(features)
    inference_time_ms = (time.perf_counter() - start) * 1000

    predictions: list[int | float] = [
        int(p) if isinstance(p, (np.integer, int)) else float(p)
        for p in predictions_raw
    ]

    probabilities: list[list[float]] | None = None
    if ml_model.task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features)
            probabilities = proba.tolist()
        except Exception:
            pass

    return MLPredictResponse(
        predictions=predictions,
        probabilities=probabilities,
        model_id=request.model_id,
        inference_time_ms=round(inference_time_ms, 4),
    )


async def get_model(db: AsyncSession, model_id: int, user_id: int) -> MLModel | None:
    """Get an ML model owned by the user."""
    result = await db.execute(
        select(MLModel).where(
            MLModel.id == model_id,
            MLModel.owner_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def list_models(
    db: AsyncSession, user_id: int, page: int = 1, page_size: int = 20
) -> tuple[list[MLModel], int]:
    """List ML models for a user with pagination."""
    count_result = await db.execute(
        select(func.count()).select_from(
            select(MLModel).where(MLModel.owner_id == user_id).subquery()
        )
    )
    total = count_result.scalar_one()

    offset = (page - 1) * page_size
    result = await db.execute(
        select(MLModel)
        .where(MLModel.owner_id == user_id)
        .order_by(MLModel.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    return list(result.scalars().all()), total


async def delete_model(db: AsyncSession, model_id: int, user_id: int) -> bool:
    """Delete an ML model and its file."""
    ml_model = await get_model(db, model_id, user_id)
    if ml_model is None:
        return False

    if os.path.exists(ml_model.model_path):
        os.remove(ml_model.model_path)

    await db.delete(ml_model)
    await db.commit()
    return True
