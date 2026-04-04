# RL Gym API — Backend API Contract

**Version:** 1.1.0
**Generated:** 2026-04-04
**Base URL:** `https://<host>/api/v1`

---

## Table of Contents

1. [Auth](#1-auth)
2. [Environments](#2-environments)
3. [Training](#3-training)
4. [Experiments](#4-experiments)
5. [Benchmarks](#5-benchmarks)
6. [Model Versions](#6-model-versions)
7. [Model Registry](#7-model-registry)
8. [A/B Testing](#8-ab-testing)
9. [Algorithms](#9-algorithms)
10. [Inference](#10-inference)
11. [Video](#11-video)
12. [Datasets](#12-datasets)
13. [ML Training](#13-ml-training)
14. [Distributed Training](#14-distributed-training)
15. [Comparison](#15-comparison)
16. [Artifacts](#16-artifacts)
17. [Multi-Agent](#17-multi-agent)
18. [Optimization (Optuna)](#18-optimization-optuna)
19. [Population-Based Training](#19-population-based-training)
20. [Pipelines](#20-pipelines)
21. [Organizations](#21-organizations)
22. [OAuth](#22-oauth)
23. [RBAC](#23-rbac)
24. [Billing](#24-billing)
25. [Vectorized Environments](#25-vectorized-environments)
26. [Evaluation](#26-evaluation)
27. [Audit Logs](#27-audit-logs)
28. [Custom Environments](#28-custom-environments)
29. [Ray Training (Legacy)](#29-ray-training-legacy)
30. [WebSocket](#30-websocket)
31. [Health & Metrics](#31-health--metrics)
32. [Status](#32-status)

---

## 1. Auth

**Prefix:** `/api/v1/auth`

### POST `/auth/register`
- **Auth:** No
- **Content-Type:** application/json
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | username | string | Yes | — |
  | email | string (email) | Yes | — |
  | password | string (min 8 chars) | Yes | — |
- **Response (201):** `UserResponse`
  | Field | Type |
  |-------|------|
  | id | int |
  | username | string |
  | email | string |
  | is_active | bool |
  | created_at | datetime |
- **Error Codes:** 400 (email/username exists), 422 (validation)
- **Implementation:** Full | **External deps:** None

### POST `/auth/login`
- **Auth:** No
- **Content-Type:** application/x-www-form-urlencoded
- **Request Body:** OAuth2 form: `username`, `password`
- **Response (200):** `Token`
  | Field | Type |
  |-------|------|
  | access_token | string |
  | token_type | string |
- **Error Codes:** 401 (invalid credentials)
- **Implementation:** Full | **External deps:** None

### POST `/auth/logout`
- **Auth:** Yes (JWT)
- **Response (200):** `{"message": "Successfully logged out"}`
- **Error Codes:** 401
- **Implementation:** Full | **External deps:** Redis (token blacklist, fallback to memory)

### GET `/auth/me`
- **Auth:** Yes (JWT)
- **Response (200):** `UserResponse`
- **Error Codes:** 401
- **Implementation:** Full | **External deps:** None

---

## 2. Environments

**Prefix:** `/api/v1/environments`

### GET `/environments/available`
- **Auth:** No
- **Response (200):** `list[{id: string, display_name: string}]`
- **Implementation:** Full | **External deps:** None

### POST `/environments/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | environment_id | string | Yes | — |
  | render_mode | string \| null | No | null |
- **Response (201):** `EnvironmentResponse`
  | Field | Type |
  |-------|------|
  | env_key | string |
  | environment_id | string |
  | observation_space | dict |
  | action_space | dict |
  | status | string |
- **Error Codes:** 400, 401, 422
- **Implementation:** Full | **External deps:** None

### GET `/environments/`
- **Auth:** Yes
- **Response (200):** `list[EnvironmentResponse]`

### GET `/environments/{env_key}`
- **Auth:** Yes
- **Response (200):** `EnvironmentResponse`
- **Error Codes:** 401, 404

### POST `/environments/{env_key}/reset`
- **Auth:** Yes
- **Response (200):** `ResetResponse`
  | Field | Type |
  |-------|------|
  | observation | list[float] |
  | info | dict |
- **Error Codes:** 401, 404

### POST `/environments/{env_key}/step`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | action | int \| list[float] | Yes |
- **Response (200):** `StepResponse`
  | Field | Type |
  |-------|------|
  | observation | list[float] |
  | reward | float |
  | terminated | bool |
  | truncated | bool |
  | info | dict |
- **Error Codes:** 401, 404, 422

### DELETE `/environments/{env_key}`
- **Auth:** Yes
- **Response:** 204
- **Error Codes:** 401, 404

---

## 3. Training

**Prefix:** `/api/v1/training`

### POST `/training/`
- **Auth:** Yes
- **Rate Limit:** 10/minute
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | environment_id | string | Yes | — |
  | algorithm | string | No | "PPO" |
  | total_timesteps | int | No | 10000 |
  | hyperparameters | dict | No | {} |
  | n_envs | int | No | 1 |
  | experiment_name | string \| null | No | null |
- **Response (202):** `TrainingStatus`
  | Field | Type |
  |-------|------|
  | experiment_id | int |
  | status | string |
  | environment_id | string |
  | algorithm | string |
  | total_timesteps | int |
  | job_id | string \| null |
  | elapsed_time | float \| null |
  | mean_reward | float \| null |
  | std_reward | float \| null |
- **Error Codes:** 401, 422, 429
- **Implementation:** Full | **External deps:** Redis (arq job queue)

### GET `/training/`
- **Auth:** Yes
- **Response (200):** `list[TrainingStatus]`

### GET `/training/{experiment_id}`
- **Auth:** Yes
- **Response (200):** `TrainingStatus`
- **Error Codes:** 401, 404

### GET `/training/{experiment_id}/job`
- **Auth:** Yes
- **Response (200):** `JobResponse`
- **Error Codes:** 401, 404

### GET `/training/{experiment_id}/result`
- **Auth:** Yes
- **Response (200):** `TrainingResult` (extends TrainingStatus + model_path, completed_at)
- **Error Codes:** 401, 404

---

## 4. Experiments

**Prefix:** `/api/v1/experiments`

### POST `/experiments/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | name | string | Yes | — |
  | environment_id | string | Yes | — |
  | algorithm | string | Yes | — |
  | hyperparameters | dict | No | {} |
  | total_timesteps | int | No | 10000 |
- **Response (201):** `ExperimentResponse`
  | Field | Type |
  |-------|------|
  | id | int |
  | name | string |
  | environment_id | string |
  | algorithm | string |
  | status | string |
  | total_timesteps | int |
  | user_id | int |
  | created_at | datetime |
  | updated_at | datetime \| null |
  | completed_at | datetime \| null |
  | mean_reward | float \| null |
  | std_reward | float \| null |
  | hyperparameters | dict |
  | metrics_summary | dict \| null |

### GET `/experiments/`
- **Auth:** Yes
- **Query Params:** `page` (int, default 1), `page_size` (int, default 20), `status` (string, optional)
- **Response (200):** `ExperimentListResponse`
  | Field | Type |
  |-------|------|
  | items | list[ExperimentResponse] |
  | total | int |
  | page | int |
  | page_size | int |

### GET `/experiments/{experiment_id}`
- **Auth:** Yes
- **Response (200):** `ExperimentResponse`
- **Error Codes:** 401, 404

### PATCH `/experiments/{experiment_id}`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | name | string \| null | No |
  | status | string \| null | No |
  | hyperparameters | dict \| null | No |
- **Response (200):** `ExperimentResponse`
- **Error Codes:** 401, 404

### DELETE `/experiments/{experiment_id}`
- **Auth:** Yes
- **Response:** 204
- **Error Codes:** 401, 404

### GET `/experiments/{experiment_id}/episodes`
- **Auth:** Yes
- **Response (200):** `list[Episode]`

---

## 5. Benchmarks

**Prefix:** `/api/v1/benchmarks`

### POST `/benchmarks/run`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | environments | list[string] | Yes | — |
  | algorithms | list[string] | Yes | — |
  | total_timesteps | int | No | 5000 |
  | n_eval_episodes | int | No | 5 |
- **Response (200):** `BenchmarkResponse`
  | Field | Type |
  |-------|------|
  | benchmark_id | string |
  | results | list[BenchmarkResult] |
  | total_combinations | int |
  | completed_at | string |
- **BenchmarkResult:**
  | Field | Type |
  |-------|------|
  | environment_id | string |
  | algorithm | string |
  | mean_reward | float |
  | std_reward | float |
  | training_time_seconds | float |
  | total_timesteps | int |
- **Implementation:** Full | **External deps:** None

### GET `/benchmarks/environments`
- **Auth:** No
- **Response (200):** `dict[str, list[str]]`

### GET `/benchmarks/algorithms`
- **Auth:** No
- **Response (200):** `dict[str, list[dict]]`

---

## 6. Model Versions

**Prefix:** `/api/v1/models`

### GET `/models/experiments/{experiment_id}`
- **Auth:** Yes
- **Response (200):** `ModelVersionListResponse`
  | Field | Type |
  |-------|------|
  | items | list[ModelVersionResponse] |
  | total | int |
- **ModelVersionResponse:**
  | Field | Type |
  |-------|------|
  | id | int |
  | experiment_id | int |
  | version | int |
  | storage_path | string |
  | storage_backend | string |
  | algorithm | string |
  | total_timesteps | int \| null |
  | mean_reward | float \| null |
  | file_size_bytes | int \| null |
  | metadata | dict \| null |
  | created_at | datetime |
  | download_url | string \| null |

### GET `/models/{version_id}`
- **Auth:** Yes
- **Response (200):** `ModelVersionResponse`
- **Error Codes:** 401, 404

### GET `/models/{version_id}/download`
- **Auth:** Yes
- **Response (200):** FileResponse (application/zip) or RedirectResponse (S3 presigned URL)
- **Error Codes:** 401, 404
- **External deps:** S3/Boto3 (for S3 backend)

### DELETE `/models/{version_id}`
- **Auth:** Yes
- **Response:** 204
- **Error Codes:** 401, 404

---

## 7. Model Registry

**Prefix:** `/api/v1/registry`

### POST `/registry/register`
- **Auth:** Yes
- **Request Body (query params):**
  | Field | Type | Required |
  |-------|------|----------|
  | model_version_id | int | Yes |
  | environment_id | string | Yes |
  | algorithm | string | Yes |
- **Response (201):** `RegistryEntry`
  | Field | Type |
  |-------|------|
  | id | int |
  | name | string |
  | environment_id | string |
  | algorithm | string |
  | stage | string |
  | model_version_id | int |
  | previous_production_id | int \| null |
  | mean_reward | float \| null |
  | promoted_by | int \| null |
  | promotion_comment | string \| null |
  | is_current | bool |
  | created_at | datetime |
  | updated_at | datetime |

### GET `/registry/`
- **Auth:** Yes
- **Query Params:** `stage` (string, optional)
- **Response (200):** `RegistryListResponse { items, total }`

### GET `/registry/production/{environment_id}/{algorithm}`
- **Auth:** No
- **Response (200):** `RegistryEntry`
- **Error Codes:** 404

### POST `/registry/{registry_id}/promote`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | model_version_id | int | Yes |
  | target_stage | string | Yes |
  | comment | string \| null | No |
- **Response (200):** `RegistryEntry`
- **Error Codes:** 400 (invalid transition), 401

### POST `/registry/rollback/{environment_id}/{algorithm}`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | comment | string \| null | No |
- **Response (200):** `RegistryEntry`
- **Error Codes:** 401, 404

### GET `/registry/{registry_id}/compare`
- **Auth:** Yes
- **Response (200):** `ComparisonResult`
  | Field | Type |
  |-------|------|
  | current_production | RegistryEntry \| null |
  | candidate | RegistryEntry |
  | mean_reward_delta | float \| null |
  | recommendation | string |

---

## 8. A/B Testing

**Prefix:** `/api/v1/ab-testing`

### POST `/ab-testing/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | name | string | Yes | — |
  | description | string \| null | No | null |
  | environment_id | string | Yes | — |
  | model_version_a_id | int | Yes | — |
  | model_version_b_id | int | Yes | — |
  | traffic_split_a | float | No | 0.5 |
  | n_eval_episodes_per_model | int | No | 100 |
  | significance_level | float | No | 0.05 |
  | statistical_test | string | No | "ttest" |
- **Response (201):** `ABTestResponse`
- **Implementation:** Full | **External deps:** None

### GET `/ab-testing/`
- **Auth:** Yes
- **Response (200):** `ABTestListResponse { items, total }`

### GET `/ab-testing/{test_id}`
- **Auth:** Yes
- **Response (200):** `ABTestResponse`

### POST `/ab-testing/{test_id}/run`
- **Auth:** Yes
- **Response (202):** `ABTestResponse`

### POST `/ab-testing/{test_id}/stop`
- **Auth:** Yes
- **Response (200):** `ABTestResponse`

### GET `/ab-testing/{test_id}/results`
- **Auth:** Yes
- **Query Params:** `page`, `page_size`
- **Response (200):** `list[ABTestResultResponse]`

### GET `/ab-testing/{test_id}/statistics`
- **Auth:** Yes
- **Response (200):** `ABTestStatistics`
  | Field | Type |
  |-------|------|
  | model_a_mean_reward | float \| null |
  | model_a_std_reward | float \| null |
  | model_a_n_episodes | int |
  | model_b_mean_reward | float \| null |
  | model_b_std_reward | float \| null |
  | model_b_n_episodes | int |
  | p_value | float \| null |
  | test_statistic | float \| null |
  | is_significant | bool |
  | winner | string \| null |
  | confidence_level | float |
  | effect_size | float \| null |

---

## 9. Algorithms

**Prefix:** `/api/v1/algorithms`

### GET `/algorithms/`
- **Auth:** No
- **Response (200):** `list[{name, display_name, description, action_space_types, ...}]`

### GET `/algorithms/{algorithm_name}`
- **Auth:** No
- **Response (200):** `dict` (algorithm details)
- **Error Codes:** 404

### GET `/algorithms/compatible/{environment_id}`
- **Auth:** No
- **Response (200):** `list[dict]` (compatible algorithms for the environment)

---

## 10. Inference

**Prefix:** `/api/v1/inference`

### POST `/inference/{environment_id}/predict`
- **Auth:** Yes
- **Rate Limit:** 60/minute
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | observation | list[float] \| dict | Yes | — |
  | algorithm | string \| null | No | null (uses production model) |
  | deterministic | bool | No | true |
- **Response (200):** `InferenceResponse`
  | Field | Type |
  |-------|------|
  | action | int \| list[float] |
  | action_probability | float \| null |
  | latency_ms | float |
  | model_version_id | int |
  | algorithm | string |
  | environment_id | string |
- **Error Codes:** 401, 404 (no production model), 422, 429
- **Implementation:** Full | **External deps:** None (SB3 models loaded from disk)

### GET `/inference/{environment_id}/info`
- **Auth:** No
- **Query Params:** `algorithm` (string, optional)
- **Response (200):** `dict` (production model info)
- **Error Codes:** 404

### GET `/inference/cache`
- **Auth:** Yes
- **Response (200):** `list[ModelCacheInfo]`
  | Field | Type |
  |-------|------|
  | model_path | string |
  | algorithm | string |
  | environment_id | string |
  | loaded_at | datetime |
  | memory_mb | float |

### DELETE `/inference/cache`
- **Auth:** Yes
- **Response (200):** `{"message": "Cache cleared", "models_evicted": int}`

---

## 11. Video

**Prefix:** `/api/v1/video`

### POST `/video/record`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default | Constraints |
  |-------|------|----------|---------|-------------|
  | environment_id | string | Yes | — | — |
  | algorithm | string | No | "PPO" | — |
  | num_episodes | int | No | 1 | 1-5 |
  | max_steps | int | No | 500 | 1-10000 |
  | fps | int | No | 30 | 1-60 |
- **Response (202):** `VideoStatus`
  | Field | Type |
  |-------|------|
  | video_id | string |
  | status | string |
  | progress | float |
  | error | string \| null |
- **Error Codes:** 401, 404 (no production model), 422
- **Implementation:** Full | **External deps:** Redis (metadata, fallback to memory), imageio/ffmpeg

### GET `/video/{video_id}/status`
- **Auth:** Yes
- **Response (200):** `VideoStatus`
- **Error Codes:** 401, 404

### GET `/video/{video_id}/download`
- **Auth:** Yes
- **Response (200):** FileResponse (`video/mp4`)
- **Error Codes:** 400 (not ready), 401, 404

### GET `/video/`
- **Auth:** Yes
- **Response (200):** `list[VideoStatus]`

### DELETE `/video/{video_id}`
- **Auth:** Yes
- **Response:** 204
- **Error Codes:** 401, 404

---

## 12. Datasets

**Prefix:** `/api/v1/datasets`

### POST `/datasets/` (trajectory dataset)
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | name | string | Yes | — |
  | description | string \| null | No | null |
  | environment_id | string | Yes | — |
  | algorithm | string \| null | No | null |
  | storage_format | string | No | "json" |
  | is_public | bool | No | false |
  | tags | list[string] | No | [] |
  | metadata | dict | No | {} |
- **Response (201):** `DatasetResponse`

### POST `/datasets/upload` (file upload)
- **Auth:** Yes
- **Content-Type:** multipart/form-data
- **Form Fields:**
  | Field | Type | Required |
  |-------|------|----------|
  | file | UploadFile | Yes |
  | name | string | Yes |
  | description | string \| null | No |
- **Response (201):** `FileDatasetResponse`
  | Field | Type |
  |-------|------|
  | id | int |
  | name | string |
  | description | string \| null |
  | dataset_type | string \| null |
  | num_samples | int \| null |
  | num_features | int \| null |
  | columns | list[string] \| null |
  | file_size_mb | float \| null |
  | owner_id | int |
  | created_at | datetime |
- **Error Codes:** 401, 413 (file too large), 422 (invalid extension)
- **Allowed extensions:** .csv, .json, .zip
- **Max file size:** 100 MB (configurable)
- **Implementation:** Full | **External deps:** None

### GET `/datasets/`
- **Auth:** Yes
- **Query Params:** `include_public` (bool, default true)
- **Response (200):** `DatasetListResponse { items, total }`

### GET `/datasets/{dataset_id}`
- **Auth:** No
- **Response (200):** `DatasetResponse`
- **Error Codes:** 404

### GET `/datasets/file/{dataset_id}`
- **Auth:** Yes (owner only)
- **Response (200):** `FileDatasetResponse`
- **Error Codes:** 401, 404

### GET `/datasets/file/{dataset_id}/preview`
- **Auth:** Yes (owner only)
- **Query Params:** `limit` (int, default 10, max 100)
- **Response (200):** `DatasetPreview`
  | Field | Type |
  |-------|------|
  | rows | list[dict] |
  | total_rows | int |
  | columns | list[string] |
- **Error Codes:** 401, 404

### GET `/datasets/file/{dataset_id}/statistics`
- **Auth:** Yes (owner only)
- **Response (200):** `list[DatasetStatistics]`
  | Field | Type |
  |-------|------|
  | column_name | string |
  | dtype | string |
  | mean | float \| null |
  | std | float \| null |
  | min | float \| null |
  | max | float \| null |
  | null_count | int |
  | unique_count | int |
- **Error Codes:** 401, 404

### POST `/datasets/{dataset_id}/episodes`
- **Auth:** Yes
- **Request Body:** `list[DatasetEpisodeCreate]`
- **Response (200):** `DatasetResponse`

### GET `/datasets/{dataset_id}/episodes`
- **Auth:** Yes
- **Query Params:** `page`, `page_size`
- **Response (200):** `list[DatasetEpisodeResponse]`

### GET `/datasets/{dataset_id}/stats`
- **Auth:** Yes
- **Response (200):** `DatasetStatsResponse`

### GET `/datasets/{dataset_id}/export`
- **Auth:** Yes
- **Query Params:** `format` (json/csv/hdf5)
- **Response (200):** Binary data with appropriate Content-Type

### POST `/datasets/{dataset_id}/collect`
- **Auth:** Yes
- **Request Body:** `CollectRequest { environment_id, n_episodes, algorithm?, model_version_id? }`
- **Response (202):** `DatasetResponse`

### DELETE `/datasets/{dataset_id}`
- **Auth:** Yes
- **Response:** 204
- **Error Codes:** 401, 404

---

## 13. ML Training

**Prefix:** `/api/v1/ml`

### POST `/ml/train`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default | Constraints |
  |-------|------|----------|---------|-------------|
  | dataset_id | int | Yes | — | — |
  | algorithm | string | Yes | — | See /ml/algorithms |
  | target_column | string \| null | No | null | Required for supervised |
  | hyperparameters | dict | No | {} | — |
  | test_split | float | No | 0.2 | 0.1-0.5 |
  | task_type | string | Yes | — | classification/regression/clustering/dimensionality_reduction |
- **Response (201):** `MLTrainResponse`
  | Field | Type |
  |-------|------|
  | model_id | int |
  | algorithm | string |
  | task_type | string |
  | metrics | dict |
  | training_time_seconds | float |
  | feature_importance | list[dict] \| null |
  | nan_rows_dropped | int |
- **Metrics by task_type:**
  - classification: accuracy, precision, recall, f1, confusion_matrix
  - regression: mse, rmse, mae, r2
  - clustering: silhouette_score, calinski_harabasz_score, n_clusters
  - dimensionality_reduction: explained_variance_ratio, total_explained_variance
- **Error Codes:** 401, 404 (dataset not found), 422 (bad algorithm/column)
- **Implementation:** Full | **External deps:** scikit-learn, joblib, pandas

### POST `/ml/predict`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | model_id | int | Yes |
  | features | list[list[float]] | Yes |
- **Response (200):** `MLPredictResponse`
  | Field | Type |
  |-------|------|
  | predictions | list[int \| float] |
  | probabilities | list[list[float]] \| null |
  | model_id | int |
  | inference_time_ms | float |
- **Error Codes:** 401, 404, 422 (wrong feature shape)

### GET `/ml/models`
- **Auth:** Yes
- **Query Params:** `page` (int, default 1), `page_size` (int, default 20, max 100)
- **Response (200):** `{ items: list[MLModelInfo], total, page, page_size }`

### GET `/ml/models/{model_id}`
- **Auth:** Yes
- **Response (200):** `MLModelInfo`
  | Field | Type |
  |-------|------|
  | id | int |
  | name | string |
  | algorithm | string |
  | task_type | string |
  | dataset_id | int \| null |
  | metrics | dict \| null |
  | created_at | datetime |
- **Error Codes:** 401, 404

### DELETE `/ml/models/{model_id}`
- **Auth:** Yes
- **Response:** 204
- **Error Codes:** 401, 404

### GET `/ml/algorithms`
- **Auth:** No
- **Response (200):**
  ```json
  {
    "classification": ["RandomForestClassifier", "GradientBoostingClassifier", "SVC", "KNeighborsClassifier", "LogisticRegression", "DecisionTreeClassifier"],
    "regression": ["RandomForestRegressor", "GradientBoostingRegressor", "SVR", "KNeighborsRegressor", "LinearRegression", "DecisionTreeRegressor"],
    "clustering": ["KMeans", "DBSCAN", "AgglomerativeClustering"],
    "dimensionality_reduction": ["PCA", "TruncatedSVD"]
  }
  ```

---

## 14. Distributed Training

**Prefix:** `/api/v1/distributed`

### POST `/distributed/train`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default | Constraints |
  |-------|------|----------|---------|-------------|
  | environment_id | string | Yes | — | — |
  | algorithm | string | No | "PPO" | — |
  | total_timesteps | int | No | 50000 | — |
  | num_workers | int | No | 2 | 1-4 |
  | num_envs_per_worker | int | No | 4 | 1-8 |
  | hyperparameters | dict | No | {} | — |
  | experiment_name | string \| null | No | null | — |
- **Response (202):** `DistributedTrainResponse`
  | Field | Type |
  |-------|------|
  | job_id | string |
  | status | string |
  | num_workers | int |
  | total_envs | int |
  | estimated_speedup | float |
- **Error Codes:** 401, 422, 503 (distributed disabled)
- **Implementation:** Full | **External deps:** Ray, Redis (fallback to memory)

### GET `/distributed/{job_id}/status`
- **Auth:** Yes
- **Response (200):** `DistributedStatus`
  | Field | Type |
  |-------|------|
  | job_id | string |
  | status | string |
  | progress | float |
  | metrics | dict \| null |
  | elapsed_seconds | float |
  | num_workers_active | int |
  | error | string \| null |
- **Status values:** queued, initializing, training, completed, failed, cancelled
- **Error Codes:** 401, 404

### POST `/distributed/{job_id}/cancel`
- **Auth:** Yes
- **Response (200):** `{"message": "Job cancellation requested", "job_id": string}`
- **Error Codes:** 401, 404

### GET `/distributed/jobs`
- **Auth:** Yes
- **Response (200):** `list[DistributedStatus]`

### GET `/distributed/cluster`
- **Auth:** Yes
- **Response (200):**
  | Field | Type |
  |-------|------|
  | initialized | bool |
  | num_cpus | int |
  | num_gpus | int |
  | nodes | int |

---

## 15. Comparison

**Prefix:** `/api/v1/comparison`

### POST `/comparison/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | experiment_ids | list[int] | Yes (2-10) |
- **Response (200):** `ComparisonResponse`
  | Field | Type |
  |-------|------|
  | experiments | list[dict] |
  | diffs | list[ExperimentDiff] |
  | best_experiment_id | int \| null |
  | comparison_metric | string |

### GET `/comparison/diff/{exp_id_a}/{exp_id_b}`
- **Auth:** Yes
- **Response (200):** `ExperimentDiff`
  | Field | Type |
  |-------|------|
  | experiment_id_a | int |
  | experiment_id_b | int |
  | name_a | string |
  | name_b | string |
  | hyperparameter_diff | dict |
  | metrics_diff | dict |
  | status_a | string |
  | status_b | string |
  | winner | string \| null |
  | improvement_pct | float \| null |

### GET `/comparison/lineage/{experiment_id}`
- **Auth:** Yes
- **Response (200):** `LineageGraph`

### PATCH `/comparison/experiments/{experiment_id}/tags`
- **Auth:** Yes
- **Request Body:** `list[string]`
- **Response (200):** `dict`

### GET `/comparison/experiments/{experiment_id}/export`
- **Auth:** Yes
- **Query Params:** `format` (json/csv)
- **Response (200):** JSON or CSV data

---

## 16. Artifacts

**Prefix:** `/api/v1/artifacts`

### POST `/artifacts/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | name | string | Yes | — |
  | artifact_type | string | Yes | — |
  | experiment_id | int \| null | No | null |
  | metadata | dict | No | {} |
- **Response (201):** `ArtifactResponse`
- **Implementation:** Full | **External deps:** S3/Boto3 (for S3 backend)

### GET `/artifacts/`
- **Auth:** Yes
- **Query Params:** `experiment_id` (int, optional)
- **Response (200):** `list[ArtifactResponse]`

### GET `/artifacts/{artifact_id}`
- **Auth:** Yes
- **Response (200):** `ArtifactResponse`

### DELETE `/artifacts/{artifact_id}`
- **Auth:** Yes
- **Response:** 204

### POST `/artifacts/{artifact_id}/lineage`
- **Auth:** Yes
- **Response (200):** `dict`

---

## 17. Multi-Agent

**Prefix:** `/api/v1/multi-agent`

### GET `/multi-agent/environments`
- **Auth:** No
- **Response (200):** `list[dict]` (available multi-agent environments)
- **Implementation:** Full | **External deps:** PettingZoo

### POST `/multi-agent/train`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | environment_id | string | Yes | — |
  | algorithm | string | No | "PPO" |
  | total_timesteps | int | No | 50000 |
  | n_eval_episodes | int | No | 10 |
  | experiment_name | string \| null | No | null |
  | hyperparameters | dict | No | {} |
  | shared_policy | bool | No | false |
- **Response (202):** `MultiAgentExperimentResponse`

### GET `/multi-agent/experiments`
- **Auth:** Yes
- **Response (200):** `MultiAgentExperimentListResponse { items, total }`

### GET `/multi-agent/experiments/{experiment_id}`
- **Auth:** Yes
- **Response (200):** `MultiAgentExperimentResponse`

### GET `/multi-agent/experiments/{experiment_id}/agents`
- **Auth:** Yes
- **Response (200):** `list[AgentPolicyResponse]`

---

## 18. Optimization (Optuna)

**Prefix:** `/api/v1/optimization`

### POST `/optimization/run`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | environment_id | string | Yes | — |
  | algorithm | string | No | "PPO" |
  | total_timesteps | int | No | 10000 |
  | n_trials | int | No | 20 |
  | n_eval_episodes | int | No | 5 |
  | experiment_name | string \| null | No | null |
  | timeout_seconds | int \| null | No | null |
  | pruning_enabled | bool | No | true |
  | hyperparameter_space | dict \| null | No | null |
- **Response (202):** `OptimizationResponse`
- **Implementation:** Full | **External deps:** Optuna

### GET `/optimization/`
- **Auth:** Yes
- **Response (200):** `list[OptimizationResponse]`

### GET `/optimization/{study_id}`
- **Auth:** Yes
- **Response (200):** `OptimizationResponse`

### GET `/optimization/{study_id}/history`
- **Auth:** Yes
- **Response (200):** `list[dict]`

### GET `/optimization/algorithms/spaces`
- **Auth:** No
- **Response (200):** `dict` (hyperparameter search spaces per algorithm)

---

## 19. Population-Based Training

**Prefix:** `/api/v1/pbt`

### POST `/pbt/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default | Constraints |
  |-------|------|----------|---------|-------------|
  | environment_id | string | Yes | — | — |
  | algorithm | string | No | "PPO" | — |
  | population_size | int | No | 8 | 2-20 |
  | total_timesteps_per_member | int | Yes | — | — |
  | exploit_interval | int | Yes | — | — |
  | mutation_rate | float | No | 0.2 | 0.0-1.0 |
  | experiment_name | string \| null | No | null | — |
  | initial_hyperparameter_ranges | dict \| null | No | null | — |
- **Response (202):** `PBTExperimentResponse`
- **Implementation:** Full | **External deps:** None

### GET `/pbt/`
- **Auth:** Yes
- **Response (200):** `PBTListResponse { items, total }`

### GET `/pbt/{pbt_id}`
- **Auth:** Yes
- **Response (200):** `PBTExperimentResponse`

### GET `/pbt/{pbt_id}/members`
- **Auth:** Yes
- **Response (200):** `list[PBTMemberResponse]`

### GET `/pbt/{pbt_id}/best`
- **Auth:** Yes
- **Response (200):** `PBTMemberResponse`

---

## 20. Pipelines

**Prefix:** `/api/v1/pipelines`

### GET `/pipelines/health`
- **Auth:** No
- **Response (200):** `dict` (Prefect connection status)
- **Implementation:** Full | **External deps:** Prefect

### POST `/pipelines/run`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | environment_id | string | Yes | — |
  | algorithm | string | No | "PPO" |
  | total_timesteps | int | No | 10000 |
  | hyperparameters | dict | No | {} |
  | experiment_name | string \| null | No | null |
  | min_reward_threshold | float \| null | No | null |
  | retrain_if_exists | bool | No | true |
  | schedule_cron | string \| null | No | null |
- **Response (202):** `PipelineRunResponse`

### POST `/pipelines/search`
- **Auth:** Yes
- **Request Body:** Same as `/run`
- **Response (202):** `PipelineRunResponse`

### GET `/pipelines/`
- **Auth:** Yes
- **Response (200):** `list[PipelineRunResponse]`

### GET `/pipelines/{pipeline_id}`
- **Auth:** Yes
- **Response (200):** `PipelineRunResponse`

---

## 21. Organizations

**Prefix:** `/api/v1/organizations`

### POST `/organizations/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | name | string | Yes |
  | slug | string \| null | No |
- **Response (201):** `OrganizationResponse`

### GET `/organizations/`
- **Auth:** Yes
- **Response (200):** `OrganizationListResponse { items, total }`

### GET `/organizations/{org_id}`
- **Auth:** Yes
- **Response (200):** `OrganizationResponse`

### POST `/organizations/{org_id}/members`
- **Auth:** Yes
- **Request Body:** `AddMemberRequest { user_id, role }`
- **Response (201):** `OrganizationMemberResponse`

### DELETE `/organizations/{org_id}/members/{user_id}`
- **Auth:** Yes
- **Response:** 204

### GET `/organizations/{org_id}/usage`
- **Auth:** Yes
- **Response (200):** `UsageResponse`

---

## 22. OAuth

**Prefix:** `/api/v1/oauth`

### GET `/oauth/google/login`
- **Auth:** No
- **Response (200):** `OAuthLoginResponse { authorization_url, state }`
- **Implementation:** Full | **External deps:** Google OAuth2

### GET `/oauth/google/callback`
- **Auth:** No
- **Query Params:** `code`, `state`
- **Response (200):** `OAuthTokenResponse { access_token, token_type, user, is_new_user }`

### GET `/oauth/github/login`
- **Auth:** No
- **Response (200):** `OAuthLoginResponse { authorization_url, state }`
- **Implementation:** Full | **External deps:** GitHub OAuth2

### GET `/oauth/github/callback`
- **Auth:** No
- **Query Params:** `code`, `state`
- **Response (200):** `OAuthTokenResponse`

### GET `/oauth/accounts`
- **Auth:** Yes
- **Response (200):** `list[OAuthAccountResponse]`

### DELETE `/oauth/accounts/{provider}`
- **Auth:** Yes
- **Response:** 204

---

## 23. RBAC

**Prefix:** `/api/v1/rbac`

### GET `/rbac/my-permissions`
- **Auth:** Yes
- **Query Params:** `organization_id` (int, optional)
- **Response (200):** `UserPermissionsResponse`
  | Field | Type |
  |-------|------|
  | user_id | int |
  | role | string |
  | permissions | list[string] |
  | organization_id | int \| null |

### POST `/rbac/check`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | permission | string | Yes |
  | resource_type | string \| null | No |
  | resource_id | int \| null | No |
- **Response (200):** `PermissionCheckResponse { allowed, role, permission, reason }`

### POST `/rbac/assign`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | user_id | int | Yes |
  | role | string | Yes |
  | organization_id | int \| null | No |
- **Response (200):** `dict`

### GET `/rbac/roles`
- **Auth:** No
- **Response (200):** `dict[str, list[str]]` (roles → permissions)

---

## 24. Billing

**Prefix:** `/api/v1/billing`

### GET `/billing/plans`
- **Auth:** No
- **Response (200):** `list[PlanInfo]`
  | Field | Type |
  |-------|------|
  | name | string |
  | price_monthly_usd | float |
  | max_experiments | int |
  | max_environments | int |
  | max_timesteps | int |
  | features | list[string] |
- **Implementation:** Full | **External deps:** Stripe

### GET `/billing/subscription/{org_id}`
- **Auth:** Yes
- **Response (200):** `SubscriptionResponse`

### POST `/billing/checkout`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | plan | string | Yes |
  | success_url | string | Yes |
  | cancel_url | string | Yes |
- **Response (200):** `CheckoutSessionResponse { checkout_url, session_id }`

### POST `/billing/webhook`
- **Auth:** No (Stripe signature verification)
- **Response (200):** `dict`

### POST `/billing/cancel/{org_id}`
- **Auth:** Yes
- **Response (200):** `SubscriptionResponse`

---

## 25. Vectorized Environments

**Prefix:** `/api/v1/vec-environments`

### POST `/vec-environments/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default | Constraints |
  |-------|------|----------|---------|-------------|
  | environment_id | string | Yes | — | — |
  | n_envs | int | No | 4 | 1-32 |
  | use_subprocess | bool | No | false | — |
  | normalize_observations | bool | No | false | — |
  | normalize_rewards | bool | No | false | — |
  | frame_stack | int \| null | No | null | — |
  | seed | int \| null | No | null | — |
- **Response (201):** `VecEnvironmentResponse`

### GET `/vec-environments/`
- **Auth:** Yes
- **Response (200):** `list[dict]`

### GET `/vec-environments/{vec_key}`
- **Auth:** Yes
- **Response (200):** `VecEnvironmentResponse`

### POST `/vec-environments/{vec_key}/reset`
- **Auth:** Yes
- **Response (200):** `VecResetResponse { observations, infos, n_envs }`

### POST `/vec-environments/{vec_key}/step`
- **Auth:** Yes
- **Request Body:** `VecStepRequest { actions: list[Any] }`
- **Response (200):** `VecStepResponse`
  | Field | Type |
  |-------|------|
  | observations | list[list[float]] |
  | rewards | list[float] |
  | terminated | list[bool] |
  | truncated | list[bool] |
  | infos | list[dict] |
  | n_envs | int |

### DELETE `/vec-environments/{vec_key}`
- **Auth:** Yes
- **Response:** 204

---

## 26. Evaluation

**Prefix:** `/api/v1/evaluation`

### POST `/evaluation/run`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required | Default |
  |-------|------|----------|---------|
  | experiment_id | int | Yes | — |
  | n_eval_episodes | int | No | 10 |
  | deterministic | bool | No | true |
  | environment_id | string \| null | No | null |
- **Response (200):** `EvaluationResponse`
  | Field | Type |
  |-------|------|
  | experiment_id | int |
  | environment_id | string |
  | algorithm | string |
  | n_eval_episodes | int |
  | mean_reward | float |
  | std_reward | float |
  | min_reward | float |
  | max_reward | float |
  | episodes | list[EpisodeMetrics] |
  | evaluated_at | string |

### GET `/evaluation/experiments/{experiment_id}/episodes`
- **Auth:** Yes
- **Response (200):** `list[Episode]`

---

## 27. Audit Logs

**Prefix:** `/api/v1/audit`

### GET `/audit/logs`
- **Auth:** Yes
- **Query Params:** `user_id`, `event_type`, `action`, `status`, `ip_address`, `from_date`, `to_date`, `page` (default 1), `page_size` (default 50, max 200)
- **Response (200):** `AuditLogListResponse { items, total, page, page_size }`
- **Note:** Users can only see their own logs

### GET `/audit/logs/me`
- **Auth:** Yes
- **Response (200):** `list[AuditLogResponse]` (last 100 events)

### GET `/audit/logs/{log_id}`
- **Auth:** Yes
- **Response (200):** `AuditLogResponse`
- **Error Codes:** 401, 404

---

## 28. Custom Environments

**Prefix:** `/api/v1/custom-environments`

### POST `/custom-environments/`
- **Auth:** Yes
- **Request Body:**
  | Field | Type | Required |
  |-------|------|----------|
  | name | string | Yes (pattern: `^[A-Za-z0-9_-]+-v\d+$`) |
  | description | string \| null | No |
  | source_code | string | Yes |
  | observation_space_spec | dict \| null | No |
  | action_space_spec | dict \| null | No |
- **Response (201):** `CustomEnvironmentResponse`
- **Limitation:** Sandboxed execution (configurable timeout, memory, CPU limits)

### GET `/custom-environments/`
- **Auth:** Yes
- **Response (200):** `CustomEnvironmentListResponse { items, total }`

### GET `/custom-environments/{env_id}`
- **Auth:** Yes
- **Response (200):** `CustomEnvironmentResponse`

### POST `/custom-environments/{env_id}/validate`
- **Auth:** Yes
- **Response (200):** `CustomEnvironmentResponse`

### DELETE `/custom-environments/{env_id}`
- **Auth:** Yes
- **Response:** 204

---

## 29. Ray Training (Legacy)

**Prefix:** `/api/v1/distributed` (shared with Distributed Training)

### GET `/distributed/status`
- **Auth:** No
- **Response (200):** `dict { ray_available, ray_address, dashboard_url, active_trials }`
- **Note:** Legacy endpoint from ray_training module

### GET `/distributed/trials/{job_id}`
- **Auth:** Yes
- **Response (200):** `DistributedTrainingResponse`
- **Error Codes:** 401, 404

---

## 30. WebSocket

### WS `/api/v1/ws/training/{experiment_id}`
- **Auth:** No (public WebSocket)
- **Protocol:** WebSocket
- **Messages (server → client):**
  ```json
  {
    "experiment_id": 1,
    "timestep": 1000,
    "mean_reward": 195.5,
    "std_reward": 12.3,
    "fps": 250
  }
  ```
- **Note:** Uses `starlette.testclient.TestClient` for testing (not httpx)

---

## 31. Health & Metrics

### GET `/health`
- **Auth:** No (requires DB session dependency)
- **Response (200):**
  | Field | Type |
  |-------|------|
  | status | string |
  | database | string |
  | version | string |
  | environment | string |
  | uptime_seconds | float |

### GET `/metrics`
- **Auth:** Token or IP-based (`X-Metrics-Token` header or allowed IP)
- **Response (200):** Prometheus text format
- **Content-Type:** text/plain

---

## 32. Status

### GET `/api/v1/status`
- **Auth:** No
- **Response (200):**
  | Field | Type |
  |-------|------|
  | api_version | string |
  | environment | string |
  | active_environments | int |
  | uptime_seconds | float |

### GET `/api/v1/status/grpc`
- **Auth:** No
- **Response (200):**
  | Field | Type |
  |-------|------|
  | grpc_host | string |
  | grpc_port | int |
  | status | string |

---

## External Service Dependencies Summary

| Service | Used By | Required? |
|---------|---------|-----------|
| **Redis** | Auth (token blacklist), Training (arq), Video, Distributed, Video metadata | No (in-memory fallback) |
| **Ray** | Distributed Training | No (503 if disabled) |
| **Stripe** | Billing | No (only if billing enabled) |
| **Google OAuth2** | OAuth (Google login) | No (optional) |
| **GitHub OAuth2** | OAuth (GitHub login) | No (optional) |
| **AWS S3 / Boto3** | Model Storage, Artifacts | No (local storage fallback) |
| **Optuna** | Optimization | Yes (for optimization endpoints) |
| **Prefect** | Pipelines | Yes (for pipeline endpoints) |
| **PostgreSQL** | Database | No (SQLite for dev) |
| **ffmpeg** | Video encoding | Yes (for video endpoints) |

---

## Global Error Response Format

All error responses follow this structure:

```json
{
  "detail": "Error message string"
}
```

Validation errors (422):
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "Error description",
      "type": "error_type"
    }
  ]
}
```

---

## Authentication

All authenticated endpoints require a JWT Bearer token in the `Authorization` header:

```
Authorization: Bearer <access_token>
```

Tokens expire after 30 minutes (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`).

---

## Rate Limits

| Scope | Limit |
|-------|-------|
| Default (all endpoints) | 100/minute |
| Training (`POST /training/`) | 10/minute |
| Inference (`POST /inference/{env}/predict`) | 60/minute |

Rate limit exceeded returns HTTP 429:
```json
{
  "detail": "Rate limit exceeded: ..."
}
```

---

## Endpoint Count Summary

| Category | Count |
|----------|-------|
| Auth | 4 |
| Environments | 7 |
| Training | 5 |
| Experiments | 6 |
| Benchmarks | 3 |
| Model Versions | 4 |
| Model Registry | 6 |
| A/B Testing | 7 |
| Algorithms | 3 |
| Inference | 4 |
| Video | 5 |
| Datasets | 13 |
| ML Training | 6 |
| Distributed Training | 5 |
| Comparison | 5 |
| Artifacts | 5 |
| Multi-Agent | 5 |
| Optimization | 5 |
| PBT | 5 |
| Pipelines | 5 |
| Organizations | 6 |
| OAuth | 6 |
| RBAC | 4 |
| Billing | 5 |
| Vec Environments | 6 |
| Evaluation | 2 |
| Audit Logs | 3 |
| Custom Environments | 5 |
| Ray Training (Legacy) | 2 |
| WebSocket | 1 |
| Health & Metrics | 2 |
| Status | 2 |
| **Total** | **155** |
