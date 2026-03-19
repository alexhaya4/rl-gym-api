from prometheus_client import Counter, Gauge, Histogram

# Training metrics
training_jobs_total = Counter(
    name="rl_gym_training_jobs_total",
    documentation="Total training jobs started",
    labelnames=["algorithm", "environment", "status"],
)

training_duration_seconds = Histogram(
    name="rl_gym_training_duration_seconds",
    documentation="Training job duration in seconds",
    labelnames=["algorithm", "environment"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
)

# Environment metrics
active_environments = Gauge(
    name="rl_gym_active_environments",
    documentation="Number of active Gymnasium environments",
)

episode_reward = Histogram(
    name="rl_gym_episode_reward",
    documentation="Episode reward distribution",
    labelnames=["environment", "algorithm"],
    buckets=[-500, -200, -100, 0, 50, 100, 200, 300, 400, 500],
)

# gRPC metrics
grpc_requests_total = Counter(
    name="rl_gym_grpc_requests_total",
    documentation="Total gRPC inference requests",
    labelnames=["method", "status"],
)

grpc_latency_microseconds = Histogram(
    name="rl_gym_grpc_latency_microseconds",
    documentation="gRPC inference latency",
    labelnames=["method"],
    buckets=[100, 500, 1000, 5000, 10000, 50000],
)

# Model metrics
model_versions_total = Gauge(
    name="rl_gym_model_versions_total",
    documentation="Total model versions stored",
)

# Experiment metrics
experiments_total = Counter(
    name="rl_gym_experiments_total",
    documentation="Total experiments created",
    labelnames=["status"],
)
