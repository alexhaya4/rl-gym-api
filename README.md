# RL Gym API

[![CI](https://github.com/alexhaya4/rl-gym-api/actions/workflows/ci.yml/badge.svg)](https://github.com/alexhaya4/rl-gym-api/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

A production-grade RESTful API platform for reinforcement learning experiments built with FastAPI, OpenAI Gymnasium, and Stable Baselines3.

| Resource | Link |
|---|---|
| API | https://rl-gym-api-production.up.railway.app |
| Swagger UI | https://rl-gym-api-production.up.railway.app/docs |
| Dashboard | https://rl-gym-dashboard-production.up.railway.app |
| GitHub | https://github.com/alexhaya4/rl-gym-api |

## Key Features

- **JWT Authentication** with token blacklist, logout, and Redis-backed session invalidation
- **Environment Management** for Gymnasium environments (CartPole, LunarLander, MountainCar, Acrobot, Pendulum)
- **10 RL Algorithms**: PPO, A2C, DQN, SAC, TD3, DDPG, TQC, TRPO, ARS, RecurrentPPO via Stable Baselines3
- **Real-time WebSocket Streaming** of training metrics
- **Experiment Tracking** with full CRUD, pagination, and status filtering
- **Benchmarking** across multiple algorithms and environments in a single request
- **Model Registry** with staging, production, and archived lifecycle management
- **A/B Testing** with statistical significance testing (t-test, Mann-Whitney U)
- **Custom Environment Registration** with Docker sandbox validation
- **Multi-Agent RL** with PettingZoo integration
- **Vectorized Environments** supporting up to 32 parallel instances
- **Population-Based Training (PBT)** for adaptive hyperparameter tuning
- **Bayesian Hyperparameter Optimization** with Optuna
- **Distributed Training** with Ray
- **Data Versioning and Dataset Management**
- **Experiment Comparison, Diff, and Artifact Lineage**
- **gRPC Inference Endpoint** with Protocol Buffers
- **Prometheus Metrics and Grafana Dashboard** for observability
- **Multi-Tenancy with Stripe Billing** (free, pro, enterprise tiers)
- **OAuth2 Login** (Google, GitHub) with account linking and RBAC with fine-grained permissions
- **Prefect Pipeline Orchestration** for complex training workflows
- **Immutable Audit Logging** for all user and system actions

## Security

Eight security phases completed:

- Token blacklist with Redis fallback for session invalidation
- Request size limits (10 MB) and security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options)
- gRPC API key authentication
- Dependabot automated dependency updates
- Bandit SAST scanning
- Trivy container vulnerability scanning with SARIF output to GitHub Security tab
- Gitleaks secret scanning
- Docker sandbox isolation for custom environment code execution
- RBAC with role-based permissions (owner, admin, member, viewer)

## Tech Stack

| Category | Technologies |
|---|---|
| Language & Framework | Python 3.12, FastAPI, Uvicorn |
| RL Libraries | Stable Baselines3, Gymnasium, PettingZoo |
| Database | PostgreSQL, SQLAlchemy Async, Alembic |
| Cache & Queue | Redis, ARQ job queue |
| APIs | REST, gRPC with Protocol Buffers, WebSockets |
| Optimization | Optuna, Ray, Prefect |
| Monitoring | Prometheus, Grafana |
| Auth | JWT (python-jose), OAuth2 (Google, GitHub), RBAC |
| Billing | Stripe |
| CI/CD | GitHub Actions, Docker, Railway |
| Code Quality | Ruff, mypy, pytest, Bandit, Trivy, Gitleaks |

## Getting Started

```bash
# Clone the repo
git clone https://github.com/alexhaya4/rl-gym-api.git
cd rl-gym-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install hatchling
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and set a strong SECRET_KEY (>= 32 characters)

# Run database migrations
alembic upgrade head

# Start the dev server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## API Reference

Full interactive documentation is available at the Swagger UI:

**https://rl-gym-api-production.up.railway.app/docs**

All endpoints are grouped by tag: auth, environments, training, experiments, benchmarks, algorithms, registry, ab-testing, custom-environments, organizations, rbac, audit, oauth, and more.

## Testing

- **228 tests** with **62% coverage**
- In-memory SQLite database (no external services required)

```bash
# Full test suite with coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Run a specific test file
pytest tests/test_experiments.py -v

# Lint and format
ruff check app/ tests/ --fix
ruff format app/ tests/

# Type check
mypy app/ --ignore-missing-imports
```

## Project Structure

```
app/
├── api/v1/           # Route handlers (auth, environments, training, experiments,
│                     #   benchmarks, registry, ab_testing, algorithms, audit,
│                     #   oauth, rbac, custom_environments, organizations, ...)
├── core/             # Cross-cutting: logging, rate limiting, security, health,
│                     #   permissions, algorithms, Prometheus metrics
├── db/               # SQLAlchemy async engine, session factory, Base
├── models/           # ORM models (User, Experiment, Episode, Job, ABTest, ...)
├── schemas/          # Pydantic v2 request/response validation
├── services/         # Business logic layer
├── grpc_server/      # gRPC inference endpoint with Protocol Buffers
├── config.py         # pydantic-settings, reads from .env
├── dependencies.py   # FastAPI deps: auth, DB session, ARQ Redis
└── main.py           # App factory, middleware stack, router registration
tests/                # 228 tests (pytest, async, in-memory SQLite)
alembic/              # Database migration scripts
```

## Deployment

The API is deployed on **Railway** using Docker. The CI/CD pipeline (GitHub Actions) runs linting, type checking, tests, security scans (Bandit, Trivy, Gitleaks), and Docker build verification on every push to `master`.

```bash
# Docker
docker compose up --build

# Run in background
docker compose up -d
```

## Related Projects

- [rl-gym-dashboard](https://github.com/alexhaya4/rl-gym-dashboard) - Frontend dashboard for RL Gym API

## Author

**Alex Odhiambo Haya** - [alexhaya19@gmail.com](mailto:alexhaya19@gmail.com)

## License

This project is licensed under the MIT License. See [LICENSE](https://opensource.org/licenses/MIT) for details.
