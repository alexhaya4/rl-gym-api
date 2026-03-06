# RL Gym API

A production-ready REST API for reinforcement learning experiments with OpenAI Gymnasium and Stable Baselines3.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://github.com/alexhaya4/rl-gym-api/actions/workflows/ci.yml/badge.svg)

## Overview

RL Gym API exposes Gymnasium environments as API endpoints, enabling remote training, evaluation, and real-time interaction with RL agents over HTTP and WebSockets. It supports multiple algorithms (PPO, A2C, DQN), cross-environment benchmarking, and full experiment lifecycle management. Built for researchers and engineers who need a scalable, containerized RL backend with JWT authentication and structured logging.

## Features

- **RL Environments** - Create, reset, step, and manage Gymnasium environments via REST
- **Training** - Train agents with PPO, A2C, and DQN using Stable Baselines3
- **Benchmarking** - Compare algorithms across multiple environments in a single request
- **Experiments** - Full CRUD for experiment tracking with pagination and filtering
- **WebSockets** - Real-time training metrics streaming
- **JWT Authentication** - Secure endpoints with token-based auth
- **Docker** - Multi-stage builds with PostgreSQL support
- **CI/CD** - GitHub Actions pipeline with linting, testing, and Docker build verification

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.115+ | Web framework and OpenAPI docs |
| Uvicorn | 0.34+ | ASGI server |
| Gymnasium | 1.0+ | RL environment interface |
| Stable Baselines3 | 2.4+ | RL algorithm implementations |
| PyTorch | 2.5+ | Neural network backend |
| SQLAlchemy | 2.0+ | Async ORM and database access |
| Alembic | 1.14+ | Database migrations |
| Pydantic | 2.10+ | Request/response validation |
| python-jose | 3.3+ | JWT token encoding/decoding |
| slowapi | 0.1.9+ | Rate limiting |
| Ruff | 0.8+ | Linting and formatting |
| pytest | 8.3+ | Testing framework |

## Project Structure

```
app/
├── api/v1/              # Route handlers
│   ├── auth.py          #   Registration and login
│   ├── environments.py  #   Gymnasium environment management
│   ├── training.py      #   RL agent training
│   ├── experiments.py   #   Experiment CRUD
│   ├── benchmarks.py    #   Algorithm benchmarking
│   └── websockets.py    #   Real-time metrics streaming
├── core/                # Cross-cutting concerns
│   ├── health.py        #   Health check with DB connectivity
│   ├── logging.py       #   JSON logging and RequestID middleware
│   ├── rate_limit.py    #   Rate limiting configuration
│   └── security.py      #   JWT token utilities
├── db/                  # Database layer
│   ├── session.py       #   Async engine and session factory
│   └── init_db.py       #   Table initialization
├── models/              # SQLAlchemy ORM models
│   ├── user.py          #   User model
│   ├── experiment.py    #   Experiment model
│   └── episode.py       #   Episode model
├── schemas/             # Pydantic request/response schemas
├── services/            # Business logic layer
├── config.py            # Settings from environment variables
├── dependencies.py      # FastAPI dependency injection (auth, DB)
└── main.py              # App factory, middleware, router setup
```

## Quick Start

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
# Edit .env with your SECRET_KEY

# Run the dev server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs` (Swagger) and `/redoc` (ReDoc).

## API Endpoints

### Authentication

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/auth/register` | No | Register a new user |
| POST | `/api/v1/auth/login` | No | Login and receive JWT token |
| GET | `/api/v1/auth/me` | Yes | Get current user profile |

### Environments

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/environments/available` | No | List available Gymnasium environments |
| POST | `/api/v1/environments/` | Yes | Create an environment instance |
| POST | `/api/v1/environments/{id}/reset` | Yes | Reset environment to initial state |
| POST | `/api/v1/environments/{id}/step` | Yes | Take a step with an action |
| DELETE | `/api/v1/environments/{id}` | Yes | Close and remove environment |

### Training

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/training/` | Yes | Start a training session |
| GET | `/api/v1/training/{id}` | Yes | Get training status |
| GET | `/api/v1/training/` | Yes | List all training sessions |

### Experiments

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/experiments` | Yes | Create a new experiment |
| GET | `/api/v1/experiments` | Yes | List experiments (paginated, filterable) |
| GET | `/api/v1/experiments/{id}` | Yes | Get experiment details |
| PATCH | `/api/v1/experiments/{id}` | Yes | Update experiment fields |
| DELETE | `/api/v1/experiments/{id}` | Yes | Delete an experiment |
| GET | `/api/v1/experiments/{id}/episodes` | Yes | List episodes for an experiment |

### Benchmarks

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/benchmarks/run` | Yes | Run benchmark across envs and algorithms |
| GET | `/api/v1/benchmarks/environments` | No | List available benchmark environments |
| GET | `/api/v1/benchmarks/algorithms` | No | List supported algorithms |

### Other

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Health check with DB status |
| GET | `/api/v1/status` | No | API status, uptime, active environments |
| WS | `/api/v1/ws/training/{id}` | No | Real-time training metrics stream |

## Docker

```bash
# Start API and PostgreSQL
docker compose up --build

# Run in background
docker compose up -d
```

The compose file starts a PostgreSQL 16 database alongside the API. Set the database URL in your `.env`:

```
DATABASE_URL=postgresql+asyncpg://rl_gym:rl_gym@postgres:5432/rl_gym
```

## Running Tests

```bash
# Full test suite with coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Run a specific test file
pytest tests/test_experiments.py -v

# Lint
ruff check app/ tests/

# Type check
mypy app/ --ignore-missing-imports
```

Tests use an in-memory SQLite database and require no external services.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change-me-to-a-random-secret-key` | JWT signing key |
| `DATABASE_URL` | `sqlite+aiosqlite:///./rl_gym.db` | Database connection string |
| `ENVIRONMENT` | `development` | Environment name (development/test/production) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS allowed origins (comma-separated) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT token expiry |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Run `ruff check app/ tests/ --fix` and `pytest tests/ -v`
5. Commit and push to your branch
6. Open a pull request against `master`

## License

This project is licensed under the MIT License. See [LICENSE](https://opensource.org/licenses/MIT) for details.
