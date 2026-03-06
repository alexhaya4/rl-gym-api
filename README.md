# rl-gym-api

A production-grade FastAPI platform for serving and managing OpenAI Gymnasium reinforcement learning environments via a RESTful API.

## Overview

rl-gym-api exposes RL environments as API endpoints, enabling remote training, evaluation, and real-time interaction with agents over HTTP and WebSockets. Built for researchers and engineers who need a scalable, containerized RL backend.

## Features

- REST and WebSocket APIs for environment step, reset, and observation
- JWT-based authentication and role-based access control
- Async SQLAlchemy with Alembic migrations
- Integration with Stable-Baselines3 for training and inference
- Structured logging and health monitoring
- Docker-ready with multi-stage builds

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | FastAPI, Uvicorn |
| RL | Gymnasium, Stable-Baselines3, PyTorch |
| Data | NumPy, Pandas |
| Database | SQLAlchemy (async), Alembic, aiosqlite |
| Auth | python-jose (JWT), Passlib (bcrypt) |
| Testing | pytest, pytest-asyncio, HTTPX |
| Quality | Ruff, mypy |

## Setup

```bash
# Clone and enter the project
git clone <repo-url> && cd rl-gym-api

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your SECRET_KEY
```

## Running Dev Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Running Tests

```bash
pytest --cov=app --cov-report=term-missing
```

## API Docs

Once the server is running, interactive documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d
```
