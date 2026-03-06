# RL Gym API — Claude Code Instructions

FastAPI platform for reinforcement learning experiments using Gymnasium and Stable Baselines3.

## Key Commands

```bash
uvicorn app.main:app --reload          # Dev server on :8000
pytest tests/ -v --cov=app             # Run tests with coverage
ruff check app/ tests/ --fix           # Lint and auto-fix
ruff format app/ tests/                # Format code
mypy app/ --ignore-missing-imports     # Type checking
alembic revision --autogenerate -m ""  # Generate migration
alembic upgrade head                   # Apply migrations
```

## Architecture

```
app/
├── api/v1/          # Route handlers (auth, environments, training, experiments, benchmarks, websockets)
├── core/            # Cross-cutting: logging (JSON + RequestID middleware), rate_limit (slowapi), health, security (JWT)
├── db/              # SQLAlchemy async engine, session factory, Base declarative base
├── models/          # ORM models: User, Experiment, Episode
├── schemas/         # Pydantic v2 models for request/response validation
├── services/        # Business logic: environment mgmt, training (SB3), benchmarking, experiment CRUD
├── config.py        # pydantic-settings, reads from .env
├── dependencies.py  # FastAPI deps: get_current_user, get_current_active_user (JWT + OAuth2)
└── main.py          # App factory with lifespan, middleware stack, router registration
```

## Important Patterns

- **Async SQLAlchemy**: Uses `AsyncSession` with `async_sessionmaker`. All DB operations use `select()` with `await db.execute()`.
- **Dependency injection**: DB sessions via `Depends(get_db)`, auth via `Depends(get_current_active_user)`. Chain: OAuth2 token → decode JWT → lookup user.
- **Training**: CPU-bound SB3 training runs via `loop.run_in_executor()` to avoid blocking the event loop.
- **Middleware order**: RequestIDMiddleware → CORSMiddleware (added in reverse due to Starlette's LIFO stack).
- **Rate limiting**: slowapi `Limiter` on `app.state.limiter`, 100/min default, 10/min for training.

## Test Strategy

- In-memory SQLite (`sqlite+aiosqlite://`) configured in `tests/conftest.py`.
- `client` fixture: creates tables, overrides `get_db`, yields `httpx.AsyncClient`, tears down.
- Auth fixtures: register + login helper returning `{"Authorization": "Bearer <token>"}`.
- WebSocket tests use `starlette.testclient.TestClient` (not httpx).
- `asyncio_mode = "auto"` in pyproject.toml — no `@pytest.mark.asyncio` needed.

## Known Gotchas

- **ruff B008 suppressed**: `Depends()` calls in function defaults trigger this rule; it's ignored in pyproject.toml.
- **passlib[bcrypt]**: Listed as dependency but verify bcrypt works — passlib can have compatibility issues with newer bcrypt versions.
- **Experiment model**: Has `mean_reward`/`std_reward` Float columns added to match `ExperimentResponse` schema.
- **WebSocket router**: Included separately in `main.py` at `/api/v1` prefix (router itself adds `/ws`).

## Environment Variables

Required in `.env` for local development:

```
SECRET_KEY=your-random-secret-key
DATABASE_URL=sqlite+aiosqlite:///./rl_gym.db
ENVIRONMENT=development
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000
```

For Docker/PostgreSQL: `DATABASE_URL=postgresql+asyncpg://rl_gym:rl_gym@postgres:5432/rl_gym`
