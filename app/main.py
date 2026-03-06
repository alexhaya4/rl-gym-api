from fastapi import FastAPI

from app.api.v1 import router as v1_router

VERSION = "0.1.0"


def create_app() -> FastAPI:
    app = FastAPI(title="RL Gym API", version=VERSION)

    app.include_router(v1_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": VERSION}

    return app


app = create_app()
