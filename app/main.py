from fastapi import FastAPI

VERSION = "0.1.0"


def create_app() -> FastAPI:
    app = FastAPI(title="RL Gym API", version=VERSION)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": VERSION}

    return app


app = create_app()
