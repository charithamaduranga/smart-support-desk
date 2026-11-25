"""One-command start: `uv run python -m app`.

Loads .env (if present), trains the classifier on first run, then starts the server.
"""
import os

from dotenv import load_dotenv

from app.config import PROJECT_ROOT


def main() -> None:
    # <repo>/.env, whatever the working directory; real environment variables take precedence.
    load_dotenv(PROJECT_ROOT / ".env")
    from app.config import Settings
    from ml.model import MODEL_PATH

    settings = Settings.from_env()
    if not settings.model_path.exists():
        if settings.model_path != MODEL_PATH:
            raise SystemExit(f"Model not found at {settings.model_path}")
        print("No trained model found: downloading the dataset and training (first run only)...", flush=True)
        from ml.train import main as train

        train()
    if not settings.team_password:
        print("WARNING: TEAM_PASSWORD is not set, so team pages are disabled. See .env.example.", flush=True)
    if not (settings.openai_api_key and settings.openai_model):
        print("Note: OPENAI_API_KEY / OPENAI_MODEL not set, so AI reply drafts are disabled.", flush=True)

    import uvicorn

    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    print(f"Customer page: http://{host}:{port}/    Team page: http://{host}:{port}/team/tickets", flush=True)
    uvicorn.run("app.main:app", host=host, port=port)


if __name__ == "__main__":
    main()
