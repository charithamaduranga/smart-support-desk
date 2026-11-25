FROM python:3.12-slim

# Pinned uv binary, same version used for development.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY ml ./ml
COPY app ./app
COPY data ./data
RUN uv sync --frozen --no-dev

# Train the classifier at build time so the image ships with the model.
# Downloads the pinned Bitext dataset revision; the HF cache is removed afterwards.
RUN uv run --no-sync python -m ml.train && rm -rf /root/.cache/huggingface

# SQLite lives on a volume; secrets (OPENAI_API_KEY, TEAM_PASSWORD, ...) come at runtime only.
RUN mkdir -p /data
ENV DATABASE_URL=sqlite:////data/support_desk.db
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
