"""Application settings read from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

from ml.model import MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Absolute path so the database location does not depend on the working directory.
DEFAULT_DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'support_desk.db').as_posix()}"


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    model_path: Path = MODEL_PATH

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            model_path=Path(os.getenv("MODEL_PATH", str(MODEL_PATH))),
        )
