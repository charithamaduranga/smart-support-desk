"""Application settings read from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

from ml.model import MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Absolute path so the database location does not depend on the working directory.
DEFAULT_DATABASE_URL = f"sqlite:///{(PROJECT_ROOT / 'support_desk.db').as_posix()}"
DEFAULT_FAQ_PATH = PROJECT_ROOT / "data" / "faq.json"


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    model_path: Path = MODEL_PATH
    team_username: str = "team"
    team_password: str | None = None  # team pages are unavailable until this is set
    faq_path: Path = DEFAULT_FAQ_PATH
    openai_api_key: str | None = None  # no key -> AI reply drafts disabled
    openai_model: str | None = None
    openai_base_url: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            model_path=Path(os.getenv("MODEL_PATH", str(MODEL_PATH))),
            team_username=os.getenv("TEAM_USERNAME", "team"),
            team_password=os.getenv("TEAM_PASSWORD") or None,
            faq_path=Path(os.getenv("FAQ_PATH", str(DEFAULT_FAQ_PATH))),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
