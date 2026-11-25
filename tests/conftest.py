import pytest
from fastapi.testclient import TestClient

from app.classifier import Prediction
from app.config import Settings
from app.main import create_app

TEAM_AUTH = ("team", "test-pass")


class FakeClassifier:
    """Deterministic stand-in: records every call and the instance that served it."""

    model_version = "fake-v1"

    def __init__(self, category="REFUND", confidence=0.87):
        self.category, self.confidence = category, confidence
        self.calls = []

    def classify(self, text):
        self.calls.append((id(self), text))
        return Prediction(self.category, self.confidence, self.model_version)


@pytest.fixture(autouse=True)
def _no_real_openai(monkeypatch):
    # A developer's shell key must never make the test suite call a real API.
    for var in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def settings(tmp_path):
    """A separate SQLite file per test; team password set; no model on disk."""
    return Settings(database_url=f"sqlite:///{(tmp_path / 'app.db').as_posix()}",
                    model_path=tmp_path / "no-model.joblib", team_password=TEAM_AUTH[1])


@pytest.fixture
def fake_classifier_cls():
    return FakeClassifier


@pytest.fixture
def fake_classifier():
    return FakeClassifier()


@pytest.fixture
def client(settings, fake_classifier):
    with TestClient(create_app(settings, classifier=fake_classifier, drafter=None),
                    follow_redirects=False) as c:
        yield c


@pytest.fixture
def team_auth():
    return TEAM_AUTH
