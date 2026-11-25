import pytest
from fastapi.testclient import TestClient

from app.classifier import Prediction
from app.config import Settings
from app.main import create_app


class FakeClassifier:
    """Deterministic stand-in: records every call and the instance that served it."""

    model_version = "fake-v1"

    def __init__(self, category="REFUND", confidence=0.87):
        self.category, self.confidence = category, confidence
        self.calls = []

    def classify(self, text):
        self.calls.append((id(self), text))
        return Prediction(self.category, self.confidence, self.model_version)


@pytest.fixture
def settings(tmp_path):
    """A separate SQLite file per test."""
    return Settings(database_url=f"sqlite:///{(tmp_path / 'app.db').as_posix()}",
                    model_path=tmp_path / "no-model.joblib")


@pytest.fixture
def fake_classifier_cls():
    return FakeClassifier


@pytest.fixture
def fake_classifier():
    return FakeClassifier()


@pytest.fixture
def client(settings, fake_classifier):
    with TestClient(create_app(settings, classifier=fake_classifier), follow_redirects=False) as c:
        yield c
