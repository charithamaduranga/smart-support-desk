"""Thin wrapper around the saved sklearn pipeline for use by the web app.

The app creates one TicketClassifier at startup and reuses it for every
request; `classify()` never touches the disk.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from ml.model import MODEL_PATH, load_model


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float  # top-class probability (not calibrated)
    model_version: str


class TicketClassifier:
    def __init__(self, pipeline, model_version: str = "unknown"):
        self._pipeline = pipeline
        self.model_version = model_version

    @classmethod
    def from_disk(cls, path: Path = MODEL_PATH) -> "TicketClassifier":
        """Load the pipeline once (raises FileNotFoundError if it hasn't been trained)."""
        path = Path(path)
        pipeline = load_model(path)
        meta_path = path.with_suffix(".meta.json")
        version = "unknown"
        if meta_path.exists():
            version = json.loads(meta_path.read_text(encoding="utf-8")).get("trained_at", "unknown")
        return cls(pipeline, version)

    def classify(self, text: str) -> Prediction:
        if not text or not text.strip():
            raise ValueError("cannot classify an empty message")
        proba = self._pipeline.predict_proba([text])[0]
        best = int(proba.argmax())
        # Plain Python types so values store cleanly in SQLAlchemy and render in Jinja.
        return Prediction(
            category=str(self._pipeline.classes_[best]),
            confidence=float(proba[best]),
            model_version=self.model_version,
        )
