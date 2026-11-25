import json
import shutil

import joblib
import numpy as np
import pytest

import app.classifier as classifier_module
from app.classifier import Prediction, TicketClassifier
from ml.model import MODEL_PATH
from ml.train import build_pipeline

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not present; run ml.train")


class FakePipeline:
    classes_ = np.array(["ORDER", "REFUND", "PAYMENT"])

    def __init__(self, proba):
        self.proba = np.array(proba)
        self.calls = []

    def predict_proba(self, texts):
        self.calls.append(list(texts))
        return np.array([self.proba])


@pytest.fixture
def toy_model_path(tmp_path):
    """A tiny real pipeline saved with a metadata file, mirroring ml.train's output layout."""
    X = ["track my order", "where is my order", "order status", "i want a refund", "refund my money",
         "give my refund"] * 2
    y = ["ORDER"] * 3 + ["REFUND"] * 3
    path = tmp_path / "toy.joblib"
    joblib.dump(build_pipeline().fit(X, y * 2), path)
    path.with_suffix(".meta.json").write_text(json.dumps({"trained_at": "2026-01-01T00:00:00+00:00"}))
    return path


def test_classify_returns_top_class_and_confidence():
    clf = TicketClassifier(FakePipeline([0.2, 0.7, 0.1]), model_version="v1")
    assert clf.classify("where is my money") == Prediction("REFUND", pytest.approx(0.7), "v1")


def test_classify_returns_plain_python_types():
    p = TicketClassifier(FakePipeline([0.6, 0.3, 0.1])).classify("x")
    assert type(p.category) is str and type(p.confidence) is float


def test_classify_passes_raw_text_to_pipeline():
    pipe = FakePipeline([1.0, 0.0, 0.0])
    TicketClassifier(pipe).classify("  Track ORDER #12!  ")
    assert pipe.calls == [["  Track ORDER #12!  "]]  # normalisation is the pipeline's job


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_classify_rejects_blank_text(blank):
    with pytest.raises(ValueError):
        TicketClassifier(FakePipeline([1.0, 0.0, 0.0])).classify(blank)


def test_from_disk_reads_model_version(toy_model_path):
    clf = TicketClassifier.from_disk(toy_model_path)
    assert clf.model_version == "2026-01-01T00:00:00+00:00"
    assert clf.classify("where is my refund").category == "REFUND"


def test_from_disk_without_metadata_uses_unknown_version(toy_model_path):
    toy_model_path.with_suffix(".meta.json").unlink()
    assert TicketClassifier.from_disk(toy_model_path).model_version == "unknown"


def test_from_disk_missing_model_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="ml.train"):
        TicketClassifier.from_disk(tmp_path / "missing.joblib")


def test_model_loaded_once_and_reused_across_predictions(toy_model_path, monkeypatch):
    load_calls = []
    real_load = classifier_module.load_model

    def counting_load(path):
        load_calls.append(path)
        return real_load(path)

    monkeypatch.setattr(classifier_module, "load_model", counting_load)
    clf = TicketClassifier.from_disk(toy_model_path)
    pipeline_before = clf._pipeline

    # Remove the artifact: any per-prediction reload would now fail.
    toy_model_path.unlink()
    toy_model_path.with_suffix(".meta.json").unlink()
    results = [clf.classify(t).category for t in ["track my order", "refund my money"] * 10]

    assert len(load_calls) == 1
    assert clf._pipeline is pipeline_before
    assert results == ["ORDER", "REFUND"] * 10


@needs_model
def test_real_model_loaded_once_from_copied_artifact(tmp_path):
    # Same guarantee with the real trained model: copy it, load, delete the copy, keep predicting.
    copy = tmp_path / MODEL_PATH.name
    shutil.copy(MODEL_PATH, copy)
    shutil.copy(MODEL_PATH.with_suffix(".meta.json"), copy.with_suffix(".meta.json"))
    clf = TicketClassifier.from_disk(copy)
    copy.unlink()
    copy.with_suffix(".meta.json").unlink()

    refund = clf.classify("I want my money back for order 123")
    account = clf.classify("how do I reset my password")
    assert (refund.category, account.category) == ("REFUND", "ACCOUNT")
    assert 0.0 <= refund.confidence <= 1.0
    assert clf.model_version != "unknown"
