"""Load the saved category classifier (used by evaluation and, later, the web app)."""
import json
import warnings
from pathlib import Path

import joblib
import sklearn

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "category_classifier.joblib"
META_PATH = MODEL_PATH.with_suffix(".meta.json")


def load_model(path: Path = MODEL_PATH):
    """Load the fitted Pipeline. Importing ml.text first guarantees the pickled
    reference to ml.text.normalize resolves."""
    import ml.text  # noqa: F401

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; train it with `uv run python -m ml.train`")
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        trained_with = json.loads(meta_path.read_text(encoding="utf-8")).get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(f"model trained with scikit-learn {trained_with}, running {sklearn.__version__}")
    return joblib.load(path)
