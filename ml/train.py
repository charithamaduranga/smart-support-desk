"""Train the ticket-category classifier on the grouped training split only.

Run `uv run python -m ml.train`. Saves the full fitted sklearn Pipeline
(normalisation + TF-IDF + LogisticRegression) to models/.
"""
import json
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.dataset import LABEL_COL, REVISION, SEED, TEXT_COL, load_split
from ml.model import META_PATH, MODEL_PATH, MODELS_DIR
from ml.text import normalize


def build_pipeline(preprocessor=normalize) -> Pipeline:
    """Fixed, untuned configuration chosen a priori (no test-set tuning).

    Word n-grams capture key phrases; char n-grams (within word boundaries)
    add robustness to typos and colloquial spellings ("pasword", "u", "ur").
    `preprocessor` is overridable only for offline comparisons.
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=preprocessor, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(preprocessor=preprocessor, analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


def main() -> None:
    split = load_split()
    pipe = build_pipeline()
    pipe.fit(split.train[TEXT_COL], split.train[LABEL_COL])  # training rows only

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_revision": REVISION,
        "train_rows": len(split.train),
        "labels": list(pipe.classes_),
        "sklearn_version": sklearn.__version__,
        "params": {"C": 10.0, "class_weight": "balanced", "word_ngrams": [1, 2], "char_wb_ngrams": [2, 5], "min_df": 2},
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Trained on {len(split.train)} rows; saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
