import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.dataset import LABEL_COL, build_groups, deduplicate, grouped_split, load_split
from ml.evaluate import CUSTOM_PATH
from ml.model import MODEL_PATH, load_model
from ml.text import normalize
from ml.train import build_pipeline

CATEGORIES = {"ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE", "ORDER", "PAYMENT",
              "REFUND", "SHIPPING", "SUBSCRIPTION"}
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not present; run ml.train")


def test_normalize_placeholders_case_punctuation():
    assert normalize("Cancel  ORDER {{Order Number}}!!") == "cancel order <ph>"
    assert normalize("Where's my REFUND?") == "where s my refund"  # punctuation -> space, not deleted


def test_deduplicate_removes_normalized_duplicates():
    df = pd.DataFrame({"instruction": ["Track my order", "track my order!", "refund please"],
                       "category": ["ORDER", "ORDER", "REFUND"]})
    out = deduplicate(df)
    assert len(out) == 2 and set(out["category"]) == {"ORDER", "REFUND"}


def test_deduplicate_rejects_conflicting_labels():
    df = pd.DataFrame({"instruction": ["help me", "Help me."], "category": ["ORDER", "REFUND"]})
    with pytest.raises(ValueError):
        deduplicate(df)


def test_build_groups_links_near_duplicates_only():
    texts = pd.Series(["i want to track my order now", "i want to track my order now please",
                       "how do i delete my account", "what payment methods do you accept"])
    groups, _, _ = build_groups(texts, threshold=0.7)
    assert groups[0] == groups[1]
    assert len({groups[0], groups[2], groups[3]}) == 3


def _toy_frame(n_groups=50, per_group=3):
    rng = np.random.default_rng(0)
    cats = ["A", "B", "C", "D", "E"]
    rows = [{"instruction": f"text {g} {i}", "category": cats[g % 5]} for g in range(n_groups) for i in range(per_group)]
    df = pd.DataFrame(rows)
    groups = np.repeat(np.arange(n_groups), per_group)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True), groups[perm]


def test_grouped_split_has_no_group_overlap_and_is_deterministic():
    df, groups = _toy_frame()
    tr1, te1 = grouped_split(df, groups)
    tr2, te2 = grouped_split(df, groups)
    assert not set(groups[tr1]) & set(groups[te1])
    assert np.array_equal(te1, te2)
    assert len(tr1) + len(te1) == len(df)


def test_pipeline_fits_predicts_and_roundtrips(tmp_path):
    X = ["track my order", "where is my order", "order status please", "i want a refund",
         "refund my money", "get my refund back"] * 3
    y = ["ORDER"] * 3 + ["REFUND"] * 3
    pipe = build_pipeline().fit(X, y * 3)
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    assert loaded.predict(["where's my ordr"])[0] == "ORDER"
    assert np.allclose(loaded.predict_proba(["refund"]).sum(), 1.0)


@needs_model
def test_saved_model_sanity():
    model = load_model()
    assert set(model.classes_) == CATEGORIES
    assert model.predict(["I want to delete my account"])[0] == "ACCOUNT"


@needs_model
def test_saved_model_loads_with_plain_joblib_from_other_cwd(tmp_path):
    # Regression test: unpickling used to fail with "No module named 'ml'" outside the repo root.
    code = f"import joblib; m = joblib.load(r'{MODEL_PATH}'); print(m.predict(['where is my refund'])[0])"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "REFUND"


def test_custom_messages_file_integrity():
    df = pd.read_csv(CUSTOM_PATH, keep_default_na=False)
    assert len(df) == 20 and df["text"].is_unique
    assert set(df["expected"]) <= CATEGORIES
    for r in df.itertuples():
        alts = [a for a in r.acceptable.split("|") if a]
        assert set(alts) <= CATEGORIES and r.expected not in alts


def test_real_split_integrity():
    try:
        split = load_split(local_files_only=True)
    except Exception as e:  # dataset not cached locally / offline
        pytest.skip(f"pinned dataset not available locally: {e}")
    assert set(split.train[LABEL_COL]) == set(split.test[LABEL_COL]) == CATEGORIES
    assert not set(split.train["group"]) & set(split.test["group"])
    assert len(split.train) == 19061 and len(split.test) == 4765
