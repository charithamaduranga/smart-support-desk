"""Load the pinned Bitext dataset and build a leakage-aware grouped train/test split.

Steps: normalise -> drop exact/normalised duplicates -> link texts whose TF-IDF
cosine >= 0.9 into near-duplicate groups (connected components) ->
StratifiedGroupKFold (5 folds, seed 42), fold 0 is the test set.

Run `uv run python -m ml.dataset` to print and save the split report.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from ml.text import normalize

REPO_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"  # pinned commit (2024-07-18)
FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
FILE_SHA256 = "6f81102b0100b97b8468eb04368033a23206bf1fde9d53500d5806ec1001a434"

SEED = 42
N_SPLITS = 5
TEST_FOLD = 0
NEAR_DUP_THRESHOLD = 0.9
TEXT_COL, LABEL_COL = "instruction", "category"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def load_raw(local_files_only: bool = False) -> pd.DataFrame:
    """Download (cached by huggingface_hub) the pinned CSV. Adds a stable `row_id`."""
    path = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset", revision=REVISION,
                           local_files_only=local_files_only)
    df = pd.read_csv(path, keep_default_na=False)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each normalised instruction.

    Inspection found no normalised text with conflicting labels; we assert it
    so a dataset change cannot silently drop a contradictory example.
    """
    df = df.assign(norm=df[TEXT_COL].map(normalize))
    conflicts = df.groupby("norm")[LABEL_COL].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"{(conflicts > 1).sum()} normalised texts have conflicting labels")
    return df.drop_duplicates("norm").reset_index(drop=True)


def _grouping_vectorizer() -> TfidfVectorizer:
    # Unsupervised and used only for grouping/diagnostics, never as model features.
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def build_groups(norm_texts: pd.Series, threshold: float = NEAR_DUP_THRESHOLD, chunk: int = 1500):
    """Connected components over the graph `cosine >= threshold`.

    Returns (group labels, fitted vectorizer, nearest-neighbour similarity per text).
    """
    vec = _grouping_vectorizer()
    X = vec.fit_transform(norm_texts)
    n = X.shape[0]
    rows, cols = [], []
    nn_sim = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        S = (X[s : s + chunk] @ X.T).toarray()
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -1  # ignore self
        nn_sim[s : s + chunk] = S.max(1)
        r, c = np.nonzero(S >= threshold)
        rows.append(r + s)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels, vec, nn_sim


def grouped_split(df: pd.DataFrame, groups: np.ndarray, seed: int = SEED):
    """Return (train_idx, test_idx) positional indices; no group crosses the boundary."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, df[LABEL_COL], groups=groups))
    train_idx, test_idx = folds[TEST_FOLD]
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise AssertionError(f"{len(overlap)} groups cross the train/test boundary")
    return train_idx, test_idx


SPLIT_FILE = Path(__file__).resolve().parent.parent / "data" / "split.csv"


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    data: pd.DataFrame  # deduplicated data with `group` and `split` columns
    vectorizer: TfidfVectorizer | None = None  # grouping vectorizer (diagnostics only)


def compute_split(write: bool = True) -> Split:
    """Full computation (~2 min). Optionally writes data/split.csv (row_id, group, split).

    Groups are built from normalised text only; the category label is used
    solely to stratify fold assignment in StratifiedGroupKFold.
    """
    data = deduplicate(load_raw())
    groups, vec, nn_sim = build_groups(data["norm"])
    data = data.assign(group=groups, nn_sim=nn_sim, split="train")
    tr, te = grouped_split(data, groups)
    data.loc[te, "split"] = "test"
    if write:
        SPLIT_FILE.parent.mkdir(exist_ok=True)
        data[["row_id", "group", "split"]].to_csv(SPLIT_FILE, index=False)
    return Split(data.iloc[tr].reset_index(drop=True), data.iloc[te].reset_index(drop=True), data, vec)


def load_split(local_files_only: bool = False) -> Split:
    """Use the cached split assignment if present (regenerate with `python -m ml.dataset`)."""
    if not SPLIT_FILE.exists():
        return compute_split()
    data = deduplicate(load_raw(local_files_only))
    assign = pd.read_csv(SPLIT_FILE)
    if set(assign["row_id"]) != set(data["row_id"]):
        raise ValueError(f"{SPLIT_FILE} does not match the deduplicated dataset; rerun ml.dataset")
    data = data.merge(assign, on="row_id", how="left")
    if set(data.loc[data.split == "train", "group"]) & set(data.loc[data.split == "test", "group"]):
        raise AssertionError("cached split has groups crossing the train/test boundary")
    train = data[data.split == "train"].reset_index(drop=True)
    test = data[data.split == "test"].reset_index(drop=True)
    return Split(train, test, data)


def test_to_train_similarity(vec, train_norm, test_norm) -> np.ndarray:
    """Max cosine similarity of each test text to any training text."""
    Xtr, Xte = vec.transform(train_norm), vec.transform(test_norm)
    best = np.zeros(Xte.shape[0], dtype=np.float32)
    for s in range(0, Xte.shape[0], 2000):
        best[s : s + 2000] = (Xte[s : s + 2000] @ Xtr.T).max(1).toarray().ravel()
    return best


def similarity_summary(best: np.ndarray) -> dict:
    return {
        "share_ge_0.9": round(float(np.mean(best >= 0.9)), 4),
        "share_ge_0.8": round(float(np.mean(best >= 0.8)), 4),
        "median": round(float(np.median(best)), 4),
    }


def chaining_report(split: Split, top: int = 5) -> list[dict]:
    """For the largest groups, check whether transitive linking joined dissimilar texts."""
    data = split.data
    sizes = data["group"].value_counts().head(top)
    out = []
    for g, size in sizes.items():
        members = data[data["group"] == g]
        X = split.vectorizer.transform(members["norm"])
        S = (X @ X.T).toarray()
        iu = np.triu_indices(len(members), 1)
        i, j = iu[0][S[iu].argmin()], iu[1][S[iu].argmin()]
        out.append({
            "group": int(g),
            "size": int(size),
            "categories": sorted(members[LABEL_COL].unique()),
            "intents": sorted(members["intent"].unique()),
            "min_pairwise_cos": round(float(S[iu].min()), 3),
            "mean_pairwise_cos": round(float(S[iu].mean()), 3),
            "least_similar_pair": [members[TEXT_COL].iloc[i], members[TEXT_COL].iloc[j]],
        })
    return out


def _dist(s: pd.Series) -> dict:
    vc = s.value_counts()
    return {k: {"n": int(v), "pct": round(100 * v / len(s), 2)} for k, v in vc.items()}


def main() -> None:
    raw = load_raw()
    split = compute_split()
    data = split.data
    group_sizes = data["group"].value_counts()
    best = test_to_train_similarity(split.vectorizer, split.train["norm"], split.test["norm"])
    report = {
        "dataset": {"repo": REPO_ID, "revision": REVISION, "file": FILENAME, "sha256": FILE_SHA256},
        "rows_raw": len(raw),
        "rows_after_dedup": len(data),
        "duplicates_removed": len(raw) - len(data),
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "n_groups": int(len(group_sizes)),
        "singleton_groups": int((group_sizes == 1).sum()),
        "largest_group_sizes": group_sizes.head(10).tolist(),
        "groups_spanning_multiple_categories": int((data.groupby("group")[LABEL_COL].nunique() > 1).sum()),
        "split": {"method": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={SEED}), fold {TEST_FOLD} = test",
                  "train_rows": len(split.train), "test_rows": len(split.test),
                  "train_groups": int(split.train["group"].nunique()), "test_groups": int(split.test["group"].nunique()),
                  "groups_crossing_boundary": len(set(split.train["group"]) & set(split.test["group"]))},
        "test_to_train_nn_similarity": similarity_summary(best),
        "class_distribution": {"overall": _dist(data[LABEL_COL]), "train": _dist(split.train[LABEL_COL]),
                               "test": _dist(split.test[LABEL_COL])},
        "largest_groups_chaining_check": chaining_report(split),
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "dataset_split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"raw={report['rows_raw']} dedup={report['rows_after_dedup']} removed={report['duplicates_removed']}")
    print(f"groups={report['n_groups']} singletons={report['singleton_groups']} largest={report['largest_group_sizes']}")
    print(f"groups spanning >1 category: {report['groups_spanning_multiple_categories']}")
    print("split:", report["split"])
    print("test->train NN similarity:", report["test_to_train_nn_similarity"])
    print(f"\n{'category':14s} {'overall%':>9s} {'train%':>8s} {'test%':>8s} {'test n':>7s}")
    cd = report["class_distribution"]
    for k in cd["overall"]:
        print(f"{k:14s} {cd['overall'][k]['pct']:9.2f} {cd['train'][k]['pct']:8.2f} {cd['test'][k]['pct']:8.2f} {cd['test'][k]['n']:7d}")
    print("\nLargest groups (chaining check):")
    for g in report["largest_groups_chaining_check"]:
        print(f"  size={g['size']} intents={g['intents']} min_cos={g['min_pairwise_cos']} mean_cos={g['mean_pairwise_cos']}")
        print(f"    least similar: {g['least_similar_pair'][0]!r}  <->  {g['least_similar_pair'][1]!r}")
    print(f"\nSaved {REPORTS_DIR / 'dataset_split.json'}")


if __name__ == "__main__":
    main()
