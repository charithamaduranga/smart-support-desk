"""Exploratory inspection of the Bitext customer-support dataset (no training).

Original first-pass analysis, kept as-is for reproducibility. Usage:
    uv run python scripts/inspect_dataset.py <path-to-pinned-csv> > reports/dataset_inspection.txt
The pinned CSV path is returned by `ml.dataset.load_raw`'s hf_hub_download call.
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)

df = pd.read_csv(sys.argv[1], keep_default_na=False, na_values=[""])

print("=== SHAPE / COLUMNS ===")
print(df.shape)
print(df.dtypes)
print(df.head(3).to_string())

print("\n=== MISSING VALUES (NaN or empty/whitespace) ===")
for c in df.columns:
    s = df[c]
    blank = s.isna() | s.astype(str).str.strip().eq("")
    print(f"{c:12s} missing/blank={blank.sum()}")

print("\n=== CATEGORY DISTRIBUTION ===")
vc = df["category"].value_counts()
print(pd.DataFrame({"n": vc, "pct": (vc / len(df) * 100).round(2)}))
print("n categories:", df["category"].nunique(), " max/min ratio:", round(vc.max() / vc.min(), 2))

print("\n=== INTENTS PER CATEGORY ===")
ic = df.groupby("category")["intent"].unique()
for cat, intents in ic.items():
    print(f"{cat:14s} {sorted(intents)}")
print("n intents:", df["intent"].nunique())
cats_per_intent = df.groupby("intent")["category"].nunique()
print("intents mapping to >1 category:", list(cats_per_intent[cats_per_intent > 1].index))

print("\n=== FLAGS (Bitext linguistic-variation tags) ===")
flag_letters = Counter(ch for f in df["flags"].astype(str) for ch in f)
print(dict(sorted(flag_letters.items(), key=lambda x: -x[1])))
print("distinct flag strings:", df["flags"].nunique())

print("\n=== INSTRUCTION LENGTH (words) ===")
wl = df["instruction"].str.split().str.len()
print(wl.describe().round(1).to_dict())
print("placeholder rows ({{...}}):", df["instruction"].str.contains(r"\{\{").sum())
print("top placeholders:", Counter(re.findall(r"\{\{[^}]+\}\}", " ".join(df["instruction"]))).most_common(8))

print("\n=== EXACT DUPLICATES ===")
print("full-row duplicates:", df.duplicated().sum())
print("instruction duplicates (extra copies):", df.duplicated("instruction").sum())
print("instruction+category duplicates:", df.duplicated(["instruction", "category"]).sum())
print("instruction+intent duplicates:", df.duplicated(["instruction", "intent"]).sum())


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\{\{[^}]+\}\}", " <ph> ", t)  # unify placeholders
    t = re.sub(r"[^a-z0-9<> ]+", " ", t)  # drop punctuation
    return re.sub(r"\s+", " ", t).strip()


df["norm"] = df["instruction"].map(norm)
print("normalized-instruction duplicates (extra copies):", df.duplicated("norm").sum())
print("unique normalized instructions:", df["norm"].nunique())

print("\n=== SAME TEXT, DIFFERENT LABEL ===")
for key in ["instruction", "norm"]:
    g = df.groupby(key).agg(ncat=("category", "nunique"), nint=("intent", "nunique"), n=("category", "size"))
    cc = g[g.ncat > 1]
    ci = g[g.nint > 1]
    print(f"[{key}] texts with >1 category: {len(cc)} (rows {cc.n.sum()}); with >1 intent: {len(ci)} (rows {ci.n.sum()})")
    if len(cc):
        ex = df[df[key].isin(cc.index[:8])].sort_values(key)[["instruction", "category", "intent"]]
        print(ex.to_string())
    if key == "norm" and len(ci):
        ex = df[df[key].isin(ci.index[:6]) & ~df[key].isin(cc.index)].sort_values(key)[["instruction", "category", "intent"]]
        print("  intent-only conflicts (same category):")
        print(ex.head(12).to_string())

print("\n=== NEAR DUPLICATES (TF-IDF word 1-2gram cosine on normalized text, nearest neighbour) ===")
uniq = df.drop_duplicates("norm").reset_index(drop=True)
vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
Xw = vec_w.fit_transform(uniq["norm"])
n = Xw.shape[0]
nn_sim = np.zeros(n, dtype=np.float32)
nn_idx = np.zeros(n, dtype=np.int64)
THRESH_EDGE = 0.9
rows, cols = [], []
chunk = 1500
for s in range(0, n, chunk):
    S = (Xw[s : s + chunk] @ Xw.T).toarray()
    for i in range(S.shape[0]):
        S[i, s + i] = -1
    nn_idx[s : s + chunk] = S.argmax(1)
    nn_sim[s : s + chunk] = S.max(1)
    r, c = np.nonzero(S >= THRESH_EDGE)
    rows.append(r + s)
    cols.append(c)
uniq["nn_sim"] = nn_sim
uniq["nn_idx"] = nn_idx
for t in [0.7, 0.8, 0.9, 0.95, 0.99]:
    print(f"unique texts with a neighbour at cos>={t}: {(nn_sim >= t).sum()} ({(nn_sim >= t).mean() * 100:.1f}%)")
print("nn similarity quantiles:", np.quantile(nn_sim, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3))

same_cat = uniq["category"].values == uniq["category"].values[nn_idx]
same_int = uniq["intent"].values == uniq["intent"].values[nn_idx]
print(f"nearest neighbour has same category: {same_cat.mean() * 100:.2f}%  same intent: {same_int.mean() * 100:.2f}%")
for t in [0.8, 0.9, 0.95]:
    m = nn_sim >= t
    print(f"  among pairs cos>={t}: different category {(~same_cat[m]).sum()} / {m.sum()}, different intent {(~same_int[m]).sum()} / {m.sum()}")

print("\nExamples: high-similarity neighbours with DIFFERENT category:")
diff = uniq[(~same_cat) & (nn_sim >= 0.8)].sort_values("nn_sim", ascending=False).head(12)
for _, r in diff.iterrows():
    o = uniq.iloc[r.nn_idx]
    print(f"  {r.nn_sim:.3f} | [{r.category}/{r.intent}] {r.instruction!r}\n        | [{o.category}/{o.intent}] {o.instruction!r}")

print("\nExamples: high-similarity neighbours, same intent (template paraphrases):")
for _, r in uniq[(same_int) & (nn_sim.round(3) >= 0.9) & (nn_sim < 0.99)].sample(6, random_state=0).iterrows():
    o = uniq.iloc[r.nn_idx]
    print(f"  {r.nn_sim:.3f} | {r.instruction!r}\n        | {o.instruction!r}")

print("\n=== NEAR-DUPLICATE GROUPS (connected components at cos >= %.2f) ===" % THRESH_EDGE)
r = np.concatenate(rows)
c = np.concatenate(cols)
A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
ncomp, labels = connected_components(A, directed=False)
uniq["group"] = labels
gs = pd.Series(labels).value_counts()
print("groups:", ncomp, " singleton groups:", (gs == 1).sum(), " largest group sizes:", gs.head(8).tolist())
mixed = uniq.groupby("group")["category"].nunique()
print("groups spanning >1 category:", (mixed > 1).sum(), " rows in them:", uniq["group"].isin(mixed[mixed > 1].index).sum())
df["group"] = df["norm"].map(dict(zip(uniq["norm"], uniq["group"])))

print("\n=== LEAKAGE: fraction of TEST rows whose text has a cos>=0.9 neighbour / exact normalized match in TRAIN ===")
Xall = vec_w.transform(df["norm"])


def leakage(train_idx, test_idx, name):
    tr_norm = set(df["norm"].iloc[train_idx])
    exact = df["norm"].iloc[test_idx].isin(tr_norm).mean()
    Xtr, Xte = Xall[train_idx], Xall[test_idx]
    best = np.zeros(len(test_idx), dtype=np.float32)
    for s in range(0, len(test_idx), 2000):
        best[s : s + 2000] = (Xte[s : s + 2000] @ Xtr.T).max(1).toarray().ravel()
    print(f"{name:32s} test={len(test_idx):5d} exact-norm-in-train={exact * 100:5.1f}%  nn>=0.9={np.mean(best >= 0.9) * 100:5.1f}%  nn>=0.8={np.mean(best >= 0.8) * 100:5.1f}%  median nn={np.median(best):.3f}")
    return best


idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, stratify=df["category"], random_state=42)
leakage(tr, te, "random stratified (row-level)")

dedup = df.drop_duplicates("norm")
didx = dedup.index.values
tr2, te2 = train_test_split(didx, test_size=0.2, stratify=dedup["category"], random_state=42)
leakage(tr2, te2, "dedup(norm) + random stratified")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
dsub = df.loc[didx]
trp, tep = next(sgkf.split(dsub, dsub["category"], groups=dsub["group"]))
tr3, te3 = didx[trp], didx[tep]
leakage(tr3, te3, "dedup + StratifiedGroupKFold")
print("group split category share (test %):")
print((df.loc[te3, "category"].value_counts(normalize=True) * 100).round(2).to_dict())
print("overall category share (%):")
print((df["category"].value_counts(normalize=True) * 100).round(2).to_dict())
