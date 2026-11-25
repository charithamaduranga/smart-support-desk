"""Independent verification of the grouped split (run: uv run python scripts/verify_split.py).

Checks: all categories on both sides; no group crosses the boundary; groups
are explained by text similarity (every multi-member group's members have a
>=0.9 neighbour inside the group, every singleton has none anywhere); a fresh
recomputation reproduces data/split.csv exactly.
"""
import json

import numpy as np
import pandas as pd

from ml.dataset import LABEL_COL, NEAR_DUP_THRESHOLD, REPORTS_DIR, SPLIT_FILE, compute_split

cached = pd.read_csv(SPLIT_FILE)
fresh = compute_split(write=False)  # recomputed from scratch in this process
data, vec = fresh.data, fresh.vectorizer
res = {}

cats = sorted(data[LABEL_COL].unique())
res["n_categories"] = len(cats)
res["categories_missing_in_train"] = sorted(set(cats) - set(fresh.train[LABEL_COL]))
res["categories_missing_in_test"] = sorted(set(cats) - set(fresh.test[LABEL_COL]))
res["min_test_count_per_category"] = int(fresh.test[LABEL_COL].value_counts().min())
res["groups_crossing_boundary"] = len(set(fresh.train["group"]) & set(fresh.test["group"]))

# Similarity explains the groups.
bad_members = 0
multi = data.groupby("group").filter(lambda g: len(g) > 1)
for _, g in multi.groupby("group"):
    X = vec.transform(g["norm"])
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, -1)
    bad_members += int((S.max(1) < NEAR_DUP_THRESHOLD - 1e-6).sum())
singletons = data.groupby("group").filter(lambda g: len(g) == 1)
res["multi_member_groups"] = int(multi["group"].nunique())
res["members_without_in_group_neighbour_ge_0.9"] = bad_members
res["singletons_with_any_neighbour_ge_0.9"] = int((singletons["nn_sim"] >= NEAR_DUP_THRESHOLD).sum())
res["groups_spanning_multiple_categories"] = int((data.groupby("group")[LABEL_COL].nunique() > 1).sum())
res["group_builder_inputs"] = "normalised text only (build_groups(norm_texts)); labels used only for fold stratification"

# Reproducibility: fresh computation vs the cached file written by an earlier process.
merged = cached.merge(data[["row_id", "group", "split"]], on="row_id", suffixes=("_cached", "_fresh"))
res["rows_cached"], res["rows_fresh"] = len(cached), len(data)
res["group_ids_identical"] = bool((merged.group_cached == merged.group_fresh).all())
res["split_identical"] = bool((merged.split_cached == merged.split_fresh).all())

(REPORTS_DIR / "split_verification.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
ok = (not res["categories_missing_in_train"] and not res["categories_missing_in_test"]
      and res["groups_crossing_boundary"] == 0 and bad_members == 0
      and res["singletons_with_any_neighbour_ge_0.9"] == 0
      and res["group_ids_identical"] and res["split_identical"] and len(merged) == len(cached) == len(data))
print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
raise SystemExit(0 if ok else 1)
