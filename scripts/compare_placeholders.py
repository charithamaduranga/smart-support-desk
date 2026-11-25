"""Compare placeholder-normalisation variants for the classifier (does not change the saved model).

Selection metric: 5-fold StratifiedGroupKFold CV on the TRAINING split only
(groups reused, so no near-duplicate leakage between CV folds). Grouped-test
and custom-message scores are reported for information, not for selection.

A  current: every {{Placeholder}} -> <ph>
B  typed:   {{Delivery City}} -> phdeliverycity (placeholder type preserved)
C  A + digit runs (2+ digits) -> <ph>, so real order numbers resemble training placeholders

Run: uv run python scripts/compare_placeholders.py
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.dataset import LABEL_COL, REPORTS_DIR, SEED, TEXT_COL, load_split
from ml.evaluate import CUSTOM_PATH
from ml.text import normalize
from ml.train import build_pipeline


def typed(text: str) -> str:
    t = re.sub(r"\{\{([^}]+)\}\}", lambda m: " ph" + re.sub(r"[^a-z]", "", m.group(1).lower()) + " ", text)
    return normalize(t)


def digits_as_ph(text: str) -> str:
    return normalize(re.sub(r"\d{2,}", " {{num}} ", text))


VARIANTS = {"A_current": normalize, "B_typed_placeholders": typed, "C_digits_as_placeholder": digits_as_ph}

split = load_split()
train, test = split.train, split.test
custom = pd.read_csv(CUSTOM_PATH, keep_default_na=False)
lenient = [{r.expected, *filter(None, r.acceptable.split("|"))} for r in custom.itertuples()]
ph_mask = test[TEXT_COL].str.contains(r"\{\{").values
cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(train, train[LABEL_COL], train["group"]))

results = {}
for name, fn in VARIANTS.items():
    cv_f1 = []
    for tr, va in cv:
        p = build_pipeline(fn).fit(train[TEXT_COL].iloc[tr], train[LABEL_COL].iloc[tr])
        cv_f1.append(f1_score(train[LABEL_COL].iloc[va], p.predict(train[TEXT_COL].iloc[va]), average="macro"))
    p = build_pipeline(fn).fit(train[TEXT_COL], train[LABEL_COL])
    pred = p.predict(test[TEXT_COL])
    y = test[LABEL_COL].values
    cpred = p.predict(custom["text"])
    results[name] = {
        "cv_macro_f1_mean": round(float(np.mean(cv_f1)), 4),
        "cv_macro_f1_std": round(float(np.std(cv_f1)), 4),
        "cv_errors_total": None,  # filled below
        "test_accuracy": round(float(accuracy_score(y, pred)), 4),
        "test_errors": int((pred != y).sum()),
        "test_errors_on_placeholder_rows": int((pred[ph_mask] != y[ph_mask]).sum()),
        "test_placeholder_rows": int(ph_mask.sum()),
        "custom_strict": round(float(np.mean(cpred == custom["expected"].values)), 2),
        "custom_lenient": round(float(np.mean([c in ok for c, ok in zip(cpred, lenient)])), 2),
        "custom_predictions": list(cpred),
    }
    results[name].pop("cv_errors_total")
    r = results[name]
    print(f"{name:26s} CV macroF1={r['cv_macro_f1_mean']:.4f}±{r['cv_macro_f1_std']:.4f}  test errors={r['test_errors']} "
          f"(on placeholder rows {r['test_errors_on_placeholder_rows']}/{r['test_placeholder_rows']})  "
          f"custom strict={r['custom_strict']} lenient={r['custom_lenient']}")

base = results["A_current"]["custom_predictions"]
for name, r in results.items():
    changed = [(custom["text"][i][:60], base[i], r["custom_predictions"][i]) for i in range(len(base)) if r["custom_predictions"][i] != base[i]]
    if changed:
        print(f"\n{name}: custom predictions that differ from A:")
        for t, a, b in changed:
            print(f"   A={a:12s} {name[:1]}={b:12s} | {t}")
(REPORTS_DIR / "placeholder_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
