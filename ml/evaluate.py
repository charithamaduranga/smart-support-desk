"""Evaluate the saved classifier.

1. Grouped test split (primary, leakage-aware result).
2. Random stratified split on raw rows (DIAGNOSTIC ONLY: duplicates and
   near-duplicates leak across the split, so this score is likely optimistic).
3. 20 hand-written messages in data/custom_messages.csv (separate from Bitext).

Run `uv run python -m ml.evaluate` after `ml.train`. Writes reports/*.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ml.dataset import LABEL_COL, REPORTS_DIR, SEED, TEXT_COL, load_raw, load_split
from ml.model import load_model
from ml.train import build_pipeline

CUSTOM_PATH = Path(__file__).resolve().parent.parent / "data" / "custom_messages.csv"


def metrics(y_true, y_pred, labels) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)), 4),
    }


def evaluate_grouped(model, test: pd.DataFrame) -> dict:
    labels = list(model.classes_)
    proba = model.predict_proba(test[TEXT_COL])
    pred = model.classes_[proba.argmax(1)]
    y = test[LABEL_COL].values
    cm = confusion_matrix(y, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(REPORTS_DIR / "confusion_matrix_grouped.csv")
    errors = test.assign(predicted=pred, confidence=proba.max(1).round(3))[y != pred]
    errors[[TEXT_COL, LABEL_COL, "intent", "predicted", "confidence", "flags"]].sort_values(
        "confidence", ascending=False).to_csv(REPORTS_DIR / "grouped_test_errors.csv", index=False)
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score", "support")}
                 for k, v in report.items() if k in labels}
    return {**metrics(y, pred, labels), "per_class": per_class,
            "confusion_matrix": {"labels": labels, "matrix": cm.tolist()}, "n_errors": int((y != pred).sum())}


def evaluate_random_diagnostic(labels) -> dict:
    raw = load_raw()  # raw rows, duplicates included, as a naive split would use
    tr, te = train_test_split(raw, test_size=0.2, stratify=raw[LABEL_COL], random_state=SEED)
    pipe = build_pipeline().fit(tr[TEXT_COL], tr[LABEL_COL])
    pred = pipe.predict(te[TEXT_COL])
    exact_overlap = te[TEXT_COL].isin(set(tr[TEXT_COL])).mean()
    return {"label": "DIAGNOSTIC ONLY - random stratified split on raw rows; duplicates/near-duplicates "
                     "leak across train/test, so this is potentially optimistic. Not the headline result.",
            **metrics(te[LABEL_COL].values, pred, labels),
            "test_rows_with_exact_text_in_train": round(float(exact_overlap), 4)}


def evaluate_custom(model) -> dict:
    df = pd.read_csv(CUSTOM_PATH, keep_default_na=False)
    proba = model.predict_proba(df["text"])
    order = np.argsort(-proba, axis=1)
    rows = []
    for i, r in df.iterrows():
        pred = model.classes_[order[i, 0]]
        ok_lenient = {r.expected, *filter(None, r.acceptable.split("|"))}
        rows.append({
            "text": r.text, "expected": r.expected, "acceptable": r.acceptable, "predicted": pred,
            "confidence": round(float(proba[i, order[i, 0]]), 3),
            "top3": [(model.classes_[j], round(float(proba[i, j]), 3)) for j in order[i, :3]],
            "strict_correct": pred == r.expected, "lenient_correct": pred in ok_lenient, "note": r.note,
        })
    out = pd.DataFrame(rows)
    return {"n": len(out), "file_sha256": hashlib.sha256(CUSTOM_PATH.read_bytes()).hexdigest(),
            "n_with_accepted_alternatives": int((df["acceptable"] != "").sum()),
            "strict_accuracy": round(float(out.strict_correct.mean()), 4),
            "lenient_accuracy": round(float(out.lenient_correct.mean()), 4),
            "note": "strict = expected label only; lenient = expected or a listed acceptable alternative",
            "predictions": rows}


def main() -> None:
    model = load_model()
    split = load_split()
    REPORTS_DIR.mkdir(exist_ok=True)
    results = {
        "grouped_test": evaluate_grouped(model, split.test),
        "custom_messages": evaluate_custom(model),
        "random_split_diagnostic": evaluate_random_diagnostic(list(model.classes_)),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    g = results["grouped_test"]
    print(f"GROUPED TEST  n={g['n']} acc={g['accuracy']} macroF1={g['macro_f1']} weightedF1={g['weighted_f1']} errors={g['n_errors']}")
    print(f"{'class':14s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>5s}")
    for k, v in g["per_class"].items():
        print(f"{k:14s} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1-score']:6.3f} {int(v['support']):5d}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(g["confusion_matrix"]["matrix"], index=g["confusion_matrix"]["labels"],
                       columns=[label[:4] for label in g["confusion_matrix"]["labels"]]).to_string())
    d = results["random_split_diagnostic"]
    print(f"\nRANDOM SPLIT (diagnostic, potentially optimistic) acc={d['accuracy']} macroF1={d['macro_f1']} "
          f"weightedF1={d['weighted_f1']} exact-text-overlap={d['test_rows_with_exact_text_in_train']}")
    c = results["custom_messages"]
    print(f"\nCUSTOM MESSAGES n={c['n']} strict={c['strict_accuracy']} lenient={c['lenient_accuracy']}")
    for p in c["predictions"]:
        mark = "OK " if p["strict_correct"] else ("~  " if p["lenient_correct"] else "XX ")
        print(f"  {mark} exp={p['expected']:12s} pred={p['predicted']:12s} conf={p['confidence']:.2f} top3={p['top3']} | {p['text']}")
    print(f"\nSaved reports to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
