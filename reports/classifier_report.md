# Ticket category classifier: data, method and results

The classifier predicts one of 11 categories from the ticket message. Every number below comes from the scripts in this repository; the raw outputs are in `reports/`.

## 1. Data inspection: problems found and what was done

The dataset is the Bitext customer-support dataset, pinned to revision `430d1a8` (file SHA-256 checked in `ml/dataset.py`). It has 26,872 rows; the columns are `flags`, `instruction`, `category`, `intent` and `response`. The full inspection output is in `reports/dataset_inspection.txt` (script: `scripts/inspect_dataset.py`).

| Problem | Finding | What I did |
|---|---|---|
| Exact duplicates | 2,237 repeated `instruction` texts; after normalising (lowercase, unified `{{placeholders}}`, no punctuation), 3,046 duplicate rows | Removed, leaving 23,826 unique texts. Safe because no duplicate has conflicting labels. |
| Near-duplicates | 18.1% of unique texts have a TF-IDF cosine neighbour ≥ 0.9, and 46.1% one ≥ 0.8. The data is template-generated, with many paraphrases differing by one word or a typo. | Linked texts with cosine ≥ 0.9 into groups (21,192 groups, the largest with 20 texts); kept each group entirely on one side of the split. |
| Contradictory labels | None: no identical or near-identical (≥ 0.9) text has two categories | Added a check (`deduplicate` fails if any normalised text has two labels) |
| Doubtful or surprising labels | **Cancelling an order is ORDER**, while **CANCEL only means the cancellation fee**. "Where is my order" is ORDER, but "when will it arrive" is DELIVERY. Pairs like "error with **signup**" (ACCOUNT) vs "error with **payment**" (PAYMENT) differ by one word. | Kept the dataset's labels (no relabelling), but used these rules when labelling the custom messages, and flagged genuinely ambiguous messages with accepted alternatives |
| Documentation mismatch | The dataset card lists category names (`CANCELLATION_FEE`, `NEWSLETTER`, `SHIPPING_ADDRESS`) and intents that differ from the data (`CANCEL`, `SUBSCRIPTION`, `SHIPPING`, 27 intents) | Trusted the data |
| Class imbalance | ACCOUNT is 22.3% of rows and CANCEL 3.5% (6.3 : 1) | Used class-balanced logistic regression, with macro-F1 as the main metric |

## 2. Why the test score does not leak

- **Split:** `StratifiedGroupKFold` (5 folds, seed 42), with fold 0 as the test set. That gives 19,061 training and 4,765 test rows, with category proportions within 0.01 points of each other. The assignment is saved in `data/split.csv`.
- **Built from text, not labels:** the groups are formed from text similarity only; the category is used only to balance the folds.
- **Checked by `scripts/verify_split.py`** (`reports/split_verification.json`):
  - no group crosses train/test;
  - all 11 categories are on both sides;
  - every member of a multi-text group has a ≥ 0.9 neighbour inside its group;
  - no single-text group has a ≥ 0.9 neighbour anywhere;
  - a fresh recomputation reproduces the split exactly.
- **Effect on leakage:** a naive random split puts the exact (normalised) text of 14.5% of test rows into the training data, and a ≥ 0.9 neighbour for 27.5%. With the grouped split both are 0%.
- **What remains:** 34.5% of test texts still have a training neighbour at ≥ 0.8. That paraphrase overlap is inherent to template-generated data.
- **Model choice was fixed in advance:** TF-IDF on words (1–2 word phrases) and characters (2–5), plus logistic regression with C = 10 and balanced class weights. Fitted on the training rows only; parameters were not tuned on the test set.

## 3. Results

| Evaluation | n | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| **Grouped test split (headline)** | 4,765 | 0.9996 | 0.9995 | 0.9996 |
| Random split on raw rows (diagnostic only) | 5,375 | 0.9994 | 0.9995 | 0.9994 |
| Custom real-style messages, expected label only | 20 | **0.80** | – | – |
| Custom real-style messages, with accepted alternatives | 20 | **0.85** | – | – |

- **The Bitext score doesn't measure real-world accuracy.** The random split scores no higher than the grouped one, even though 11.2% of its test rows appear word-for-word in its training data. The synthetic task is so easy that leakage makes no measurable difference, so the near-perfect Bitext score says little about real tickets.
- **The 20 custom messages** (`data/custom_messages.csv`, labelled before the model was run) are the more realistic estimate. With only 20, one message is worth 5 points.
- **Origin of the custom messages:** they were drafted with an AI assistant, not independently written by a person or taken from real customers. The same tool also built the model, so they are not a fully independent test. A larger set of independently written messages is a next step.
- **Raw outputs:** per-class scores are in `reports/metrics.json` and the confusion matrix in `reports/confusion_matrix_grouped.csv`.

## 4. Five wrong predictions explained

Examples 1–4 are the four strict errors from the 20 custom messages. Example 4 counts as correct under lenient scoring, which is why lenient accuracy is 85% rather than 80%. Example 5 is one of the 2 errors from the grouped test set. The explanations come from the model's per-feature contributions (TF-IDF value × class weight).

1. **"just wanted to say the support agent was super helpful, thanks!"** (custom message) was predicted **CONTACT** (0.95); expected FEEDBACK.
   - The words "agent" and "support" are strong CONTACT signals, and the FEEDBACK score was negative.
   - Bitext's FEEDBACK examples are requests *to leave* a review ("I want to write a review"), not actual praise, so a real compliment is unlike anything in training.
   - It's the most dangerous kind of error: confidently wrong.
2. **"my card got declined at checkout but the money still left my bank account"** (custom message) was predicted **ACCOUNT** (0.75); expected PAYMENT (REFUND accepted).
   - The word "account" (here meaning a *bank* account) is a strong ACCOUNT signal; PAYMENT scored almost nothing despite "card" and "declined".
   - The model reads keywords, not meaning.
3. **"I was charged twice for the same purchase on my visa. Please fix this asap"** (custom message) was predicted **ORDER** (0.40); expected PAYMENT (REFUND accepted).
   - Bitext uses "purchase" as a synonym for order, so it pushes strongly toward ORDER.
   - "charged twice" and "visa" are rare or absent in training and contribute almost nothing. The low confidence reflects this.
4. **"hey my order #48213 still hasnt shipped, its been 9 days?? whats going on"** (custom message) was predicted **DELIVERY** (0.60); expected ORDER, with DELIVERY accepted.
   - Character n-grams from "shipped" and the word "going" push toward DELIVERY.
   - This is mainly a labelling-convention problem: the dataset puts "where is my order" under ORDER and "when will it arrive" under DELIVERY, and a person could reasonably choose either.
5. **"can I ordrr from {{Delivery City}}?"** (grouped test set) was predicted **INVOICE** (0.71); expected DELIVERY.
   - Normalisation replaces every `{{placeholder}}` with one generic token, which removes the "Delivery City" hint. The typo "ordrr" removes "order".
   - What remains ("can I … from <placeholder>") happens to lean towards INVOICE in training.
   - Keeping placeholder types would fix this Bitext example, but real tickets don't contain placeholders. In a train-only cross-validation comparison (`scripts/compare_placeholders.py`, `reports/placeholder_comparison.json`), all variants were within noise, so the simpler normalisation was kept.

## 5. Limitations

- Training data is synthetic and template-generated, so real-world accuracy will be lower than the Bitext test score.
- The 0.9 near-duplicate threshold was chosen by inspecting examples, not tuned.
- The custom set is small: 20 messages, so one message is worth 5 points of accuracy.
- Confidence scores are not calibrated: error 1 was wrong at 0.95.
