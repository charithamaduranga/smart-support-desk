# NOTES

Working notes for the Smart Support Desk assessment: decisions, results,
limitations, and a log of mistakes made by the AI coding tool.

## How I worked

I used Claude Code as my AI coding assistant and worked in small, reviewed
steps: dataset inspection → classifier → classifier review → web app
(Part A).
I did not move to the next step until I had read the previous one's output.
I asked for real numbers from scripts I can re-run, not summaries, so every
result below is backed by a file in `reports/`.

## Key decisions

- **Pinned dataset.** Bitext revision `430d1a8`. The CSV's SHA-256 is
  recorded in `ml/dataset.py` and matches the file I inspected.
- **Leakage-aware split.** I remove exact and normalised duplicates (3,046
  rows) and link near-duplicates (TF-IDF cosine ≥ 0.9) into groups. Then I
  use `StratifiedGroupKFold` (5 folds, seed 42), so no group is on both
  sides. The split is cached in `data/split.csv` and re-checked by
  `scripts/verify_split.py`.
- **Simple model.** TF-IDF on words and characters plus LogisticRegression.
  I fixed the parameters up front and did no tuning on the test set.
- **Placeholder normalisation.** Before changing anything, I compared three
  variants with grouped cross-validation on the training data only. The
  differences were within noise, so I kept the original.
- **No "needs review" threshold.** The model's confidence scores are not
  calibrated, so I show the category and confidence only and don't invent a
  cut-off.
- **SQLite without migrations.** The Part B reply columns (`suggested_reply`,
  `reply_error`) are in the table from the start, so I don't need a schema
  change later.
- **Classifier loaded once.** The model is loaded at app startup and shared
  by every request; it is never reloaded per prediction. I made this a
  tested requirement, not an assumption.
- **Swappable parts for testing.** `create_app(settings, classifier)` lets
  tests pass in a fake classifier and a separate SQLite file each, so the
  app tests don't need the trained model, the network or an API key.
- **Plain, safe web flow.** Server-rendered pages, server-side validation
  (1–5000 characters), a redirect after each submission so a refresh can't
  resubmit, and Jinja's automatic HTML escaping.

## Results so far

| Evaluation | n | Accuracy | Macro-F1 |
|---|---|---|---|
| Grouped test split | 4,765 | 0.9996 | 0.9995 |
| Random split (diagnostic only) | 5,375 | 0.9994 | 0.9995 |
| My 20 custom messages: expected label only / accepted alternatives too | 20 | 0.80 / 0.85 | – |

The random split did not score higher, even though 11% of its test rows
appear word-for-word in its training data. The synthetic data is so easy
that leakage makes no measurable difference. So the Bitext test score can't
tell models apart and won't predict accuracy on real tickets.

The 20 custom messages are a more honest check, but too few to rely on. The
assistant drafted them and their labels before the model was run. I kept
the labels fixed afterwards; the file's hash is recorded in
`reports/metrics.json`.

## Mistakes made by the AI tool

For each one: what went wrong, how it was found, and what I did about it.

1. **Mislabelled analysis output.** The first inspection script printed a
   heading saying it used word *and* character n-grams; the code only used
   word n-grams.
   - *Found:* Claude Code spotted it while re-reading its own output, before
     reporting.
   - *Fix:* corrected the label. The numbers were unaffected; only the
     description was wrong.

2. **Wrong explanation of a class-distribution shift.** In its inspection
   report, the assistant said the grouped split made ORDER less common in
   the test set (12.7% vs 14.8%).
   - *Found:* `ml/dataset.py` later printed the train/test distribution next
     to the deduplicated one, and they matched within 0.01 points. The
     earlier comparison had been against the raw data.
   - *Fix:* corrected the claim. The shift comes from deduplication, not
     grouping.

3. **Unit test with the wrong expectation.** A test assumed `normalize()`
   deletes apostrophes; it actually replaces them with a space.
   - *Found:* the test failed on the first pytest run.
   - *Fix:* I changed the test, not the function. Changing `normalize()`
     would have changed the duplicate counts the agreed split depends on.

4. **Saved model only loaded from the repo root.** The saved pipeline
   depends on `ml.text.normalize`, so loading it anywhere else failed with
   `ModuleNotFoundError: No module named 'ml'`.
   - *Found:* the assistant's error-analysis script crashed with it, and the
     assistant flagged it as a risk for the app.
   - *Fix:* I made portability a required item in my classifier review. The
     project is now an installable package, with a `load_model()` helper and
     a regression test that loads the model from another directory.

5. **Ran a script before fixing the problem it depended on.** The assistant
   started the split verification after diagnosing that same import problem,
   but before fixing it.
   - *Found:* it crashed immediately with the same error.
   - *Fix:* re-ran it after the packaging fix; every check passed.

6. **Test that passed for the wrong reason.** The `created_at` test checked
   the timestamp on the in-memory object before it was saved. SQLite actually
   returns it without a timezone after a reload.
   - *Found:* Claude Code checked a real reload directly and saw the
     timezone was gone.
   - *Fix:* a small column type that stores UTC and returns timezone-aware
     values, and tests that read back in a fresh session.

7. **Line endings would have broken a recorded hash.** Git on Windows
   converts files to Windows line endings on checkout. That would change the
   custom-message file byte-for-byte, so its recorded SHA-256 would no longer
   match.
   - *Found:* from Git's line-ending warnings during the first commit.
   - *Fix:* a `.gitattributes` that keeps Unix line endings; I confirmed the
     committed file still matches the recorded hash.

8. **Proposed an arbitrary review threshold.** The app plan included a
   "needs review" badge below 0.5 confidence.
   - *Found:* I rejected it when reviewing the plan. The probabilities aren't
     calibrated, and on the custom messages 0.5 would have caught only one of
     the four errors; one wrong answer had 0.95 confidence.
   - *Fix:* the app shows category and confidence only.

9. **A check that tested nothing.** To prove the load-once test works, the
   assistant tried to temporarily make `classify()` reload the model on
   every call. Its shell command was broken, so the code was never changed,
   and it reported "2 passed".
   - *Found:* the "mutated" confirmation it was supposed to print never
     appeared, so the assistant re-checked instead of trusting the pass.
   - *Fix:* re-ran it as a script that fails if the code to change isn't
     found. With the reload in place, both load-once tests failed as they
     should; the file was then restored and they pass again.

10. **Fragile test import.** The app tests imported the fake classifier
    directly from `tests/conftest.py`. That only worked by accident of how
    pytest sets up imports, and can load the module twice.
    - *Found:* Claude Code noticed it right after the first app tests passed.
    - *Fix:* the fake is now provided through a pytest fixture.

11. **Unused code in the health check.** `/health` asked for a database
    session but never used it, so it didn't actually check the database.
    - *Found:* Claude Code noticed it in the same review.
    - *Fix:* `/health` now runs `SELECT 1`.

12. **Test status not confirmed.** A combined command showed "57 passed" but
    not whether the real-model test had run or been skipped.
    - *Found:* the expected line for that test was missing from the output.
    - *Fix:* re-ran that test on its own; it ran and passed.

13. **Browser tool left files in the repo.** While taking screenshots of the
    running app, the browser tool wrote logs and images into a new
    `.playwright-mcp/` folder inside the project.
    - *Found:* its first attempt to save outside the repo was refused, and it
      then wrote into the project folder instead.
    - *Fix:* moved the screenshots out, deleted the folder, and added it to
      `.gitignore`. I checked `git status` before committing to make sure no
      databases, screenshots or logs were included.

## What I checked in review

- That the split really is grouped by text, not by category; that all 11
  categories appear on both sides; and that a fresh recomputation gives an
  identical split. All checks passed (`reports/split_verification.json`).
- That the model was not changed without a comparison (the placeholder
  experiment).
- That custom-message labels were not edited to improve accuracy.
- For the app, I asked for tests of:
  - the classifier being loaded once and reused across requests;
  - a separate database for each test;
  - the full flow: submit → classify → save → redirect → detail page;
  - HTML escaping and invalid submissions.
- I also had the app run for real in a browser. A `<script>` typed into a
  ticket shows as plain text.

## Known limitations

- The Bitext test set can't distinguish models; real-world accuracy will be
  lower than the headline score.
- The 0.9 similarity threshold was chosen by inspecting examples, not tuned.
  About a third of test texts still have a close paraphrase in the training
  data.
- The custom set is small (20) and was drafted by the same tool that built
  the model.
- No database migrations. Schema changes during development mean deleting
  `support_desk.db`.
- The saved model loads only inside the project environment (`uv run` /
  `uv sync`).
- No login and no CSRF protection; fine for a demo, not for production.
- An unknown ticket ID returns FastAPI's default JSON 404, not an HTML page.
- Bootstrap loads from a CDN, so the pages need internet to be styled.
- The ticket list has no pagination.
