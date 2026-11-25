"""B3: generate reply drafts for the 10 fixed evaluation tickets and write a grading sheet.

Run (needs OPENAI_API_KEY and OPENAI_MODEL in <repo>/.env):
    uv run python scripts/eval_replies.py
Writes reports/reply_eval.csv with empty `grade` (good/bad) and `reason` columns,
to be filled in by a human. The automatic `numbers_not_in_cited_faq` column is
only a hint for the grader: it cannot prove a draft is correct.
"""
import csv
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

from app.config import PROJECT_ROOT, Settings
from app.replies import build_drafter

CASES = PROJECT_ROOT / "data" / "reply_eval_cases.csv"
OUT = PROJECT_ROOT / "reports" / "reply_eval.csv"
_NUM = re.compile(r"\d+(?:[.:]\d+)?")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings.from_env()
    drafter = build_drafter(settings)
    if drafter is None:
        raise SystemExit("Set OPENAI_API_KEY and OPENAI_MODEL in .env to run the reply evaluation.")
    answers = {e.id: e.answer for e in drafter.retriever.entries}
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    with CASES.open(encoding="utf-8") as f:
        for case in csv.DictReader(f):
            hits = drafter.retriever.search(case["ticket"])
            try:
                draft = drafter.draft(case["ticket"])
                status, text, used, error = draft.status, draft.text, [s["id"] for s in draft.sources], ""
            except Exception as exc:  # recorded, not hidden
                status, text, used, error = "error", "", [], f"{type(exc).__name__}: {exc}"
            # Compare whole numbers, not substrings ("99" must not match "$4.99").
            cited_numbers = set(_NUM.findall(" ".join(answers[i] for i in used)))
            unsupported = sorted(set(_NUM.findall(text)) - cited_numbers)
            rows.append({**case, "retrieved": " ".join(f"{h.entry.id}:{h.score}" for h in hits),
                         "status": status, "used_faq_ids": " ".join(used), "draft": text, "error": error,
                         "numbers_not_in_cited_faq": " ".join(unsupported),
                         "model": settings.openai_model, "run_at": run_at, "grade": "", "reason": ""})
            print(f"{case['case_id']} [{status}] used={used} retrieved={rows[-1]['retrieved']}\n   {text or error}\n")
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT}. Fill in `grade` (good/bad) and `reason` for each row.")


if __name__ == "__main__":
    main()
