# Smart Support Desk

A small support-desk web application with built-in machine learning:

- **Customers** send a support message through a simple web form.
- **Every ticket is categorised automatically** by a text classifier trained on the public [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
- **An AI-drafted reply**, grounded only in a small FAQ, is prepared for the support team, showing which FAQ entries it relied on.
- **The team** reviews tickets behind a login, filters them by category, and updates their status and priority.

## Features

- **Customer contact form:** name, email and message, with validation and clear error messages.
- **Automatic categorisation** into 11 categories (ORDER, REFUND, DELIVERY, PAYMENT, ACCOUNT, …), with the model's confidence. The model is loaded once at startup and runs on every new ticket.
- **AI reply drafts:**
  - The best-matching FAQ entries are retrieved, and an OpenAI model writes a draft that may only use facts from those entries.
  - The FAQ entries used are listed with the draft.
  - If no FAQ entry fits, the draft says so instead of inventing an answer, and no LLM call is made.
  - Instructions inside a ticket ("ignore your rules…") are treated as customer text, not commands.
- **Team dashboard** (login required):
  - a ticket list with category, priority and status, filterable by category;
  - a ticket page with the draft and its sources;
  - status (`open`, `in_progress`, `resolved`) and priority (`low`, `normal`, `high`) controls.
- **REST API** for creating, listing, reading and updating tickets.
- **Graceful degradation:** without an OpenAI key the app works normally and only drafts are disabled. If the OpenAI API fails, the ticket is still saved and the error is shown.

## Tech stack

- **Backend:** Python 3.12, FastAPI, SQLAlchemy, SQLite
- **Frontend:** server-rendered Jinja2 templates with Bootstrap 5 (loaded from a CDN)
- **ML:** scikit-learn (TF-IDF + logistic regression classifier; TF-IDF FAQ retrieval)
- **LLM:** OpenAI API (model configurable)
- **Tooling:** uv, pytest, Docker Compose

## Prerequisites

- **Docker:** Docker with Compose v2.
- **Or local:** Python 3.12 and [uv](https://docs.astral.sh/uv/).
- **Internet access** on the first run, to download the dataset (about 19 MB, a pinned revision from Hugging Face). Pages also load Bootstrap from a CDN.
- **Optional:** an OpenAI API key, for AI reply drafts.

## Configuration

Copy the example file and fill in the values:

```bash
cp .env.example .env
```

`.env` is gitignored. Never commit real keys.

| Variable | Required | Description |
|---|---|---|
| `TEAM_PASSWORD` | yes, for the team pages | Password for the team login. Team pages and the team API are unavailable until it is set. |
| `TEAM_USERNAME` | no (default `team`) | Username for the team login |
| `OPENAI_API_KEY` | for AI drafts | OpenAI API key |
| `OPENAI_MODEL` | for AI drafts | Chat model name, e.g. `gpt-4o`. Drafts are enabled only when both OpenAI variables are set. |
| `OPENAI_BASE_URL` | no | Alternative OpenAI-compatible endpoint (untested) |
| `DATABASE_URL` | no | SQLAlchemy URL; default is `support_desk.db` in the project folder |
| `MODEL_PATH`, `FAQ_PATH` | no | Override the classifier file (`models/category_classifier.joblib`) or the FAQ file (`data/faq.json`) |
| `HOST`, `PORT` | no | Bind address for local runs (default `127.0.0.1:8000`) |

Only the ticket **message** is sent to OpenAI, never the customer's name or email.

## Running with Docker Compose

```bash
cp .env.example .env        # set TEAM_PASSWORD; optionally OPENAI_API_KEY and OPENAI_MODEL
docker compose up --build
```

Then open <http://localhost:8000>.

- **Build:** the image downloads the dataset and trains the classifier during the build, so the first build takes a few minutes.
- **Secrets:** `.env` values are passed in when the container starts, not at build time, so no keys end up in the image.
- **Storage:** tickets are stored in SQLite on the named volume `app-data` and survive restarts. `docker compose down -v` deletes them.
- **Port:** if port 8000 is already in use, choose another host port with `APP_PORT=8010 docker compose up --build`.

## Running locally

```bash
cp .env.example .env        # set TEAM_PASSWORD; optionally OPENAI_API_KEY and OPENAI_MODEL
uv run python -m app
```

- **First run:** uv installs the dependencies. If no trained model exists yet, the dataset is downloaded and the classifier trained (about 15 seconds of training).
- **Startup message:** the command prints the customer and team URLs, and warns if `TEAM_PASSWORD` or the OpenAI settings are missing.
- **Train separately:** `uv run python -m ml.train`.

## Using the application

### Customer page: <http://localhost:8000/>

Fill in name, email and message and press **Send**. You get a ticket number. Categories and AI drafts are internal and are not shown to the customer.

### Team page: <http://localhost:8000/team/tickets>

1. **Log in:** your browser asks for a username and password. Use `TEAM_USERNAME` (default `team`) and `TEAM_PASSWORD`.
2. **Ticket list:** shows the customer, message, predicted category with confidence, priority, status and creation time (UTC). Use the drop-down to filter by category.
3. **Ticket page:** open a ticket to see the full message and contact details, and the **suggested reply** with the FAQ entries it used and their match scores.
4. **Update:** change the **status** or **priority** and press **Save**.

AI drafts are written when a ticket is submitted. Tickets created while drafts were disabled don't get one later.

### REST API

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/tickets` | public | Create a ticket; returns `201 {"id", "status"}` |
| `GET` | `/api/tickets?category=REFUND` | team | List tickets, newest first, optionally filtered |
| `GET` | `/api/tickets/{id}` | team | Ticket details, including the draft and its FAQ sources |
| `PATCH` | `/api/tickets/{id}` | team | Update `status` and/or `priority` |
| `GET` | `/health` | public | Service status and model version |

Team endpoints use HTTP Basic authentication. Invalid input returns `422` with a message for each field, and an unknown ticket returns `404`. Interactive API docs are available at <http://localhost:8000/docs>.

```bash
curl -X POST http://localhost:8000/api/tickets \
  -H "Content-Type: application/json" \
  -d '{"name": "Ana", "email": "ana@example.com", "message": "How long does express delivery take?"}'

curl -u team:YOUR_TEAM_PASSWORD http://localhost:8000/api/tickets/1

curl -u team:YOUR_TEAM_PASSWORD -X PATCH http://localhost:8000/api/tickets/1 \
  -H "Content-Type: application/json" -d '{"status": "in_progress", "priority": "high"}'
```

## Running the tests

```bash
uv run pytest
```

- **No API key or network needed:** the tests never call OpenAI, use fake classifier and reply components where needed, and give each test its own temporary database.
- **Some tests skip** if their inputs are missing: the trained model, or the downloaded dataset. Run `uv run python -m ml.train` first for the full suite.

## Evaluations

### Classifier

```bash
uv run python -m ml.evaluate
```

The data findings, why the split doesn't leak, the results, and five wrong predictions explained are summarised in [`reports/classifier_report.md`](reports/classifier_report.md).

The command needs a trained model, and writes `reports/metrics.json`, `reports/confusion_matrix_grouped.csv` and `reports/grouped_test_errors.csv`. It reports:
- accuracy, macro-F1 and per-class scores on a held-out test split where near-duplicate texts never cross between training and test data;
- a random-split comparison, which is likely optimistic and labelled as a diagnostic only;
- scores on 20 AI-drafted test messages in `data/custom_messages.csv`, labelled before the model was run.

Related scripts:

```bash
uv run python -m ml.dataset            # rebuild the train/test split and data/split.csv (about 2 min)
uv run python scripts/verify_split.py  # check the split: no group crosses train/test, reproducible
```

### Reply drafts

```bash
uv run python scripts/eval_replies.py
```

Requires `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`.
- **Cases:** the 10 test tickets in `data/reply_eval_cases.csv`, including deliberately hard ones such as a prompt-injection attempt and questions the FAQ doesn't cover. Each has a description of what a good draft should do.
- **Output:** `reports/reply_eval.csv`, for manual good/bad grading.
- **Graded results** and how reply quality would be measured at scale: [`reports/reply_eval.md`](reports/reply_eval.md).

## Project structure

```
app/         web application: routes, ticket logic, database models, FAQ retrieval, reply drafts, templates
ml/          dataset loading and split, classifier training and evaluation
data/        FAQ, train/test split, test messages and reply-evaluation cases
scripts/     dataset inspection, split verification, reply evaluation
reports/     evaluation outputs
tests/       pytest suite
```

## Limitations

- **Training data is synthetic.** The Bitext dataset is template-generated, so held-out test scores are near-perfect and overstate real-world accuracy. The 20 AI-drafted test messages give a more realistic but small-sample estimate; they were not independently written by real customers.
- **The FAQ is fictional and small:** 15 entries for a made-up shop. Retrieval matches words, so heavy typos, synonyms or other languages can miss a relevant entry. The draft then falls back to "not covered".
- **AI drafts are suggestions for a human to review.** They can still be wrong or leave out a caveat, and they need an OpenAI key.
- **Simple authentication:** one shared team account via HTTP Basic, and no CSRF protection. To log out, close the browser. Not intended for production as is.
- **SQLite without migrations:** if the database schema changes, delete the local `support_desk.db` (or the Docker volume) to recreate it.

More background on design choices and evaluation results is in [NOTES.md](NOTES.md).
