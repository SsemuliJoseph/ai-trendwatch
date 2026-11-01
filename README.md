# AI TrendWatch

AI TrendWatch is a local Python pipeline that watches AI and technology news, filters and scores items with Gemini, stores them in SQLite, and delivers the most relevant stories to a WhatsApp group via a local OpenWA gateway.

The project is deliberately safety-first: everything defaults to dry-run mode, delivery is capped per run, the destination must be a WhatsApp group ID, and each item is tracked through a SQLite lifecycle so duplicate or interrupted sends are avoided.

## What it does

- Collects AI/tech items from RSS feeds and arXiv
- Optionally reads GitHub data when a token is configured
- Validates and deduplicates incoming articles
- Stores everything in SQLite so runs are repeatable and bounded
- Sends items through Gemini for relevance scoring and concise summaries
- Triage-filtering by score threshold
- Formats the top items into WhatsApp-friendly text, with optional images
- Delivers them through the local OpenWA WhatsApp gateway
- Schedules repeated runs with APScheduler
- Protects against overlapping runs with a lock file

## High-level pipeline

```text
RSS / arXiv / GitHub
        ↓
collect → validate → deduplicate → store
        ↓
Gemini analysis (score + summary)
        ↓
triage by relevance threshold
        ↓
select top items for delivery
        ↓
OpenWA → WhatsApp group
```

## Architecture overview

The codebase is split into clear layers:

| Path              | Responsibility                                                  |
| ----------------- | --------------------------------------------------------------- |
| `main.py`         | Composition root and end-to-end runner                          |
| `app/config/`     | `.env` loading and strict runtime settings                      |
| `app/models/`     | Pure data model (`TrendItem`)                                   |
| `app/collectors/` | RSS and arXiv collection logic                                  |
| `app/processing/` | Validation and deduplication rules                              |
| `app/database/`   | SQLite repository and item lifecycle                            |
| `app/ai/`         | Gemini analysis service                                         |
| `app/agents/`     | LangGraph orchestration workflow                                |
| `app/messaging/`  | Message formatting and OpenWA delivery client                   |
| `app/media/`      | Generated image cards and preview image handling                |
| `app/utils/`      | Logging, retry, and run locking                                 |
| `scripts/`        | OpenWA setup and scheduler utilities                            |
| `tests/`          | Pytest coverage for collectors, pipeline, and delivery behavior |

## Requirements

- Python 3.10+
- Git
- Docker and Docker Compose for the local OpenWA gateway
- A Google AI Studio API key for Gemini
- An actual WhatsApp group to target (never an individual chat)

## Quick start

### 1) Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare configuration

Copy the example env file and fill in real values:

```bash
cp .env.example .env
```

Required minimum values for a real run:

- `GEMINI_API_KEY`
- `OPENWA_API_KEY`
- `OPENWA_SESSION_ID`
- `WHATSAPP_GROUP_ID`

The default runtime configuration is intentionally safe:

- `WHATSAPP_DRY_RUN=true`
- messages are capped with `MAX_MESSAGES_PER_RUN`
- only a group ID is accepted (`@g.us`)

### 3) Set up the local WhatsApp gateway

This project does not talk directly to WhatsApp. It uses a separate OpenWA service that must be started locally.

```bash
python scripts/setup_openwa.py up
```

Useful commands:

```bash
python scripts/setup_openwa.py status
python scripts/setup_openwa.py key
python scripts/setup_openwa.py sessions
python scripts/setup_openwa.py groups <SESSION_ID>
python scripts/setup_openwa.py down
```

OpenWA is cloned into the workspace under `./openwa` and its own env settings are stored in `openwa/.env` based on `openwa.env.example`.

### 4) Verify and configure OpenWA values

After starting OpenWA, note:

- the admin key using `python scripts/setup_openwa.py key`
- the session UUID using `python scripts/setup_openwa.py sessions`
- the target group ID using `python scripts/setup_openwa.py groups <SESSION_ID>`

Then set these values in `.env`:

```env
GEMINI_API_KEY=...
OPENWA_BASE_URL=http://127.0.0.1:2785
OPENWA_API_KEY=...
OPENWA_SESSION_ID=...
WHATSAPP_GROUP_ID=123456789012-123456789@g.us
WHATSAPP_DRY_RUN=true
```

### 5) Run a dry-run preview

This is the safest way to test the full pipeline without sending messages:

```bash
python main.py
```

Equivalent force-dry-run style:

```bash
python main.py --dry-run
```

The project logs the run and prints a summary. In dry-run mode, the pipeline still calls Gemini and inserts analyzed results into SQLite, but it does not send any WhatsApp messages.

### 6) Go live only when ready

Once you have validated the draft content and are confident about the recipient, switch off dry-run in `.env`:

```env
WHATSAPP_DRY_RUN=false
```

Then run:

```bash
python main.py
```

> Safety rule: the app refuses to send live unless `WHATSAPP_GROUP_ID` and `OPENWA_API_KEY` are set and the session is present. It also blocks individual chat IDs.

## Scheduling

The project includes a scheduler that runs the pipeline repeatedly:

```bash
python scripts/scheduler.py --once --dry-run
python scripts/scheduler.py
python scripts/scheduler.py --interval 5
python scripts/scheduler.py --skip-first-run
```

Behavior:

- runs immediately by default if `SCHEDULE_RUN_IMMEDIATELY=true`
- repeats every `SCHEDULE_INTERVAL_MINUTES`
- prevents overlapping runs with a lock file
- supports dry-run mode for scheduled execution too

The scheduler is built on APScheduler and is designed to keep one pipeline instance active at a time.

## Configuration reference

The project reads `.env` through `Settings` and validates values centrally in `app/config/settings.py`.

### Core environment variables

```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_REQUESTS_PER_MINUTE=10

DATABASE_PATH=data/trendwatch.db
MAX_ITEMS_PER_SOURCE=20
REQUEST_TIMEOUT_SECONDS=15
GITHUB_TOKEN=

OPENWA_BASE_URL=http://127.0.0.1:2785
OPENWA_API_KEY=...
OPENWA_SESSION_ID=...
WHATSAPP_GROUP_ID=...
WHATSAPP_DRY_RUN=true
MAX_MESSAGES_PER_RUN=5
MESSAGE_DELAY_SECONDS=8
MESSAGE_IMAGES=true

MIN_RELEVANCE_SCORE=6
MAX_ANALYZE_PER_RUN=20

RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY_SECONDS=2
RUN_LOCK_STALE_MINUTES=30

LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_DIR=logs
LOG_MAX_MB=2
LOG_BACKUP_COUNT=5
SCHEDULE_INTERVAL_MINUTES=60
SCHEDULE_RUN_IMMEDIATELY=true
```

### Important policy rules

- `WHATSAPP_GROUP_ID` must end with `@g.us`
- sending to a private chat is rejected by configuration validation
- `WHATSAPP_DRY_RUN=false` requires a group id, OpenWA key, and session id
- API keys are stored as `SecretStr` and redacted in logs
- a stale run lock is treated as crashed and can be reclaimed safely

## Data model and storage

The app persists each seen source item in SQLite, not just those that are sent.

Key statuses:

- `new` — collected and waiting for analysis
- `analyzed` — scored and summarized by Gemini
- `sending` — claimed before send attempt
- `sent` — successfully delivered
- `rejected` — dropped because irrelevant, invalid, or not safe to resend

This design prevents duplicate posting and helps recover interrupted runs.

## Delivery lifecycle and safety

The sending layer is intentionally strict:

1. Format item for WhatsApp
2. Claim item in SQLite before sending
3. Send through OpenWA
4. Record success or failure
5. Mark rejected items so they are never retried unexpectedly
6. Treat unknown outcomes as not safe to re-send

This means a message that may have gone through but cannot be confirmed is not re-posted; instead, it is marked rejected and handled as a one-time-at-most-once event.

## Tests

The repository includes a pytest suite covering collectors, validation, Gemini integration, delivery safety, scheduler behavior, and setup scripts.

Run the full suite:

```bash
pytest -q
```

Run a subset for faster verification:

```bash
pytest tests/test_arxiv_collector.py -q
pytest tests/test_rss_collector.py -q
pytest tests/test_pipeline_delivery.py -q
pytest tests/test_setup_openwa.py -q
```

## Example command flow

A typical local workflow looks like this:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill GEMINI_API_KEY and OpenWA values
python scripts/setup_openwa.py up
python main.py --dry-run
# review the summary
# once happy, set WHATSAPP_DRY_RUN=false
python main.py
```

## Troubleshooting

### Gemini errors

- ensure `GEMINI_API_KEY` is valid and set in `.env`
- check the model name and quota limits in Google AI Studio
- verify `max_requests_per_minute` is low enough for your free-tier plan

### OpenWA not reachable

- confirm the Docker gateway is started
- check `OPENWA_BASE_URL` is `http://127.0.0.1:2785`
- use `python scripts/setup_openwa.py status`
- inspect `docker compose` logs in the OpenWA checkout if needed

### Missing or invalid group ID

- use `python scripts/setup_openwa.py groups <SESSION_ID>`
- the ID must end with `@g.us`
- do not use a personal chat ID

### Scheduler is skipping runs

- make sure no other `python main.py` or scheduler process already holds the run lock
- inspect the logs in `logs/`
- the project deliberately avoids overlapping runs to protect quota and prevent duplicate sends

## Security notes

- `.env` is gitignored and should never be committed
- secrets are redacted in logs
- the app is designed to minimize exposure and avoid accidental mass messaging
- the OpenWA gateway is configured to run locally and not expose its API to the wider network by default

## Project status

This project is set up as a local AI news delivery pipeline for controlled, low-risk message distribution. It emphasizes correctness, safety, and idempotency over speed or broad deployment complexity.

If you want to run it in production-like mode, keep the following in place:

- valid Gemini API quota and limits
- stable local OpenWA session
- verified WhatsApp group target
- dry-run validation before switching to live mode
- monitoring via the generated log files

## License

This repository does not currently include a root-level license file. If you plan to distribute or publish it, add a license before public release and make sure your legal team or project owner approves the chosen terms.

---

This README is intended to reflect the actual current project behavior in this repository. If you want a shorter operational version for quick onboarding or a more developer-focused architecture guide, I can also prepare those as companion docs.
