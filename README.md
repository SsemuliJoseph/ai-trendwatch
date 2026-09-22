# AI TrendWatch

Agent-based real-time AI & technology update system (academic project).

```
RSS/APIs → collect → validate → dedup (SQLite) → LangGraph → Gemini → format → OpenWA → WhatsApp group
```

## Stack (100% free tier)
Python · Google Gemini API (free tier) · LangGraph · SQLite · RSS/arXiv/GitHub · OpenWA (local) · Git

## Layout
| Path | Responsibility |
|---|---|
| `app/config/` | `.env` → typed settings |
| `app/models/` | `TrendItem` schema (pure data) |
| `app/collectors/` | RSS / arXiv / GitHub fetching |
| `app/processing/` | validation + deduplication |
| `app/database/` | SQLite repository |
| `app/ai/` | Gemini relevance + summary |
| `app/agents/` | LangGraph workflow (orchestration only) |
| `app/messaging/` | formatter + OpenWA HTTP adapter |
| `app/utils/` | logging, retry, rate limiting |
| `tests/` | pytest suite |

Dependency rules are documented in each package's `__init__.py`.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python init_structure.py
cp .env.example .env             # then add your GEMINI_API_KEY
```

## Safety
Dedicated test number only · one class group · rate-limited · `WHATSAPP_DRY_RUN=true` until the controlled delivery test.
