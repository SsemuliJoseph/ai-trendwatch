"""
FILE: init_structure.py
MODULE 1 — Project scaffolding.

Run ONCE from the project root:   python init_structure.py
Idempotent: never overwrites an existing file, so re-running is safe.

WHY a script instead of manual mkdir?
  * Reproducible: anyone (or CI) rebuilds the identical tree.
  * Each package's __init__.py records its dependency rules, so the
    architecture is documented next to the code it constrains.

DEPENDENCY RULES (enforced by convention, verified by review/tests later):

    agents  ──►  collectors, processing, ai, database, messaging   (orchestrator)
    processing ──► models, database
    database   ──► models
    collectors ──► models
    ai         ──► models, config
    messaging  ──► models, config
    everything ──► config, utils, models   (leaf layers: import nothing upward)

    FORBIDDEN:  collectors ✗ ai/messaging | ai ✗ database/messaging
                messaging ✗ ai/database   | only `agents` sees every layer
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# package path -> module docstring written into that package's __init__.py
PACKAGES: dict[str, str] = {
    "app": "AI TrendWatch application root package.",
    "app/config": (
        "Configuration layer: loads .env into a typed Settings object.\n"
        "Leaf layer — imports NOTHING from other app packages."
    ),
    "app/models": (
        "Domain schemas (TrendItem). Pure data, zero I/O.\n"
        "Leaf layer — imports NOTHING from other app packages."
    ),
    "app/collectors": (
        "Deterministic data collection (RSS, arXiv/GitHub) -> TrendItem.\n"
        "ALLOWED: models.  FORBIDDEN: ai, messaging, database, agents."
    ),
    "app/processing": (
        "Deterministic validation & deduplication (no LLM involved).\n"
        "ALLOWED: models, database.  FORBIDDEN: ai, messaging, collectors."
    ),
    "app/database": (
        "SQLite persistence via repository pattern.\n"
        "ALLOWED: models.  FORBIDDEN: ai, messaging, collectors."
    ),
    "app/ai": (
        "Gemini integration: relevance scoring + summarization (the only 'LLM' code).\n"
        "ALLOWED: models, config.  FORBIDDEN: database, messaging\n"
        "(must know nothing about WhatsApp or SQLite)."
    ),
    "app/agents": (
        "LangGraph orchestration ONLY: nodes, state, edges.\n"
        "Wires the other layers together but embeds no core logic.\n"
        "The single package permitted to import every other layer."
    ),
    "app/messaging": (
        "WhatsApp message formatter + OpenWA HTTP adapter.\n"
        "ALLOWED: models, config.  FORBIDDEN: ai, database, collectors.\n"
        "Talks to OpenWA exclusively over HTTP — no Node/WhatsApp internals."
    ),
    "app/utils": (
        "Cross-cutting helpers (logging, retry, rate limiting).\n"
        "Leaf layer — may import config only."
    ),
    "tests": "Unit and integration tests (pytest).",
}

# Directories that must exist but hold runtime artifacts (contents git-ignored).
RUNTIME_DIRS = ["data", "logs"]


def write_if_missing(path: Path, content: str) -> bool:
    """Create `path` with `content` only if absent. Returns True if created."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    created, skipped = [], []

    for pkg, doc in PACKAGES.items():
        init_file = ROOT / pkg / "__init__.py"
        target = created if write_if_missing(init_file, f'"""{doc}\n"""\n') else skipped
        target.append(init_file.relative_to(ROOT).as_posix())

    for d in RUNTIME_DIRS:
        keep = ROOT / d / ".gitkeep"  # lets git track an otherwise-empty dir
        target = created if write_if_missing(keep, "") else skipped
        target.append(keep.relative_to(ROOT).as_posix())

    print(f"Created ({len(created)}):")
    for p in created:
        print(f"  + {p}")
    if skipped:
        print(f"Already present, untouched ({len(skipped)}):")
        for p in skipped:
            print(f"  = {p}")


if __name__ == "__main__":
    main()
