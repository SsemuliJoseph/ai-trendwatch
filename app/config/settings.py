"""
FILE: app/config/settings.py
MODULE 2 — Configuration management & .env security.

ARCHITECTURE NOTES
  * LEAF layer: imports nothing from other `app.*` packages, so any layer
    may depend on it without creating cycles.
  * SOURCE PRECEDENCE (highest wins):
        real OS environment variables  >  .env file  >  defaults below
    -> CI / Docker can override .env without editing files.
  * FAIL-FAST vs FAIL-LATE (deliberate split):
        - Malformed values (bad int, bad URL, bad group id) fail at load time.
        - Missing *secrets* do NOT fail at load; the component that needs one
          calls `require_*()` at startup. This lets you develop/test collectors
          (Modules 4-7) before a Gemini key exists.
  * SECRETS: stored as `SecretStr`, so print(), repr() and logs show '**********'.
    Only `.get_secret_value()` reveals them — grep for it to audit every leak point.
  * IMMUTABLE (`frozen=True`) + cached singleton: config cannot be mutated
    mid-run by accident, and every module sees identical values.
  * SAFETY GUARDS (encode the project's "no mass messaging" rule in code):
        - WHATSAPP_GROUP_ID must be a group id ('@g.us'); individual chats are rejected.
        - WHATSAPP_DRY_RUN=false is refused unless a group id and OpenWA key exist.

USAGE
    from app.config.settings import get_settings
    settings = get_settings()
    api_key = settings.require_gemini_key()

INSPECT (secrets masked):   python -m app.config.settings
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/config/settings.py -> parents[0]=config, [1]=app, [2]=project root.
# Anchoring to __file__ makes paths independent of the current working directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Values that mean "the user never replaced the template value".
_PLACEHOLDERS = frozenset({"", "your-gemini-api-key-here", "change-me"})


class ConfigError(RuntimeError):
    """Configuration is invalid, or a required setting is missing at point of use."""


def _is_set(secret: SecretStr) -> bool:
    """True if the secret holds a real value (not empty / not a template placeholder)."""
    return secret.get_secret_value().strip().lower() not in _PLACEHOLDERS


class Settings(BaseSettings):
    """All runtime configuration. Field `gemini_api_key` <- env var GEMINI_API_KEY (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown keys in .env are ignored, not fatal
        frozen=True,
    )

    # ---- Google Gemini ----
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash"
    # Ceiling for the rate limiter built in Module 15. Keep below your AI Studio RPM.
    gemini_max_requests_per_minute: int = Field(default=10, ge=1, le=1000)

    # ---- Storage ----
    # Relative paths are anchored to PROJECT_ROOT (see validator below).
    database_path: Path = Path("data/trendwatch.db")

    # ---- Collection ----
    max_items_per_source: int = Field(default=20, ge=1, le=100)
    request_timeout_seconds: int = Field(default=15, ge=1, le=120)
    github_token: SecretStr = SecretStr("")  # optional

    # ---- OpenWA / WhatsApp ----
    openwa_base_url: str = "http://127.0.0.1:2785"  # OpenWA default port; 127.0.0.1 avoids Windows IPv6 surprises
    openwa_api_key: SecretStr = SecretStr("")
    openwa_session_id: str = ""  # the session ID (a UUID) OpenWA returns when the session is created
    whatsapp_group_id: str = ""
    whatsapp_dry_run: bool = True  # SAFE DEFAULT: never send until explicitly disabled
    max_messages_per_run: int = Field(default=5, ge=1, le=20)
    message_delay_seconds: int = Field(default=8, ge=1, le=300)

    # ---- Pipeline policy (Module 9) ----
    min_relevance_score: int = Field(default=6, ge=0, le=10)  # only items scoring >= this may be delivered
    max_analyze_per_run: int = Field(default=20, ge=1, le=200)  # Gemini quota guard (5 items = 1 request)

    # ---- Resilience (Module 15) ----
    retry_max_attempts: int = Field(default=3, ge=1, le=6)  # total tries for idempotent reads (feeds, Gemini); 1 = never retry
    retry_base_delay_seconds: float = Field(default=2.0, ge=0.1, le=60)  # first backoff step; doubles per attempt
    run_lock_stale_minutes: int = Field(default=30, ge=1, le=1440)  # a run lock older than this is treated as crashed

    # ---- Runtime ----
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = True  # false = console only (containers, quick experiments)
    log_dir: Path = Path("logs")  # relative paths are anchored to PROJECT_ROOT (see validator below)
    log_max_mb: int = Field(default=2, ge=1, le=100)  # per file; with the backups this bounds total disk use
    log_backup_count: int = Field(default=5, ge=1, le=50)
    schedule_interval_minutes: int = Field(default=60, ge=1, le=1440)
    schedule_run_immediately: bool = True  # scripts/scheduler.py: run a pass at startup, not just after the first interval

    # ------------------------------------------------------------------
    # Field validators (deterministic normalisation/validation, no I/O)
    # ------------------------------------------------------------------
    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, v: object) -> object:
        # Accept 'info', ' Info ' etc.; Literal then enforces the allowed set.
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("openwa_base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")  # no trailing slash -> f"{base}/path" is always safe
        if not v.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return v

    @field_validator("whatsapp_group_id")
    @classmethod
    def _validate_group_id(cls, v: str) -> str:
        v = v.strip()
        if v and not v.endswith("@g.us"):
            raise ValueError(
                "must be a WhatsApp GROUP id ending in '@g.us' "
                "(sending to individual chats is blocked by design)"
            )
        return v

    @field_validator("database_path", "log_dir")
    @classmethod
    def _anchor_database_path(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _live_sending_requires_target(self) -> "Settings":
        """Refuse to leave dry-run mode without a fully specified, single destination."""
        if not self.whatsapp_dry_run:
            if not self.whatsapp_group_id:
                raise ValueError("WHATSAPP_DRY_RUN=false requires WHATSAPP_GROUP_ID")
            if not _is_set(self.openwa_api_key):
                raise ValueError("WHATSAPP_DRY_RUN=false requires OPENWA_API_KEY")
            if not self.openwa_session_id.strip():
                raise ValueError("WHATSAPP_DRY_RUN=false requires OPENWA_SESSION_ID")
        return self

    # ------------------------------------------------------------------
    # Point-of-use guards: call these where the secret is actually needed
    # ------------------------------------------------------------------
    def require_gemini_key(self) -> str:
        """Return the Gemini key, or raise a clear, actionable ConfigError."""
        if not _is_set(self.gemini_api_key):
            raise ConfigError(
                "GEMINI_API_KEY is missing or still the template placeholder. "
                "Create a key at https://aistudio.google.com/apikey and put it in .env"
            )
        return self.gemini_api_key.get_secret_value()

    def secret_values(self) -> list[str]:
        """The REAL secret values (template placeholders excluded), for log redaction.

        The one method that hands secrets out in bulk: call it only from the composition root (main.py)."""
        return [
            s.get_secret_value()
            for s in (self.gemini_api_key, self.openwa_api_key, self.github_token)
            if _is_set(s)
        ]

    def safe_dump(self) -> dict[str, object]:
        """Loggable view: secrets become '<set>' / '<missing>'; never the value."""
        out: dict[str, object] = {}
        for name, value in self.model_dump().items():
            if isinstance(value, SecretStr):
                out[name] = "<set>" if _is_set(value) else "<missing>"
            elif isinstance(value, Path):
                out[name] = str(value)
            else:
                out[name] = value
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton accessor. Tests: call `get_settings.cache_clear()` after changing env."""
    try:
        return Settings()
    except ValidationError as exc:
        # Convert pydantic's verbose trace into one readable, env-var-oriented message.
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']).upper() or 'SETTINGS'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration (.env / environment):\n{problems}") from None


if __name__ == "__main__":
    import json

    print(json.dumps(get_settings().safe_dump(), indent=2))
