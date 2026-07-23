"""Central runtime configuration, sourced from the environment (12-factor).

Every subsystem reads its defaults from :class:`OmniSettings`; constructors
keep explicit parameters so tests and embedders can override without touching
the environment. Use :func:`get_settings` for the cached process-wide
instance.

All variables use the ``OMNIAI_`` prefix, e.g.::

    OMNIAI_DATABASE_URL=postgresql+asyncpg://omni:secret@db:5432/omniai
    OMNIAI_API_KEYS=key1,key2
    OMNIAI_ENGINE_BASE_URL=http://vllm:8000
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Parsed from comma-separated env strings by _split_csv, not JSON.
CsvList = Annotated[list[str], NoDecode]


class OmniSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMNIAI_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- persistence -------------------------------------------------------
    # Any SQLAlchemy async URL: postgresql+asyncpg:// (production),
    # sqlite+aiosqlite:// (zero-config dev), mysql+asyncmy://, ...
    database_url: str = "sqlite+aiosqlite:///interactions.db"

    # --- security ----------------------------------------------------------
    api_keys: CsvList = Field(default_factory=list)
    auth_disabled: bool = False  # must be set explicitly to run without keys
    rate_limit_rps: float = 10.0  # sustained requests/second per API key
    rate_limit_burst: int = 20  # bucket capacity
    cors_origins: CsvList = Field(default_factory=list)
    max_body_bytes: int = 1_000_000

    # --- engine ------------------------------------------------------------
    engine_base_url: str | None = None  # set => attach to external server
    engine_managed: bool = True  # False in compose: vLLM is its own service
    request_timeout_s: float = 120.0
    engine_retries: int = 3
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0
    supervisor_max_restarts: int = 5

    # --- observability -----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    otlp_endpoint: str | None = None
    service_name: str = "omniai-gateway"

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def auth_enabled(self) -> bool:
        return not self.auth_disabled

    def validate_security(self) -> None:
        """Fail closed: refuse to serve with auth on but no keys configured."""
        if self.auth_enabled and not self.api_keys:
            raise RuntimeError(
                "No API keys configured (OMNIAI_API_KEYS). Set keys, or set "
                "OMNIAI_AUTH_DISABLED=true to explicitly run without auth."
            )


@lru_cache(maxsize=1)
def get_settings() -> OmniSettings:
    return OmniSettings()


def reset_settings_cache() -> None:
    """Testing hook: force re-read of the environment."""
    get_settings.cache_clear()
