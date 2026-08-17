"""
Delve Configuration Module
──────────────────────────
Loads environment variables via pydantic-settings.
All API keys and paths are centralized here.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── LLM (Google Gemini Free Tier Engine) ─────────────────────────────
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key from Google AI Studio (Free Tier).",
    )
    llm_api_key: str = Field(
        default="",
        description="Alias for Gemini API key.",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Base URL for the Google Gemini v1beta REST API.",
    )
    gemini_primary_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Primary workhorse model (15 RPM / 250k TPM / 500 RPD).",
    )
    gemini_fallback_models: list[str] = Field(
        default_factory=lambda: [
            "gemini-3.5-flash-lite",  # 15 RPM / 250k TPM / 500 RPD (Alternative Workhorse)
            "gemma-4-31b",            # 30 RPM / 16k TPM / 14,400 RPD (Massive Volume)
            "gemma-4-26b",            # 30 RPM / 16k TPM / 14,400 RPD (Massive Volume)
            "gemini-3.7-flash",       # 5 RPM / 250k TPM / 20 RPD (Heavy Reasoning)
            "gemini-3.6-flash",       # 5 RPM / 250k TPM / 20 RPD (Heavy Reasoning)
            "gemini-3.5-flash",       # 5 RPM / 250k TPM / 20 RPD (Heavy Reasoning)
            "gemini-3-flash",         # 5 RPM / 250k TPM / 20 RPD (Heavy Reasoning)
            "gemini-2.5-flash",       # 5 RPM / 250k TPM / 20 RPD
            "gemini-2.5-flash-lite",  # 10 RPM / 250k TPM / 20 RPD
        ],
        description="Priority cascade across all available free-tier models.",
    )
    llm_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Current active model name.",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        description="Per-request timeout in seconds for Gemini API calls.",
    )

    # ── Tavily Web Search ─────────────────────────────────────────────────
    tavily_api_key: str = Field(
        default="",
        description="Tavily API key for web search (optional).",
    )

    # ── Retrieval Tuning ──────────────────────────────────────────────────
    max_search_queries: int = Field(default=4)
    max_source_results_per_query: int = Field(default=4)
    semantic_scholar_max_retries: int = Field(default=3)
    semantic_scholar_base_backoff_seconds: float = Field(default=2.0)
    max_debate_rounds: int = Field(default=1)
    strict_synthesis_mode: bool = Field(default=False)
    crossref_contact_email: str = Field(default="")

    # ── Supabase (required for all deployments) ───────────────────────────
    app_environment: str = Field(default="production")
    frontend_origins: list[str] = Field(default_factory=lambda: [])
    supabase_url: str = Field(default="")
    supabase_service_role_key: str = Field(default="")
    supabase_jwt_secret: str = Field(default="")
    supabase_jwt_audience: str = Field(default="authenticated")
    supabase_storage_bucket: str = Field(default="delve-documents")
    ws_ticket_secret: str = Field(default="")
    job_dispatch_secret: str = Field(default="")

    # ── Upstash (QStash + Redis) ──────────────────────────────────────────
    qstash_token: str = Field(default="")
    qstash_current_signing_key: str = Field(default="")
    qstash_next_signing_key: str = Field(default="")
    public_api_base_url: str = Field(default="")
    upstash_redis_rest_url: str = Field(default="")
    upstash_redis_rest_token: str = Field(default="")

    # ── Safety limits ─────────────────────────────────────────────────────
    max_concurrent_jobs_per_user: int = Field(default=10, ge=1, le=50)
    max_research_jobs_per_day: int = Field(default=25, ge=1, le=100)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024 * 1024)
    max_upload_pages: int = Field(default=100, ge=1, le=1000)
    max_extracted_text_bytes: int = Field(default=5 * 1024 * 1024, ge=100_000)
    max_documents_per_user: int = Field(default=20, ge=1, le=200)
    max_llm_calls_per_job: int = Field(default=60, ge=5, le=200)
    max_llm_calls_in_flight: int = Field(default=6, ge=1, le=20)

    # ── Server ────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=10000)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "enable_decoding": False,
    }

    @field_validator("frontend_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def effective_gemini_api_key(self) -> str:
        key = self.gemini_api_key or self.llm_api_key
        return key.strip()

    def gemini_endpoint(self, model: str) -> str:
        return f"{self.gemini_base_url.rstrip('/')}/models/{model}:generateContent"

    @property
    def is_production(self) -> bool:
        return self.app_environment.lower() in {"production", "staging"}

    @property
    def supabase_rest_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/rest/v1" if self.supabase_url else ""

    @property
    def supabase_storage_url(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/storage/v1" if self.supabase_url else ""


# Singleton instance
settings = Settings()
