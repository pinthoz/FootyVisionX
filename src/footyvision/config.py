"""Central application settings, loaded from environment / .env."""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from footyvision import aws_secrets


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    postgres_user: str = "footy"
    postgres_password: str = "footy"
    postgres_db: str = "footyvision"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    # --- ETL ---
    min_minutes: int = Field(default=600, description="Min season minutes to include a player.")
    # Seasons below this share of a full double round-robin are fragments: a per-90 rate
    # from 34 games of a 306-game season is noisy, and ranking it alongside complete
    # seasons moves everybody else's percentile. Default 0 keeps everything the database
    # holds — raise it (0.9 is the natural value) to exclude fragments from the pool.
    min_season_coverage: float = Field(
        default=0.0,
        description="Minimum season completeness for a player to enter the pool.",
    )

    # --- LLM (local, OpenAI-compatible endpoint) ---
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_embed_model: str = "text-embedding-embeddinggemma-300m-qat"
    # Embedding models are trained with task prefixes and lose accuracy without them.
    # These are EmbeddingGemma's; nomic-embed-text uses "search_document: "/"search_query: ".
    # See scripts/eval_embeddings.py for the measurements behind this default.
    llm_embed_document_prefix: str = "title: none | text: "
    llm_embed_query_prefix: str = "task: search result | query: "
    llm_api_key: str = "not-needed-for-local"

    # --- Serving ---
    # Comma-separated origins allowed to call the API from a browser. Empty means local
    # development only: a public deployment must set this to its own frontend, because
    # "*" lets any site on the internet spend this instance's LLM budget.
    cors_origins: str = ""
    # The Vercel scope (team or account slug) whose preview deployments may call the API,
    # e.g. "pinthozs-projects" for footy-vision-<hash>-pinthozs-projects.vercel.app. Empty
    # admits no previews. It used to admit every *.vercel.app: any stranger's free app
    # could then call these endpoints from its visitors' browsers, each visitor bringing a
    # fresh IP and so a fresh rate-limit bucket.
    cors_preview_scope: str = ""
    # Calls per minute per client to the endpoints that invoke an LLM. 0 disables it.
    rate_limit_per_minute: int = 20

    # --- Fine-tuned retriever (optional) ---
    # Path to the sentence-transformers model produced by scripts/finetune_embeddings.py.
    # When set, BOTH the index and the queries are embedded with it — mixing it with the
    # stock model would compare vectors from two different spaces. Empty means the app
    # behaves exactly as before, with no torch dependency at runtime.
    finetuned_embed_path: str = ""

    # --- Cloud LLM (Fallback / Production) ---
    cloud_llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    cloud_llm_model: str = "gemini-2.5-flash"
    # Comma-separated models tried in order when the one above is out of quota (429) or
    # overloaded (5xx). Each Gemini model has its own free daily quota, so a chain of
    # them multiplies how many questions the dashboard answers per day at no cost.
    cloud_llm_fallback_models: str = ""
    cloud_llm_embed_model: str = "gemini-embedding-001"
    cloud_llm_api_key: str = ""
    gemini_api_key: str = ""

    @property
    def active_cloud_api_key(self) -> str:
        return (
            self.gemini_api_key
            or self.cloud_llm_api_key
            or (self.llm_api_key if self.llm_api_key != "not-needed-for-local" else "")
        )

    @property
    def allowed_origins(self) -> list[str]:
        """Origins for CORS, defaulting to the local dashboard when unset."""
        origins = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return origins or ["http://localhost:3000", "http://127.0.0.1:3000"]

    @property
    def allowed_origin_regex(self) -> str:
        """Local development, the production dashboard, and this project's own previews.

        The production alias is fixed here rather than left to CORS_ORIGINS: it was only
        ever admitted by the old *.vercel.app wildcard, so a deployment whose environment
        did not also list it would have locked its own dashboard out on the first deploy.
        """
        allowed = (
            r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://footy-vision-tau\.vercel\.app$"
        )
        scope = re.escape(self.cors_preview_scope.strip())
        if not scope:
            return allowed
        return allowed + rf"|^https://footy-vision-[a-z0-9-]+-{scope}\.vercel\.app$"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return _with_psycopg2(self.database_url)
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    # On AWS the database URL and the Gemini key live in Secrets Manager; this puts them
    # in the environment before Settings reads it. Elsewhere it does nothing.
    aws_secrets.load()
    return Settings()


def _with_psycopg2(url: str) -> str:
    """Name the PostgreSQL driver in the URL rather than leaving it to SQLAlchemy.

    A bare `postgresql://` URL means whatever driver the installed SQLAlchemy defaults to,
    and that changed underneath this project: 2.0 picked psycopg2, 2.1 picks psycopg 3.
    Render installs the newest release on every build, so a deploy that changed nothing
    about the database failed at startup with "No module named 'psycopg'" — psycopg2 is
    the driver this package depends on. Hosted providers also hand out the older
    `postgres://` scheme, which SQLAlchemy has not accepted since 1.4.

    Any URL that already names a driver, or is not PostgreSQL, is left alone.
    """
    for scheme in ("postgresql://", "postgres://"):
        if url.startswith(scheme):
            return "postgresql+psycopg2://" + url[len(scheme) :]
    return url
