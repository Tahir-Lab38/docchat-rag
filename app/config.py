"""
Application configuration using pydantic-settings.

All settings are loaded from environment variables or a `.env` file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for DocChat."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Groq LLM ────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    QUERY_REWRITE_MODEL: str = "llama-3.1-8b-instant"
    QUERY_REWRITE_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
    ANSWER_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"

    # ── Server ──────────────────────────────────────────────────
    CORS_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"

    # ── ChromaDB ────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # ── File storage ────────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_MB: int = 50

    # ── Chunking ────────────────────────────────────────────────
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ───────────────────────────────────────────────
    TOP_K: int = 8
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 4

    # ── Embedding ───────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Generation ──────────────────────────────────────────────
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 768
    MAX_CONVERSATIONS: int = 500

    @property
    def cors_origins(self) -> list[str]:
        """Parse a comma-separated CORS origin list."""
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return [origin for origin in origins if origin]


# Singleton – import this everywhere
settings = Settings()
