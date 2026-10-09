"""
Application configuration.

pydantic-settings reads environment variables (and the .env file) and
validates them into a typed object. If DATABASE_URL is missing or
CHUNK_SIZE isn't an integer, the app fails loudly at startup instead of
crashing mysteriously later.

Usage anywhere in the app:
    from app.core.config import settings
    settings.LLM_MODEL
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:7b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSIONS: int = 768
    # nomic-embed-text was trained with task prefixes and scores noticeably
    # worse without them. Other models may want "" for both.
    EMBEDDING_DOCUMENT_PREFIX: str = "search_document: "
    EMBEDDING_QUERY_PREFIX: str = "search_query: "

    # Document processing
    UPLOAD_DIR: str = "./uploads"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    # Retrieval
    HYBRID_SEARCH: bool = True         # fuse vector + keyword results (see services/retrieval.py)

    # Answer quality
    RELEVANCE_THRESHOLD: float = 0.55  # best cosine similarity below this -> "not covered", skip LLM
    LLM_TIMEOUT_SECONDS: float = 180   # CPU inference is slow; but not infinitely slow

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # don't error on unrelated env vars
    )


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and cache it — reading .env on every access is wasteful."""
    return Settings()


settings = get_settings()
