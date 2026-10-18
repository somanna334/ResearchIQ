from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    app_name: str = "ResearchIQ"
    environment: str = "development"
    # Comma-separated list of allowed browser origins.
    cors_origins: str = "http://localhost:5173"

    # Storage (paths are relative to the directory you start the server from: backend/)
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/researchiq.db"

    # Upload / ingestion limits
    max_upload_mb: int = 25
    max_pages: int = 500

    # Default chunking (can be overridden per request for experiments)
    chunk_strategy: Literal["fixed", "sentence"] = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # LLM (SecretStr keeps the key out of logs and repr output)
    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.5-flash-lite"
    llm_max_output_tokens: int = 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RESEARCHIQ_",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()