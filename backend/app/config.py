from functools import lru_cache
from pathlib import Path

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