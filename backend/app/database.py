from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    if settings.database_url.startswith("sqlite"):
        # FastAPI runs sync endpoints in worker threads.
        return create_engine(
            settings.database_url, connect_args={"check_same_thread": False}
        )
    return create_engine(settings.database_url, pool_pre_ping=True)


def _alembic_config() -> Config:
    """Load the backend's Alembic configuration independent of the CWD."""
    backend_dir = Path(__file__).resolve().parent.parent
    config = Config(str(backend_dir / "alembic.ini"))
    # env.py obtains the effective URL from pydantic-settings.
    return config


def init_db() -> None:
    """Apply versioned migrations, adopting a complete pre-Alembic schema safely."""
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    app_tables = {"users", "auth_sessions", "documents", "pages", "chunks", "embeddings"}

    config = _alembic_config()
    if "alembic_version" not in tables:
        existing_app_tables = tables & app_tables
        if existing_app_tables:
            # Milestone 9 databases were created with metadata.create_all. Stamp only
            # when the complete current schema is present; never silently bless a
            # partially upgraded or legacy schema. This does not alter user data.
            complete = app_tables.issubset(tables)
            has_owner = complete and "owner_id" in {
                column["name"] for column in inspector.get_columns("documents")
            }
            if not (complete and has_owner):
                missing = sorted(app_tables - tables)
                raise RuntimeError(
                    "Existing ResearchIQ database has a partial/legacy schema and "
                    "cannot be auto-stamped. Back it up and migrate it deliberately. "
                    f"Missing tables: {missing}; documents.owner_id present: {has_owner}."
                )
            command.stamp(config, "head")

    command.upgrade(config, "head")


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one database session per request."""
    with Session(get_engine()) as session:
        yield session
