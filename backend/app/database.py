from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
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


def init_schema(engine: Engine) -> None:
    """Create tables (and, on PostgreSQL, the vector extension and HNSW index)."""
    from app import models  # noqa: F401  (registers the models on Base)

    postgres = engine.dialect.name == "postgresql"
    if postgres:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    if postgres:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_embeddings_vector_hnsw "
                    "ON embeddings USING hnsw (vector vector_cosine_ops)"
                )
            )


def init_db() -> None:
    """Called at startup (Alembic migrations come later)."""
    init_schema(get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one database session per request."""
    with Session(get_engine()) as session:
        yield session