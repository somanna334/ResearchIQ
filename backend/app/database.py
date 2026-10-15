from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        # FastAPI runs sync endpoints in worker threads.
        connect_args["check_same_thread"] = False
    return create_engine(settings.database_url, connect_args=connect_args)


def init_db() -> None:
    """Create tables if they don't exist (Alembic migrations come later)."""
    from app import models  # noqa: F401  (registers the models on Base)

    Base.metadata.create_all(get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one database session per request."""
    with Session(get_engine()) as session:
        yield session