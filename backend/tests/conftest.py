from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401  (registers tables)
from app.config import Settings, get_settings
from app.database import Base, get_db
from app.main import app


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A test client with an in-memory database and a temporary upload folder."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    settings = Settings(
        data_dir=tmp_path, max_upload_mb=1, max_pages=5, _env_file=None
    )

    def override_db() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: settings
    # Deliberately not using `with TestClient(...)`: that would run the app's
    # startup and create the real database file.
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Build a small PDF; an empty string makes a blank (no-text) page."""

    def _make(pages: list[str], name: str = "sample.pdf") -> Path:
        path = tmp_path / name
        pdf = canvas.Canvas(str(path))
        for text in pages:
            if text:
                pdf.drawString(72, 750, text)
            pdf.showPage()
        pdf.save()
        return path

    return _make


@pytest.fixture
def upload(client: TestClient) -> Callable[..., object]:
    def _upload(filename: str, content: bytes):
        return client.post(
            "/documents", files={"file": (filename, content, "application/pdf")}
        )

    return _upload