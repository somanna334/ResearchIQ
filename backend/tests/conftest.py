import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401  (registers tables)
from app.config import Settings, get_settings
from app.database import Base, get_db
from app.embeddings.embedder import get_embedder
from app.main import app
from app.ratelimit import reset_limiters
from app.retrieval.bm25 import tokenize

TEST_PASSWORD = "correct-horse-battery"


class FakeEmbedder:
    """Deterministic stand-in: hashes words into 256 buckets, then normalizes.

    It only captures word overlap, not meaning. It tests the plumbing
    (storage, ranking, cleanup), not the quality of a real model.
    """

    model_name = "fake-hash-256"
    dimension = 256

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            vec[zlib.crc32(token.encode()) % self.dimension] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def _signed_in_client(email: str) -> TestClient:
    """A new client, registered through the real endpoint and carrying its token."""
    other = TestClient(app)
    response = other.post("/auth/register", json={"email": email, "password": TEST_PASSWORD})
    assert response.status_code == 201, response.text
    other.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
    return other


@pytest.fixture(autouse=True)
def _fresh_limiters() -> Iterator[None]:
    """Rate-limit counters live in memory, so every test starts from zero."""
    reset_limiters()
    yield
    reset_limiters()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A signed-in client (admin@example.com) with an in-memory database and temp uploads."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    settings = Settings(
        data_dir=tmp_path,
        max_upload_mb=1,
        max_pages=5,
        admin_emails="admin@example.com",
        argon2_time_cost=1,
        argon2_memory_kib=1024,
        argon2_parallelism=1,
        # Generous limits, so only the tests that tune them ever hit one.
        auth_requests_per_minute=1000,
        llm_requests_per_minute=1000,
        upload_requests_per_minute=1000,
        login_max_failures=50,
        _env_file=None,
    )
    embedder = FakeEmbedder()

    def override_db() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder] = lambda: embedder
    # Deliberately not using `with TestClient(...)`: that would run the app's
    # startup and create the real database file.
    yield _signed_in_client("admin@example.com")
    app.dependency_overrides.clear()


@pytest.fixture
def tune(client: TestClient) -> Callable[..., Settings]:
    """Change settings for the rest of a test: tune(max_documents_per_user=2)."""

    def _tune(**changes: object) -> Settings:
        current = app.dependency_overrides[get_settings]()
        patched = current.model_copy(update=changes)
        app.dependency_overrides[get_settings] = lambda: patched
        return patched

    return _tune


@pytest.fixture
def make_client(client: TestClient) -> Callable[[str], TestClient]:
    """Create another signed-in user (shares the same in-memory database)."""
    return _signed_in_client


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
