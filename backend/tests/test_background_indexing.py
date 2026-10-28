"""Background indexing (Milestone 9C).

Tests run with background_indexing switched on via tune(). Starlette's TestClient
runs background tasks before it hands the response back, so each test checks the
response BODY (what the user would have seen) and then GETs the document again
(what it became).
"""

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import resolve_document_ids
from app.embeddings.embedder import get_embedder_factory
from app.ingestion.pipeline import recover_interrupted_documents
from app.main import app
from app.models import Document, User

TEXT = ["Alpha content on page one", "Beta content on page two"]


def _upload(client, make_pdf, pages=None, name="doc.pdf"):
    pdf = make_pdf(pages or TEXT, name=name)
    return client.post(
        "/documents", files={"file": (name, pdf.read_bytes(), "application/pdf")}
    )


def _factory(embedder):
    return lambda: (lambda: embedder)


def _break_embedder():
    """Make the indexing worker's embedder loader raise, and return the secret text."""
    secret = "SECRET-internal C:\\models\\weights.bin"

    def boom():
        raise RuntimeError(secret)

    app.dependency_overrides[get_embedder_factory] = lambda: boom
    return secret


def test_upload_returns_processing_then_ready_and_embedded(client, tune, make_pdf):
    tune(background_indexing=True)

    response = _upload(client, make_pdf)

    assert response.status_code == 201
    assert response.json()["status"] == "processing"  # what the user saw at once
    document_id = response.json()["id"]

    document = client.get(f"/documents/{document_id}").json()
    assert document["status"] == "ready"
    assert document["page_count"] == 2
    assert document["error_message"] is None

    # "ready" means searchable: every chunk already has its vector.
    stats = client.post(f"/documents/{document_id}/embeddings").json()
    assert stats["total_chunks"] >= 1
    assert stats["newly_embedded"] == 0


def test_indexing_failure_hides_internals_and_retry_recovers(
    client, tune, make_pdf, tmp_path, fake_embedder
):
    tune(background_indexing=True)
    secret = _break_embedder()

    response = _upload(client, make_pdf)
    assert response.json()["status"] == "processing"
    document_id = response.json()["id"]

    failed = client.get(f"/documents/{document_id}").json()
    assert failed["status"] == "failed"
    assert failed["error_message"]
    assert "SECRET" not in json.dumps(failed)
    assert "weights" not in json.dumps(failed)
    assert secret not in response.text
    # The file is kept, so the failure can be retried.
    assert (tmp_path / "uploads" / f"{document_id}.pdf").exists()

    app.dependency_overrides[get_embedder_factory] = _factory(fake_embedder)
    retry = client.post(f"/documents/{document_id}/retry")

    assert retry.status_code == 202
    assert retry.json()["status"] == "processing"
    after = client.get(f"/documents/{document_id}").json()
    assert after["status"] == "ready"
    assert after["error_message"] is None


def test_retry_rejects_a_ready_document(client, make_pdf, upload):
    document_id = upload("a.pdf", make_pdf(TEXT).read_bytes()).json()["id"]
    assert client.get(f"/documents/{document_id}").json()["status"] == "ready"

    assert client.post(f"/documents/{document_id}/retry").status_code == 409


def test_retry_rejects_a_missing_file(client, tune, make_pdf, tmp_path):
    tune(background_indexing=True)
    _break_embedder()
    document_id = _upload(client, make_pdf).json()["id"]
    assert client.get(f"/documents/{document_id}").json()["status"] == "failed"

    (tmp_path / "uploads" / f"{document_id}.pdf").unlink()

    assert client.post(f"/documents/{document_id}/retry").status_code == 409


def test_retry_of_someone_elses_document_is_404(client, make_client, tune, make_pdf):
    tune(background_indexing=True)
    _break_embedder()
    document_id = _upload(client, make_pdf).json()["id"]
    assert client.get(f"/documents/{document_id}").json()["status"] == "failed"

    other = make_client("other@example.com")

    assert other.post(f"/documents/{document_id}/retry").status_code == 404


def test_retry_needs_a_signed_in_user(client):
    from fastapi.testclient import TestClient

    anonymous = TestClient(app)

    assert anonymous.post("/documents/anything/retry").status_code == 401


def test_pdf_without_text_never_touches_the_embedder(client, tune, make_pdf, fake_embedder):
    tune(background_indexing=True)
    loads: list[int] = []

    def factory():
        loads.append(1)
        return fake_embedder

    app.dependency_overrides[get_embedder_factory] = lambda: factory

    response = _upload(client, make_pdf, pages=["", ""], name="scan.pdf")

    assert response.json()["status"] == "processing"
    document = client.get(f"/documents/{response.json()['id']}").json()
    assert document["status"] == "no_text"
    assert document["page_count"] == 2
    assert loads == []  # the embedder (the slow model) was never loaded


def test_recovery_fails_leftover_processing_documents(client, db_engine):
    with Session(db_engine) as db:
        owner = db.scalar(select(User).where(User.email == "admin@example.com"))
        db.add_all(
            [
                Document(
                    id="stuck", owner_id=owner.id, filename="stuck.pdf",
                    size_bytes=1, sha256="0" * 64, status="processing",
                ),
                Document(
                    id="fine", owner_id=owner.id, filename="fine.pdf",
                    size_bytes=1, sha256="1" * 64, status="ready",
                ),
            ]
        )
        db.commit()

    assert recover_interrupted_documents(db_engine) == 1

    with Session(db_engine) as db:
        stuck = db.get(Document, "stuck")
        fine = db.get(Document, "fine")
        assert stuck.status == "failed"
        assert "Retry" in (stuck.error_message or "")
        assert fine.status == "ready"
        assert fine.error_message is None


def test_processing_documents_are_not_in_the_all_documents_scope(client, db_engine):
    with Session(db_engine) as db:
        owner = db.scalar(select(User).where(User.email == "admin@example.com"))
        db.add_all(
            [
                Document(
                    id="busy", owner_id=owner.id, filename="busy.pdf",
                    size_bytes=1, sha256="0" * 64, status="processing",
                ),
                Document(
                    id="done", owner_id=owner.id, filename="done.pdf",
                    size_bytes=1, sha256="1" * 64, status="ready",
                ),
            ]
        )
        db.commit()
        db.refresh(owner)

        assert resolve_document_ids(db, owner, None) == ["done"]
        assert resolve_document_ids(db, owner, []) == ["done"]
        # Naming a document explicitly still works for any document the user owns.
        assert resolve_document_ids(db, owner, ["busy"]) == ["busy"]
