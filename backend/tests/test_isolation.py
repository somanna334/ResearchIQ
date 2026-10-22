import pytest

from app.database import get_db
from app.embeddings.embedder import get_embedder
from app.llm.base import LLMResult
from app.llm.factory import get_llm_provider
from app.main import app
from app.retrieval.keyword import keyword_search
from app.retrieval.semantic import semantic_search

BIO = [
    "Mitochondria produce energy for the cell",
    "Photosynthesis happens inside chloroplasts",
]
KEY = "fixed-800-150"


class FakeLLM:
    name = "fake"
    model_name = "fake-1"

    def __init__(self):
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        return LLMResult("Photosynthesis happens inside chloroplasts [S1].", 10, 5)


@pytest.fixture
def llm(client):
    fake = FakeLLM()
    app.dependency_overrides[get_llm_provider] = lambda: fake
    return fake


def add_document(who, make_pdf, name="bio.pdf", pages=BIO):
    response = who.post(
        "/documents",
        files={"file": (name, make_pdf(pages, name=name).read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    doc_id = response.json()["id"]
    assert who.post(f"/documents/{doc_id}/embeddings").status_code == 200
    return doc_id


def test_users_only_see_their_own_documents(client, make_client, make_pdf):
    bob = make_client("bob@example.com")
    alice_doc = add_document(client, make_pdf)

    assert [d["id"] for d in client.get("/documents").json()] == [alice_doc]
    assert bob.get("/documents").json() == []


def test_other_users_documents_look_nonexistent(client, make_client, make_pdf, llm):
    bob = make_client("bob@example.com")
    doc_id = add_document(client, make_pdf)
    attempts = [
        ("GET", f"/documents/{doc_id}"),
        ("GET", f"/documents/{doc_id}/pages/1"),
        ("GET", f"/documents/{doc_id}/chunks"),
        ("POST", f"/documents/{doc_id}/chunks"),
        ("POST", f"/documents/{doc_id}/embeddings"),
        ("POST", f"/documents/{doc_id}/summary"),
        ("DELETE", f"/documents/{doc_id}"),
    ]
    for method, path in attempts:
        assert bob.request(method, path).status_code == 404, f"{method} {path} leaked"

    missing = bob.get("/documents/does-not-exist")
    assert missing.json() == bob.get(f"/documents/{doc_id}").json()  # indistinguishable
    assert client.get(f"/documents/{doc_id}").status_code == 200  # still there
    assert llm.calls == 0


def test_search_never_returns_other_users_passages(client, make_client, make_pdf):
    bob = make_client("bob@example.com")
    alice_doc = add_document(client, make_pdf)
    query = {"q": "chloroplasts photosynthesis"}

    for path in ("/search", "/search/semantic", "/search/hybrid"):
        assert client.get(path, params=query).json(), f"owner got nothing from {path}"
        assert bob.get(path, params=query).json() == [], f"{path} leaked"
        named = bob.get(path, params={**query, "document_id": alice_doc})
        assert named.status_code == 404, f"{path} accepted someone else's document id"


def test_ask_by_another_user_never_reaches_the_llm(client, make_client, make_pdf, llm):
    bob = make_client("bob@example.com")
    alice_doc = add_document(client, make_pdf)

    body = bob.post("/ask", json={"question": "chloroplasts photosynthesis"}).json()
    assert body["status"] == "insufficient_evidence" and body["sources"] == []
    assert llm.calls == 0

    named = bob.post("/ask", json={"question": "anything", "document_ids": [alice_doc]})
    assert named.status_code == 404

    assert client.post("/ask", json={"question": "chloroplasts photosynthesis"}).json()["status"] == "answered"


def test_compare_rejects_documents_the_user_does_not_own(client, make_client, make_pdf, llm):
    bob = make_client("bob@example.com")
    alice_doc = add_document(client, make_pdf, "a.pdf")
    bob_doc = add_document(bob, make_pdf, "b.pdf")

    for who, ids in ((bob, [bob_doc, alice_doc]), (client, [alice_doc, bob_doc])):
        response = who.post("/compare", json={"document_ids": ids, "question": "energy"})
        assert response.status_code == 404
    assert llm.calls == 0


def test_an_empty_document_scope_fails_closed(client, make_pdf):
    add_document(client, make_pdf)
    db = next(app.dependency_overrides[get_db]())
    embedder = app.dependency_overrides[get_embedder]()

    assert keyword_search(db, "chloroplasts", 5, KEY, None)  # no filter finds it
    assert keyword_search(db, "chloroplasts", 5, KEY, []) == []  # empty list finds nothing
    assert semantic_search(db, embedder, "chloroplasts", 5, KEY, None)
    assert semantic_search(db, embedder, "chloroplasts", 5, KEY, []) == []


def test_admins_get_no_special_access_to_documents(client, make_client, make_pdf):
    bob = make_client("bob@example.com")
    bob_doc = add_document(bob, make_pdf)

    assert client.get("/auth/me").json()["is_admin"] is True
    assert client.get("/documents").json() == []
    assert client.get(f"/documents/{bob_doc}").status_code == 404


def test_the_same_file_uploaded_by_two_users_stays_separate(client, make_client, make_pdf):
    bob = make_client("bob@example.com")
    alice_doc = add_document(client, make_pdf)
    bob_doc = add_document(bob, make_pdf)
    query = {"q": "chloroplasts photosynthesis"}

    assert alice_doc != bob_doc
    assert {h["document_id"] for h in client.get("/search/semantic", params=query).json()} == {alice_doc}
    assert {h["document_id"] for h in bob.get("/search/semantic", params=query).json()} == {bob_doc}
    assert client.delete(f"/documents/{alice_doc}").status_code == 204
    assert bob.get(f"/documents/{bob_doc}").status_code == 200  # untouched


def test_evaluation_endpoints_require_an_administrator(client, make_client):
    bob = make_client("bob@example.com")

    assert bob.get("/eval/runs").status_code == 403
    assert bob.get("/eval/runs/20261019T100000Z").status_code == 403
    assert client.get("/eval/runs").status_code == 200