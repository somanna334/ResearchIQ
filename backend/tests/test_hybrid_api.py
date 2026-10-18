import pytest

from app.main import app
from app.retrieval.rerank import get_reranker

BIO = [
    "Mitochondria produce energy for the cell",
    "Photosynthesis happens inside chloroplasts",
]


class FakeReranker:
    """Prefers passages that mention mitochondria, regardless of the query."""

    model_name = "fake-reranker"

    def score(self, query, passages):
        return [1.0 if "mitochondria" in p.lower() else 0.0 for p in passages]


@pytest.fixture
def indexed(client, make_pdf, upload):
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    doc_id = upload("bio.pdf", make_pdf(BIO, name="bio.pdf").read_bytes()).json()["id"]
    client.post(f"/documents/{doc_id}/embeddings")
    return doc_id


def test_hybrid_returns_component_scores(client, indexed):
    hits = client.get("/search/hybrid", params={"q": "chloroplasts photosynthesis", "k": 2}).json()

    assert hits[0]["page_number"] == 2
    assert hits[0]["bm25_score"] is not None
    assert hits[0]["dense_score"] is not None
    assert hits[0]["rerank_score"] is None


def test_rerank_can_reorder_the_candidates(client, indexed):
    hits = client.get(
        "/search/hybrid",
        params={"q": "chloroplasts photosynthesis", "k": 2, "rerank": "true"},
    ).json()

    assert hits[0]["page_number"] == 1
    assert hits[0]["rerank_score"] == 1.0


def test_hybrid_still_works_when_nothing_is_embedded(client, make_pdf, upload):
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    upload("bio.pdf", make_pdf(BIO, name="bio.pdf").read_bytes())

    hits = client.get("/search/hybrid", params={"q": "chloroplasts", "k": 2}).json()

    assert hits[0]["page_number"] == 2
    assert hits[0]["dense_score"] is None  # keyword-only hit