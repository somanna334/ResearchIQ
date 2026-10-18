import pytest

from app.llm.base import LLMResult
from app.llm.factory import get_llm_provider
from app.main import app
from app.retrieval.rerank import get_reranker

BIO = [
    "Mitochondria produce energy for the cell",
    "Photosynthesis happens inside chloroplasts",
]


class FakeLLM:
    name = "fake"
    model_name = "fake-1"

    def __init__(self):
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        return LLMResult("Photosynthesis happens inside chloroplasts [S1].", 100, 20)


class FakeReranker:
    model_name = "fake-reranker"

    def score(self, query, passages):
        return [1.0 if "mitochondria" in p.lower() else 0.0 for p in passages]


@pytest.fixture
def setup(client, make_pdf, upload):
    llm = FakeLLM()
    app.dependency_overrides[get_llm_provider] = lambda: llm
    app.dependency_overrides[get_reranker] = lambda: FakeReranker()
    doc_id = upload("bio.pdf", make_pdf(BIO, name="bio.pdf").read_bytes()).json()["id"]
    client.post(f"/documents/{doc_id}/embeddings")
    return llm


def test_weak_retrieval_skips_the_llm(client, setup):
    body = client.post("/ask", json={"question": "zzzqqq", "k": 2}).json()

    assert body["status"] == "insufficient_evidence"
    assert body["retrieval"]["gated"] is True
    assert len(body["sources"]) == 2  # still returned, for inspection
    assert setup.calls == 0


def test_strong_retrieval_reaches_the_llm(client, setup):
    body = client.post("/ask", json={"question": "photosynthesis chloroplasts", "k": 2}).json()

    assert body["status"] == "answered"
    assert body["retrieval"]["gated"] is False
    assert setup.calls == 1


@pytest.mark.parametrize("mode", ["semantic", "hybrid", "hybrid_rerank"])
def test_every_retrieval_mode_can_answer(client, setup, mode):
    body = client.post(
        "/ask", json={"question": "photosynthesis chloroplasts", "k": 2, "mode": mode}
    ).json()

    assert body["status"] == "answered"
    assert body["retrieval"]["mode"] == mode