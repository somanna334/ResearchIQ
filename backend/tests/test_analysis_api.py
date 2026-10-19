from types import SimpleNamespace

import pytest

from app.generation.prompt import Source, build_compare_prompt
from app.generation.tasks import select_chunks
from app.llm.base import LLMResult
from app.llm.factory import get_llm_provider
from app.main import app

A_PAGES = [
    "Mitochondria produce energy for the cell",
    "Photosynthesis happens inside chloroplasts",
]
B_PAGES = [
    "Mitochondria are found in animal cells and fungi",
    "Chloroplasts are found only in plants and algae",
]


class FakeLLM:
    name = "fake"
    model_name = "fake-1"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        return LLMResult(self.reply, 100, 20)


@pytest.fixture
def setup(client, make_pdf, upload):
    """Upload and embed two small documents; install a fake LLM with a canned reply."""

    def _setup(reply):
        llm = FakeLLM(reply)
        app.dependency_overrides[get_llm_provider] = lambda: llm
        ids = []
        for name, pages in (("a.pdf", A_PAGES), ("b.pdf", B_PAGES)):
            doc_id = upload(name, make_pdf(pages, name=name).read_bytes()).json()["id"]
            client.post(f"/documents/{doc_id}/embeddings")
            ids.append(doc_id)
        return llm, ids

    return _setup


def compare_body(ids, question="mitochondria energy"):
    return {"document_ids": ids, "question": question, "k": 2}


# ---- unit tests ----

def test_select_chunks_keeps_everything_under_budget():
    chunks = [SimpleNamespace(text="x" * 100) for _ in range(5)]

    assert select_chunks(chunks, 1000) == chunks


def test_select_chunks_samples_evenly_over_budget():
    chunks = [SimpleNamespace(text=f"c{i}".ljust(100, "x")) for i in range(10)]
    picked = select_chunks(chunks, 350)

    assert [c.text[:2] for c in picked] == ["c0", "c3", "c6"]


def test_compare_prompt_labels_each_document():
    sources = [
        Source("S1", 1, "d1", "a.pdf", 1, 0.5, "alpha"),
        Source("S2", 2, "d2", "b.pdf", 3, 0.4, "beta"),
    ]
    _, prompt = build_compare_prompt("What differs?", sources, {"d1": "A", "d2": "B"})

    assert 'document="A: a.pdf"' in prompt
    assert 'document="B: b.pdf"' in prompt


# ---- summary ----

def test_summary_cites_chunks_and_reports_coverage(client, setup):
    _, ids = setup("- Cells make energy [S1].\n- Plants photosynthesize [S2].")
    body = client.post(f"/documents/{ids[0]}/summary").json()

    assert body["status"] == "answered"
    assert [s["cited"] for s in body["sources"]] == [True, True]
    assert body["coverage"] == {
        "chunks_used": 2, "chunks_total": 2, "pages_covered": [1, 2],
        "page_count": 2, "sampled": False,
    }


def test_summary_with_invented_tag_is_withheld(client, setup):
    _, ids = setup("- Something [S9].")
    body = client.post(f"/documents/{ids[0]}/summary").json()

    assert body["status"] == "citation_error"
    assert not any(s["cited"] for s in body["sources"])


def test_summary_of_unknown_document_returns_404(client, setup):
    setup("unused")

    assert client.post("/documents/nope/summary").status_code == 404


def test_summary_of_scanned_document_returns_409(client, setup, make_pdf, upload):
    llm, _ = setup("unused")
    scan = upload("scan.pdf", make_pdf(["", ""], name="scan.pdf").read_bytes()).json()["id"]

    assert client.post(f"/documents/{scan}/summary").status_code == 409
    assert llm.calls == 0


# ---- comparison ----

def test_compare_labels_documents_and_reports_who_was_cited(client, setup):
    _, ids = setup("Both discuss mitochondria [S1][S3].")
    body = client.post("/compare", json=compare_body(ids)).json()

    assert body["status"] == "answered"
    assert [d["label"] for d in body["documents"]] == ["A", "B"]
    assert [d["cited"] for d in body["documents"]] == [True, True]
    assert len(body["sources"]) == 4  # 2 passages per document


def test_compare_flags_an_answer_that_cites_only_one_document(client, setup):
    _, ids = setup("Only A is covered [S1].")
    body = client.post("/compare", json=compare_body(ids)).json()

    assert [d["cited"] for d in body["documents"]] == [True, False]


def test_compare_skips_the_llm_when_every_document_is_weak(client, setup):
    llm, ids = setup("should not be used")
    body = client.post("/compare", json=compare_body(ids, "zzzqqq")).json()

    assert body["status"] == "insufficient_evidence"
    assert body["retrieval"]["gated"] is True
    assert len(body["sources"]) == 4  # still returned for inspection
    assert llm.calls == 0


def test_compare_rejects_duplicate_documents(client, setup):
    _, ids = setup("unused")

    assert client.post("/compare", json=compare_body([ids[0], ids[0]])).status_code == 400


def test_compare_needs_at_least_two_documents(client, setup):
    _, ids = setup("unused")

    assert client.post("/compare", json=compare_body([ids[0]])).status_code == 422


def test_compare_unknown_document_returns_404(client, setup):
    _, ids = setup("unused")

    assert client.post("/compare", json=compare_body([ids[0], "nope"])).status_code == 404