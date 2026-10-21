import json
from types import SimpleNamespace

import pytest

from app.chunking.strategies import ChunkConfig
from app.config import get_settings
from app.database import get_db
from app.embeddings.embedder import get_embedder
from app.evaluation.check import check_floor, make_floor
from app.evaluation.dataset import Question, load_dataset
from app.evaluation.report import render_report
from app.evaluation.runner import ExperimentConfig, run_experiments
from app.evaluation.validate import label_problems
from app.main import app

A_PAGES = ["Mitochondria produce energy for the cell", "Photosynthesis happens inside chloroplasts"]
B_PAGES = ["Gradient descent minimizes a loss function", "Backpropagation computes gradients in neural networks"]


def _line(qid, text, relevant=None, split="test", qtype="factoid"):
    item = {"id": qid, "question": text, "answerable": relevant is not None, "type": qtype, "split": split}
    if relevant is not None:
        item["relevant"] = [{"document": d, "pages": p} for d, p in relevant]
    return json.dumps(item)


@pytest.fixture
def corpus(client, make_pdf, upload, tmp_path):
    for name, pages in (("a.pdf", A_PAGES), ("b.pdf", B_PAGES)):
        upload(name, make_pdf(pages, name=name).read_bytes())
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "\n".join(
            [
                _line("q1", "chloroplasts photosynthesis", [("a.pdf", [2])]),
                _line("q2", "gradient descent loss function", [("b.pdf", [1])]),
                _line("q3", "neural networks backpropagation gradients", [("b.pdf", [2])], split="dev"),
                _line("q4", "zzzqqq", None, qtype="off_topic"),
            ]
        ),
        encoding="utf-8",
    )
    questions, sha = load_dataset(path)
    return SimpleNamespace(
        client=client,
        db=next(app.dependency_overrides[get_db]()),
        settings=app.dependency_overrides[get_settings](),
        embedder=app.dependency_overrides[get_embedder](),
        questions=questions,
        sha=sha,
    )


def run(c, experiments):
    return run_experiments(
        c.db, c.settings, c.embedder, object(), c.questions, experiments, 5,
        dataset_path="d.jsonl", dataset_sha=c.sha, log=lambda _: None,
    )


BASELINE = ExperimentConfig("baseline", ChunkConfig(), "semantic")


def test_metrics_exist_for_answerable_questions_only(corpus):
    base = run(corpus, [BASELINE])["experiments"][0]
    by_id = {q["id"]: q for q in base["questions"]}

    assert by_id["q1"]["metrics"]["recall@1"] == 1.0
    assert by_id["q1"]["metrics"]["mrr"] == 1.0
    assert "metrics" not in by_id["q4"]
    assert by_id["q4"]["best_dense"] is not None


def test_a_new_chunk_set_is_built_without_touching_the_default(corpus):
    small = ExperimentConfig("small", ChunkConfig(chunk_size=400, overlap=80), "semantic")
    run(corpus, [BASELINE, small])

    for doc in corpus.client.get("/documents").json():
        default = corpus.client.get(f"/documents/{doc['id']}/chunks").json()
        custom = corpus.client.get(
            f"/documents/{doc['id']}/chunks", params={"chunk_config": "fixed-400-80"}
        ).json()
        assert len(default) == 2
        assert len(custom) > 0


def test_report_has_the_expected_sections(corpus):
    hybrid = ExperimentConfig("hybrid", ChunkConfig(), "hybrid")
    text = render_report(run(corpus, [BASELINE, hybrid]), 0.58)

    assert "Retrieval quality (test split)" in text
    assert "Recall@5" in text
    assert "Refusal threshold" in text
    assert "Only 2 answerable test questions" in text  # the small-sample warning


def test_floor_check_passes_then_catches_a_regression(corpus):
    results = run(corpus, [BASELINE])
    floor = make_floor(results, "baseline", "test", margin=0.0)
    assert check_floor(results, floor) == []

    floor["min"]["recall@5"] = 1.1
    assert any("recall@5" in f for f in check_floor(results, floor))

    floor["dataset_sha256"] = "something-else"
    assert any("dataset changed" in f for f in check_floor(results, floor))


def test_label_problems_flags_unknown_documents_and_pages(corpus):
    assert label_problems(corpus.db, corpus.questions) == []

    bad = Question("x1", "q?", True, "factoid", "dev", frozenset({("missing.pdf", 1), ("a.pdf", 99)}))
    problems = label_problems(corpus.db, [bad])

    assert any("missing.pdf" in p and "not uploaded" in p for p in problems)
    assert any("a.pdf has 2 pages" in p for p in problems)