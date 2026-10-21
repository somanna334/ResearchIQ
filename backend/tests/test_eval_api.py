import json

import pytest

from app.config import Settings, get_settings
from app.main import app


def _retrieval(run_id):
    def question(qid, split, answerable, recall, best):
        item = {
            "id": qid, "question": qid, "type": "factoid" if answerable else "off_topic",
            "split": split, "answerable": answerable, "best_dense": best,
            "latency_ms": 10.0, "retrieved": [],
        }
        if answerable:
            item["relevant"] = [["a.pdf", 1]]
            item["metrics"] = {
                "recall@5": recall, "precision@5": recall / 5, "ndcg@5": recall, "mrr": recall,
            }
        return item

    questions = [
        question("q1", "test", True, 1.0, 0.7),
        question("q2", "test", True, 0.0, 0.6),
        question("q3", "test", False, 0, 0.5),
        question("q4", "dev", True, 1.0, 0.65),
    ]
    return {
        "schema": 1,
        "meta": {
            "run_id": run_id, "created_utc": "2026-10-19T10:00:00+00:00", "git_commit": "abc1234",
            "git_dirty": False, "python": "3.11", "k": 5,
            "dataset": {"path": "d.jsonl", "sha256": "x" * 64, "questions": 4},
            "corpus": [], "settings": {"min_top_score": 0.58},
        },
        "experiments": [
            {"name": "baseline", "config": {}, "questions": questions},
            {"name": "hybrid", "config": {}, "questions": questions},
        ],
    }


def _record(qid, split, answerable, status, verdict=None, supported=None):
    return {
        "id": qid, "question": qid, "type": "factoid" if answerable else "off_topic",
        "split": split, "answerable": answerable, "has_reference": answerable,
        "status": status, "gated": status == "insufficient_evidence", "answer": "x",
        "best_dense": 0.6, "retrieved": [],
        "citation": (
            {"cited_pages": [], "precision": 1.0, "recall": 1.0, "hits_relevant": True}
            if status == "answered" and answerable else None
        ),
        "correctness": {"verdict": verdict, "reason": ""} if verdict else None,
        "faithfulness": {"claims": 2, "supported": supported} if supported is not None else None,
        "judge_error": False, "error": None,
        "usage": {
            "prompt_tokens": 100, "completion_tokens": 10,
            "judge_prompt_tokens": 50, "judge_completion_tokens": 5,
        },
        "latency_ms": {"retrieval": 10.0, "llm": 100.0},
    }


def _answers(run_id):
    return {
        "kind": "answers",
        "schema": 1,
        "meta": {
            "run_id": run_id, "created_utc": "2026-10-20T10:00:00+00:00", "git_commit": "abc1234",
            "k": 5, "mode": "semantic", "chunk_config": "fixed-800-150",
            "judge": {"enabled": True, "model": "j"}, "generation": {"model": "g"},
            "dataset": {"path": "d.jsonl", "sha256": "x" * 64, "questions": 4},
            "corpus": [], "settings": {"min_top_score": 0.58},
            "pricing": {"generation": None, "judge": None},
        },
        "questions": [
            _record("q1", "test", True, "answered", verdict="correct", supported=2),
            _record("q2", "test", True, "answered", verdict="incorrect", supported=1),
            _record("q3", "test", True, "insufficient_evidence"),
            _record("q4", "test", False, "insufficient_evidence"),
        ],
    }


@pytest.fixture
def results_dir(client, tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    settings = Settings(data_dir=tmp_path, eval_results_dir=folder, _env_file=None)
    app.dependency_overrides[get_settings] = lambda: settings
    (folder / "20261019T100000Z.json").write_text(json.dumps(_retrieval("20261019T100000Z")))
    (folder / "20261020T100000Z-answers.json").write_text(json.dumps(_answers("20261020T100000Z-answers")))
    (folder / "notes.json").write_text("{}")  # not a run: ignored
    return folder


def test_runs_are_listed_newest_first_and_other_files_are_skipped(client, results_dir):
    (results_dir / "20261021T100000Z.json").write_text("{broken")
    runs = client.get("/eval/runs").json()

    assert [r["id"] for r in runs] == ["20261020T100000Z-answers", "20261019T100000Z"]
    assert [r["kind"] for r in runs] == ["answers", "retrieval"]


def test_retrieval_run_detail_has_split_tables_and_gate_rows(client, results_dir):
    body = client.get("/eval/runs/20261019T100000Z").json()
    rows = body["splits"]["test"]

    assert body["kind"] == "retrieval" and body["k"] == 5
    assert [r["name"] for r in rows] == ["baseline", "hybrid"]
    assert rows[0]["n"] == 2
    assert rows[0]["metrics"]["recall@5"]["mean"] == pytest.approx(0.5)
    assert rows[0]["delta"] is None and rows[1]["delta"] is not None
    assert body["gate"]["test"]["answerable"] == 2 and body["gate"]["test"]["negatives"] == 1
    assert body["gate"]["dev"] is None  # the dev split has no unanswerable questions


def test_answer_run_detail_computes_the_summary(client, results_dir):
    summary = client.get("/eval/runs/20261020T100000Z-answers").json()["splits"]["test"]

    assert summary["correctness"]["end_to_end_correct"]["mean"] == pytest.approx(1 / 3)
    assert summary["answerable"]["refused"]["mean"] == pytest.approx(1 / 3)
    assert summary["unanswerable"]["refused"]["mean"] == 1.0
    assert summary["faithfulness"]["per_answer"]["mean"] == pytest.approx(0.75)
    assert summary["faithfulness"]["claim_level"] == {"supported": 3, "total": 4}
    assert summary["tokens"]["generation_prompt"] == 400
    assert summary["cost"] is None  # no prices configured


def test_unknown_and_malformed_run_ids_return_404(client, results_dir):
    assert client.get("/eval/runs/20250101T000000Z").status_code == 404
    assert client.get("/eval/runs/not-a-run").status_code == 404
    assert client.get("/eval/runs/..").status_code == 404


def test_unreadable_results_file_returns_422(client, results_dir):
    (results_dir / "20261021T100000Z.json").write_text("{broken")

    assert client.get("/eval/runs/20261021T100000Z").status_code == 422