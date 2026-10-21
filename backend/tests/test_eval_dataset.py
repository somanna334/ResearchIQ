import json

import pytest

from app.evaluation.dataset import DatasetError, load_dataset


def line(qid, text, answerable=True, qtype="factoid", split="dev", relevant=None, **extra):
    item = {"id": qid, "question": text, "answerable": answerable, "type": qtype, "split": split}
    # An explicit `relevant` is always written, even for unanswerable questions,
    # so tests can build deliberately invalid lines.
    if relevant is not None or answerable:
        item["relevant"] = relevant or [{"document": "a.pdf", "pages": [1]}]
    item.update(extra)
    return json.dumps(item)


def write(tmp_path, *lines):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_valid_file_loads_with_a_hash(tmp_path):
    path = write(
        tmp_path,
        line("q1", "What is A?", reference_answer="It is A."),
        line("q2", "Unrelated?", answerable=False, qtype="off_topic", split="test"),
    )
    questions, sha = load_dataset(path)

    assert [q.id for q in questions] == ["q1", "q2"]
    assert questions[0].relevant == frozenset({("a.pdf", 1)})
    assert questions[0].reference_answer == "It is A."
    assert len(sha) == 64


def test_every_problem_is_reported_at_once(tmp_path):
    bad_type = line("q1", "First?", qtype="banana")
    duplicate = line("q1", "Second?")
    no_pages = json.dumps(
        {"id": "q3", "question": "Third?", "answerable": True, "type": "factoid", "split": "dev"}
    )
    with pytest.raises(DatasetError) as caught:
        load_dataset(write(tmp_path, bad_type, duplicate, no_pages))

    message = str(caught.value)
    assert "line 1" in message and "'type'" in message
    assert "duplicate id 'q1'" in message
    assert "line 3" in message and "need 'relevant' pages" in message


def test_unanswerable_question_cannot_list_pages(tmp_path):
    bad = line(
        "q1", "Nope?", answerable=False, qtype="off_topic",
        relevant=[{"document": "a.pdf", "pages": [1]}],
    )
    with pytest.raises(DatasetError, match="must not list 'relevant' pages"):
        load_dataset(write(tmp_path, bad))


def test_multi_doc_needs_two_documents(tmp_path):
    with pytest.raises(DatasetError, match="at least two documents"):
        load_dataset(write(tmp_path, line("q1", "Both?", qtype="multi_doc")))


def test_invalid_json_reports_the_line_number(tmp_path):
    with pytest.raises(DatasetError, match="line 2: invalid JSON"):
        load_dataset(write(tmp_path, line("q1", "Fine?"), "{not json"))