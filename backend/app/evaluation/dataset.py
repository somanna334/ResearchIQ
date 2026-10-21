import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = ("factoid", "paraphrase", "multi_doc", "near_miss", "off_topic")
SPLITS = ("dev", "test")


class DatasetError(Exception):
    """The dataset file has problems. The message lists all of them."""


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool
    qtype: str
    split: str
    relevant: frozenset[tuple[str, int]]  # (document filename, page number)
    reference_answer: str | None = None


def _is_page(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _parse(item: object, number: int, errors: list[str]) -> Question | None:
    start = len(errors)

    def err(message: str) -> None:
        errors.append(f"line {number}: {message}")

    if not isinstance(item, dict):
        err("each line must be a JSON object")
        return None

    qid, text = item.get("id"), item.get("question")
    if not isinstance(qid, str) or not qid.strip():
        err("'id' must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        err("'question' must be a non-empty string")
    answerable = item.get("answerable")
    if not isinstance(answerable, bool):
        err("'answerable' must be true or false")
    qtype, split = item.get("type"), item.get("split")
    if qtype not in QUESTION_TYPES:
        err(f"'type' must be one of: {', '.join(QUESTION_TYPES)}")
    if split not in SPLITS:
        err(f"'split' must be one of: {', '.join(SPLITS)}")

    relevant: set[tuple[str, int]] = set()
    documents: set[str] = set()
    entries = item.get("relevant", [])
    if not isinstance(entries, list):
        err("'relevant' must be a list")
        entries = []
    for entry in entries:
        doc = entry.get("document") if isinstance(entry, dict) else None
        pages = entry.get("pages") if isinstance(entry, dict) else None
        if (
            not isinstance(doc, str)
            or not doc
            or not isinstance(pages, list)
            or not pages
            or not all(_is_page(p) for p in pages)
        ):
            err("each 'relevant' entry needs a 'document' and a non-empty list of page numbers >= 1")
            continue
        documents.add(doc)
        relevant.update((doc, p) for p in pages)

    if answerable is True and not relevant:
        err("answerable questions need 'relevant' pages")
    if answerable is False and relevant:
        err("unanswerable questions must not list 'relevant' pages")
    if qtype in ("near_miss", "off_topic") and answerable is True:
        err(f"type '{qtype}' must have answerable=false")
    if qtype in ("factoid", "paraphrase", "multi_doc") and answerable is False:
        err(f"type '{qtype}' must have answerable=true")
    if qtype == "multi_doc" and len(documents) < 2:
        err("multi_doc questions need relevant pages in at least two documents")
    reference = item.get("reference_answer")
    if reference is not None and not isinstance(reference, str):
        err("'reference_answer' must be a string")

    if len(errors) > start:
        return None
    return Question(
        id=qid,
        question=text.strip(),
        answerable=answerable,
        qtype=qtype,
        split=split,
        relevant=frozenset(relevant),
        reference_answer=reference.strip() if isinstance(reference, str) and reference.strip() else None,
    )


def load_dataset(path: Path) -> tuple[list[Question], str]:
    """Returns (questions, sha256 of the file). Raises DatasetError listing every problem."""
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        content = raw.decode("utf-8-sig")  # tolerate the BOM some Windows editors add
    except UnicodeDecodeError as exc:
        raise DatasetError("file is not valid UTF-8") from exc

    errors: list[str] = []
    questions: list[Question] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON ({exc.msg})")
            continue

        qid = item.get("id") if isinstance(item, dict) else None
        if isinstance(qid, str):
            if qid in seen_ids:
                errors.append(f"line {number}: duplicate id '{qid}'")
            seen_ids.add(qid)
        text = item.get("question") if isinstance(item, dict) else None
        if isinstance(text, str):
            normalized = " ".join(text.lower().split())
            if normalized in seen_text:
                errors.append(f"line {number}: duplicate question text")
            seen_text.add(normalized)

        question = _parse(item, number, errors)
        if question is not None:
            questions.append(question)

    if errors:
        raise DatasetError("\n".join(errors))
    if not questions:
        raise DatasetError("the dataset is empty")
    return questions, sha