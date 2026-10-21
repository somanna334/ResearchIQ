"""Check the dataset against your uploaded documents.

    python -m app.evaluation.validate            # problems and balance warnings
    python -m app.evaluation.validate --show     # also print the text of every labelled page
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_engine
from app.evaluation.dataset import DatasetError, Question, load_dataset
from app.models import Document, Page


def label_problems(db: Session, questions: list[Question]) -> list[str]:
    docs = list(db.scalars(select(Document).where(Document.status == "ready")))
    by_name: dict[str, list[Document]] = {}
    for doc in docs:
        by_name.setdefault(doc.filename, []).append(doc)

    problems: list[str] = []
    for q in questions:
        for name, page in sorted(q.relevant):
            matches = by_name.get(name, [])
            if not matches:
                problems.append(f"{q.id}: document '{name}' is not uploaded (or not ready)")
            elif len(matches) > 1:
                problems.append(f"{q.id}: several uploaded documents are named '{name}'")
            elif page > matches[0].page_count:
                problems.append(
                    f"{q.id}: {name} has {matches[0].page_count} pages, but page {page} is listed"
                )
    return list(dict.fromkeys(problems))


def _print_pages(db: Session, questions: list[Question]) -> None:
    for q in questions:
        for name, page in sorted(q.relevant):
            row = db.scalar(
                select(Page)
                .join(Document, Document.id == Page.document_id)
                .where(Document.filename == name, Page.page_number == page)
            )
            text = " ".join(row.text.split())[:300] if row else "(page not found)"
            print(f"[{q.id}] {q.question}\n    {name} p.{page}: {text}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", default="eval/datasets/v1.jsonl")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles choke on some PDF characters

    try:
        questions, sha = load_dataset(Path(args.dataset))
    except FileNotFoundError:
        print(f"{args.dataset} not found. Run this from the backend folder.")
        return 1
    except DatasetError as exc:
        print("Dataset problems:\n" + str(exc))
        return 1

    with Session(get_engine()) as db:
        problems = label_problems(db, questions)
        if args.show:
            _print_pages(db, questions)

    counts = Counter((q.split, q.answerable) for q in questions)
    types = Counter(q.qtype for q in questions)
    print(f"{len(questions)} questions, sha256 {sha[:12]}")
    print(f"  dev:  {counts[('dev', True)]} answerable, {counts[('dev', False)]} unanswerable")
    print(f"  test: {counts[('test', True)]} answerable, {counts[('test', False)]} unanswerable")
    print("  types: " + ", ".join(f"{t}={n}" for t, n in sorted(types.items())))

    warnings: list[str] = []
    if counts[("test", True)] < 15:
        warnings.append("fewer than 15 answerable test questions: results will be inconclusive")
    if sum(1 for q in questions if not q.answerable) < 0.2 * len(questions):
        warnings.append("less than 20% unanswerable questions: refusal behaviour is under-tested")
    if types["near_miss"] < 5:
        warnings.append("fewer than 5 near_miss questions: the hard negatives are missing")
    missing = sum(1 for q in questions if q.answerable and not q.reference_answer)
    if missing:
        warnings.append(f"{missing} answerable questions have no reference_answer (needed for answer scoring)")
    for warning in warnings:
        print("WARNING:", warning)

    if problems:
        print("\nLabel problems:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("\nAll labels point at uploaded documents and valid pages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())