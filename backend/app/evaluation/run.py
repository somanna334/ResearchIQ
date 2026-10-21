"""Run all experiments:  python -m app.evaluation.run [--detail NAME]"""
import argparse
import json
import sys
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_engine
from app.embeddings.embedder import get_embedder
from app.evaluation.dataset import DatasetError, load_dataset
from app.evaluation.report import render_report
from app.evaluation.runner import EvalSetupError, load_experiments, run_experiments
from app.evaluation.validate import label_problems
from app.retrieval.rerank import get_reranker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="eval/experiments.json")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--detail", default=None, help="experiment for the type and threshold sections")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        dataset_path, k, experiments = load_experiments(Path(args.experiments))
        questions, sha = load_dataset(Path(dataset_path))
    except FileNotFoundError as exc:
        print(f"File not found: {exc.filename}. Run this from the backend folder.")
        return 1
    except (EvalSetupError, DatasetError) as exc:
        print("Setup problem:\n" + str(exc))
        return 1

    settings = get_settings()
    with Session(get_engine()) as db:
        problems = label_problems(db, questions)
        if problems:
            print("Fix these label problems first (run validate for details):")
            for problem in problems:
                print(" -", problem)
            return 1
        try:
            results = run_experiments(
                db, settings, get_embedder(), get_reranker(), questions, experiments, k,
                dataset_path=dataset_path, dataset_sha=sha,
            )
        except EvalSetupError as exc:
            print("Setup problem:", exc)
            return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = results["meta"]["run_id"]
    json_path = out_dir / f"{run_id}.json"
    md_path = out_dir / f"{run_id}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_report(results, settings.min_top_score, args.detail), encoding="utf-8")
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())