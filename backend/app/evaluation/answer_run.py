"""Evaluate full answers:  python -m app.evaluation.answer_run [options]"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.chunking.service import default_chunk_config
from app.config import get_settings
from app.database import get_engine
from app.embeddings.embedder import get_embedder
from app.evaluation.answer_report import render_answer_report, render_audit
from app.evaluation.answers import RateLimiter, ThrottledLLM, run_answer_eval
from app.evaluation.cost import load_pricing
from app.evaluation.dataset import DatasetError, load_dataset
from app.evaluation.runner import EvalSetupError
from app.evaluation.validate import label_problems
from app.llm.base import LLMError
from app.llm.factory import build_provider
from app.retrieval.rerank import get_reranker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/datasets/v1.jsonl")
    parser.add_argument("--mode", choices=["semantic", "hybrid", "hybrid_rerank"], default=None,
                        help="default: the server's retrieval mode")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke tests)")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge (cheaper, fewer metrics)")
    parser.add_argument("--resume", default=None, metavar="RUN_ID", help="continue an interrupted run")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--pricing", default="eval/pricing.json")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    mode = args.mode or settings.retrieval_mode
    try:
        questions, sha = load_dataset(Path(args.dataset))
    except FileNotFoundError:
        print(f"{args.dataset} not found. Run this from the backend folder.")
        return 1
    except DatasetError as exc:
        print("Dataset problems:\n" + str(exc))
        return 1

    if args.split != "all":
        questions = [q for q in questions if q.split == args.split]
    if args.limit:
        questions = questions[: args.limit]
        print(f"NOTE: only the first {len(questions)} questions. Not representative; use for smoke tests.")
    if not questions:
        print("No questions selected.")
        return 1

    use_judge = not args.no_judge
    try:
        gen_raw = build_provider(settings)
        judge_raw = (
            build_provider(settings.model_copy(update={"llm_model": settings.judge_model or settings.llm_model}))
            if use_judge
            else gen_raw
        )
    except LLMError as exc:
        print("LLM not available:", exc)
        return 1

    limiter = RateLimiter(settings.eval_requests_per_minute)
    gen = ThrottledLLM(gen_raw, limiter)
    judge = ThrottledLLM(judge_raw, limiter) if use_judge else gen

    upper = sum(3 if q.answerable else 2 for q in questions) if use_judge else len(questions)
    rate = max(1, settings.eval_requests_per_minute)
    print(f"{len(questions)} questions, up to {upper} model calls at {rate}/minute "
          f"(at most about {upper / rate:.0f} minutes). Mode: {mode}, k={args.k}.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_config = default_chunk_config(settings).key
    header = {
        "dataset_sha256": sha, "mode": mode, "k": args.k, "split": args.split,
        "limit": args.limit, "use_judge": use_judge, "chunk_config": chunk_config,
    }
    run_id = args.resume or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-answers"
    partial = out_dir / f"{run_id}.partial.jsonl"

    done: dict[str, dict] = {}
    if args.resume:
        if not partial.exists():
            print(f"No checkpoint found at {partial}")
            return 1
        lines = partial.read_text(encoding="utf-8").splitlines()
        saved = json.loads(lines[0]).get("_header") if lines else None
        if saved != header:
            print("The checkpoint was made with different settings or a different dataset; start a new run.")
            return 1
        for line in lines[1:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line from an interruption
            done[record["id"]] = record
        print(f"Resuming: {len(done)} questions already done.")
    else:
        partial.write_text(json.dumps({"_header": header}) + "\n", encoding="utf-8")

    def save(record: dict) -> None:
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    pricing = {
        "generation": load_pricing(Path(args.pricing), gen.model_name),
        "judge": load_pricing(Path(args.pricing), judge.model_name) if use_judge else None,
    }
    if pricing["generation"] is None:
        print(f"NOTE: no price for '{gen.model_name}' in {args.pricing}; cost will not be estimated.")

    with Session(get_engine()) as db:
        problems = label_problems(db, questions)
        if problems:
            print("Fix these label problems first (run validate for details):")
            for problem in problems:
                print(" -", problem)
            return 1
        try:
            results = run_answer_eval(
                db, settings, get_embedder(), get_reranker(), gen, judge, questions,
                mode=mode, k=args.k, chunk_config=chunk_config,
                dataset_path=args.dataset, dataset_sha=sha, run_id=run_id,
                use_judge=use_judge, pricing=pricing, done=done, on_record=save,
            )
        except EvalSetupError as exc:
            print("Setup problem:", exc)
            return 1
        except KeyboardInterrupt:
            print(f"\nInterrupted. Continue with:  python -m app.evaluation.answer_run --resume {run_id} "
                  "(and the same other options)")
            return 130

    json_path = out_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / f"{run_id}.md").write_text(render_answer_report(results), encoding="utf-8")
    audit_path = out_dir / f"{run_id}-audit.md"
    audit_path.write_text(render_audit(results), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"\nWrote {json_path}\nWrote {out_dir / (run_id + '.md')}\nWrote {audit_path}  <- read this and check the judge")
    return 0


if __name__ == "__main__":
    sys.exit(main())