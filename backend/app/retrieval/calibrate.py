"""Run from backend/:  python -m app.retrieval.calibrate

Compares the three retrieval modes on a small labelled question set and shows
where a refusal threshold could go. Illustrative only: 14 questions over one
document is far too small for real conclusions (Milestone 8 does that properly).
"""
import json
import sys
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.chunking.service import default_chunk_config
from app.config import get_settings
from app.database import get_engine
from app.embeddings.embedder import get_embedder
from app.retrieval.pipeline import retrieve
from app.retrieval.rerank import get_reranker

QUERIES_PATH = Path("eval/calibration.json")
MODES = ("semantic", "hybrid", "hybrid_rerank")
SHORT = {"semantic": "sem", "hybrid": "hyb", "hybrid_rerank": "rr "}
K = 5


def main() -> int:
    if not QUERIES_PATH.exists():
        print(f"{QUERIES_PATH} not found. Run this from the backend/ folder.")
        return 1
    items = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    settings = get_settings()
    key = default_chunk_config(settings).key
    embedder, reranker = get_embedder(), get_reranker()

    hits = {m: 0 for m in MODES}
    elapsed = {m: 0.0 for m in MODES}
    failed: set[str] = set()
    with_expected = sum(1 for i in items if i.get("expected_pages"))
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    with Session(get_engine()) as db:
        for mode in MODES:  # warm-up: the first call loads each model
            try:
                retrieve(db, embedder, reranker, items[0]["q"], K, key, None, mode,
                         settings.candidate_pool, settings.rrf_k)
            except Exception as exc:
                print(f"Mode {mode} failed to run: {exc}")
                failed.add(mode)

        print(f"{'A/N':<4}{'dense':>7}  " + " ".join(SHORT[m] for m in MODES) + "  question")
        for item in items:
            marks = []
            best_dense = None
            for mode in MODES:
                if mode in failed:
                    marks.append(" - ")
                    continue
                t0 = time.perf_counter()
                result = retrieve(db, embedder, reranker, item["q"], K, key, None, mode,
                                  settings.candidate_pool, settings.rrf_k)
                elapsed[mode] += (time.perf_counter() - t0) * 1000
                if mode == "semantic":
                    best_dense = result.best_dense
                expected = item.get("expected_pages")
                if expected:
                    found = any(h.chunk.page_number in expected for h in result.hits)
                    hits[mode] += found
                    marks.append("YES" if found else "NO ")
                else:
                    marks.append(" . ")
            if best_dense is not None:
                (pos_scores if item["answerable"] else neg_scores).append(best_dense)
            tag = "A" if item["answerable"] else "N"
            dense_text = f"{best_dense:.3f}" if best_dense is not None else "  n/a"
            print(f"{tag:<4}{dense_text:>7}  " + " ".join(marks) + f"  {item['q'][:60]}")

    print(f"\nPage hit@{K} (any expected page in the top {K}) and average time per query:")
    for mode in MODES:
        if mode in failed:
            print(f"  {mode:<14} not available")
            continue
        print(f"  {mode:<14} {hits[mode]}/{with_expected}   {elapsed[mode] / len(items):.0f} ms")

    print("\nRefusal threshold on the best dense cosine score:")
    if pos_scores and neg_scores:
        lowest_pos, highest_neg = min(pos_scores), max(neg_scores)
        print(f"  lowest answerable top score : {lowest_pos:.3f}")
        print(f"  highest unanswerable score  : {highest_neg:.3f}")
        if lowest_pos > highest_neg:
            print(f"  separable. A threshold near {(lowest_pos + highest_neg) / 2:.3f} splits them "
                  f"(margin {lowest_pos - highest_neg:.3f}).")
        else:
            print("  OVERLAP: no single threshold separates them. Unanswerable questions can "
                  "score as high as weak answerable ones, so the gate alone is not enough.")
    print(f"  current min_top_score setting: {settings.min_top_score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())