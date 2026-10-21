import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.chunking.service import build_chunks
from app.chunking.strategies import ChunkConfig
from app.config import Settings
from app.embeddings.embedder import Embedder, SentenceTransformerEmbedder
from app.embeddings.service import embed_chunk_set
from app.evaluation.dataset import Question
from app.evaluation.metrics import KS, all_metrics
from app.models import EMBEDDING_DIM, Chunk, Document
from app.retrieval.pipeline import Mode, retrieve
from app.retrieval.rerank import Reranker

MODES = ("semantic", "hybrid", "hybrid_rerank")


class EvalSetupError(Exception):
    """The experiment setup or corpus can't be evaluated."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    chunk: ChunkConfig
    mode: Mode
    embedding_model: str | None = None  # None means the app's configured model
    query_prefix: str | None = None  # only used with a non-default embedding_model


def load_experiments(path: Path) -> tuple[str, int, list[ExperimentConfig]]:
    """Returns (dataset path, headline k, experiments). The first experiment is the baseline."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    experiments: list[ExperimentConfig] = []
    names: set[str] = set()
    for item in data.get("experiments", []):
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise EvalSetupError("every experiment needs a 'name'")
        if name in names:
            raise EvalSetupError(f"duplicate experiment name '{name}'")
        names.add(name)
        mode = item.get("mode", "semantic")
        if mode not in MODES:
            raise EvalSetupError(f"experiment '{name}': mode must be one of {', '.join(MODES)}")
        try:
            chunk = ChunkConfig(**item.get("chunk", {}))
        except ValidationError as exc:
            raise EvalSetupError(f"experiment '{name}': invalid chunk settings ({exc})") from exc
        if item.get("query_prefix") is not None and item.get("embedding_model") is None:
            raise EvalSetupError(f"experiment '{name}': query_prefix needs embedding_model")
        experiments.append(
            ExperimentConfig(name, chunk, mode, item.get("embedding_model"), item.get("query_prefix"))
        )
    if not experiments:
        raise EvalSetupError("no experiments defined")
    k = data.get("k", 5)
    if k not in KS:
        raise EvalSetupError(f"'k' must be one of {KS}")
    return data.get("dataset", "eval/datasets/v1.jsonl"), k, experiments


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        return done.stdout.strip() if done.returncode == 0 else None
    except Exception:
        return None


def run_experiments(
    db: Session,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker,
    questions: list[Question],
    experiments: list[ExperimentConfig],
    k: int,
    *,
    dataset_path: str,
    dataset_sha: str,
    make_embedder: Callable[[str, str], Embedder] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run every experiment over every question and return a JSON-ready result."""
    docs = list(
        db.scalars(select(Document).where(Document.status == "ready").order_by(Document.filename))
    )
    if not docs:
        raise EvalSetupError("no ready documents: upload your corpus first")
    names = [d.filename for d in docs]
    if len(set(names)) != len(names):
        raise EvalSetupError("two uploaded documents share a filename; labels need unique names")
    filename_by_id = {d.id: d.filename for d in docs}

    cache: dict[tuple[str, str], Embedder] = {}

    def embedder_for(exp: ExperimentConfig) -> Embedder:
        if exp.embedding_model is None or exp.embedding_model == embedder.model_name:
            return embedder
        prefix = exp.query_prefix or ""
        key = (exp.embedding_model, prefix)
        if key not in cache:
            factory = make_embedder or (
                lambda model, pre: SentenceTransformerEmbedder(
                    model, settings.embedding_batch_size, pre
                )
            )
            built = factory(exp.embedding_model, prefix)
            if built.dimension != EMBEDDING_DIM:
                raise EvalSetupError(
                    f"{exp.embedding_model} has {built.dimension} dimensions; "
                    f"the database column holds {EMBEDDING_DIM}"
                )
            cache[key] = built
        return cache[key]

    # Build any missing chunk sets and vectors. Existing ones are reused untouched.
    prepared: set[tuple[str, str]] = set()
    for exp in experiments:
        emb = embedder_for(exp)
        marker = (exp.chunk.key, emb.model_name)
        if marker in prepared:
            continue
        prepared.add(marker)
        log(f"Preparing {exp.chunk.key} with {emb.model_name}")
        for doc in docs:
            existing = db.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_id == doc.id, Chunk.chunk_config == exp.chunk.key)
            )
            if not existing:
                build_chunks(db, doc, exp.chunk)
            embed_chunk_set(db, emb, doc.id, exp.chunk.key)

    deepest = max(KS)
    results = []
    for exp in experiments:
        emb = embedder_for(exp)
        key = exp.chunk.key
        log(f"Running {exp.name} ({key}, {exp.mode})")
        # Warm-up so model loading doesn't count as latency.
        retrieve(db, emb, reranker, questions[0].question, deepest, key, None, exp.mode,
                 settings.candidate_pool, settings.rrf_k)

        records = []
        for q in questions:
            started = time.perf_counter()
            found = retrieve(db, emb, reranker, q.question, deepest, key, None, exp.mode,
                             settings.candidate_pool, settings.rrf_k)
            latency_ms = (time.perf_counter() - started) * 1000

            retrieved = [
                {
                    "document": filename_by_id[h.chunk.document_id],
                    "page": h.chunk.page_number,
                    "chunk_index": h.chunk.chunk_index,
                    "score": round(h.score, 4),
                }
                for h in found.hits
            ]
            record: dict = {
                "id": q.id,
                "question": q.question,
                "type": q.qtype,
                "split": q.split,
                "answerable": q.answerable,
                "best_dense": round(found.best_dense, 4) if found.best_dense is not None else None,
                "latency_ms": round(latency_ms, 1),
                "retrieved": retrieved,
            }
            if q.answerable:
                record["relevant"] = [[doc, page] for doc, page in sorted(q.relevant)]
                chunk_pages = [(r["document"], r["page"]) for r in retrieved]
                record["metrics"] = {
                    name: round(value, 6)
                    for name, value in all_metrics(chunk_pages, q.relevant).items()
                }
            records.append(record)

        results.append(
            {
                "name": exp.name,
                "config": {
                    "chunk_config": key,
                    "chunk": exp.chunk.model_dump(),
                    "mode": exp.mode,
                    "embedding_model": emb.model_name,
                },
                "questions": records,
            }
        )

    now = datetime.now(timezone.utc)
    return {
        "schema": 1,
        "meta": {
            "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
            "created_utc": now.isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "k": k,
            "dataset": {"path": dataset_path, "sha256": dataset_sha, "questions": len(questions)},
            "corpus": [{"filename": d.filename, "sha256": d.sha256, "pages": d.page_count} for d in docs],
            "settings": {
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "candidate_pool": settings.candidate_pool,
                "rrf_k": settings.rrf_k,
                "min_top_score": settings.min_top_score,
            },
        },
        "experiments": results,
    }