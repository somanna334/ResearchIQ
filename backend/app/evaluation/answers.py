import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.embeddings.embedder import Embedder
from app.evaluation.cost import Pricing
from app.evaluation.dataset import Question
from app.evaluation.judge import Judged, judge_correctness, judge_faithfulness
from app.evaluation.runner import EvalSetupError, _git
from app.generation.service import answer_question
from app.llm.base import LLMError, LLMProvider, LLMResult
from app.models import Document
from app.retrieval.pipeline import Mode
from app.retrieval.rerank import Reranker


class RateLimiter:
    """Spaces calls at least 60/per_minute seconds apart. per_minute=0 disables it."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=time.sleep) -> None:
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0

    def wait(self) -> None:
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self._interval


class ThrottledLLM:
    """Wraps a provider with a shared rate limit and backoff on retryable errors."""

    def __init__(
        self,
        llm: LLMProvider,
        limiter: RateLimiter,
        backoffs: tuple[float, ...] = (20.0, 60.0, 120.0),
        sleep=time.sleep,
    ) -> None:
        self._llm = llm
        self._limiter = limiter
        self._backoffs = backoffs
        self._sleep = sleep
        self.name = llm.name
        self.model_name = llm.model_name
        self.calls = 0
        self.retries = 0
        self.last_call_ms = 0.0  # provider time of the last call, excluding waits and retries

    def generate(self, system: str, prompt: str) -> LLMResult:
        for attempt in range(len(self._backoffs) + 1):
            self._limiter.wait()
            self.calls += 1
            started = time.perf_counter()
            try:
                result = self._llm.generate(system, prompt)
            except LLMError as exc:
                if not exc.retryable or attempt == len(self._backoffs):
                    raise
                self.retries += 1
                self._sleep(self._backoffs[attempt])
                continue
            self.last_call_ms = (time.perf_counter() - started) * 1000
            return result
        raise AssertionError("unreachable")


def _add_judge_tokens(record: dict, judged: Judged) -> None:
    record["usage"]["judge_prompt_tokens"] += judged.prompt_tokens or 0
    record["usage"]["judge_completion_tokens"] += judged.completion_tokens or 0


def evaluate_question(
    q: Question,
    *,
    db: Session,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker,
    gen_llm: LLMProvider,
    judge_llm: LLMProvider,
    mode: Mode,
    k: int,
    chunk_config: str,
    use_judge: bool,
) -> dict:
    record: dict = {
        "id": q.id,
        "question": q.question,
        "type": q.qtype,
        "split": q.split,
        "answerable": q.answerable,
        "has_reference": bool(q.reference_answer),
        "reference_answer": q.reference_answer,
        "status": None,
        "gated": False,
        "answer": None,
        "best_dense": None,
        "retrieved": [],
        "citation": None,
        "correctness": None,
        "faithfulness": None,
        "judge_error": False,
        "error": None,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
        },
        "latency_ms": None,
    }

    calls_before = getattr(gen_llm, "calls", None)
    try:
        result = answer_question(
            db, settings, embedder, reranker, gen_llm, q.question, k, chunk_config, None, mode
        )
    except LLMError as exc:
        record["error"] = str(exc)[:300]
        return record

    llm_ms = result.llm_ms
    if hasattr(gen_llm, "last_call_ms"):
        # Use the provider's own call time, not our throttling waits.
        llm_ms = gen_llm.last_call_ms if gen_llm.calls != calls_before else 0.0

    record.update(
        status=result.status,
        gated=result.gated,
        answer=result.answer,
        best_dense=round(result.best_dense, 4) if result.best_dense is not None else None,
        retrieved=[
            {"document": s.filename, "page": s.page_number, "cited": s.tag in result.cited_tags}
            for s in result.sources
        ],
        latency_ms={"retrieval": round(result.retrieval_ms, 1), "llm": round(llm_ms, 1)},
    )
    record["usage"]["prompt_tokens"] = result.prompt_tokens or 0
    record["usage"]["completion_tokens"] = result.completion_tokens or 0

    if q.answerable and result.status == "answered":
        cited = [s for s in result.sources if s.tag in result.cited_tags]
        cited_pages = {(s.filename, s.page_number) for s in cited}
        on_target = sum(1 for s in cited if (s.filename, s.page_number) in q.relevant)
        record["citation"] = {
            "cited_pages": [[d, p] for d, p in sorted(cited_pages)],
            "precision": on_target / len(cited) if cited else 0.0,
            "recall": len(cited_pages & q.relevant) / len(q.relevant),
            "hits_relevant": bool(cited_pages & q.relevant),
        }

    if use_judge and result.status == "answered":
        try:
            if q.answerable and q.reference_answer:
                judged = judge_correctness(judge_llm, q.question, q.reference_answer, result.answer)
                _add_judge_tokens(record, judged)
                if judged.ok:
                    record["correctness"] = {"verdict": judged.verdict, "reason": judged.reason}
                else:
                    record["judge_error"] = True
            judged = judge_faithfulness(judge_llm, result.answer, result.sources)
            _add_judge_tokens(record, judged)
            if judged.ok:
                record["faithfulness"] = {"claims": judged.claims, "supported": judged.supported}
            else:
                record["judge_error"] = True
        except LLMError as exc:
            record["judge_error"] = True
            record["judge_message"] = str(exc)[:200]
    return record


def _pricing_dict(pricing: Pricing | None) -> dict | None:
    return asdict(pricing) if pricing else None


def run_answer_eval(
    db: Session,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker,
    gen_llm: LLMProvider,
    judge_llm: LLMProvider,
    questions: list[Question],
    *,
    mode: Mode,
    k: int,
    chunk_config: str,
    dataset_path: str,
    dataset_sha: str,
    run_id: str,
    use_judge: bool = True,
    pricing: dict[str, Pricing | None] | None = None,
    done: dict[str, dict] | None = None,
    on_record: Callable[[dict], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    docs = list(
        db.scalars(select(Document).where(Document.status == "ready").order_by(Document.filename))
    )
    if not docs:
        raise EvalSetupError("no ready documents: upload your corpus first")
    done = done or {}
    pricing = pricing or {}

    if any(q.id not in done for q in questions):
        # Warm-up, so model loading doesn't count as latency.
        embedder.embed_query("warm up")
        if mode == "hybrid_rerank":
            reranker.score("warm up", ["warm up"])

    records: list[dict] = []
    for index, q in enumerate(questions, start=1):
        if q.id in done:
            records.append(done[q.id])
            continue
        log(f"[{index}/{len(questions)}] {q.id}: {q.question[:70]}")
        record = evaluate_question(
            q, db=db, settings=settings, embedder=embedder, reranker=reranker,
            gen_llm=gen_llm, judge_llm=judge_llm, mode=mode, k=k,
            chunk_config=chunk_config, use_judge=use_judge,
        )
        records.append(record)
        if on_record:
            on_record(record)

    return {
        "kind": "answers",
        "schema": 1,
        "meta": {
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "mode": mode,
            "k": k,
            "chunk_config": chunk_config,
            "dataset": {"path": dataset_path, "sha256": dataset_sha, "questions": len(questions)},
            "corpus": [{"filename": d.filename, "sha256": d.sha256, "pages": d.page_count} for d in docs],
            "generation": {"model": gen_llm.model_name},
            "judge": {"enabled": use_judge, "model": judge_llm.model_name if use_judge else None},
            "settings": {
                "min_top_score": settings.min_top_score,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "candidate_pool": settings.candidate_pool,
                "rrf_k": settings.rrf_k,
                "requests_per_minute": settings.eval_requests_per_minute,
            },
            "pricing": {
                "generation": _pricing_dict(pricing.get("generation")),
                "judge": _pricing_dict(pricing.get("judge")),
            },
            "llm_calls_this_session": {
                "generation": getattr(gen_llm, "calls", None),
                "judge": getattr(judge_llm, "calls", None) if use_judge else 0,
                "retries": getattr(gen_llm, "retries", None),
            },
        },
        "questions": records,
    }