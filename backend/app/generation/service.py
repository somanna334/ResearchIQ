import logging
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import Settings
from app.embeddings.embedder import Embedder
from app.generation.citations import Status, parse_answer
from app.generation.prompt import Source, build_prompt
from app.llm.base import LLMProvider
from app.retrieval.pipeline import Mode, retrieve
from app.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

INSUFFICIENT_MESSAGE = "I can't answer this confidently from the selected documents."
WITHHELD_MESSAGE = (
    "The model's answer could not be verified against the retrieved sources, "
    "so it was withheld."
)


@dataclass(frozen=True)
class AnswerResult:
    status: Status
    answer: str
    sources: list[Source]
    cited_tags: set[str]
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_ms: float
    llm_ms: float
    mode: str
    best_dense: float | None
    gated: bool  # True when weak retrieval stopped us before calling the LLM


def answer_question(
    db: Session,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLMProvider,
    question: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None,
    mode: Mode,
) -> AnswerResult:
    started = time.perf_counter()
    retrieved = retrieve(
        db, embedder, reranker, question, k, chunk_config, document_ids, mode,
        settings.candidate_pool, settings.rrf_k,
    )
    sources = [
        Source(
            tag=f"S{i}",
            chunk_id=h.chunk.id,
            document_id=h.chunk.document_id,
            filename=h.chunk.document.filename,
            page_number=h.chunk.page_number,
            score=round(h.score, 4),
            text=h.chunk.text,
        )
        for i, h in enumerate(retrieved.hits, start=1)
    ]
    retrieval_ms = (time.perf_counter() - started) * 1000

    def declined(gated: bool) -> AnswerResult:
        return AnswerResult(
            "insufficient_evidence", INSUFFICIENT_MESSAGE, sources, set(), llm.model_name,
            None, None, retrieval_ms, 0.0, mode, retrieved.best_dense, gated,
        )

    if not sources:
        logger.info("ask status=insufficient_evidence reason=no_sources mode=%s", mode)
        return declined(False)

    # Code-level refusal: never rely on the prompt alone to catch weak evidence.
    # Skipped when nothing is embedded (best_dense is None), because we can't judge.
    if retrieved.best_dense is not None and retrieved.best_dense < settings.min_top_score:
        logger.info(
            "ask status=insufficient_evidence reason=weak_retrieval mode=%s best_dense=%.4f threshold=%.2f",
            mode, retrieved.best_dense, settings.min_top_score,
        )
        return declined(True)

    system, prompt = build_prompt(question, sources)
    llm_started = time.perf_counter()
    result = llm.generate(system, prompt)  # may raise LLMError; the API maps it
    llm_ms = (time.perf_counter() - llm_started) * 1000

    parsed = parse_answer(result.text, len(sources))
    if parsed.status == "answered":
        answer = result.text
    elif parsed.status == "insufficient_evidence":
        answer = INSUFFICIENT_MESSAGE
    else:
        answer = WITHHELD_MESSAGE
        logger.warning(
            "citation_error invalid=%s cited=%s raw=%r",
            parsed.invalid, parsed.cited, result.text[:200],
        )

    logger.info(
        "ask status=%s mode=%s k=%d retrieval_ms=%.0f llm_ms=%.0f prompt_tokens=%s completion_tokens=%s",
        parsed.status, mode, len(sources), retrieval_ms, llm_ms,
        result.prompt_tokens, result.completion_tokens,
    )
    return AnswerResult(
        status=parsed.status,
        answer=answer,
        sources=sources,
        cited_tags={f"S{n}" for n in parsed.cited} if parsed.status == "answered" else set(),
        model=llm.model_name,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        mode=mode,
        best_dense=retrieved.best_dense,
        gated=False,
    )