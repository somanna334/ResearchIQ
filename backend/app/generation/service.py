import logging
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.embeddings.embedder import Embedder
from app.generation.citations import Status, parse_answer
from app.generation.prompt import Source, build_prompt
from app.llm.base import LLMProvider
from app.retrieval.semantic import semantic_search

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


def answer_question(
    db: Session,
    embedder: Embedder,
    llm: LLMProvider,
    question: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None,
) -> AnswerResult:
    started = time.perf_counter()
    hits = semantic_search(db, embedder, question, k, chunk_config, document_ids)
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
        for i, h in enumerate(hits, start=1)
    ]
    retrieval_ms = (time.perf_counter() - started) * 1000

    if not sources:
        # Nothing retrieved (e.g. nothing embedded yet): don't spend an LLM call.
        logger.info("ask status=insufficient_evidence reason=no_sources")
        return AnswerResult(
            "insufficient_evidence", INSUFFICIENT_MESSAGE, [], set(), llm.model_name,
            None, None, retrieval_ms, 0.0,
        )

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
        "ask status=%s k=%d retrieval_ms=%.0f llm_ms=%.0f prompt_tokens=%s completion_tokens=%s",
        parsed.status, len(sources), retrieval_ms, llm_ms,
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
    )