import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.embeddings.embedder import Embedder
from app.generation.citations import Status, parse_answer
from app.generation.prompt import Source, build_compare_prompt, build_summary_prompt
from app.generation.service import INSUFFICIENT_MESSAGE, WITHHELD_MESSAGE
from app.llm.base import LLMProvider
from app.models import Chunk, Document
from app.retrieval.pipeline import Mode, retrieve
from app.retrieval.rerank import Reranker

logger = logging.getLogger(__name__)

# About 7,500 tokens. Longer documents are sampled, and the response says so.
SUMMARY_BUDGET_CHARS = 30_000


class AnalysisInputError(Exception):
    """The request can't be processed (for example, the document has no chunks)."""


class _HasText(Protocol):
    @property
    def text(self) -> str: ...


T = TypeVar("T", bound=_HasText)


def select_chunks(chunks: Sequence[T], budget: int) -> list[T]:
    """All chunks if they fit the budget, otherwise evenly spaced ones that do."""
    total = sum(len(c.text) for c in chunks)
    if total <= budget:
        return list(chunks)
    average = total / len(chunks)
    wanted = max(1, min(len(chunks), int(budget // average)))
    step = len(chunks) / wanted
    picked: list[T] = []
    used = 0
    for i in range(wanted):
        chunk = chunks[int(i * step)]
        if picked and used + len(chunk.text) > budget:
            break
        picked.append(chunk)
        used += len(chunk.text)
    return picked


def _judge(text: str, source_count: int) -> tuple[Status, str, set[str]]:
    """Classify a model reply: (status, text to show, cited tags)."""
    parsed = parse_answer(text, source_count)
    if parsed.status == "answered":
        return parsed.status, text, {f"S{n}" for n in parsed.cited}
    if parsed.status == "insufficient_evidence":
        return parsed.status, INSUFFICIENT_MESSAGE, set()
    logger.warning(
        "citation_error invalid=%s cited=%s raw=%r", parsed.invalid, parsed.cited, text[:200]
    )
    return parsed.status, WITHHELD_MESSAGE, set()


# ---------- summary ----------

@dataclass(frozen=True)
class Coverage:
    chunks_used: int
    chunks_total: int
    pages_covered: list[int]
    page_count: int
    sampled: bool


@dataclass(frozen=True)
class SummaryResult:
    status: Status
    summary: str
    sources: list[Source]
    cited_tags: set[str]
    coverage: Coverage
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    prep_ms: float
    llm_ms: float


def summarize_document(
    db: Session, llm: LLMProvider, document: Document, chunk_config: str
) -> SummaryResult:
    started = time.perf_counter()
    chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document.id, Chunk.chunk_config == chunk_config)
            .order_by(Chunk.chunk_index)
        )
    )
    if not chunks:
        raise AnalysisInputError(f"No chunks found for '{chunk_config}'")

    selected = select_chunks(chunks, SUMMARY_BUDGET_CHARS)
    sources = [
        Source(
            tag=f"S{i}",
            chunk_id=c.id,
            document_id=c.document_id,
            filename=document.filename,
            page_number=c.page_number,
            score=0.0,  # not applicable: nothing was ranked
            text=c.text,
        )
        for i, c in enumerate(selected, start=1)
    ]
    prep_ms = (time.perf_counter() - started) * 1000

    system, prompt = build_summary_prompt(sources)
    llm_started = time.perf_counter()
    result = llm.generate(system, prompt)  # may raise LLMError; the API maps it
    llm_ms = (time.perf_counter() - llm_started) * 1000

    status, summary, cited = _judge(result.text, len(sources))
    coverage = Coverage(
        chunks_used=len(selected),
        chunks_total=len(chunks),
        pages_covered=sorted({s.page_number for s in sources}),
        page_count=document.page_count,
        sampled=len(selected) < len(chunks),
    )
    logger.info(
        "summary status=%s chunks=%d/%d llm_ms=%.0f prompt_tokens=%s completion_tokens=%s",
        status, len(selected), len(chunks), llm_ms, result.prompt_tokens, result.completion_tokens,
    )
    return SummaryResult(
        status, summary, sources, cited, coverage, llm.model_name,
        result.prompt_tokens, result.completion_tokens, prep_ms, llm_ms,
    )


# ---------- comparison ----------

@dataclass(frozen=True)
class DocumentCoverage:
    document_id: str
    filename: str
    label: str
    passages: int
    best_dense: float | None
    cited: bool


@dataclass(frozen=True)
class CompareResult:
    status: Status
    answer: str
    sources: list[Source]
    cited_tags: set[str]
    documents: list[DocumentCoverage]
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_ms: float
    llm_ms: float
    mode: str
    gated: bool


def compare_documents(
    db: Session,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLMProvider,
    documents: list[Document],
    question: str,
    k: int,
    chunk_config: str,
    mode: Mode,
) -> CompareResult:
    labels = {doc.id: chr(ord("A") + i) for i, doc in enumerate(documents)}
    started = time.perf_counter()

    sources: list[Source] = []
    per_document: list[tuple[Document, int, float | None]] = []
    for doc in documents:
        # Retrieve per document so every document contributes evidence.
        retrieved = retrieve(
            db, embedder, reranker, question, k, chunk_config, [doc.id], mode,
            settings.candidate_pool, settings.rrf_k,
        )
        for hit in retrieved.hits:
            sources.append(
                Source(
                    tag=f"S{len(sources) + 1}",
                    chunk_id=hit.chunk.id,
                    document_id=hit.chunk.document_id,
                    filename=doc.filename,
                    page_number=hit.chunk.page_number,
                    score=round(hit.score, 4),
                    text=hit.chunk.text,
                )
            )
        per_document.append((doc, len(retrieved.hits), retrieved.best_dense))
    retrieval_ms = (time.perf_counter() - started) * 1000

    def coverage(cited_tags: set[str]) -> list[DocumentCoverage]:
        cited_docs = {s.document_id for s in sources if s.tag in cited_tags}
        return [
            DocumentCoverage(doc.id, doc.filename, labels[doc.id], count, best, doc.id in cited_docs)
            for doc, count, best in per_document
        ]

    def declined(gated: bool) -> CompareResult:
        return CompareResult(
            "insufficient_evidence", INSUFFICIENT_MESSAGE, sources, set(), coverage(set()),
            llm.model_name, None, None, retrieval_ms, 0.0, mode, gated,
        )

    if not sources:
        logger.info("compare status=insufficient_evidence reason=no_sources")
        return declined(False)

    judged = [best for _, _, best in per_document if best is not None]
    if judged and all(best < settings.min_top_score for best in judged):
        logger.info("compare status=insufficient_evidence reason=weak_retrieval docs=%d", len(documents))
        return declined(True)

    system, prompt = build_compare_prompt(question, sources, labels)
    llm_started = time.perf_counter()
    result = llm.generate(system, prompt)  # may raise LLMError; the API maps it
    llm_ms = (time.perf_counter() - llm_started) * 1000

    status, answer, cited = _judge(result.text, len(sources))
    logger.info(
        "compare status=%s docs=%d mode=%s retrieval_ms=%.0f llm_ms=%.0f prompt_tokens=%s",
        status, len(documents), mode, retrieval_ms, llm_ms, result.prompt_tokens,
    )
    return CompareResult(
        status, answer, sources, cited, coverage(cited), llm.model_name,
        result.prompt_tokens, result.completion_tokens, retrieval_ms, llm_ms, mode, False,
    )