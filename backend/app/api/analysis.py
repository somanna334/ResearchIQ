from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.ask import ModeName, SourceOut, UsageOut
from app.auth.deps import CurrentUser, get_owned_document
from app.chunking.service import default_chunk_config
from app.config import Settings, get_settings
from app.database import get_db
from app.embeddings.embedder import Embedder, get_embedder
from app.generation.tasks import AnalysisInputError, compare_documents, summarize_document
from app.llm.base import LLMError, LLMProvider
from app.llm.factory import get_llm_provider
from app.models import Document, User
from app.retrieval.rerank import Reranker, get_reranker

router = APIRouter(tags=["analysis"])

Status = Literal["answered", "insufficient_evidence", "citation_error"]


class CoverageOut(BaseModel):
    chunks_used: int
    chunks_total: int
    pages_covered: list[int]
    page_count: int
    sampled: bool


class SummaryResponse(BaseModel):
    status: Status
    summary: str
    sources: list[SourceOut]
    coverage: CoverageOut
    usage: UsageOut  # retrieval_ms here is the time to read the chunks


class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2, max_length=3)
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(4, ge=1, le=6, description="Passages retrieved per document")
    chunk_config: str | None = Field(None, max_length=64)
    mode: ModeName | None = None


class CompareDocumentOut(BaseModel):
    document_id: str
    filename: str
    label: str
    passages: int
    best_dense_score: float | None
    cited: bool


class CompareMetaOut(BaseModel):
    mode: ModeName
    min_top_score: float
    gated: bool


class CompareResponse(BaseModel):
    status: Status
    answer: str
    documents: list[CompareDocumentOut]
    sources: list[SourceOut]
    usage: UsageOut
    retrieval: CompareMetaOut


def _ready_document(db: Session, user: User, document_id: str) -> Document:
    document = get_owned_document(db, user, document_id)
    if document.status != "ready":
        raise HTTPException(
            status_code=409, detail=f"'{document.filename}' has no extractable text"
        )
    return document


def _sources_out(sources, cited_tags) -> list[SourceOut]:
    return [
        SourceOut(
            tag=s.tag, chunk_id=s.chunk_id, document_id=s.document_id, filename=s.filename,
            page_number=s.page_number, score=s.score, text=s.text, cited=s.tag in cited_tags,
        )
        for s in sources
    ]


@router.post("/documents/{document_id}/summary", response_model=SummaryResponse)
def summarize(
    document_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> SummaryResponse:
    """Bullet-point summary with citations. Long documents are sampled evenly."""
    document = _ready_document(db, user, document_id)
    try:
        result = summarize_document(db, llm, document, default_chunk_config(settings).key)
    except AnalysisInputError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=503 if exc.retryable else 502, detail=str(exc)) from exc

    return SummaryResponse(
        status=result.status,
        summary=result.summary,
        sources=_sources_out(result.sources, result.cited_tags),
        coverage=CoverageOut(
            chunks_used=result.coverage.chunks_used,
            chunks_total=result.coverage.chunks_total,
            pages_covered=result.coverage.pages_covered,
            page_count=result.coverage.page_count,
            sampled=result.coverage.sampled,
        ),
        usage=UsageOut(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            retrieval_ms=round(result.prep_ms, 1),
            llm_ms=round(result.llm_ms, 1),
        ),
    )


@router.post("/compare", response_model=CompareResponse)
def compare(
    body: CompareRequest,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> CompareResponse:
    """Compare 2 or 3 of the user's documents, with evidence retrieved from each."""
    if len(set(body.document_ids)) != len(body.document_ids):
        raise HTTPException(status_code=400, detail="Choose different documents to compare")
    documents = [_ready_document(db, user, document_id) for document_id in body.document_ids]
    key = body.chunk_config or default_chunk_config(settings).key
    mode = body.mode or settings.retrieval_mode
    try:
        result = compare_documents(
            db, settings, embedder, reranker, llm, documents, body.question, body.k, key, mode
        )
    except LLMError as exc:
        raise HTTPException(status_code=503 if exc.retryable else 502, detail=str(exc)) from exc

    return CompareResponse(
        status=result.status,
        answer=result.answer,
        documents=[
            CompareDocumentOut(
                document_id=d.document_id,
                filename=d.filename,
                label=d.label,
                passages=d.passages,
                best_dense_score=round(d.best_dense, 4) if d.best_dense is not None else None,
                cited=d.cited,
            )
            for d in result.documents
        ],
        sources=_sources_out(result.sources, result.cited_tags),
        usage=UsageOut(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            retrieval_ms=round(result.retrieval_ms, 1),
            llm_ms=round(result.llm_ms, 1),
        ),
        retrieval=CompareMetaOut(
            mode=result.mode, min_top_score=settings.min_top_score, gated=result.gated
        ),
    )