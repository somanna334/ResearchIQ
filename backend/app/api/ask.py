from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.chunking.service import default_chunk_config
from app.config import Settings, get_settings
from app.database import get_db
from app.embeddings.embedder import Embedder, get_embedder
from app.generation.service import answer_question
from app.llm.base import LLMError, LLMProvider
from app.llm.factory import get_llm_provider
from app.retrieval.rerank import Reranker, get_reranker

router = APIRouter(tags=["ask"])

ModeName = Literal["semantic", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=10)
    document_ids: list[str] | None = None
    chunk_config: str | None = Field(None, max_length=64)
    mode: ModeName | None = Field(None, description="Defaults to the server setting")


class SourceOut(BaseModel):
    tag: str
    chunk_id: int
    document_id: str
    filename: str
    page_number: int
    score: float  # meaning depends on the retrieval mode
    text: str
    cited: bool


class UsageOut(BaseModel):
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_ms: float
    llm_ms: float


class RetrievalOut(BaseModel):
    mode: ModeName
    best_dense_score: float | None
    min_top_score: float
    gated: bool


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "citation_error"]
    answer: str
    sources: list[SourceOut]
    usage: UsageOut
    retrieval: RetrievalOut


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> AskResponse:
    """Answer from the documents, with validated citations."""
    key = body.chunk_config or default_chunk_config(settings).key
    mode = body.mode or settings.retrieval_mode
    try:
        result = answer_question(
            db, settings, embedder, reranker, llm, body.question, body.k, key,
            body.document_ids, mode,
        )
    except LLMError as exc:
        raise HTTPException(
            status_code=503 if exc.retryable else 502, detail=str(exc)
        ) from exc

    return AskResponse(
        status=result.status,
        answer=result.answer,
        sources=[
            SourceOut(
                tag=s.tag,
                chunk_id=s.chunk_id,
                document_id=s.document_id,
                filename=s.filename,
                page_number=s.page_number,
                score=s.score,
                text=s.text,
                cited=s.tag in result.cited_tags,
            )
            for s in result.sources
        ],
        usage=UsageOut(
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            retrieval_ms=round(result.retrieval_ms, 1),
            llm_ms=round(result.llm_ms, 1),
        ),
        retrieval=RetrievalOut(
            mode=result.mode,
            best_dense_score=(
                round(result.best_dense, 4) if result.best_dense is not None else None
            ),
            min_top_score=settings.min_top_score,
            gated=result.gated,
        ),
    )