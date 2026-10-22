from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.deps import CurrentUser, resolve_document_ids
from app.chunking.service import default_chunk_config
from app.config import Settings, get_settings
from app.database import get_db
from app.embeddings.embedder import Embedder, get_embedder
from app.retrieval.pipeline import retrieve
from app.retrieval.rerank import Reranker, get_reranker

router = APIRouter(tags=["search"])


class HybridHitOut(BaseModel):
    chunk_id: int
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float  # RRF score, or the reranker score when rerank=true
    dense_score: float | None
    bm25_score: float | None
    rerank_score: float | None
    text: str


@router.get("/search/hybrid", response_model=list[HybridHitOut])
def search_hybrid(
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
    q: Annotated[str, Query(min_length=1, max_length=500)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    document_id: Annotated[list[str] | None, Query()] = None,
    chunk_config: Annotated[str | None, Query(max_length=64)] = None,
    rerank: Annotated[bool, Query(description="Rescore candidates with a cross-encoder")] = False,
) -> list[HybridHitOut]:
    """BM25 + vector search merged with RRF, optionally reranked. Shows every component score."""
    key = chunk_config or default_chunk_config(settings).key
    mode = "hybrid_rerank" if rerank else "hybrid"
    document_ids = resolve_document_ids(db, user, document_id)
    result = retrieve(
        db, embedder, reranker, q, k, key, document_ids, mode,
        settings.candidate_pool, settings.rrf_k,
    )
    return [
        HybridHitOut(
            chunk_id=h.chunk.id,
            document_id=h.chunk.document_id,
            filename=h.chunk.document.filename,
            page_number=h.chunk.page_number,
            chunk_index=h.chunk.chunk_index,
            score=round(h.score, 4),
            dense_score=round(h.dense_score, 4) if h.dense_score is not None else None,
            bm25_score=round(h.bm25_score, 4) if h.bm25_score is not None else None,
            rerank_score=round(h.rerank_score, 4) if h.rerank_score is not None else None,
            text=h.chunk.text,
        )
        for h in result.hits
    ]