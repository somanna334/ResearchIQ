import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.chunking.service import default_chunk_config
from app.config import Settings, get_settings
from app.database import get_db
from app.embeddings.embedder import Embedder, get_embedder
from app.embeddings.service import embed_chunk_set
from app.models import Chunk, Document
from app.retrieval.semantic import semantic_search
from app.schemas import SearchHitOut

router = APIRouter(tags=["embeddings"])


class EmbedStatsOut(BaseModel):
    chunk_config: str
    model_name: str
    total_chunks: int
    newly_embedded: int
    seconds: float


@router.post("/documents/{document_id}/embeddings", response_model=EmbedStatsOut)
def embed_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    chunk_config: Annotated[str | None, Query(max_length=64)] = None,
) -> EmbedStatsOut:
    """Embed a document's chunk set. Safe to repeat: only missing vectors are computed."""
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    key = chunk_config or default_chunk_config(settings).key
    chunk_total = db.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.document_id == document_id, Chunk.chunk_config == key)
    )
    if not chunk_total:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{key}'")

    started = time.perf_counter()
    total, new = embed_chunk_set(db, embedder, document_id, key)
    return EmbedStatsOut(
        chunk_config=key,
        model_name=embedder.model_name,
        total_chunks=total,
        newly_embedded=new,
        seconds=round(time.perf_counter() - started, 2),
    )


@router.get("/search/semantic", response_model=list[SearchHitOut])
def search_semantic(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    q: Annotated[str, Query(min_length=1, max_length=500)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    document_id: Annotated[list[str] | None, Query()] = None,
    chunk_config: Annotated[str | None, Query(max_length=64)] = None,
) -> list[SearchHitOut]:
    """Semantic search. `score` is cosine similarity. It never returns 'no match'."""
    key = chunk_config or default_chunk_config(settings).key
    hits = semantic_search(db, embedder, q, k, key, document_id)
    return [
        SearchHitOut(
            chunk_id=h.chunk.id,
            document_id=h.chunk.document_id,
            filename=h.chunk.document.filename,
            page_number=h.chunk.page_number,
            chunk_index=h.chunk.chunk_index,
            score=round(h.score, 4),
            text=h.chunk.text,
        )
        for h in hits
    ]