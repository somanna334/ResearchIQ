from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.chunking.service import default_chunk_config
from app.config import Settings, get_settings
from app.database import get_db
from app.retrieval.keyword import keyword_search
from app.schemas import SearchHitOut

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[SearchHitOut])
def search(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    q: Annotated[str, Query(min_length=1, max_length=500)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    document_id: Annotated[list[str] | None, Query()] = None,
    chunk_config: Annotated[str | None, Query(max_length=64)] = None,
) -> list[SearchHitOut]:
    """Keyword (BM25) search. Baseline retrieval; embeddings come in Milestone 4."""
    key = chunk_config or default_chunk_config(settings).key
    hits = keyword_search(db, q, k, key, document_id)
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