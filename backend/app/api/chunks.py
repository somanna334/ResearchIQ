from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.deps import CurrentUser, get_owned_document
from app.chunking.service import build_chunks, default_chunk_config
from app.chunking.strategies import ChunkConfig
from app.config import Settings, get_settings
from app.database import get_db
from app.models import Chunk
from app.schemas import ChunkOut, ChunkStatsOut

router = APIRouter(prefix="/documents/{document_id}/chunks", tags=["chunks"])


@router.post("", response_model=ChunkStatsOut)
def rechunk_document(
    document_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    config: Annotated[ChunkConfig | None, Body()] = None,
) -> ChunkStatsOut:
    """Build (or rebuild) a chunk set. With no body, uses the default configuration."""
    document = get_owned_document(db, user, document_id)
    if document.status != "ready":
        raise HTTPException(
            status_code=409, detail="Document has no extractable text to chunk"
        )
    stats = build_chunks(db, document, config or default_chunk_config(settings))
    return ChunkStatsOut.model_validate(stats)


@router.get("", response_model=list[ChunkOut])
def list_chunks(
    document_id: str,
    user: CurrentUser,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    chunk_config: Annotated[str | None, Query(max_length=64)] = None,
    page_number: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Chunk]:
    get_owned_document(db, user, document_id)
    key = chunk_config or default_chunk_config(settings).key
    stmt = select(Chunk).where(
        Chunk.document_id == document_id, Chunk.chunk_config == key
    )
    if page_number is not None:
        stmt = stmt.where(Chunk.page_number == page_number)
    stmt = stmt.order_by(Chunk.chunk_index).offset(offset).limit(limit)
    return list(db.scalars(stmt))