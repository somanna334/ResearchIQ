from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.chunking.strategies import ChunkConfig, chunk_pages
from app.config import Settings
from app.models import Chunk, Document, Embedding


@dataclass(frozen=True)
class ChunkStats:
    chunk_config: str
    chunk_count: int
    avg_chars: float
    min_chars: int
    max_chars: int


def default_chunk_config(settings: Settings) -> ChunkConfig:
    return ChunkConfig(
        strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )


def build_chunks(db: Session, document: Document, config: ChunkConfig) -> ChunkStats:
    """(Re)build one chunk set for a document. Other chunk sets are untouched."""
    text_chunks = chunk_pages(
        [(page.page_number, page.text) for page in document.pages], config
    )

    # Bulk deletes skip ORM cascades, so remove the old vectors explicitly.
    old_chunk_ids = select(Chunk.id).where(
        Chunk.document_id == document.id, Chunk.chunk_config == config.key
    )
    db.execute(delete(Embedding).where(Embedding.chunk_id.in_(old_chunk_ids)))
    db.execute(
        delete(Chunk).where(
            Chunk.document_id == document.id, Chunk.chunk_config == config.key
        )
    )
    db.add_all(
        Chunk(
            document_id=document.id,
            chunk_config=config.key,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            text=c.text,
            char_count=len(c.text),
        )
        for c in text_chunks
    )
    db.commit()

    lengths = [len(c.text) for c in text_chunks]
    return ChunkStats(
        chunk_config=config.key,
        chunk_count=len(lengths),
        avg_chars=round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        min_chars=min(lengths, default=0),
        max_chars=max(lengths, default=0),
    )