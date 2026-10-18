import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import Embedder
from app.models import Chunk, Embedding

logger = logging.getLogger(__name__)

BATCH = 64


def embed_chunk_set(
    db: Session, embedder: Embedder, document_id: str, chunk_config: str
) -> tuple[int, int]:
    """Embed the chunks of one chunk set that have no vector for this model yet.

    Returns (total_chunks, newly_embedded).
    """
    chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id, Chunk.chunk_config == chunk_config)
            .order_by(Chunk.chunk_index)
        )
    )
    already = set(
        db.scalars(
            select(Embedding.chunk_id)
            .join(Chunk, Chunk.id == Embedding.chunk_id)
            .where(
                Chunk.document_id == document_id,
                Chunk.chunk_config == chunk_config,
                Embedding.model_name == embedder.model_name,
            )
        )
    )
    pending = [c for c in chunks if c.id not in already]

    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        vectors = embedder.embed_documents([c.text for c in batch])
        db.add_all(
            Embedding(
                chunk_id=chunk.id,
                model_name=embedder.model_name,
                dimension=embedder.dimension,
                vector=np.asarray(vector, dtype=np.float32).tobytes(),
            )
            for chunk, vector in zip(batch, vectors)
        )
        db.commit()  # keep progress if a later batch fails

    logger.info(
        "embedded document=%s config=%s model=%s new=%d total=%d",
        document_id,
        chunk_config,
        embedder.model_name,
        len(pending),
        len(chunks),
    )
    return len(chunks), len(pending)