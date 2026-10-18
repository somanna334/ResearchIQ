import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.embedder import Embedder
from app.models import Chunk, Embedding
from app.retrieval.keyword import SearchHit


def semantic_search(
    db: Session,
    embedder: Embedder,
    query: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Brute-force cosine search. Vectors are unit length, so dot product == cosine."""
    stmt = (
        select(Chunk, Embedding)
        .join(Embedding, Embedding.chunk_id == Chunk.id)
        .where(
            Chunk.chunk_config == chunk_config,
            Embedding.model_name == embedder.model_name,
            Embedding.dimension == embedder.dimension,
        )
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    rows = db.execute(stmt).all()
    if not rows:
        return []

    matrix = np.vstack([np.frombuffer(e.vector, dtype=np.float32) for _, e in rows])
    scores = matrix @ embedder.embed_query(query)  # one similarity per chunk
    top = np.argsort(-scores)[:k]
    return [SearchHit(chunk=rows[i][0], score=float(scores[i])) for i in top]