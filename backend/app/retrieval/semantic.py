from typing import Literal

import numpy as np
from sqlalchemy import Float, select
from sqlalchemy.orm import Session

from app.embeddings.embedder import Embedder
from app.models import Chunk, Embedding
from app.retrieval.keyword import SearchHit

Method = Literal["sql", "python"]


def _base_filters(stmt, embedder: Embedder, chunk_config: str, document_ids):
    stmt = stmt.join(Embedding, Embedding.chunk_id == Chunk.id).where(
        Chunk.chunk_config == chunk_config,
        Embedding.model_name == embedder.model_name,
        Embedding.dimension == embedder.dimension,
    )
    # None means no filter. An empty list means "match nothing" (fail closed).
    if document_ids is not None:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    return stmt


def semantic_search_python(
    db: Session,
    embedder: Embedder,
    query: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Exact brute-force cosine search in NumPy (the 4A method). Works on any database."""
    stmt = _base_filters(select(Chunk, Embedding), embedder, chunk_config, document_ids)
    rows = db.execute(stmt).all()
    if not rows:
        return []

    matrix = np.vstack([e.vector for _, e in rows])
    scores = matrix @ embedder.embed_query(query)  # unit vectors: dot == cosine
    top = np.argsort(-scores)[:k]
    return [SearchHit(chunk=rows[i][0], score=float(scores[i])) for i in top]


def semantic_search_sql(
    db: Session,
    embedder: Embedder,
    query: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None = None,
) -> list[SearchHit]:
    """Cosine search inside PostgreSQL with pgvector (uses the HNSW index when worthwhile)."""
    query_vector = embedder.embed_query(query).tolist()
    distance = Embedding.vector.op("<=>", return_type=Float)(query_vector)
    stmt = _base_filters(
        select(Chunk, distance.label("distance")), embedder, chunk_config, document_ids
    )
    rows = db.execute(stmt.order_by(distance).limit(k)).all()
    return [SearchHit(chunk=chunk, score=1.0 - float(d)) for chunk, d in rows]


def semantic_search(
    db: Session,
    embedder: Embedder,
    query: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None = None,
    method: Method | None = None,
) -> list[SearchHit]:
    """Use pgvector on PostgreSQL, NumPy elsewhere, unless a method is forced."""
    if method is None:
        method = "sql" if db.get_bind().dialect.name == "postgresql" else "python"
    search = semantic_search_sql if method == "sql" else semantic_search_python
    return search(db, embedder, query, k, chunk_config, document_ids)