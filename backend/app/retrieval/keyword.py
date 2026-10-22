from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.retrieval.bm25 import BM25Index, tokenize


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def keyword_search(
    db: Session,
    query: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None = None,
) -> list[SearchHit]:
    """BM25 search over one chunk set. The index is rebuilt on every call (MVP).

    document_ids=None means no filter. An empty list means "match nothing":
    it must never fall back to searching everything.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    stmt = select(Chunk).where(Chunk.chunk_config == chunk_config)
    if document_ids is not None:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    chunks = list(db.scalars(stmt))
    if not chunks:
        return []

    index = BM25Index([tokenize(c.text) for c in chunks])
    scores = index.score(query_tokens)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [SearchHit(chunk, score) for chunk, score in ranked[:k] if score > 0]