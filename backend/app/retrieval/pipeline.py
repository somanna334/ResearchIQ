from dataclasses import dataclass, replace
from typing import Literal

from sqlalchemy.orm import Session

from app.embeddings.embedder import Embedder
from app.models import Chunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword import keyword_search
from app.retrieval.rerank import Reranker
from app.retrieval.semantic import semantic_search

Mode = Literal["semantic", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # the score used for final ordering; its meaning depends on the mode
    dense_score: float | None = None  # cosine similarity, if the vector search saw it
    bm25_score: float | None = None  # keyword score, if BM25 matched it
    rerank_score: float | None = None  # cross-encoder score, if reranked


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievedChunk]
    # Best cosine similarity across all vector candidates (None if nothing is embedded).
    best_dense: float | None


def retrieve(
    db: Session,
    embedder: Embedder,
    reranker: Reranker,
    query: str,
    k: int,
    chunk_config: str,
    document_ids: list[str] | None,
    mode: Mode,
    pool: int,
    rrf_k: int,
) -> RetrievalResult:
    if mode == "semantic":
        dense = semantic_search(db, embedder, query, k, chunk_config, document_ids)
        hits = [RetrievedChunk(h.chunk, h.score, dense_score=h.score) for h in dense]
        return RetrievalResult(hits, dense[0].score if dense else None)

    pool = max(pool, k)
    dense = semantic_search(db, embedder, query, pool, chunk_config, document_ids)
    sparse = keyword_search(db, query, pool, chunk_config, document_ids)
    dense_by_id = {h.chunk.id: h for h in dense}
    sparse_by_id = {h.chunk.id: h for h in sparse}
    chunk_by_id = {h.chunk.id: h.chunk for h in [*dense, *sparse]}

    fused = reciprocal_rank_fusion(
        [[h.chunk.id for h in dense], [h.chunk.id for h in sparse]], rrf_k
    )
    order = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
    candidates = [
        RetrievedChunk(
            chunk=chunk_by_id[chunk_id],
            score=fused[chunk_id],
            dense_score=dense_by_id[chunk_id].score if chunk_id in dense_by_id else None,
            bm25_score=sparse_by_id[chunk_id].score if chunk_id in sparse_by_id else None,
        )
        for chunk_id in order
    ]

    if mode == "hybrid_rerank" and candidates:
        scores = reranker.score(query, [c.chunk.text for c in candidates])
        candidates = [
            replace(c, rerank_score=s, score=s) for c, s in zip(candidates, scores)
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)

    best_dense = dense[0].score if dense else None
    return RetrievalResult(candidates[:k], best_dense)