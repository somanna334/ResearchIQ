import math
from collections.abc import Sequence

Page = tuple[str, int]  # (document filename, page number)

KS = (1, 3, 5, 10)


def _ranking(chunk_pages: Sequence[Page], k: int) -> list[Page]:
    """Distinct pages, in order of first appearance, from the top-k chunks."""
    seen: set[Page] = set()
    ranking: list[Page] = []
    for page in chunk_pages[:k]:
        if page not in seen:
            seen.add(page)
            ranking.append(page)
    return ranking


def _check(relevant: set[Page] | frozenset[Page], k: int) -> None:
    if not relevant:
        raise ValueError("relevant set must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")


def recall_at_k(chunk_pages: Sequence[Page], relevant: set[Page] | frozenset[Page], k: int) -> float:
    _check(relevant, k)
    return len(set(_ranking(chunk_pages, k)) & relevant) / len(relevant)


def precision_at_k(chunk_pages: Sequence[Page], relevant: set[Page] | frozenset[Page], k: int) -> float:
    """Share of the top-k chunks that sit on a relevant page (divides by k even if fewer came back)."""
    _check(relevant, k)
    return sum(1 for page in chunk_pages[:k] if page in relevant) / k


def reciprocal_rank(chunk_pages: Sequence[Page], relevant: set[Page] | frozenset[Page], k: int) -> float:
    _check(relevant, k)
    for rank, page in enumerate(_ranking(chunk_pages, k), start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunk_pages: Sequence[Page], relevant: set[Page] | frozenset[Page], k: int) -> float:
    """Binary-gain nDCG over distinct pages."""
    _check(relevant, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, page in enumerate(_ranking(chunk_pages, k), start=1)
        if page in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def all_metrics(chunk_pages: Sequence[Page], relevant: set[Page] | frozenset[Page]) -> dict[str, float]:
    """Every metric at every K in KS, plus MRR over the deepest list."""
    out: dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = recall_at_k(chunk_pages, relevant, k)
        out[f"precision@{k}"] = precision_at_k(chunk_pages, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(chunk_pages, relevant, k)
    out["mrr"] = reciprocal_rank(chunk_pages, relevant, max(KS))
    return out