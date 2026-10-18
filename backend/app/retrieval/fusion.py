from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], k: int = 60
) -> dict[T, float]:
    """Merge ranked lists using ranks only: score = sum of 1 / (k + rank).

    Items keep first-seen order in the returned dict, so a stable sort by score
    breaks ties in favour of the earlier list.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores