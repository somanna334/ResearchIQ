from collections.abc import Sequence
from dataclasses import dataclass

# 0.40, 0.42, ... 0.80
DEFAULT_THRESHOLDS = tuple(round(0.40 + 0.02 * i, 2) for i in range(21))


@dataclass(frozen=True)
class GateRow:
    threshold: float
    false_refuse: int  # answerable questions refused (best score below threshold)
    answerable: int
    false_accept: int  # unanswerable questions let through (best score at or above)
    negatives: int


def gate_sweep(
    answerable_scores: Sequence[float],
    negative_scores: Sequence[float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[GateRow]:
    """Best dense similarity per question -> error counts at each threshold."""
    return [
        GateRow(
            threshold=t,
            false_refuse=sum(1 for s in answerable_scores if s < t),
            answerable=len(answerable_scores),
            false_accept=sum(1 for s in negative_scores if s >= t),
            negatives=len(negative_scores),
        )
        for t in thresholds
    ]