import math
import random
from collections.abc import Sequence
from statistics import fmean


def bootstrap_ci(
    values: Sequence[float], resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, low, high). Percentile bootstrap over the questions; seeded, so repeatable."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(resamples))
    tail = (1 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1 - tail) * resamples))]
    return fmean(values), low, high


def paired_difference_ci(
    a: Sequence[float], b: Sequence[float], resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Interval for mean(a - b) when a[i] and b[i] are the same question."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci([x - y for x, y in zip(a, b)], resamples, confidence, seed)


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]