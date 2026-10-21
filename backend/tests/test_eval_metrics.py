import math

import pytest

from app.evaluation.gate import gate_sweep
from app.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from app.evaluation.stats import bootstrap_ci, paired_difference_ci, percentile

A1, A2 = ("a", 1), ("a", 2)
X, Y = ("x", 1), ("y", 1)
RELEVANT = {A1, A2}
RANKED = [X, A1, X, A2, Y]


def test_recall_precision_and_mrr_by_hand():
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert reciprocal_rank(RANKED, RELEVANT, 5) == 0.5
    # Only the first two chunks count at k=2: pages x, a1
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5
    assert precision_at_k(RANKED, RELEVANT, 2) == 0.5


def test_ndcg_matches_hand_calculation():
    dcg = 1 / math.log2(3) + 1 / math.log2(4)  # a1 at page rank 2, a2 at rank 3
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / ideal)


def test_duplicate_chunks_on_one_page_count_once():
    ranked = [("a", 1)] * 3 + [("b", 1)]

    assert recall_at_k(ranked, {("b", 1)}, 3) == 0.0
    assert recall_at_k(ranked, {("b", 1)}, 4) == 1.0
    assert reciprocal_rank(ranked, {("b", 1)}, 4) == 0.5  # page rank 2, not chunk rank 4


def test_empty_relevant_set_raises():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [0, 1, 0, 1, 1, 1, 0, 1]
    first = bootstrap_ci(values)

    assert first == bootstrap_ci(values)
    mean, low, high = first
    assert mean == pytest.approx(0.625)
    assert low <= mean <= high


def test_bootstrap_of_constant_values_collapses():
    assert bootstrap_ci([0.5] * 10) == pytest.approx((0.5, 0.5, 0.5))


def test_paired_difference_uses_per_question_gaps():
    mean, _, _ = paired_difference_ci([1, 1, 0, 1], [0, 1, 0, 0])

    assert mean == pytest.approx(0.5)


def test_percentile_nearest_rank():
    values = list(range(1, 11))

    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile(values, 10) == 1


def test_gate_sweep_counts_both_error_types():
    rows = gate_sweep([0.6, 0.7], [0.5, 0.62], [0.55, 0.65])

    assert (rows[0].false_refuse, rows[0].false_accept) == (0, 1)  # 0.62 slips through
    assert (rows[1].false_refuse, rows[1].false_accept) == (1, 0)  # 0.6 is refused