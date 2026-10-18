import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_matches_hand_calculation():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)

    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert sorted(scores, key=scores.get, reverse=True) == ["b", "c", "a", "d"]


def test_rrf_with_empty_rankings_is_empty():
    assert reciprocal_rank_fusion([[], []]) == {}