import pytest

from app.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cat and the HAT!") == ["cat", "hat"]


def test_idf_matches_hand_calculation():
    index = BM25Index([["cat", "zebra"], ["cat"], ["cat"]])

    # ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) and ln(1 + (3 - 3 + 0.5) / (3 + 0.5))
    assert index.idf["zebra"] == pytest.approx(0.981, abs=0.001)
    assert index.idf["cat"] == pytest.approx(0.1335, abs=0.001)


def test_rare_term_only_matches_documents_containing_it():
    index = BM25Index([tokenize(t) for t in ["cat dog", "cat bird", "cat fish zebra"]])
    scores = index.score(tokenize("zebra"))

    assert scores[2] > 0
    assert scores[0] == 0 and scores[1] == 0


def test_shorter_document_scores_higher_for_same_term_frequency():
    index = BM25Index([["cat", "x1"], ["cat", "x2", "x3", "x4", "x5", "x6"]])
    scores = index.score(["cat"])

    assert scores[0] > scores[1]


def test_empty_query_scores_zero():
    index = BM25Index([["cat"], ["dog"]])

    assert index.score([]) == [0.0, 0.0]