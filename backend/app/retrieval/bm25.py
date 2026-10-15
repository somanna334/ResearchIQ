import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the and or of to in on is are was were for with as by at it this that".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop very common words. No stemming."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over a list of already-tokenized documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0

        # term -> [(doc_index, term_frequency), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, freq in Counter(doc).items():
                self.postings[term].append((i, freq))

        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        """One BM25 score per document, in the original order."""
        scores = [0.0] * self.n
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in self.postings[term]:
                length_norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                scores[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
        return scores