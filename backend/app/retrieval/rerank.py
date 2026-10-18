import threading
from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Reranker(Protocol):
    """Scores each passage against the query. Higher means more relevant."""

    model_name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Loads the model lazily, so requests that don't rerank never pay for it."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = self._load()
        scores = model.predict(
            [(query, passage) for passage in passages], show_progress_bar=False
        )
        return [float(s) for s in scores]


@lru_cache
def get_reranker() -> Reranker:
    """FastAPI dependency. Cheap to resolve; the model loads on first use."""
    return CrossEncoderReranker(get_settings().reranker_model)