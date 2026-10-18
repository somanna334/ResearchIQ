from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import get_settings


class Embedder(Protocol):
    """Anything that turns text into unit-length vectors."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int, query_prefix: str) -> None:
        # Imported here so the app (and tests) start without loading PyTorch.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._model = SentenceTransformer(model_name)
        # The method name changed across sentence-transformers versions.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dimension = int(get_dim())

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # unit length: dot product == cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]


@lru_cache
def get_embedder() -> Embedder:
    """FastAPI dependency. The model loads once, on first use."""
    settings = get_settings()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_batch_size,
        settings.embedding_query_prefix,
    )