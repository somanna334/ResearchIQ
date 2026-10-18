import os

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app import models  # noqa: F401  (registers tables)
from app.database import Base, init_schema
from app.models import EMBEDDING_DIM, Chunk, Document, Embedding
from app.retrieval.semantic import semantic_search_python, semantic_search_sql

TEST_URL = os.environ.get("RESEARCHIQ_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_URL, reason="set RESEARCHIQ_TEST_DATABASE_URL to run PostgreSQL tests"
)


class StubEmbedder:
    model_name = "stub"
    dimension = EMBEDDING_DIM

    def __init__(self, query_vector: np.ndarray) -> None:
        self._query_vector = query_vector

    def embed_query(self, text: str) -> np.ndarray:
        return self._query_vector


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


@pytest.fixture
def pg_session():
    engine = create_engine(TEST_URL)
    init_schema(engine)
    Base.metadata.drop_all(engine)
    init_schema(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_pgvector_ranking_matches_numpy_brute_force(pg_session):
    rng = np.random.default_rng(0)
    document = Document(filename="t.pdf", size_bytes=1, sha256="0" * 64)
    pg_session.add(document)
    pg_session.flush()
    pg_session.add_all(
        Chunk(
            document_id=document.id,
            chunk_config="t",
            chunk_index=i,
            page_number=1,
            text=f"chunk {i}",
            char_count=7,
            embeddings=[
                Embedding(
                    model_name="stub",
                    dimension=EMBEDDING_DIM,
                    vector=_unit(rng.normal(size=EMBEDDING_DIM)),
                )
            ],
        )
        for i in range(40)
    )
    pg_session.commit()

    embedder = StubEmbedder(_unit(rng.normal(size=EMBEDDING_DIM)))
    sql_hits = semantic_search_sql(pg_session, embedder, "q", 5, "t")
    py_hits = semantic_search_python(pg_session, embedder, "q", 5, "t")

    assert [h.chunk.id for h in sql_hits] == [h.chunk.id for h in py_hits]
    for a, b in zip(sql_hits, py_hits):
        assert a.score == pytest.approx(b.score, abs=1e-4)