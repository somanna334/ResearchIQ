from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from app.database import init_schema

OLD_DOCUMENTS_TABLE = """
CREATE TABLE documents (
    id VARCHAR(36) PRIMARY KEY, filename VARCHAR(255) NOT NULL, size_bytes INTEGER NOT NULL,
    sha256 VARCHAR(64) NOT NULL, status VARCHAR(20) NOT NULL, error_message TEXT,
    page_count INTEGER NOT NULL, pages_with_text INTEGER NOT NULL, created_at DATETIME NOT NULL
)
"""


def test_a_pre_accounts_database_gets_an_owner_column_and_keeps_its_rows():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    with engine.begin() as conn:
        conn.execute(text(OLD_DOCUMENTS_TABLE))
        conn.execute(
            text(
                "INSERT INTO documents VALUES "
                "('d1', 'old.pdf', 1, 'x', 'ready', NULL, 1, 1, '2026-01-01 00:00:00')"
            )
        )

    init_schema(engine)

    assert "owner_id" in {c["name"] for c in inspect(engine).get_columns("documents")}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT owner_id FROM documents WHERE id = 'd1'")).scalar() is None
    init_schema(engine)  # running it again must be harmless