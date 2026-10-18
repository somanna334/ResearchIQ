import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import LargeBinary
from sqlalchemy.types import TypeDecorator


class VectorType(TypeDecorator):
    """A float32 vector: native pgvector on PostgreSQL, raw bytes elsewhere (SQLite).

    Python always sees a NumPy float32 array. The bytes format matches what
    Milestone 4A stored, so existing SQLite data stays readable.
    """

    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        return array if dialect.name == "postgresql" else array.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return np.asarray(value, dtype=np.float32)
        return np.frombuffer(value, dtype=np.float32)