# Database migrations

ResearchIQ uses Alembic. Application startup applies `alembic upgrade head`.

- New empty database: the initial revision creates the schema and pgvector HNSW index.
- Existing Milestone 9 database: startup stamps revision `0001_initial` only when all six application tables exist and `documents.owner_id` exists, then runs `upgrade head`. Stamping records the existing schema; it does not delete or rewrite rows.
- Partial or older schemas fail closed with a clear error. Back up and plan a dedicated migration before upgrading those databases.

Manual commands (run from `backend/` with the virtual environment active):

```powershell
alembic current
alembic history
alembic upgrade head
```

Create future revisions after changing SQLAlchemy models:

```powershell
alembic revision --autogenerate -m "describe schema change"
```

Review generated migrations before applying them. Never edit an already-applied revision; create a new revision instead.
