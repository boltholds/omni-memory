# Memory audit persistence

OmniMemory can optionally persist writeback audit data to SQL. This is disabled by default, so local in-memory development works without a database.

## Local SQLite smoke test

Set these environment variables before starting the API:

```text
MEMORY_DATABASE_URL=sqlite:///data/omni_memory.db
MEMORY_AUDIT_ENABLED=true
MEMORY_AUDIT_AUTO_CREATE=true
```

Then start the server:

```bash
poetry run uvicorn omni_memory.main:app --reload --host 127.0.0.1 --port 8000
```

`/v1/memories/remember` will still return the normal writeback result. When persistence is enabled, it also adds:

```json
{
  "audit_persistence": {
    "enabled": true,
    "configured": true,
    "auto_create": true,
    "persisted": true,
    "error": null
  }
}
```

## Postgres

Use a SQLAlchemy URL compatible with psycopg, then run migrations:

```text
MEMORY_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME
MEMORY_AUDIT_ENABLED=true
```

```bash
poetry run alembic upgrade head
```

## Tables

```text
memory_records       -> saved memory objects
memory_operations    -> remember operation envelopes
policy_decisions     -> conversion, write-policy and repository decisions
review_candidates    -> rejected requires_review candidates for future Memory Inbox
```

This is the first persistence step toward a product dashboard and governance UI.
