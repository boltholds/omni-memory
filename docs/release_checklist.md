# Release checklist

Use this checklist before merging `dev` into `main`.

## 1. Local checks

```bash
poetry install
poetry run pytest -q
poetry run python benchmarks/memory_eval/run_benchmark.py --provider fake
poetry run python benchmarks/memory_eval/report.py
```

## 2. SQLite persistence smoke test

PowerShell:

```powershell
$env:MEMORY_DATABASE_URL="sqlite:///data/omni_memory.db"
$env:MEMORY_AUDIT_ENABLED="true"
$env:MEMORY_AUDIT_AUTO_CREATE="true"
$env:LLM_PROVIDER="none"
$env:EMBEDDING_BACKEND="hash"
$env:NER_BACKEND="regex"
poetry run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Then run the demo from `docs/demo_end_to_end.md`.

## 3. API smoke checks

```text
POST /v1/memories/remember
POST /v1/memories/search
POST /v1/context
GET  /v1/memories
GET  /v1/audit/operations
GET  /v1/audit/decisions
GET  /v1/audit/reviews
```

Expected minimum:

```text
remember returns saved/rejected/errors
remember returns policy_decisions and operations
remember returns audit_persistence block
read endpoints do not crash when persistence is disabled
read endpoints return rows when persistence is enabled
```

## 4. Postgres migration smoke check

```bash
export MEMORY_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME
poetry run alembic upgrade head
```

## 5. Documentation check

Confirm these files are current:

```text
README.md
docs/architecture.md
docs/memory_persistence.md
docs/demo_end_to_end.md
docs/release_checklist.md
```

## 6. Stable vs experimental boundary

Stable surface:

```text
domain/models.py
domain/writeback.py
domain/operations.py
app/memory.py
app/writeback/service.py
app/retriever.py
app/context_builder.py
app/memory_planner.py
app/api_v1.py
benchmarks/memory_eval/
infra/db/
```

Experimental surface:

```text
app/mcp_server.py
app/integrations/mcp.py
app/agent_cycle.py
app/services/answer_chain.py
app/fact_maintenance.py
```

Do not promise experimental APIs as stable in release notes.

## 7. Merge

Recommended GitHub flow:

```text
open PR: dev -> main
wait for CI
review changed files
merge PR
create tag: v0.2.0-memory-governance
```

Suggested release title:

```text
v0.2.0 - Policy-first memory governance
```

Suggested release summary:

```text
Adds auditable writeback, conflict policy modes, decision and experience memory, intent-aware retrieval, benchmark diagnostics, and optional SQL persistence for memory governance.
```
