# End-to-end demo

This demo shows OmniMemory as a product memory layer, not just vector search.

It covers:

```text
remember fact
inspect writeback audit
reject duplicate fact
reject or review conflict
search memory
build context
record experience
retrieve by intent
persist audit data
```

## 1. Start with optional SQLite audit persistence

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

## 2. Remember a fact

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "source": "demo",
    "policy_mode": "permissive",
    "items": [
      {
        "type": "fact",
        "subject": "project",
        "predicate": "backend_framework",
        "object": "FastAPI",
        "meta": {"confidence": 1.0}
      }
    ]
  }'
```

Expected behavior:

```text
saved: 1
rejected: 0
policy_decisions include conversion, provenance, ttl, pii, conflict, confidence, dedup, repository save
audit_persistence.persisted: true, if persistence is enabled
```

## 3. Try to save the same fact again

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "source": "demo",
    "policy_mode": "permissive",
    "items": [
      {
        "type": "fact",
        "subject": "project",
        "predicate": "backend_framework",
        "object": "FastAPI",
        "meta": {"confidence": 1.0}
      }
    ]
  }'
```

Expected behavior:

```text
saved: 0
rejected reason: duplicate_fact
rejected policy: dedup
```

## 4. Try a conflicting fact in strict mode

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "source": "demo",
    "policy_mode": "strict",
    "items": [
      {
        "type": "fact",
        "subject": "project",
        "predicate": "backend_framework",
        "object": "Django",
        "meta": {"confidence": 1.0}
      }
    ]
  }'
```

Expected behavior:

```text
saved: 0
rejected reason: fact_conflict
rejected policy: conflict
```

## 5. Try a conflicting fact in review mode

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "source": "demo",
    "policy_mode": "review",
    "items": [
      {
        "type": "fact",
        "subject": "project",
        "predicate": "backend_framework",
        "object": "Flask",
        "meta": {"confidence": 1.0}
      }
    ]
  }'
```

Expected behavior:

```text
saved: 0
rejected reason: requires_review
rejected policy: conflict
```

If audit persistence is enabled, the rejected item is stored as a future Memory Inbox candidate.

## 6. Search memory

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"q": "What backend framework does the project use?", "intent": "answer_question"}'
```

Expected behavior:

```text
facts include project backend_framework FastAPI
beliefs include current belief for project.backend_framework
```

## 7. Build context

```bash
curl -X POST http://127.0.0.1:8000/v1/context \
  -H "Content-Type: application/json" \
  -d '{"q": "What backend framework does the project use?", "intent": "answer_question"}'
```

Expected behavior:

```text
context.sections include Current Beliefs and Facts
```

## 8. Inspect persistent audit records

```bash
curl http://127.0.0.1:8000/v1/audit/operations
curl http://127.0.0.1:8000/v1/audit/decisions
curl http://127.0.0.1:8000/v1/audit/reviews
curl http://127.0.0.1:8000/v1/memories
```

If persistence is disabled, these endpoints return an empty list with audit_persistence.enabled=false.

## 9. Run benchmark smoke test

```bash
poetry run python benchmarks/memory_eval/run_benchmark.py --provider fake
poetry run python benchmarks/memory_eval/report.py
```

The benchmark should report memory lift, context score, privacy violations, write failures and answer failures where context was already correct.
