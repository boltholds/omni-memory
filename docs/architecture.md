# OmniMemory architecture

OmniMemory is a policy-first long-term memory layer for LLM agents.

## Core flow

```text
interaction / event / command
  -> writeback conversion
  -> write policies
  -> typed memory repositories
  -> retrieval planner
  -> retrieval bundle
  -> context builder
  -> LLM / agent
```

## Stable core modules

These modules form the stable architecture surface:

```text
domain/models.py          typed memory records: Fact, Episode, DecisionRecord, ExperienceRecord
domain/writeback.py       writeback request/result/decision contracts
domain/operations.py      operation and policy-decision audit envelopes
app/memory.py             public OmniMemory facade
app/writeback/service.py  policy-driven writeback pipeline
app/retriever.py          semantic, graph, episodic, decision and experience retrieval
app/context_builder.py    structured context assembly
app/memory_planner.py     intent-aware retrieval/context profiles
app/api_v1.py             product-facing HTTP API
```

## Product layer

```text
benchmarks/memory_eval/   memory-vs-no-memory benchmark
infra/db/                 optional SQL audit persistence
docs/                     architecture, persistence and demo docs
migrations/               Alembic migrations for audit persistence
```

## Experimental layer

These modules are useful, but should be treated as experimental until the API stabilizes:

```text
app/mcp_server.py
app/integrations/mcp.py
app/agent_cycle.py
app/services/answer_chain.py
app/fact_maintenance.py
```

They should remain behind explicit entrypoints and should not be required for the minimal memory core.

## Memory types

```text
Fact              current/historical structured knowledge
MemoryObject      semantic notes and generic memories
Episode           session or event-level memories
DecisionRecord    decisions, alternatives and consequences
ExperienceRecord  goal/action/outcome/lesson reusable experience
```

## Policy-first writeback

Every write passes through:

```text
conversion policy
provenance policy
TTL policy
PII policy
conflict policy
confidence policy
dedup policy
repository router
```

The writeback result includes both saved/rejected objects and an audit trail:

```text
policy_decisions
operations
```

## Intent-aware retrieval

`MemoryPlanner` controls which memory channels are used for each intent:

```text
answer_question -> facts, beliefs, conflicts, semantic notes
make_decision   -> decision records, relevant experience
debug_failure   -> relevant experience
plan_task        -> beliefs, facts, decisions, experience, notes
write_code       -> decisions, experience, semantic notes
```

## Persistence

Persistence is optional. When enabled, `/v1/memories/remember` stores the writeback result into SQL:

```text
memory_records
memory_operations
policy_decisions
review_candidates
```

See `docs/memory_persistence.md` for setup.
