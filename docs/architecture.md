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
domain/models.py          typed memory records, MemoryScope and DomainGraph records
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
infra/repo/               in-memory repositories for facts, decisions, experience, domains, skills and failure patterns
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
Fact                  current/historical structured knowledge
MemoryObject          semantic notes and generic memories
Episode               session or event-level memories
DecisionRecord        decisions, alternatives and consequences
ExperienceRecord      goal/action/outcome/lesson reusable experience
SkillRecord           reusable procedure promoted from repeated successful experience
FailurePatternRecord  symptom/cause/fix/detection pattern promoted from repeated failures
```

## Memory hygiene and domain graph

Every memory object can carry canonical scope metadata in `meta.scope`:

```text
tenant_id
agent_id
domain_ids
environment: prod | dev | test | benchmark | sandbox
durability: durable | ephemeral | session
visibility: private | shared | global
exclude_from_consolidation
```

Projects are not stored as a separate `project_id`. A project is a `DomainNode(kind="project")` in the domain graph, and memory records refer to project/area/environment nodes through `domain_ids`.

The first hygiene policy is conservative:

```text
source=test / pytest       -> environment=test, durability=ephemeral, exclude_from_consolidation=true
source=benchmark / eval    -> environment=benchmark, durability=ephemeral, exclude_from_consolidation=true
durable private memory with no domain_ids -> scope warning, not rejection
```

This keeps test fixtures and benchmark artifacts out of long-term consolidation while avoiding a hard breaking change for existing write paths.

## Scope-aware retrieval

Domain graph is used as a ranking signal by default:

```text
query text
  -> infer matching DomainNode ids by id/name/aliases
  -> expand through DomainGraphRepo.reachable_domain_ids(...)
  -> boost memories whose meta.scope.domain_ids overlap
  -> downrank test / benchmark / sandbox / ephemeral / session memories
```

Callers can also pass an optional `scope` filter to `OmniMemory.retrieve`, `/v1/memories/search` and `/v1/context`:

```json
{
  "domain_ids": ["domain:project:omni-memory"],
  "strict_domains": true,
  "environments": ["dev"],
  "durabilities": ["durable"],
  "memory_types": ["skill", "failure_pattern"],
  "include_ephemeral": false,
  "expand_domains": true
}
```

Filter behavior:

```text
domain_ids + strict_domains=true -> hard keep only matching domain_ids
domain_ids + strict_domains=false -> boost matching domains and related domains
environments                    -> hard filter by meta.scope.environment
durabilities                    -> hard filter by meta.scope.durability
memory_types                    -> disable unrequested retrieval channels
include_ephemeral=false         -> hide ephemeral/session memories
```

This means a query about `OmniMemory dependency issue` can prefer OmniMemory-scoped skills over Persona-scoped skills even if both match the lexical query. If no domain is detected and no scope is provided, retrieval still works like ordinary memory retrieval, with only hygiene penalties applied.

## Production-like fact mining

Fact mining is not a regex write path. It is a gated candidate pipeline:

```text
source text
  -> FactExtractor port
  -> LLMFactExtractor JSON schema extraction
  -> candidate schema validation
  -> evidence quote grounding
  -> confidence / PII / secret / uncertainty checks
  -> WriteBackService policy dry-run
  -> candidate status: policy_accepted | requires_review | policy_rejected | saved
```

The default production extractor is `LLMFactExtractor`, which uses the configured BYO-LLM provider and asks for strict JSON candidates. Tests and offline integrations can inject `StaticFactExtractor`, but the core service does not depend on deterministic regex mining.

Fact mining is dry-run by default:

```text
OmniMemory.mine_facts(..., dry_run=True)
POST /v1/facts/mine { "dry_run": true }
```

Dry-run still runs conversion and write policies, including provenance, scope, TTL, PII, conflict, confidence and dedup checks. It returns auditable operations and policy decisions, but does not save to repositories. Applying candidates requires an explicit `dry_run=false` call.

Conflict handling defaults to `policy_mode=review`, so conflicting mined facts become review candidates instead of silently overwriting durable memory.

## Policy-first writeback

Every write passes through:

```text
conversion policy
provenance + scope policy
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

## Cognitive memory loop

The practical learning loop is:

```text
development action
  -> ExperienceRecord
  -> consolidation
  -> SkillRecord / FailurePatternRecord
  -> retrieval by intent
  -> better next action
```

Consolidation ignores memories marked as `exclude_from_consolidation=true`, as well as `test`, `benchmark`, `sandbox`, `ephemeral` and `session` memories. This prevents test fixtures and temporary runtime artifacts from becoming durable skills or failure patterns.

## Intent-aware retrieval

`MemoryPlanner` controls which memory channels are used for each intent:

```text
answer_question -> facts, beliefs, conflicts, semantic notes
make_decision   -> decision records, relevant experience
fail/debug      -> relevant experience, failure patterns, skills
plan_task        -> beliefs, facts, decisions, experience, notes
write_code       -> decisions, experience, skills, semantic notes
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
