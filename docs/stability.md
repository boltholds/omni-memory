# Stability policy

OmniMemory currently has two layers: stable core and experimental extensions.

## Stable core

The stable core is the part that should remain backward-compatible between minor releases:

```text
domain/models.py
domain/writeback.py
domain/operations.py
app/memory.py
app/writeback/service.py
app/writeback/memory_policies.py
app/writeback/writeback_policies.py
app/retriever.py
app/context_builder.py
app/memory_planner.py
app/api_v1.py
benchmarks/memory_eval/
infra/db/
```

Stable does not mean frozen. It means changes should be intentional, documented and covered by tests.

## Experimental extensions

The experimental layer is allowed to change quickly:

```text
app/mcp_server.py
app/integrations/mcp.py
app/agent_cycle.py
app/services/answer_chain.py
app/fact_maintenance.py
```

These modules can be used for demos and research, but should not be treated as public API yet.

## Why not move files immediately?

The current goal is to stabilize behavior before a release. Physically moving modules into an `experimental/` package would create import churn and may break existing tests. The boundary is documented first; package restructuring can happen in a later cleanup release.

## Rule of thumb

```text
If a module is required for remember/search/context/writeback, it belongs to stable core.
If a module adds an integration, workflow, agent loop, or research feature, it belongs to experimental.
```
